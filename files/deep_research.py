#!/usr/bin/env python3
"""
Deep Research Pipeline -- Agentic Book Generator
=================================================
Stage 1 of the Agentic Deep Research roadmap: section-by-section book generation
with prior-section memory, optional LLM-as-judge review, and structural sanitization.
Future stages add retrieval, planning, and re-search loops (see WORKPLAN.md).

Current implementation:
  - 12 chapters x N passes (atomic LLM calls, ~1,300 words each)
  - Prior-section tail + chapter-so-far titles fed forward as continuity context
  - Optional --review pass with regenerate-once on low scores
  - Sanitization strips model-hallucinated H1/H2 / References / Conclusion blocks
  - Auto-resume from checkpoint, macOS notification, PDF render
"""
import json, os, re, sys, time, signal, argparse, subprocess, threading, tempfile, atexit, errno
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import fcntl  # POSIX only; we degrade to no-op locks on Windows
    _HAVE_FCNTL = True
except ImportError:
    _HAVE_FCNTL = False

# Stage 2 -- agentic research layer. Optional: pipeline degrades to Stage 1 if
# the package fails to import (e.g. missing httpx).
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import research as _research
    RESEARCH_AVAILABLE = True
except Exception as _research_import_err:
    _research = None
    RESEARCH_AVAILABLE = False
    print(f"[deep_research] research layer unavailable: {_research_import_err}", flush=True)

# === CONFIG ===
OLLAMA_BASE  = "http://localhost:11434"
# Writer model: override via env DEEP_RESEARCH_WRITER_MODEL.
# Recommended upgrades for better synthesis:
#   - qwen2.5:7b  (~4.7 GB, better instruction-following)
#   - qwen2.5:14b (~9 GB,   noticeably stronger synthesis if RAM allows)
#   - mixtral:8x7b (~26 GB, only if you have the headroom)
MODEL = os.environ.get("DEEP_RESEARCH_WRITER_MODEL", "gemma3:4b")
DEFAULT_TIMEOUT = 600

BASE_DIR  = Path(__file__).parent
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

STATE_FILE   = OUT_DIR / "state.json"
FINAL_MD     = OUT_DIR / "book.md"
FINAL_HTML   = OUT_DIR / "book.html"
FINAL_PDF    = OUT_DIR / "book.pdf"
CLEAN_MD     = OUT_DIR / "book.clean.md"
REPORT_FILE  = OUT_DIR / "report.json"
LOG_FILE     = OUT_DIR / "pipeline.log"


def _rebind_output_paths(out_name: str):
    """Repoint every output path so a single run can produce its own family of
    artifacts (book1.md / book1.pdf / book1.state.json / ...). Idempotent.

    Mirror this convention in runner.py via the DEEP_RESEARCH_OUT_NAME env var.
    """
    global STATE_FILE, FINAL_MD, FINAL_HTML, FINAL_PDF, CLEAN_MD, REPORT_FILE, LOG_FILE
    name = out_name.strip().strip("/")
    if not name:
        return
    STATE_FILE  = OUT_DIR / f"{name}.state.json"
    FINAL_MD    = OUT_DIR / f"{name}.md"
    FINAL_HTML  = OUT_DIR / f"{name}.html"
    FINAL_PDF   = OUT_DIR / f"{name}.pdf"
    CLEAN_MD    = OUT_DIR / f"{name}.clean.md"
    REPORT_FILE = OUT_DIR / f"{name}.report.json"
    LOG_FILE    = OUT_DIR / f"{name}.pipeline.log"

# Legacy filenames migrated to the new structure on first run.
_LEGACY_MAP = {
    "llm400_state.json":  STATE_FILE,
    "llm400_report.json": REPORT_FILE,
    "llm400_book.md":     FINAL_MD,
    "llm400_book.html":   FINAL_HTML,
    "llm400_book.pdf":    FINAL_PDF,
    "llm400_clean.md":    CLEAN_MD,
    "llm400.log":         LOG_FILE,
}


def migrate_legacy_outputs():
    """One-shot rename of any pre-existing llm400_* artifacts to the new neutral names.

    Safe to call repeatedly. Does NOT overwrite an existing new-name file; if both exist
    the legacy copy is left for the user to resolve manually.
    """
    moved = []
    for old, new in _LEGACY_MAP.items():
        old_p = OUT_DIR / old
        if old_p.exists() and not new.exists():
            old_p.rename(new)
            moved.append(f"{old} -> {new.name}")
    if moved:
        print("[MIGRATE] renamed legacy outputs: " + ", ".join(moved))


# W1: target length emerges from evidence count, not a static budget. The old
# `WORD_BUDGET = 4200` + "Target: 1800-2500 words" prompt pressure caused 4B
# writers to loop, hallucinate, and drop citations to game the verifier.
# `WORD_BUDGET` is now a *ceiling* (the upper cap on the dynamic target), not
# a target. The dict `"w"` field on each section is kept for backward compat
# but is ignored at runtime in favor of compute_target_words().
WORD_BUDGET = 1500              # ceiling -- never ask the writer for more
WORD_TARGET_PER_SOURCE = 220    # per retained evidence source
WORD_TARGET_FLOOR = 400         # minimum target so empty results still produce a stub
WORD_TARGET_NO_EVIDENCE = 900   # fallback when research layer is off


# === CHAPTERS: 12 chapters × 6 passes = 72 sections ===
CHAPTERS = [
  {"n": 1,  "t": "Introduction to Large Language Models", "passes": [
    {"p": 1, "t": "History and Evolution",           "w": 4200, "pr": "Write a comprehensive section on the HISTORY AND EVOLUTION OF LARGE LANGUAGE MODELS (LLMs). Cover: Statistical language models (N-gram, HMM), Neural language models (RNN, LSTM, GRU), The Transformer revolution 2017 (Attention is All You Need), Pre-training era (ELMo, GPT-1, BERT, GPT-2, GPT-3), Scaling laws (Kaplan, Chinchilla), The LLM era (PaLM, LLaMA, GPT-4, Claude, Gemini, Mistral). Include key paper citations with year and authors. Target: 1800-2500 words with technical depth."},
    {"p": 2, "t": "Mathematical Foundations",         "w": 4200, "pr": "Write a deep technical section on MATHEMATICAL FOUNDATIONS FOR LLMs. Cover: Maximum Likelihood Estimation (MLE) formula and derivation, Cross-entropy loss with derivation, Perplexity as model evaluation metric, Chain rule of probability for sequence modeling, Information theory basics (entropy, cross-entropy, KL divergence with formulas), Neural language model formulation with matrix notation, Backpropagation through time (BPTT) for RNNs. Include Python code for MLE and perplexity calculation. Target: 1800-2500 words."},
    {"p": 3, "t": "Pre-training Paradigm",            "w": 4200, "pr": "Write a section on THE PRE-TRAINING PARADIGM. Cover: Word embeddings (Word2Vec CBOW, Skip-gram, GloVe), Contextual embeddings (ELMo, BERT), Pre-training objectives: Next Sentence Prediction (NSP), Masked Language Modeling (MLM), Causal Language Modeling (CLM), PrefixLM and encoder-decoder T5, Contrastive learning (SimCSE, SentenceBERT), Scaling laws and compute-optimal training (Chinchilla). Target: 1800-2500 words with formulas."},
    {"p": 4, "t": "The LLM Ecosystem",                "w": 4200, "pr": "Write an overview section on THE LLM ECOSYSTEM AND LANDSCAPE. Cover: Open-source vs closed-source models, Major model families (GPT, Claude, Gemini, LLaMA, Mistral, Gemma, Qwen, DeepSeek), Commercial API providers (OpenAI, Anthropic, Google, Cohere), Open-source ecosystem (HuggingFace, llama.cpp, Ollama, vLLM, TGI), Model benchmarks (MMLU, HELM, BIG-bench, GSM8K, HumanEval), Evaluation protocols, Specialized vs general-purpose LLMs. Target: 1800-2500 words."},
    {"p": 5, "t": "LLM Capabilities and Limitations", "w": 4200, "pr": "Write a section on LLM CAPABILITIES AND LIMITATIONS. Cover: What LLMs do well (text generation, summarization, translation, reasoning), Emergent capabilities at scale (few-shot, chain-of-thought), Hallucination: causes, examples, and mitigation strategies (RAG, grounding, constitutional AI), Context window limitations and attention bottlenecks, Compositionality and systematic generalization failures, Benchmark saturation and the gap between benchmarks and real-world performance. Target: 1800-2500 words."},
    {"p": 6, "t": "Societal Impact and Responsible AI", "w": 4200, "pr": "Write a section on SOCIETAL IMPACT AND RESPONSIBLE AI DEVELOPMENT. Cover: Economic impact on knowledge work and automation, Labor market transformation and skills gap, Education and research applications, Privacy concerns with training data, Energy consumption and environmental impact (carbon footprint of LLM training), EU AI Act risk-based framework, US Executive Order on AI, International coordination (OECD), Open-source vs closed-source safety debate. Target: 1800-2500 words."},
    {"p": 7, "t": "Architecture Fundamentals and Attention Mechanism", "w": 4200, "pr": "Write a section on ARCHITECTURE FUNDAMENTALS. Cover: Encoder-decoder architecture overview, Attention mechanism intuition and motivation, Scaled dot-product attention formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, Query-Key-Value projection matrices, Why scale by sqrt(d_k), Multi-head attention (MHA), Layer normalization, Residual connections. Include PyTorch attention code. Target: 1800-2500 words."},
    {"p": 8, "t": "Practical LLM Usage and Prompt Engineering", "w": 4200, "pr": "Write a section on PRACTICAL LLM USAGE. Cover: How to interact with LLMs via API (OpenAI, Anthropic, local), Prompt engineering fundamentals (zero-shot, few-shot, CoT), Temperature, top-k, top-p sampling strategies, Token counting and cost estimation, Handling long contexts and truncation, Streaming vs non-streaming, Best practices for production deployments, Rate limiting and error handling. Include Python API code examples. Target: 1800-2500 words."},
  ]},
  {"n": 2,  "t": "The Transformer Architecture", "passes": [
    {"p": 1, "t": "Self-Attention Mechanism",          "w": 4200, "pr": "Write a deep technical section on SELF-ATTENTION MECHANISM. Cover: Scaled dot-product attention formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, Query-Key-Value projection matrices, Why scale by sqrt(d_k) (gradient stability), Multi-head attention (MHA) with full formula, Computational complexity O(n^2*d) and the long-sequence bottleneck, Flash Attention v1/v2 algorithm and memory efficiency. Include complete PyTorch code for scaled dot-product attention with multi-head support. Target: 1800-2500 words with formulas and code."},
    {"p": 2, "t": "Positional Encoding",              "w": 4200, "pr": "Write a technical section on POSITIONAL ENCODING. Cover: Why positional encoding is needed in Transformers (permutation invariance problem), Sinusoidal positional encoding (Vaswani et al.) with exact formulas, Learnable positional embeddings, Relative positional encoding (Shaw et al.), RoPE (Rotary Position Embedding) from Su et al. with rotation matrix derivation, ALiBi (Attention with Linear Biases), Comparison of PE methods. Include PyTorch code for RoPE implementation. Target: 1800-2500 words."},
    {"p": 3, "t": "FFN, Residual Connections, LayerNorm", "w": 4200, "pr": "Write a technical section on TRANSFORMER COMPONENTS. Cover: Feed-Forward Network (FFN) formula with GELU activation: FFN(x) = GELU(xW_1 + b_1)W_2 + b_2, GELU activation function and its Gaussian CDF approximation, Residual connections (skip connections) and gradient flow mathematics, Layer Normalization formula: LayerNorm(x) = gamma*(x-mu)/sqrt(sigma^2+epsilon) + beta, Pre-norm vs post-norm Transformers (Pre-LN advantages), Embedding layer, Output projection. Include complete PyTorch module code for a Transformer block. Target: 1800-2500 words."},
    {"p": 4, "t": "Transformer Variants and Efficiency", "w": 4200, "pr": "Write a section on TRANSFORMER VARIANTS AND EFFICIENCY TECHNIQUES. Cover: Sparse attention (Longformer with sliding window, BigBird with global tokens), Linear attention alternatives (Performer with random feature approximation, Linformer with low-rank projection, Reformer with locality-sensitive hashing), Mixture of Experts (Switch Transformer, Mixtral, DBRX), State Space Models (Mamba S4, RWKV), Flash Attention 1/2, Grouped Query Attention (GQA) and Multi-Query Attention (MQA), Sliding window attention, Speculative decoding. Target: 1800-2500 words."},
    {"p": 5, "t": "Training Dynamics and Optimization",  "w": 4200, "pr": "Write a section on TRANSFORMER TRAINING DYNAMICS AND OPTIMIZATION. Cover: Learning rate schedules (cosine decay, linear warmup), Gradient clipping and norm (why 1.0 threshold), Mixed precision training (FP16, BF16, FP8), Activation checkpointing (gradient recalculation trade-off), Weight decay (decoupled from bias terms) and AdamW optimizer, Training instabilities (loss spikes, NaN issues), Distributed training (Data Parallelism, ZeRO stages 1/2/3, FSDP), Pipeline parallelism (GPipe, PipeDream), Tensor parallelism (Megatron-LM). Include PyTorch training loop example. Target: 1800-2500 words."},
    {"p": 6, "t": "Inference-Time Computation",         "w": 4200, "pr": "Write a section on INFERENCE-TIME COMPUTATION AND DECODING STRATEGIES. Cover: Greedy vs sampling (temperature, top-k, top-p), Beam search with width tradeoffs, Speculative decoding (draft-then-verify with correct implementation), Medusa (multi-draft), Lookahead decoding, Early exiting (DeeLay, FastBERT), KV cache management and memory, Cascade and routing strategies, Prefix caching. Include Python code for speculative decoding. Target: 1800-2500 words."},
    {"p": 7, "t": "Advanced Attention and Architecture Innovations", "w": 4200, "pr": "Write a section on ADVANCED ATTENTION VARIANTS. Cover: Flash Attention 2 and 3 detailed algorithm with tiling, Grouped Query Attention GQA for efficiency, Multi-Query Attention MQA comparison, Sliding Window Attention for long sequences, Sparse attention patterns mixture of experts, Ring attention for distributed long-context, Flash Decoding optimization, Flash Attention with nested tensors. Include CUDA-like pseudocode for Flash Attention tiling. Target: 1800-2500 words."},
    {"p": 8, "t": "Scaling Transformers and Compute-Optimal Training", "w": 4200, "pr": "Write a section on SCALING AND TRAINING AT SCALE. Cover: Scaling laws Kaplan et al, Chinchilla optimal compute allocation, Training compute FLOPs estimation, LLaMA 3 training recipe, Data parallelism and gradient accumulation, Mixed batch size scheduling, Learning rate warmup and decay, Gradient checkpointing for memory, Zero Redundancy Optimizer ZeRO stages, FSDP fully sharded data parallel. Include scaling law formulas and descriptions. Target: 1800-2500 words."},
  ]},
  {"n": 3,  "t": "Tokenization and Text Representation", "passes": [
    {"p": 1, "t": "Tokenization Strategies",           "w": 4200, "pr": "Write a section on TOKENIZATION STRATEGIES. Cover: Character-level tokenization pros/cons, Byte Pair Encoding (BPE) algorithm with step-by-step merge table construction, WordPiece (used in BERT), Unigram Language Model (SentencePiece), Tiktoken (OpenAI's fast BPE), Token vocabulary sizes across models (GPT-4: 100K, Llama: 32K), Multi-language tokenization challenges (English-heavy bias, non-Latin scripts). Include Python BPE training code from scratch. Target: 1800-2500 words."},
    {"p": 2, "t": "Embedding Layers and Representation", "w": 4200, "pr": "Write a section on EMBEDDING LAYERS AND REPRESENTATION. Cover: Token embedding matrix (vocab_size x d_model), Positional embedding addition, Learned vs fixed (sinusoidal) embeddings, Contextual vs static embeddings, Representation geometry in embedding space (anisotropy, embedding degeneracy), Weight tying between input/output embeddings, Cross-lingual alignment (MUSE, LASER). Include PyTorch code for embedding layer. Target: 1800-2500 words."},
    {"p": 3, "t": "Tokenizer Evaluation",              "w": 4200, "pr": "Write a section on TOKENIZER EVALUATION AND COMPARISON. Cover: Vocabulary coverage metrics, Out-of-Vocabulary (OOV) rates and subword fallback, Compression ratio (tokens per character), BPE dropout for training robustness, tiktoken vs HuggingFace tokenizers benchmark, Vision tokenization (ViT patch embedding), Image tokenization (VQ-VAE, VQ-GAN, DALL-E), Byte-level BPE (GPT-2), Unicode normalization (NFC, NFD) and its impact. Target: 1800-2500 words."},
    {"p": 4, "t": "Advanced Representation Learning",   "w": 4200, "pr": "Write a section on ADVANCED REPRESENTATION LEARNING. Cover: Probing classifiers (what linguistic knowledge is encoded), Information-theoretic analysis of representations, Geometric properties (clustering, PCA, t-SNE, UMAP of embedding spaces), Analogical reasoning in embedding space (king-man+woman=queen), Representation heterogeneity across layers, Cross-lingual transfer and alignment quality, Knowledge neurons hypothesis. Target: 1800-2500 words."},
    {"p": 5, "t": "Unicode and Multilingual Challenges", "w": 4200, "pr": "Write a section on UNICODE, MULTILINGUAL AND NON-LATIN TOKENIZATION. Cover: Unicode normalization forms (NFC vs NFD), Challenges with CJK characters (each character as token vs subword), Arabic script right-to-left and diacritics, Hindi Devanagari script complexities, SentencePiece language-agnostic approach, Vocabulary size tradeoffs for multilingual models, OOV handling strategies, tiktoken vs HuggingFace for non-English, Counting tokens for multilingual text. Target: 1800-2500 words."},
    {"p": 6, "t": "Subword Regularization",            "w": 4200, "pr": "Write a section on SUBWORD REGULARIZATION AND ROBUSTNESS. Cover: BPE dropout for training regularization, Subword regularization (Kudo 2018), Vocabulary pruning techniques, Handling misspellings and typos, Code tokenization (identifiers, special characters, keywords), Domain-specific tokenization (mathematical notation, chemical formulas, music ABC notation), Practical guide: choosing a tokenizer for your use case. Target: 1800-2500 words."},
    {"p": 7, "t": "Embedding Space Analysis",          "w": 4200, "pr": "Write a section on EMBEDDING SPACE ANALYSIS. Cover: Intrinsic evaluation (word similarity, analogy), Extrinsic evaluation (downstream tasks), Principal Component Analysis (PCA) visualization, t-SNE and UMAP for embeddings, Anisotropy and representation degeneration problem, Representation learning theory, Sentence embeddings (SBERT, SimCSE), Multilingual alignment quality metrics. Target: 1800-2500 words."},
    {"p": 8, "t": "Cross-Lingual and Multimodal Representations", "w": 4200, "pr": "Write a section on CROSS-LINGUAL AND MULTIMODAL REPRESENTATIONS. Cover: Cross-lingual transfer learning with mBERT, XLM-R, Language-agnostic sentence representations, Aligned embedding spaces for cross-lingual NLP, CLIP contrastive vision-language learning, Alignment methods (Procrustes, SVD), Zero-shot cross-lingual transfer, Evaluation of cross-lingual models (XNLI, MLQA). Target: 1800-2500 words."},
  ]},
  {"n": 4,  "t": "Pre-training Objectives and Data", "passes": [
    {"p": 1, "t": "Causal Language Modeling",          "w": 4200, "pr": "Write a section on CAUSAL LANGUAGE MODELING (CLM). Cover: Autoregressive generation formulation, Teacher forcing during training, Cross-entropy loss for sequence prediction, Sequence packing for efficiency, Gradient checkpointing for memory savings, Mixed-precision training implementation, Flash Attention for causal masking. Include complete PyTorch training loop for CLM. Target: 1800-2500 words."},
    {"p": 2, "t": "Masked Language Modeling",          "w": 4200, "pr": "Write a section on MASKED LANGUAGE MODELING (MLM). Cover: BERT-style MLM with [MASK] token, Whole Word Masking (WWM) improvement, ELECTRA replaced token detection (RTD) -- more sample-efficient than MLM, DeBERTa with disentangled attention and enhanced mask decoder, Span corruption objective (T5 style), GLM (General Language Model), Denoising autoencoder objectives, Comparison: CLM vs MLM vs RTD tradeoffs. Target: 1800-2500 words."},
    {"p": 3, "t": "Pre-training Data Curation",        "w": 4200, "pr": "Write a section on PRE-TRAINING DATA ENGINEERING. Cover: Common Crawl extraction (CCNet, RefinedWeb), CC100 multilingual dataset, The Pile (EleutherAI) with its 22 sub-datasets, RedPajama-1T (replicated LLaMA training data), Data quality filtering (perplexity-based heuristics, classifier-based), Language identification and filtering, Toxicity filtering, Decontamination of benchmarks, Data mixing ratios (web vs books vs code). Target: 1800-2500 words."},
    {"p": 4, "t": "Scaling Laws",                      "w": 4200, "pr": "Write a section on SCALING LAWS FOR LLM TRAINING. Cover: Kaplan et al. (2020) scaling laws: L(N) ~ N^alpha with alpha approx -0.076 for parameters, Compute-optimal training (Chinchilla): train N ~ C^0.49, LLM training FLOPs calculation, Flash Attention for compute efficiency, Mixed batch size schedule (gradually increase), Learning rate schedule with recovery, Training stability and loss spikes, LLaMA 3 training recipe (15T tokens). Include scaling law chart description. Target: 1800-2500 words."},
    {"p": 5, "t": "Data Quality and Deduplication",    "w": 4200, "pr": "Write a section on DATA QUALITY AND DEDUPLICATION. Cover: Data quality taxonomy (text quality, deduplication, toxicity, privacy), Heuristic filters (length, language, repetitiveness), Classifier-based quality filtering (fine-tuned models), MinHash deduplication at web scale (near-duplicate detection), SimHash for near-duplicate detection, Web text extraction (trafilatura, newspaper3k), Quality vs quantity tradeoff in scaling, Data provenance and documentation. Target: 1800-2500 words."},
    {"p": 6, "t": "Efficient Training Techniques",       "w": 4200, "pr": "Write a section on EFFICIENT TRAINING TECHNIQUES. Cover: Gradient accumulation for effective large batch, ZeRO-1/2/3 (partitioned optimizer states, gradients, parameters), FSDP (Fully Sharded Data Parallel), Tensor parallelism for training (Megatron-LM), Pipeline parallelism (interleaved schedule), Sequence parallelism (ring attention for long sequences), Flash Attention 2 for training, Memory-efficient fine-tuning techniques. Include PyTorch FSDP code example. Target: 1800-2500 words."},
    {"p": 7, "t": "Training Recipes and Case Studies", "w": 4200, "pr": "Write a section on TRAINING RECIPES AND CASE STUDIES. Cover: LLaMA training recipe step by step, Mistral training details, Gemma model training, Phi model series from Microsoft, Data cleaning pipeline, Curriculum learning strategies, Learning rate scheduling in practice, Training stability tricks, Common failure modes and debugging. Target: 1800-2500 words."},
    {"p": 8, "t": "Datasets and Corpus Engineering",  "w": 4200, "pr": "Write a section on DATASETS AND CORPUS ENGINEERING. Cover: Major pre-training datasets (C4, The Pile, RedPajama, SlimPajama), Dataset composition and mixing ratios, Sampling strategy across sources, Domain-specific pre-training, Continual pre-training, Dataset documentation (datasheets), Licensing and copyright considerations, Synthetic data generation for training. Target: 1800-2500 words."},
  ]},
  {"n": 5,  "t": "Fine-tuning and Task Adaptation", "passes": [
    {"p": 1, "t": "Fine-tuning vs Parameter-Efficient", "w": 4200, "pr": "Write a section on FINE-TUNING STRATEGIES AND ADAPTER METHODS. Cover: Full fine-tuning drawbacks (catastrophic forgetting, compute cost), Adapter layers (Houlsby architecture), Prefix tuning (learnable soft prompts), Prompt tuning (continuous prompt tokens), LoRA (Low-Rank Adaptation): delta-W = BA, rank r selection, QLoRA (4-bit NF4 + LoRA), Comparison: full ft vs LoRA vs adapter vs prefix. Include PyTorch LoRA implementation. Target: 1800-2500 words."},
    {"p": 2, "t": "Instruction Tuning and SFT",         "w": 4200, "pr": "Write a section on INSTRUCTION TUNING AND SUPERVISED FINE-TUNING (SFT). Cover: SFT pipeline overview, Instruction dataset construction (FLAN, Alpaca, Dolly, OpenOrca), SFTTrainer from TRL library, Data formatting with chat templates (ChatML, Llama 3 chat template), Curriculum learning strategies, Data quality vs quantity in instruction tuning, Common pitfalls (overfitting to format, reward hacking). Target: 1800-2500 words."},
    {"p": 3, "t": "Domain Adaptation",                  "w": 4200, "pr": "Write a section on DOMAIN ADAPTATION FOR LLMs. Cover: Continual pre-training vs task-specific fine-tuning, Medical domain (PubMed, ClinicalBERT, MedLLaMA, BioBERT), Code domain (CodeBERT, StarCoder, CodeLLaMA, AlphaCode), Legal domain (ChatLaw, LexGPT, Corporate Legal documents), Scientific domain (Galactica, SciBERT, MathLLM), Domain vocabulary handling and special tokens, Mixture-of-adapters for multi-domain. Target: 1800-2500 words."},
    {"p": 4, "t": "Dataset Curation for Fine-tuning",   "w": 4200, "pr": "Write a section on DATASET CURATION FOR FINE-TUNING. Cover: Data collection strategies (human annotation, API sourcing, web scraping), Annotation quality control and inter-annotator agreement, Synthetic data generation with LLMs (self-instruct, evolution from seeds), Quality filtering pipelines, Dataset balancing across tasks/domains, Deduplication (exact and near-duplicate), Human preference data collection, RLHF data pipeline. Target: 1800-2500 words."},
    {"p": 5, "t": "LoRA Theory and Advanced Variants",  "w": 4200, "pr": "Write a section on LoRA THEORY AND ADVANCED VARIANTS. Cover: LoRA theory -- why low-rank works (singular value decomposition insight), Which layers to adapt (q_proj, v_proj vs all), Rank selection: r=4 vs r=64 tradeoffs, Alpha-rank scaling (alpha/r relationship), QLoRA with 4-bit NF4 (Normal Float 4) quantization, LoRA+: improved per-layer learning rates, DoRA (Weight-Decomposed LoRA), LongLoRA for extended context, Merging adapters (linear combination, TIES-Merging). Include PyTorch LoRA+ code. Target: 1800-2500 words."},
    {"p": 6, "t": "Prompt Tuning and Adapter Methods", "w": 4200, "pr": "Write a section on SOFT PROMPTING AND ADAPTER METHODS. Cover: Prompt tuning (learnable soft tokens in embedding space), Prefix tuning (prepended to all transformer layers), Adapter modules (Houlsby, Pfeiffer, Compacter, KronA), Series vs parallel adapter composition, Efficient Adapter Tuning (EAT), AdaptFormer, Series adapters vs LoRA comparison, Prefix tuning vs LoRA vs full fine-tuning benchmarks. Include PyTorch adapter code. Target: 1800-2500 words."},
    {"p": 7, "t": "Advanced PEFT Methods",             "w": 4200, "pr": "Write a section on ADVANCED PEFT METHODS. Cover: AdaLoRA adaptive rank allocation, LoftQ quantization-aware LoRA, VeRA very low-rank adapters, LoRA-fa layer-wise rank adaptation, Compacter advanced adapters, Mixtral of Experts fine-tuning, MoE adapter strategies, Expert routing with adapters, Training and merging multiple adapters. Target: 1800-2500 words."},
    {"p": 8, "t": "Fine-tuning Best Practices and Troubleshooting", "w": 4200, "pr": "Write a section on FINE-TUNING BEST PRACTICES. Cover: Catastrophic forgetting prevention, Learning rate selection for LoRA, Weight decay and optimizer choices, Batch size and gradient accumulation, Evaluation during fine-tuning, Hyperparameter tuning systematic approach, Overfitting detection, Model merging techniques (TIES, Elect), DARE task arithmetic for weight difference. Target: 1800-2500 words."},
  ]},
  {"n": 6,  "t": "Alignment: RLHF, DPO, and Beyond", "passes": [
    {"p": 1, "t": "RLHF (Reinforcement Learning from Human Feedback)", "w": 4200, "pr": "Write a section on RLHF. Cover: Why alignment matters (helpful, harmless, honest), The three stages of RLHF (SFT, reward model, PPO), Reward model: Bradley-Terry preference model formulation, PPO update with KL penalty to reference model, Reward hacking and goodhart's law, Mode collapse in RLHF, Challenges: reward hacking, human preference noise, Implementation of PPO for language models. Include PPO pseudocode. Target: 1800-2500 words."},
    {"p": 2, "t": "DPO (Direct Preference Optimization)", "w": 4200, "pr": "Write a section on DPO (Direct Preference Optimization). Cover: DPO objective formula derivation from RLHF, Why DPO avoids training a separate reward model, GRPO (DeepSeek) -- group relative policy optimization, KTO (Kahneman-Tversky Optimization) for loss aversion, IPO (Identity Preference Optimization), Comparison: RLHF vs DPO vs GRPO vs KTO (sample efficiency, stability, performance), CPO (Contrastive Preference Optimization). Include PyTorch DPO loss code. Target: 1800-2500 words."},
    {"p": 3, "t": "Reward Modeling",                   "w": 4200, "pr": "Write a section on REWARD MODELING. Cover: Reward model architecture (RM head on top of base model), Pointwise reward (score a single response) vs pairwise (compare two), Listwise ranking models, Reward model scaling laws, Constitutional AI (CAI): principles + self-critique, RLAIF: replacing human feedback with LLM-as-judge feedback, Self-rewarding models, Ensemble reward models. Target: 1800-2500 words."},
    {"p": 4, "t": "Safety Red-teaming",                "w": 4200, "pr": "Write a section on AI SAFETY AND RED-TEAMING. Cover: Red-teaming methodologies (manual, automated, adversarial), Harmful content categories (violence, CSAM, self-harm, misinformation), Safety benchmarks (TruthfulQA, ToxiGen, RealToxicityPrompts), Refusal learning and calibrated refusals, Guardrails (NeMo, Llama Guard, Azure Content Safety), Interpretability for safety (mechanistic analysis of refusal), Fairness and bias evaluation (BBQ, BOLD). Target: 1800-2500 words."},
    {"p": 5, "t": "Advanced Alignment Techniques",      "w": 4200, "pr": "Write a section on ADVANCED ALIGNMENT TECHNIQUES. Cover: Constitutional AI (CAI) -- principles list, self-critique, and revision training, RLAIF -- LLM-as-judge replacing human annotations, Scalable oversight (Debate, Amplification, Recursive Reward Modeling), Iterated amplification (debate between agents), Self-play for alignment, Addressing reward hacking through uncertainty, CoT-based safety (reasoning before response), Process reward models vs outcome reward models. Target: 1800-2500 words."},
    {"p": 6, "t": "DPO Variants and On-Policy Issues", "w": 4200, "pr": "Write a section on DPO VARIANTS AND ON-POLICY STABILITY. Cover: On-policy vs off-policy in DPO (importance sampling), RTF (Reinforced Fine-Tuning from OpenAI), Self-rewarding language models (Yao et al.), Iterative DPO (online preference learning), GRPO from DeepSeek (group relative), SimPO (Simple Preference Optimization), Issues with DPO: gradient collision, reward overestimation. Include PyTorch code for iterative DPO. Target: 1800-2500 words."},
    {"p": 7, "t": "Alignment in Practice",             "w": 4200, "pr": "Write a section on ALIGNMENT IN PRACTICE. Cover: Collecting preference data from human raters, Quality control for preference labels, Inter-annotator disagreement handling, Reward model training pipeline, PPO hyperparameters in practice, KL divergence regularization, KL penalty annealing, Handling distribution shift, Iterative DPO workflow. Target: 1800-2500 words."},
    {"p": 8, "t": "AI Safety and Governance",          "w": 4200, "pr": "Write a section on AI SAFETY AND GOVERNANCE. Cover: Current AI safety challenges, Scalable oversight techniques, Interpretability for alignment, Reward model robustness, Handling out-of-distribution inputs, Emerging regulatory frameworks (EU AI Act), Safety evaluation methodology, Incident response and reporting. Target: 1800-2500 words."},
  ]},
  {"n": 7,  "t": "Prompt Engineering and In-Context Learning", "passes": [
    {"p": 1, "t": "Zero-shot, Few-shot, Chain-of-Thought", "w": 4200, "pr": "Write a section on PROMPT ENGINEERING FUNDAMENTALS. Cover: Zero-shot prompting, Few-shot in-context learning (ICL) with examples, Chain-of-Thought (CoT) prompting -- step-by-step reasoning, Auto-CoT (automatic CoT generation), Self-consistency (majority vote over multiple CoT paths), Tree-of-Thought (ToT) for exploration, Best practices: clear instructions, delimiters, role assignment. Include prompt examples for each technique. Target: 1800-2500 words."},
    {"p": 2, "t": "In-Context Learning Theory",          "w": 4200, "pr": "Write a section on IN-CONTEXT LEARNING THEORY AND MECHANISMS. Cover: Inductive bias of Transformers enabling ICL, Gradient descent in hidden space hypothesis (ICL as meta-learning), Function vectors hypothesis (specific attention heads store task instructions), How demonstrations affect attention patterns, Analogical reasoning in ICL, Label semantics (flipped labels study), Example ordering effects (recency bias), ICL vs fine-tuning: when to use which. Target: 1800-2500 words."},
    {"p": 3, "t": "Retrieval-Augmented Generation (RAG)", "w": 4200, "pr": "Write a section on RAG (Retrieval-Augmented Generation). Cover: Why RAG matters (knowledge cut-off, hallucination reduction), Vector databases (FAISS, ChromaDB, Milvus, Pinecone, Qdrant), Embedding models (BGE, E5, Instructor, OpenAI text-embedding-3), Chunking strategies (size, overlap, semantic), Dense retrieval (bi-encoder), Sparse retrieval (BM25), Hybrid search (dense + sparse), Re-ranking (cross-encoder), RAG vs fine-tuning tradeoffs. Target: 1800-2500 words."},
    {"p": 4, "t": "LLM Agents and Tool Use",            "w": 4200, "pr": "Write a section on LLM AGENTS AND TOOL USE. Cover: ReAct (Reasoning + Acting) framework, Tool use with function calling / tool schema, Agent architectures (single-agent, multi-agent), Planning and task decomposition (H thought, CoT, Tree-of-Thought), Memory systems (vector store, summary, episodic), Code interpreter agents (REPL, sandboxed execution), Multi-agent debate and collaboration, Error recovery strategies. Include ReAct implementation code. Target: 1800-2500 words."},
    {"p": 5, "t": "Advanced Prompting Techniques",       "w": 4200, "pr": "Write a section on ADVANCED PROMPTING AND SELF-IMPROVEMENT. Cover: Reflexion (verbal reinforcement learning with episodic memory), Self-refine (iterative critique and revision), Prompt evolution (genetic algorithm for better prompts), Multi-agent prompting (debate, consensus, role assignment), Generated knowledge prompting (separate knowledge generation + answer), Chain-of-Verification (CoV) for factual accuracy, APE (Automatic Prompt Engineering), DSPy (Declarative Self-Improving Python). Include code examples. Target: 1800-2500 words."},
    {"p": 6, "t": "Advanced RAG Architecture",           "w": 4200, "pr": "Write a section on ADVANCED RAG ARCHITECTURES. Cover: Dense retrieval (bi-encoder) vs sparse (BM25) vs ColBERT (late interaction), ANN indexes (HNSW, IVF-PQ, ScaNN, DiskANN), Reciprocal Rank Fusion (RRF) for combining retrievers, ColBERTv2 (composable late interaction), Knowledge graph RAG (entity linking, graph traversal), Parent document retrieval, Contextual compression (LLMCompacter, RECOMP), Self-RAG (self-reflection tokens). Include RAG implementation code. Target: 1800-2500 words."},
    {"p": 7, "t": "Agent Architectures and Memory Systems", "w": 4200, "pr": "Write a section on AGENT ARCHITECTURES AND MEMORY. Cover: ReAct and Plan-and-Execute patterns, Reflection agents with self-critique, Memory types (episodic, semantic, procedural), Vector memory and summary memory, Long-term memory for agents, Tool use and API integration, Multi-agent collaboration and communication, Error handling and retry strategies, Agent evaluation frameworks. Target: 1800-2500 words."},
    {"p": 8, "t": "Advanced Agent Patterns and Production", "w": 4200, "pr": "Write a section on ADVANCED AGENT PATTERNS. Cover: LangChain and LlamaIndex for agent development, AutoGen and CrewAI multi-agent frameworks, Planning with LLMs task decomposition, Self-correcting agents with feedback loops, Code generation and execution agents, Autonomous agents (BabyAGI, AutoGPT), Evaluating agent performance, Safety considerations for autonomous agents, Production deployment patterns. Target: 1800-2500 words."},
  ]},
  {"n": 8,  "t": "Evaluation and Benchmarking", "passes": [
    {"p": 1, "t": "Core NLP Benchmarks",               "w": 4200, "pr": "Write a section on CORE LLM BENCHMARKS. Cover: MMLU (57 subjects, 5-shot), GSM8K (Grade School Math 8K), HumanEval (code generation), MATH (competition math), HellaSwag (commonsense reasoning), ARC (AI2 Reasoning Challenge), TruthfulQA (truthfulness), BIG-bench (200+ tasks), BIG-Bench Hard (BBH), Evaluation protocols (few-shot, CoT, pass@k), Limitations: gaming, contamination, saturation. Target: 1800-2500 words."},
    {"p": 2, "t": "LLM Evaluation Frameworks",          "w": 4200, "pr": "Write a section on LLM EVALUATION FRAMEWORKS AND TOOLS. Cover: lm-evaluation-harness (EleutherAI) -- supports 200+ benchmarks, LightEval (HuggingFace), OpenCompass (MMLab), HELM (Holistic Evaluation of Language Models), AlpacaEval, MT-Bench (multi-turn), BLEU, ROUGE, METEOR limitations for generation, Perplexity (intrinsic evaluation), Human evaluation best practices (paired comparison, Likert scales). Target: 1800-2500 words."},
    {"p": 3, "t": "Code and Reasoning Evaluation",      "w": 4200, "pr": "Write a section on CODE AND REASONING EVALUATION. Cover: HumanEval (OpenAI pass@1 benchmark), MBPP (Mostly Basic Python Problems), APPS (Automated Programming Progress Standard), Codeforces (competitive programming), MultiPL-E (multilingual), ToolBench (API usage), GSM8K and MATH (math reasoning), BIG-Bench Hard (harder subset), Evaluating Chain-of-Thought quality, Execution-based code evaluation (sandbox). Target: 1800-2500 words."},
    {"p": 4, "t": "Safety and Fairness Evaluation",    "w": 4200, "pr": "Write a section on SAFETY AND FAIRNESS EVALUATION. Cover: TruthfulQA variants (GPQA, TriviaQA), ToxiGen (toxicity detection), BOLD (bias in open-ended generation), RealToxicityPrompts (prompt-based toxicity), Decoding-time safety interventions, Evaluation of refusal behavior (refusal quality, calibration), BBQ (bias benchmark), BOLD demographic performance parity, Refusal instruction training evaluation. Target: 1800-2500 words."},
    {"p": 5, "t": "Statistical Evaluation Metrics",       "w": 4200, "pr": "Write a section on STATISTICAL EVALUATION METRICS. Cover: BLEU n-gram precision with brevity penalty formula, ROUGE-L (Longest Common Subsequence), METEOR (stem matching, synonymy), BERTScore (contextual embedding similarity), BARTScore (generation quality from pretrained model), G-Eval (LLM-based evaluation with CoT), Limitations of surface-level matching, When to use which metric (summarization: ROUGE+BERTScore, translation: BLEU+METEOR, generation: LLM-as-judge). Target: 1800-2500 words."},
    {"p": 6, "t": "Human Evaluation and LLM-as-Judge", "w": 4200, "pr": "Write a section on HUMAN EVALUATION AND LLM-AS-JUDGE. Cover: Human evaluation protocols (pairwise comparison, Likert scales, A/B testing), Inter-annotator agreement (Cohen's kappa, Krippendorff's alpha), LLM-as-Judge prompting strategies, Self-evaluation bias in LLMs, Position bias mitigation (swap order), reference-free evaluation, Elo and Bradley-Terry for model ranking, Chatbot Arena (mt-bench, arena), Statistical significance in LLM comparisons. Target: 1800-2500 words."},
    {"p": 7, "t": "Reasoning and Math Evaluation",    "w": 4200, "pr": "Write a section on REASONING AND MATH EVALUATION. Cover: GSM8K Grade School Math 8K, MATH benchmark, FrontierMath, ARC-C abstract reasoning, GPQA graduate-level science, Competition math evaluation, MATH-ASCII plain text math, Evaluating chain-of-thought quality, Pass@k and sampling-based evaluation, Process reward models for math. Target: 1800-2500 words."},
    {"p": 8, "t": "Comprehensive Evaluation Strategy", "w": 4200, "pr": "Write a section on COMPREHENSIVE EVALUATION STRATEGY. Cover: Designing an evaluation suite, Balancing capability and safety benchmarks, Red-teaming for specific failure modes, Evals for fine-tuned models, Evals for RAG and agentic systems, Continuous evaluation in production, A/B testing with statistical significance, Cost-efficient evaluation, Building custom benchmarks for domain-specific needs. Target: 1800-2500 words."},
  ]},
  {"n": 9,  "t": "Deployment, Inference, and Optimization", "passes": [
    {"p": 1, "t": "LLM Inference Optimization",          "w": 4200, "pr": "Write a section on LLM INFERENCE OPTIMIZATION. Cover: KV cache (key-value cache for autoregressive decoding), KV cache memory calculation (layers x 2 x batch x seq_len x d_k x bytes_per_param), Batching strategies: static vs dynamic, Continuous batching (iteration-level scheduling), PagedAttention from vLLM, Tensor parallelism for inference, Pipeline parallelism, Speculative decoding speedup, Beam search vs sampling tradeoffs. Target: 1800-2500 words."},
    {"p": 2, "t": "Quantization",                       "w": 4200, "pr": "Write a section on QUANTIZATION FOR LLM INFERENCE. Cover: Quantization fundamentals (INT8, FP16, BF16, FP8), Post-Training Quantization (PTQ): calibration, weight-only quantization, GPTQ (Gradient PTQ) with OBQ, AWQ (Activation-Aware Weight Quantization), SmoothQuant (channel-wise smoothing), GGUF formats (Q8, Q6, Q5, Q4, Q3, Q2 with tradeoffs), BitsAndBytes 8-bit and 4-bit with NF4, Impact on perplexity and accuracy per task. Include quantization code (GPTQ). Target: 1800-2500 words."},
    {"p": 3, "t": "Local Deployment Tools",             "w": 4200, "pr": "Write a section on LOCAL DEPLOYMENT TOOLS AND FRAMEWORKS. Cover: llama.cpp (C/C++, Metal GPU acceleration, CUDA, GGUF format), Ollama (local model management, OpenAI-compatible API), vLLM (PagedAttention, continuous batching), Text Generation Inference (TGI, HuggingFace), Inference endpoints comparison (Replicate, Modal, Banana), OpenAI-compatible API wrappers, Benchmarking local inference (throughput tok/s, latency ms), Memory requirements by model size and quantization. Target: 1800-2500 words."},
    {"p": 4, "t": "Pruning and Distillation",           "w": 4200, "pr": "Write a section on MODEL PRUNING AND KNOWLEDGE DISTILLATION. Cover: Structured vs unstructured pruning (magnitude, movement), Sparse attention patterns, Knowledge distillation: teacher-student framework, MiniLLM (distilling GPT into smaller LLM), Wanda (pruning by weights multiplied by activations), SparseGPT (one-shot pruning), Combining quantization + pruning, Neural Architecture Search (NAS) for efficient models, LLMLingua for prompt compression. Target: 1800-2500 words."},
    {"p": 5, "t": "Distributed Serving",                 "w": 4200, "pr": "Write a section on DISTRIBUTED SERVING SYSTEMS. Cover: vLLM architecture (PagedAttention, block manager), Continuous batching vs static batching, Tensor parallelism for inference (Megatron in vLLM), Pipeline parallelism in serving, Expert routing in MoE models, Prefix caching strategies, Latency vs throughput tradeoff (SLA-driven), P50, P95, P99 latency targets, Load balancing (weighted round-robin), Ray Serve for multi-model serving. Target: 1800-2500 words."},
    {"p": 6, "t": "Memory Optimization",                "w": 4200, "pr": "Write a section on MEMORY OPTIMIZATION FOR INFERENCE. Cover: KV cache quantization (INT8, FP8), KIVI (2-bit KV cache quantization), Automatic Prefix Caching (APC) in vLLM, Chunked prefill for long sequences, Speculative decoding (Medusa, EAGLE, cascade decoding), Flash Decoding for long-context inference, Streaming attention and sliding window, Memory allocation strategies (paged memory). Target: 1800-2500 words."},
    {"p": 7, "t": "Production Deployment Patterns",     "w": 4200, "pr": "Write a section on PRODUCTION DEPLOYMENT. Cover: Container orchestration Kubernetes for LLMs, API gateway patterns, Rate limiting and autoscaling, Canary releases and rollback, Multi-model serving, Latency optimization techniques, Cost optimization strategies, Monitoring inference quality, Error handling and fallbacks. Target: 1800-2500 words."},
    {"p": 8, "t": "Optimization Case Studies and Benchmarks", "w": 4200, "pr": "Write a section on OPTIMIZATION CASE STUDIES. Cover: Comparing quantization methods on downstream tasks, vLLM vs TGI performance comparison, llama.cpp Metal vs CUDA throughput, Speculative decoding speedup measurements, Batch size tuning for throughput, KV cache compression benchmarks, End-to-end serving benchmarks, Practical guide to choosing optimization methods. Target: 1800-2500 words."},
  ]},
  {"n": 10, "t": "Multimodal and Emerging Capabilities", "passes": [
    {"p": 1, "t": "Vision-Language Models",             "w": 4200, "pr": "Write a section on VISION-LANGUAGE MODELS (VLMs). Cover: CLIP (contrastive image-text learning), LLaVA (vision encoder + LLM connector), BLIP-2 (Q-Former architecture), Flamingo (cross-attention per layer), GPT-4V and Gemini (multimodal perception), Vision-language alignment strategies, Instruction tuning for VLMs, Evaluation (VQA, image captioning, document understanding, chart understanding), MiniGPT-4, CogVLM, InternVL. Target: 1800-2500 words."},
    {"p": 2, "t": "Audio, Long Context, Emergent Abilities", "w": 4200, "pr": "Write a section on AUDIO-LANGUAGE MODELS AND EMERGENT ABILITIES. Cover: Whisper (speech recognition), AudioPaLM (speech-to-speech), Emergent abilities (in-context learning, CoT appearing at scale), Chain-of-thought emergence phase transitions, Arithmetic and logical reasoning capabilities, Theory of mind in LLMs, RoPE scaling for long context (YaRN), Landmark attention for long sequences, State-space models for long context (Mamba). Target: 1800-2500 words."},
    {"p": 3, "t": "Interpretability",                    "w": 4200, "pr": "Write a section on LLM INTERPRETABILITY. Cover: Feature probing and activation patching, Circuit analysis (induction heads, indirect object identification), Attention head roles (induction heads, name moving heads), Sparse autoencoders (SAE) for monosemantic features, Superposition hypothesis (many features in few neurons), Gradient-based attribution (IG, LIME), Probing classifiers (syntactic dependencies, semantics), Mechanistic interpretability of refusal behavior, Activation steering (tailored). Target: 1800-2500 words."},
    {"p": 4, "t": "Future of LLM Research",              "w": 4200, "pr": "Write a section on THE FUTURE OF LLM RESEARCH. Cover: Open-source vs closed-source capability gap trend, MoE (Mixture of Experts) scaling, Test-time compute scaling (o1, o3 style reasoning), Constitutional AI development, Mechanistic interpretability roadmap, LLM compression (1-bit LLM, extreme quantization), Scientific discovery with LLMs (AlphaFold-like moments), Energy efficiency improvements, AGI debates and timeline estimates. Target: 1800-2500 words."},
    {"p": 5, "t": "Multimodal Training Evolution",       "w": 4200, "pr": "Write a section on MULTIMODAL TRAINING EVOLUTION. Cover: Early fusion vs late fusion, ALIGN (large-scale image-text pairs), Flamingo with Perceiver resampler, BLIP-2 (frozen LLM + Q-Former), LLaVA (linear projection of vision features), MiniGPT-4 (BLIP-2 strategy), GPT-4V training pipeline, Architecture comparison table, Document understanding and OCR, Video understanding integration. Target: 1800-2500 words."},
    {"p": 6, "t": "Video, World Models, Embodied AI",   "w": 4200, "pr": "Write a section on VIDEO UNDERSTANDING AND WORLD MODELS. Cover: Video LLMs (VideoChat, VideoLLaMA, LLaMA-VID), Temporal modeling in video, Action recognition and anticipation, World models from video (Gaia, Genie), Embodied AI (instructing robots with language), Sora (video generation model), Stable Video Diffusion, LWM (Long World Model, 1M context), Reasoning about physical world. Target: 1800-2500 words."},
    {"p": 7, "t": "Multimodal Architectures Deep Dive", "w": 4200, "pr": "Write a section on MULTIMODAL ARCHITECTURES. Cover: Vision transformer (ViT) architecture, SigLIP and DINOv2 vision encoders, Cross-attention vs fusion-in-decoder, Perceiver resampler and Q-Former, Gemma multimodal, InternVL architecture, Molmo and NVLM models, Training multimodal models data mixture, Evaluation of multimodal models. Target: 1800-2500 words."},
    {"p": 8, "t": "Audio Speech and Video Models",     "w": 4200, "pr": "Write a section on AUDIO SPEECH AND VIDEO MODELS. Cover: Whisper architecture for ASR, AudioPaLM for speech-to-speech, Video generation models (Sora, Lumiere), World models from video (GAIA-1), Temporal video representation, Latent video compression, Text-to-video generation, Understanding long videos, Embodied agents in 3D environments. Target: 1800-2500 words."},
  ]},
  {"n": 11, "t": "Practical Applications and Case Studies", "passes": [
    {"p": 1, "t": "Building Production RAG Systems",    "w": 4200, "pr": "Write a section on BUILDING PRODUCTION RAG SYSTEMS. Cover: End-to-end RAG architecture (indexer -> retriever -> generator), Embedding models (BGE-M3, E5-Mistral, GTE), Vector database selection, Chunking: size, overlap, semantic chunking, Hybrid search (keyword + vector), Re-ranking with cross-encoder, Guardrails and output validation, RAGAS evaluation metrics, LangChain vs LlamaIndex comparison, Complete RAG pipeline with LangChain. Target: 1800-2500 words."},
    {"p": 2, "t": "Building LLM Agents",               "w": 4200, "pr": "Write a section on BUILDING PRODUCTION LLM AGENTS. Cover: Agent loop: perceive -> plan -> act -> reflect, Tool definition and function calling schemas, ReAct implementation from scratch, Multi-agent systems (collaboration, competition), Memory: vector store + summary + episodic, Error handling and graceful degradation, Agent evaluation (success rate, tool call accuracy), LangGraph for complex workflows, Production best practices. Include ReAct agent code. Target: 1800-2500 words."},
    {"p": 3, "t": "Fine-tuning Case Studies",           "w": 4200, "pr": "Write a section on FINE-TUNING CASE STUDIES. Cover: Medical LLM: BioBERT -> MedLLaMA evolution, Code LLM: Codex -> CodeLLaMA, Legal LLM: ChatLaw, Finance: BloombergGPT, Domain adaptation recipe (data collection, preprocessing, LoRA config), LoRA hyperparameter tuning (rank, alpha, target modules), Dataset size vs quality tradeoff, Cost and time estimation for fine-tuning. Target: 1800-2500 words."},
    {"p": 4, "t": "Cost Estimation and Optimization",   "w": 4200, "pr": "Write a section on LLM COST OPTIMIZATION. Cover: API pricing comparison (OpenAI GPT-4o, Anthropic Claude 3.7, Google Gemini, open-source), Token estimation (tokens approx 0.75 x words for English), Caching strategies (semantic cache, KV cache), Batch inference for cost savings, Model selection by task complexity, Fine-tuning vs RAG vs prompt engineering cost analysis, Self-hosted vs API cost comparison, Context length cost impact. Target: 1800-2500 words."},
    {"p": 5, "t": "LLM Security",                      "w": 4200, "pr": "Write a section on LLM SECURITY. Cover: Prompt injection (direct, indirect, cross-site), Jailbreaking (role-play, ASCII art, token smuggling, universal jailbreaks), System prompt extraction attacks, Data leakage from training and retrieval systems, Membership inference attacks, Adversarial suffixes (GCG, AutoDAN), Defensive strategies (input validation, output filtering, sandboxing), OWASP LLM Top 10 vulnerabilities, Llama Guard for content safety. Include attack and defense code examples. Target: 1800-2500 words."},
    {"p": 6, "t": "Monitoring and Observability",      "w": 4200, "pr": "Write a section on PRODUCTION LLM OPERATIONS AND OBSERVABILITY. Cover: LLM observability essentials (logging, latency, cost, quality), Tracing frameworks (OpenTelemetry, LangSmith, Arize Phoenix), Prompt version management, Drift detection (input distribution shift), Quality monitoring dashboards, Cost attribution per feature/customer, A/B testing LLMs with statistical significance, Feature flags for model routing, Incident response for LLM failures. Target: 1800-2500 words."},
    {"p": 7, "t": "Enterprise Application Patterns",   "w": 4200, "pr": "Write a section on ENTERPRISE APPLICATION PATTERNS. Cover: LLM-powered search engines (semantic, hybrid), Document intelligence (extraction, classification), Customer service automation, Code generation in IDEs, Data analysis and BI with LLMs, Content moderation systems, Personalized recommendation with LLMs, Compliance and audit trails. Target: 1800-2500 words."},
    {"p": 8, "t": "End-to-End Project Case Studies", "w": 4200, "pr": "Write a section on END-TO-END CASE STUDIES. Cover: Building a medical QA system step by step, Legal document summarization pipeline, Code review automation with multi-agent, Financial report analysis system, Implementing RAG with hybrid search (production-ready), Fine-tuning for code generation from scratch, Building a multimodal document processor, Lessons learned and common pitfalls. Target: 1800-2500 words."},
  ]},
  {"n": 12, "t": "Research Frontiers and Future Directions", "passes": [
    {"p": 1, "t": "Current Research Frontiers",         "w": 4200, "pr": "Write a section on CURRENT RESEARCH FRONTIERS IN LLMs. Cover: Open-source capability gap with frontier models, MoE scaling (cost efficiency), Test-time compute scaling (o1, o3 style reasoning), Constitutional AI and value alignment, Mechanistic interpretability (circuits, features, superposition), Superposition hypothesis and sparse autoencoders, Model editing (ROME, MEMIT for targeted knowledge), 1-bit LLM (BitNet b1.58), Key papers from 2024-2026. Target: 1800-2500 words."},
    {"p": 2, "t": "Tools and Resources",               "w": 4200, "pr": "Write a section on TOOLS AND RESOURCES FOR LLM RESEARCH. Cover: HuggingFace ecosystem (Transformers, Datasets, Spaces, Hub), TRL (Transformer Reinforcement Learning), PEFT (Parameter-Efficient Fine-Tuning), LLaMA-Factory, LangChain, LlamaIndex, Weights & Biases, Ollama, PromptLayer, lm-evaluation-harness, Academic resources (arXiv, Papers with Code, Hugging Face daily papers), Community resources (subreddits, Discord, newsletters). Target: 1800-2500 words."},
    {"p": 3, "t": "Regulatory Landscape and Policy",   "w": 4200, "pr": "Write a section on REGULATORY LANDSCAPE AND POLICY for LLMs. Cover: EU AI Act risk tiers and compliance obligations, US AI Executive Order on safe & secure AI, NIST AI Risk Management Framework, UK AI Safety Institute and AISI evaluations, China's generative AI regulations, ISO/IEC standards for AI systems, Privacy-preserving training (differential privacy, federated learning) -- focus on REGULATORY angle only, International coordination (OECD, G7 Hiroshima Process), Corporate governance commitments. Do NOT re-cover AI safety research itself (already in Ch6.8); focus strictly on policy/regulatory aspects. Target: 1800-2500 words."},
    {"p": 4, "t": "Building Real-World Applications",   "w": 4200, "pr": "Write a section on BUILDING REAL-WORLD LLM APPLICATIONS. Cover: Architecture patterns: RAG vs agents vs fine-tuned, System design for production (reliability, scalability), Latency optimization (caching, speculative), Error handling and graceful degradation, Monitoring and observability stack, A/B testing for LLMs, Cost management strategies, Security (prompt injection, data leakage), Case studies: customer support, coding assistant, research assistant. Target: 1800-2500 words."},
    {"p": 5, "t": "Emergent Capabilities and Scaling", "w": 4200, "pr": "Write a section on EMERGENT CAPABILITIES AND SCALING PHENOMENA. Cover: What are emergent capabilities (sudden phase transitions), Phase transitions in model behavior (in-context learning emergence), Predicting capability emergence from scaling laws, Compute-optimal vs performance-optimal training, Data-optimal scaling (quality over quantity), Test-time compute (Strawberry o1 chain-of-thought, verify), The Bitter Lesson (compute beats hand-crafted), Current frontier models (2025-2026). Target: 1800-2500 words."},
    {"p": 6, "t": "Ecosystem and Competition",         "w": 4200, "pr": "Write a section on THE LLM ECOSYSTEM AND MARKET COMPETITION. Cover: Timeline of major releases 2020-2026 (GPT-3 to o3), Open-source leaders (LLaMA, Mistral, Gemma, Qwen, Phi, DeepSeek), Closed-source frontier (GPT-4o, Claude 3.7 Sonnet, Gemini 2.5, Grok), The closing gap between open and closed, API economics and pricing trends (per-token cost reduction), Fine-tuning ecosystem (Axolotl, LLaMA-Factory, TRL), Custom silicon (TPUs, Groq LPU, Cerebras), What open-source needs to catch up. Target: 1800-2500 words."},
    {"p": 7, "t": "1-bit LLMs and Model Efficiency Frontiers", "w": 4200, "pr": "Write a section on 1-BIT LLMS AND MODEL EFFICIENCY. Cover: 1-bit LLM BitNet b1.58 architecture, BitNet paper analysis, Ternary and binary networks, Sparse models and pruning at scale, Speculative decoding theory and practice, Early exit strategies for adaptive computation, Mixture of Experts routing efficiency, Hardware-algorithm co-design for efficient inference. Target: 1800-2500 words."},
    {"p": 8, "t": "The Path Forward: AGI and Long-term AI", "w": 4200, "pr": "Write a section on AGI AND LONG-TERM AI. Cover: Current definitions of AGI and their limitations, Benchmark saturation and what it means, Reasoning ability current state and gaps, World models and situational awareness, LLM benchmarking for general intelligence, What remains unsolved, Timeline debates and expert opinions, The role of open-source in safe AI development, Recommendations for researchers and practitioners. Target: 1800-2500 words."},
  ]},
]



# ============================================================================
# Ollama Client
# ============================================================================

class OllamaClient:
    def __init__(self, base=OLLAMA_BASE, model=MODEL, timeout=DEFAULT_TIMEOUT):
        self.base = base
        self.model = model
        self.timeout = timeout
        self.lock = threading.Lock()

    def generate(self, prompt, system="", temperature=0.7, num_predict=15000):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        effective_tokens = min(num_predict, 15000)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": msgs,
            "options": {
                "temperature": temperature,
                "num_predict": effective_tokens,
                "top_p": 0.95,
                "top_k": 20,
                "repeat_penalty": 1.05,
            },
        }
        t0 = time.time()
        with self.lock:
            import httpx
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(f"{self.base}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
        msg = data.get("message", {})
        content = msg.get("content", "").strip()
        ec = data.get("eval_count", 0)
        ed = data.get("eval_duration", 0)
        tps = ec / (ed / 1e9) if ed > 0 else 0
        return content, {
            "tokens": ec,
            "tps": round(tps, 1),
            "elapsed": round(time.time() - t0, 1),
            "done": data.get("done_reason", ""),
        }

    def health(self):
        try:
            import httpx
            with httpx.Client(timeout=5) as c:
                return c.get(f"{self.base}/api/tags").status_code == 200
        except:
            return False


SYS = (
    "You are a world-class technical book writer specializing in Large Language Models. "
    "You are writing ONE section of a larger book. The book's outer structure (chapter title, "
    "section title, numbering) is handled by the assembly pipeline -- do NOT recreate it.\n\n"
    "STRICT OUTPUT RULES:\n"
    "1. Do NOT output any H1 (`#`) or H2 (`##`) heading. The section heading is added by the "
    "assembler. You may use H3 (`###`) and H4 (`####`) for sub-topics inside the section.\n"
    "2. Do NOT start with phrases like 'In this section', 'This chapter covers', 'We will discuss', "
    "or any meta-introduction. Start directly with substantive content. Also AVOID the "
    "formulaic openings this book overuses: do NOT begin with 'The {abstract noun} of', do NOT "
    "open by restating 'Large Language Models (LLMs)', and do NOT use the word 'necessitates'. "
    "Open in medias res with a concrete fact, formula, named method, or benchmark result from the "
    "EVIDENCE -- e.g. lead with a definition, a number, or a specific paper's finding.\n"
    "3. Do NOT write a 'Conclusion', 'Summary', 'Wrap-up', or 'In summary' section at the end. "
    "End with the last technical point.\n"
    "4. Do NOT write a 'References', 'Bibliography', or 'Further Reading' section. A single "
    "References page is assembled from your [N] markers at the end of the book.\n"
    "5. CITATIONS (MANDATORY when an EVIDENCE block is present): You MUST place at least "
    "FIVE `[N]` citation markers in this section, where N is a source number from the EVIDENCE "
    "block. A section with zero `[N]` markers is REJECTED and regenerated -- citing is the single "
    "most important requirement. Anchor each `[N]` on a specific factual claim (number, date, "
    "formula, named method, benchmark, paper finding). Stack multiple `[N]` in one sentence when "
    "several sources agree. Use ONLY the numbers shown in EVIDENCE; never invent a higher index. "
    "When citing, name the real author/lab from the EVIDENCE entry (e.g. 'Wei et al. (2022) [3]') "
    "-- never write a search-tool name (Tavily/DuckDuckGo/Brave) as the author, and never write "
    "the literal placeholder `[N]` (always a concrete digit). If a single sub-point truly has no "
    "matching source, hedge it in prose WITHOUT a marker -- but the section as a whole must still "
    "carry its 5+ real citations.\n"
    "6. When NO EVIDENCE block is present, use inline author-year style "
    "`(Vaswani et al., 2017)` for well-known canonical references only -- never a search-tool name.\n"
    "7. Do NOT repeat definitions of concepts that have already been introduced earlier in the book "
    "(a context block will tell you what was covered). Reference them and build on them.\n"
    "8. Avoid filler -- prefer dense technical content over restating the topic. Quality over word count.\n"
    "9. Write in a scholarly, precise style. Include formulas in LaTeX (`$...$` or `$$...$$`) and "
    "code examples in fenced blocks where genuinely useful.\n"
    "10. MATH NOTATION: every variable, Greek letter, subscript, superscript, fraction, sum, "
    "or operator MUST be inside `$...$` (inline) or `$$...$$` (display). Do NOT use HTML "
    "`<sub>`/`<sup>` tags, do NOT italic-wrap symbols like `*alpha*` or `*beta*`, do NOT "
    "duplicate a formula in both a Unicode/plain-text form AND a LaTeX form -- write each "
    "formula ONCE in LaTeX. Bad: `Let *d<sub>k</sub>* be...`. Good: `Let $d_k$ be...`.\n"
    "Output ONLY the section body Markdown -- nothing else."
)


REVIEW_SYS = (
    "You are a strict technical editor reviewing one section of a book on Large Language Models. "
    "Score the section from 1 to 10 on three axes and respond with ONLY a single JSON object on one line, "
    "no prose, no markdown fences. Format: "
    '{\"depth\":N,\"coherence\":N,\"format\":N,\"issues\":\"...\"} '
    "where depth = technical correctness and density, coherence = follows from prior context without repeating, "
    "format = obeys structural rules (no H1/H2, no References block, no Conclusion, no meta-intro). "
    "Issues is one short sentence (<=20 words) naming the worst problem, or empty string if none."
)


CONTINUATION_WORDS = 120  # words of previous section's tail to feed forward
MIN_REVIEW_SCORE = 6      # below this on any axis triggers one regeneration


# ============================================================================
# Utilities
# ============================================================================

def wc(text):
    return len(re.findall(r"\S+", text))


# Patterns the model tends to hallucinate even when told not to.
_LEADING_HEADER_RE = re.compile(r"^\s*(#{1,2})\s+.*?$", re.MULTILINE)
_TRAILING_BLOCK_RE = re.compile(
    r"\n#{1,6}\s+(references|bibliography|further\s+reading|conclusion|summary|"
    r"wrap[- ]?up|in\s+summary|closing\s+thoughts)\b.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_META_INTRO_RE = re.compile(
    r"^\s*(in this (section|chapter)|this (section|chapter) (covers|discusses|introduces|will)|"
    r"we will (discuss|explore|cover|examine)|here (is|we present)|let'?s (explore|dive|examine))[^\n]*\n+",
    re.IGNORECASE,
)


_GREEK_TEX = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}

# Math context detectors -- only normalize symbols inside a paragraph that already
# looks math-adjacent, to avoid touching unrelated prose.
_HAS_MATH_CONTEXT = re.compile(r"[=+\-*/^_]|\\[a-zA-Z]+|\\$|<su[bp]>", re.MULTILINE)


def normalize_math(content: str) -> str:
    """Repair common model-emitted notation bugs:

    1. HTML `<sub>k</sub>` / `<sup>n</sup>` not inside a math context  -> LaTeX `_{k}` / `^{n}`
       wrapped with `$...$`. Common pattern: 'Let *d<sub>k</sub>* be...' -> 'Let $d_{k}$ be...'.
    2. Italic-wrapped single Greek letters like `*α*` or `*β<sub>θ</sub>*` -> proper LaTeX.
    3. Bare Greek letters surrounded by whitespace in math-y paragraphs -> `$\\alpha$`.

    Conservative: only rewrites tokens that are unambiguously math; leaves regular prose alone.
    Idempotent: re-running on already-normalized text is a no-op (the $-wrapped form has no
    further triggers).
    """
    if not content:
        return content

    # 1. <sub>x</sub> with optional italic asterisks around a single-letter variable.
    #    Capture leading letter so `d<sub>k</sub>` -> `$d_{k}$`. The negative lookbehind
    #    prevents mid-word matches (so `pi<sub>theta</sub>` stays untouched rather than
    #    becoming the broken `p$i_{theta}$`).
    def _sub_to_tex(m):
        prefix = m.group(1) or ""
        body = m.group(2)
        return f"${prefix}_{{{body}}}$"
    content = re.sub(
        r"(?<![A-Za-z0-9])\*?([A-Za-zͰ-Ͽ])?\*?<sub>\s*([^<>\n]{1,40}?)\s*</sub>\*?",
        _sub_to_tex, content,
    )

    def _sup_to_tex(m):
        prefix = m.group(1) or ""
        body = m.group(2)
        return f"${prefix}^{{{body}}}$"
    content = re.sub(
        r"(?<![A-Za-z0-9])\*?([A-Za-zͰ-Ͽ])?\*?<sup>\s*([^<>\n]{1,40}?)\s*</sup>\*?",
        _sup_to_tex, content,
    )

    # 2. Italic-wrapped single Greek letters: *α* -> $\alpha$
    def _italic_greek(m):
        return f"${_GREEK_TEX[m.group(1)]}$"
    greek_chars = "".join(_GREEK_TEX.keys())
    content = re.sub(rf"\*([{greek_chars}])\*", _italic_greek, content)

    # 3. Collapse degenerate empty math `$ $` (1+ whitespace required between).
    #    IMPORTANT: must NOT match `$$` -- that is the display-math delimiter.
    content = re.sub(r"\$[ \t]+\$", "", content)

    # 4. Unwrap `$$...$$` blocks the model used as ASCII flow diagrams (with `->` arrows)
    #    instead of as real math. LaTeX's math mode chokes on literal `&`, `->`, and
    #    arbitrary words, producing "Misplaced alignment tab character" errors in tectonic.
    #    Heuristic: a `$$...$$` block whose body contains `->` or `=>` AND has no
    #    backslash LaTeX commands is treated as a diagram, not math, and rewritten as
    #    a fenced code block.
    def _unwrap_fake_math(m):
        body = m.group(1).strip()
        looks_like_diagram = ("->" in body or "=>" in body) and not re.search(r"\\[A-Za-z]+", body)
        if looks_like_diagram:
            return f"\n\n```\n{body}\n```\n\n"
        return m.group(0)
    content = re.sub(r"\$\$\s*(.+?)\s*\$\$", _unwrap_fake_math, content, flags=re.DOTALL)

    # 5. Tighten inline math delimiters that have stray whitespace adjacent to `$`.
    #    Pandoc spec: the opening `$` must have a non-space char immediately to its
    #    right, the closing `$` a non-space immediately to its left. Otherwise
    #    pandoc treats the block as literal text and (when targeting LaTeX) escapes
    #    every `_` and `{`, which then breaks tectonic with "Missing $ inserted".
    #    Only fire when the body contains a `\` LaTeX command -- avoids touching
    #    incidental `$ price $`-style prose.
    content = re.sub(
        r"(?<!\$)\$[ \t]+([^$\n]*\\[A-Za-z]+[^$\n]*?)[ \t]+\$(?!\$)",
        lambda m: f"${m.group(1).strip()}$",
        content,
    )
    # Asymmetric variants (space on only one side).
    content = re.sub(
        r"(?<!\$)\$[ \t]+([^$\n]*\\[A-Za-z]+[^$\n]*?)\$(?!\$)",
        lambda m: f"${m.group(1).lstrip()}$",
        content,
    )
    content = re.sub(
        r"(?<!\$)\$([^$\n]*\\[A-Za-z]+[^$\n]*?)[ \t]+\$(?!\$)",
        lambda m: f"${m.group(1).rstrip()}$",
        content,
    )

    # 6. Rank10 -- the REAL tectonic crash. The writer glues consecutive display
    #    formulas sharing a delimiter: `$$F1$$F2$$` -> pandoc reads F1 as math,
    #    F2 as TEXT (its \frac/\text commands then break math mode -> "Missing $
    #    inserted"). First split shared delimiters (a `$$` flush against a
    #    formula-end before AND a \command after is a close+open that must become
    #    two blocks), then balance any residual odd count.
    content = _split_glued_display(content)
    content = _balance_display_math(content)

    # 7. Rank10 -- escape raw Unicode Greek INSIDE math spans to LaTeX commands
    #    (the real cause of the tectonic "no beta glyph" warning + silent glyph
    #    drop). Only touches text between $...$ / $$...$$ so prose Greek is left
    #    alone. Idempotent: \beta has no Unicode Greek to re-convert.
    content = _escape_unicode_math(content)

    return content


def _code_fence_spans(text: str):
    return [(m.start(), m.end()) for m in re.finditer(r"```.*?```", text, re.DOTALL)]


# A `$$` that is flush against a formula-ending char on the left AND a LaTeX
# `\command` on the right is a shared close+open delimiter between two glued
# display formulas. Requiring the `\command` lookahead means a `$$` that closes
# a formula and is followed by ordinary prose (" The model...") is NOT split.
_GLUED_DISPLAY_RE = re.compile(r"(?<=[}\)\]A-Za-z0-9])\$\$(?=\s*\\[A-Za-z])")


def _split_glued_display(content: str) -> str:
    """Split `$$F1$$F2$$`-style glued display formulas into separate blocks."""
    new, n = _GLUED_DISPLAY_RE.subn("$$\n\n$$", content)
    if n:
        print(f"[normalize_math] split {n} glued display-math delimiter(s)", flush=True)
    return new


def _balance_display_math(content: str) -> str:
    """Repair odd `$$` counts (outside fenced code) that crash tectonic with
    'Missing $ inserted'. With an odd count the LAST `$$` is unpaired; the
    unclosed formula can sit on EITHER side of it:
      - MISSING-OPENER: a LaTeX-bearing segment sits between the prev `$$` and
        the orphan -> wrap that segment as its own display block.
      - MISSING-CLOSER: a LaTeX-bearing segment trails AFTER the orphan (writer
        opened `$$F` and never closed) -> close it by appending `$$`.
      - otherwise (no LaTeX either side) -> drop the orphan delimiter.
    """
    fences = _code_fence_spans(content)
    def in_fence(pos):
        return any(a <= pos < b for a, b in fences)
    positions = [m.start() for m in re.finditer(r"\$\$", content) if not in_fence(m.start())]
    if len(positions) % 2 == 0:
        return content
    last = positions[-1]
    prev = positions[-2] if len(positions) >= 2 else None
    before = content[prev + 2:last] if prev is not None else ""
    after = content[last + 2:]
    has_latex = lambda s: bool(re.search(r"\\[A-Za-z]+", s))
    # Prefer the side whose dangling segment looks like a formula. Check the
    # AFTER side first (missing-closer is the common "$$F never closed" case
    # the review flagged); fall back to the BEFORE side (missing-opener).
    if has_latex(after) and not has_latex(before):
        # trailing formula opened by `last`, never closed: close it.
        # close at the end of the formula-ish run (first blank line) or doc end.
        m = re.search(r"\n\s*\n", after)
        cut = (last + 2 + m.start()) if m else len(content)
        print("[normalize_math] odd $$ count; closing unclosed trailing display formula", flush=True)
        return content[:cut] + "$$" + content[cut:]
    if prev is not None and has_latex(before):
        print("[normalize_math] odd $$ count; wrapping dangling display formula (missing opener)", flush=True)
        return (content[:prev + 2] + "\n\n$$" + before.strip() + "$$\n\n"
                + content[last + 2:])
    print(f"[normalize_math] odd $$ count ({len(positions)}); dropping orphan display delimiter", flush=True)
    return content[:last] + content[last + 2:]


_MATH_SPAN_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.DOTALL)

# Unicode math symbols that pandoc renders as math-mode-only LaTeX commands
# (e.g. ℝ -> \symbb{R}, which crashes tectonic with "allowed only in math mode"
# when the char sits in TEXT). bookv6 had ℝ, ∈, ∞, plus mathematical-alphanumeric
# styled letters (𝑑, 𝒃) handled separately via NFKC folding.
_MATH_UNICODE = {
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "𝔼": r"\mathbb{E}",
    "∈": r"\in", "∉": r"\notin", "∞": r"\infty", "∑": r"\sum",
    "∏": r"\prod", "∇": r"\nabla", "∂": r"\partial",
    "≈": r"\approx", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "×": r"\times", "⋅": r"\cdot",
    "²": r"^{2}", "³": r"^{3}", "⁴": r"^{4}", "ⁿ": r"^{n}",
    "₀": r"_{0}", "₁": r"_{1}", "₂": r"_{2}", "√": r"\sqrt{}",
}


def _escape_unicode_math(content: str) -> str:
    r"""Make Unicode math safe for tectonic:
      - NFKC-fold mathematical-alphanumeric styled letters (italic d -> d) everywhere
        (harmless in prose, fixes \symbb-style commands in math).
      - Inside math spans: Unicode Greek + math symbols -> LaTeX commands.
      - In text: stray math symbols (R-double-struck, in, infinity, ...) get
        wrapped in $...$ so they render in math mode instead of crashing tectonic.
    """
    import unicodedata
    content = "".join(
        unicodedata.normalize("NFKC", c) if 0x1D400 <= ord(c) <= 0x1D7FF else c
        for c in content
    )

    def _fix_span(m):
        span = m.group(0)
        for ch, tex in _GREEK_TEX.items():
            if ch in span:
                span = span.replace(ch, tex + " ")
        for ch, tex in _MATH_UNICODE.items():
            if ch in span:
                span = span.replace(ch, tex + " ")
        return span
    content = _MATH_SPAN_RE.sub(_fix_span, content)

    # Remaining math symbols are in TEXT -> wrap each in $...$.
    for ch, tex in _MATH_UNICODE.items():
        if ch in content:
            content = content.replace(ch, f"${tex}$")
    return content


def sanitize(content: str) -> str:
    """Strip model-hallucinated headers, meta-intros, trailing blocks, and normalize math.

    Layered cleanup:
      - structural: remove meta-intros, demote H1/H2, strip References/Conclusion
      - notation:   normalize math (HTML sub/sup -> LaTeX, italic Greek -> LaTeX)

    The assembler emits H1 (chapter) and H2 (section) headings itself, so any H1/H2 the
    model leaks must be demoted to H3 to avoid duplicate / out-of-order headings in the PDF.
    """
    if not content:
        return content
    text = content.strip()

    # Rank9: content-bleed. A leading line carrying an orphan "(ChN.M...)" outline
    # tag is a prior section's disambiguated title that leaked into this body
    # (bookv6 section 8.4 opened with "Multi-Objective Alignment... (Ch5.10.4)").
    # Drop that whole leading line, then strip any residual inline tag anywhere.
    text = re.sub(
        r"\A\s*(?:#{1,6}\s*|\*\*\s*)?[^\n]*\((?:Ch|Chapter)\s*\d+\.\d+[^)]*\)[^\n]*\n+",
        "", text,
    )
    text = re.sub(r"\s*\((?:Ch|Chapter)\s*\d+\.\d+(?:\.\d+)*[^)]*\)", "", text)

    # Structural pass.
    text = _META_INTRO_RE.sub("", text, count=1)

    def _demote(m):
        return "### " + m.group(0).lstrip("#").lstrip()
    text = _LEADING_HEADER_RE.sub(_demote, text)

    text = _TRAILING_BLOCK_RE.sub("", text)

    # Notation pass.
    text = normalize_math(text)

    return text.strip()


# Concept tracker -- pulled from headers + bold terms after each section is written,
# used to bar later sections from redefining the same thing.
_CONCEPT_HEADER_RE = re.compile(r"^#{3,4}\s+([^\n]{3,80})$", re.MULTILINE)
_CONCEPT_BOLD_RE   = re.compile(r"\*\*([A-Z][A-Za-z0-9 \-/]{2,40})\*\*")
_STOPWORDS = {"the", "and", "of", "for", "in", "to", "is", "with", "on", "by", "an", "a"}


def dedupe_outline(chapters: list, key_terms: list = None) -> list:
    """Inject 'already covered earlier' guidance into each section's prompt so the
    writer stops re-introducing concepts that show up in many sections.

    Auditing the hardcoded CHAPTERS reveals that ~15 terms (attention, scaling laws,
    fine-tuning, RAG, etc.) appear in 3+ sections each. Rather than rewrite the
    outline manually, we add a directive line to the second-and-later prompts that
    mention a given concept: 'X was introduced in Ch{N}.{M}; do NOT redefine -- focus
    only on the aspect specific to this section.'

    Idempotent: detects its own directive prefix and skips already-augmented prompts.
    """
    if key_terms is None:
        key_terms = [
            "attention", "self-attention", "multi-head attention", "scaling laws",
            "chinchilla", "transformer", "embedding", "tokeniz", "fine-tuning",
            "LoRA", "QLoRA", "RAG", "RLHF", "DPO", "quantization", "flash attention",
            "KV cache", "ZeRO", "FSDP", "PEFT", "prompt engineering",
            "chain-of-thought", "in-context learning",
        ]
    DIR_TAG = "[OUTLINE-DEDUPE]"

    first_seen = {}     # term lower -> "Ch{n}.{p}"
    out = []
    for ch in chapters:
        new_passes = []
        for pp in ch["passes"]:
            pr = pp["pr"]
            if DIR_TAG in pr:
                new_passes.append(pp)
                continue
            already_here = []
            for term in key_terms:
                if term.lower() in pr.lower():
                    if term.lower() not in first_seen:
                        first_seen[term.lower()] = f"Ch{ch['n']}.{pp['p']}"
                    elif first_seen[term.lower()] != f"Ch{ch['n']}.{pp['p']}":
                        already_here.append((term, first_seen[term.lower()]))
            if already_here:
                # Soft directive: tell the writer to spend a SHORT recap (1-2 sentences,
                # 1 citation) and then move on to what's new. Earlier wording said "do
                # NOT redefine, more than one sentence" which pushed writers into
                # uncomfortably-thin sections and contributed to citation collapse.
                directive = (
                    f" {DIR_TAG} The following concepts were introduced earlier; you may "
                    "open with a 1-2 sentence recap (with one citation if you have evidence) "
                    "but then spend the bulk of the section on the aspect specific to THIS "
                    "section -- do not re-derive from first principles or re-prove the basics. "
                    "Concepts: "
                    + "; ".join(f"{t} (first in {loc})" for t, loc in already_here) + "."
                )
                new_pp = {**pp, "pr": pr + directive}
            else:
                new_pp = pp
            new_passes.append(new_pp)
        out.append({**ch, "passes": new_passes})
    return out


def extract_concepts(content: str, limit: int = 15) -> list:
    """Pull a deduplicated list of concept names introduced in this section.

    Heuristic: take H3/H4 subheader text + bolded proper-noun-ish phrases, dedupe
    case-insensitively, drop pure stop-words. Used as 'already defined' signal
    for later sections so they don't re-introduce the same things.
    """
    if not content:
        return []
    # Strip fenced code blocks before extraction so Python comments don't leak in
    code_stripped = re.sub(r"```[\s\S]*?```", "", content)
    # Rank7: the 4B writer rarely emits H3/H4 or bold, so the header/bold regexes
    # found nothing in 144/150 bookv6 sections. Add two reliable signals:
    #   - ALL-CAPS acronyms (DPO, RLHF, RAG, MoE, PPO, ViT, ...) -- 2-6 chars
    #   - canonical method/model names present in the text (the seed aliases),
    #     which are exactly the concepts we must stop later sections redefining.
    acronyms = re.findall(r"\b[A-Z][A-Za-z]?[A-Z]{1,4}\b", code_stripped)
    canon = []
    if RESEARCH_AVAILABLE:
        try:
            low = code_stripped.lower()
            canon = [a for a in _research.canonical_seeds.SEED_MAP
                     if re.search(r"\b" + re.escape(a) + r"\b", low)]
        except Exception:
            canon = []
    raw = (_CONCEPT_HEADER_RE.findall(code_stripped)
           + _CONCEPT_BOLD_RE.findall(code_stripped)
           + acronyms + canon)
    seen = {}
    for c in raw:
        c = c.strip().rstrip(":.,").strip()
        if not c or len(c) < 3:
            continue
        # Drop code/comment-looking entries: lines starting with #, lines containing = or (
        if c.startswith("#") or c.startswith("//"):
            continue
        # Drop numbered prefixes like "1. " "1.2 "
        c = re.sub(r"^\d+(\.\d+)*[.)]\s*", "", c)
        # Drop trivial all-stopword headers
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", c)]
        if not words or all(w in _STOPWORDS for w in words):
            continue
        key = c.lower()
        if key not in seen:
            seen[key] = c
        if len(seen) >= limit:
            break
    return list(seen.values())


def prev_tail(content: str, n_words: int = CONTINUATION_WORDS) -> str:
    """Return the last ~n_words of a section, used as continuity context for the next call.

    Rank9: strip markdown headings and "(ChN.M...)" outline tags from the tail first,
    so the next section doesn't echo a heading/title it sees in the continuation
    context (the content-bleed source)."""
    if not content:
        return ""
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s.*$", "", content)
    cleaned = re.sub(r"\((?:Ch|Chapter)\s*\d+\.\d+[^)]*\)", "", cleaned)
    words = re.findall(r"\S+", cleaned)
    tail = " ".join(words[-n_words:])
    return tail


def build_context(state: dict, ch_n: int, pp_n: int) -> str:
    """Assemble a compact context block for the upcoming section.

    Includes:
      (a) Titles already covered IN THIS CHAPTER -- so the model doesn't reintroduce them
      (b) A short trailing excerpt of the immediately previous section for prose continuity
      (c) A condensed "concepts already introduced in earlier chapters" list -- cross-chapter
          knowledge map, so e.g. the agents chapter doesn't redefine attention.
    Kept short (~250 words total) to avoid blowing the model's context window.
    """
    passes = state.get("passes", {})
    in_chapter_titles = []
    earlier_chapter_titles = []
    prev_content = ""
    prev_key = (-1, -1)  # (ch, pp) -- prefer immediate previous regardless of chapter boundary
    for k, v in passes.items():
        v_ch = v.get("ch", 0)
        v_pp = v.get("pp", 0)
        v_t = v.get("title", "")
        if v_ch == ch_n and v_pp < pp_n:
            in_chapter_titles.append((v_pp, v_t))
        elif v_ch < ch_n:
            earlier_chapter_titles.append((v_ch, v_pp, v_t))
        if (v_ch, v_pp) > prev_key and (v_ch, v_pp) < (ch_n, pp_n):
            prev_key = (v_ch, v_pp)
            prev_content = v.get("content", "")
    in_chapter_titles.sort()
    earlier_chapter_titles.sort()

    parts = []
    if in_chapter_titles:
        titles = "; ".join(f"{p}. {t}" for p, t in in_chapter_titles)
        parts.append(
            "PRIOR SECTIONS IN THIS CHAPTER (do NOT reintroduce these concepts from scratch -- "
            f"reference and build on them):\n{titles}"
        )
    if earlier_chapter_titles:
        # Build a concept-first-occurrence map across all completed earlier sections.
        # Surfacing the explicit concept names (not just section titles) lets the writer
        # know exactly which terms must NOT be redefined from scratch.
        already_defined = {}  # concept lower -> "Ch{N}.{M}"
        for k, v in passes.items():
            v_ch = v.get("ch", 0)
            v_pp = v.get("pp", 0)
            if (v_ch, v_pp) >= (ch_n, pp_n):
                continue
            for c in v.get("concepts", []) or []:
                low = c.lower()
                if low not in already_defined:
                    already_defined[low] = (c, v_ch, v_pp)

        if already_defined:
            # Show top ~25 prior concepts, sorted by chapter to keep it scannable.
            items = sorted(already_defined.values(), key=lambda x: (x[1], x[2]))[:25]
            cline = "; ".join(f"\"{c}\" (Ch{ch}.{pp})" for c, ch, pp in items)
            parts.append(
                "ALREADY DEFINED in earlier sections -- DO NOT redefine, DO NOT re-derive "
                "from scratch, DO NOT spend more than one sentence on; reference by name and "
                "go straight into what's NEW for this section:\n" + cline
            )
        else:
            # Fallback to title-only signal if concept extraction hasn't populated yet.
            by_ch = {}
            for ch, pp, t in earlier_chapter_titles:
                by_ch.setdefault(ch, []).append(t)
            chap_lines = []
            for ch in sorted(by_ch.keys()):
                chap_lines.append(f"  Ch{ch}: " + "; ".join(by_ch[ch]))
            parts.append(
                "CONCEPTS INTRODUCED IN EARLIER CHAPTERS (do NOT redefine these -- "
                "reference them by name and assume the reader knows them):\n" + "\n".join(chap_lines)
            )
    if prev_content:
        tail = prev_tail(prev_content, CONTINUATION_WORDS)
        if tail:
            parts.append(
                "LAST PARAGRAPH OF THE IMMEDIATELY PRECEDING SECTION (continue the prose "
                f"naturally from here, do not echo it):\n{tail}"
            )
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n---\n\n"


def review_section(client, content: str, ch_t: str, pp_t: str) -> dict:
    """LLM-as-judge pass. Returns dict with depth/coherence/format scores + issues string.

    On any parse failure returns a permissive 'pass' so the pipeline never blocks on the
    reviewer itself -- the reviewer is advisory.
    """
    excerpt = content if len(content) < 6000 else content[:3000] + "\n...\n" + content[-2500:]
    prompt = (
        f"Section context: chapter '{ch_t}', section '{pp_t}'.\n\n"
        f"SECTION TEXT:\n{excerpt}\n\n"
        "Return the JSON object now."
    )
    try:
        raw, _ = client.generate(prompt=prompt, system=REVIEW_SYS, temperature=0.2, num_predict=200)
    except Exception as e:
        log(f"  [REVIEW] call failed: {e}")
        return {"depth": 10, "coherence": 10, "format": 10, "issues": "", "_skipped": True}

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"depth": 10, "coherence": 10, "format": 10, "issues": "", "_skipped": True}
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {"depth": 10, "coherence": 10, "format": 10, "issues": "", "_skipped": True}
    for k in ("depth", "coherence", "format"):
        try:
            data[k] = int(data.get(k, 10))
        except Exception:
            data[k] = 10
    data["issues"] = str(data.get("issues", ""))[:200]
    return data


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def notify(title: str, body: str):
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{body}" with title "{title}"'],
            timeout=5, capture_output=True,
        )
    except:
        pass


# ============================================================================
# State management
# ============================================================================

# --- W3: process-safe state I/O ---------------------------------------------
# Why: state.json was written via plain open()+json.dump (non-atomic) and
# guarded only by threading.Lock (does NOT cross processes). When the runner
# watchdog respawned a fresh pipeline before the prior one was confirmed dead,
# two writers raced on state.json -> JSON parse failure on next resume.
# Fix: atomic write (tempfile + os.replace) + cross-process fcntl.flock on a
# dedicated .lock sidecar, plus a startup PID lock that refuses double-spawn.

class _FileLock:
    """Cross-process advisory lock via fcntl.flock on a sidecar .lock file."""
    def __init__(self, path: Path, exclusive: bool = True):
        self.path = path
        self.exclusive = exclusive
        self._fh = None

    def __enter__(self):
        if not _HAVE_FCNTL:
            return self  # no-op on non-POSIX
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        fcntl.flock(self._fh.fileno(), mode)
        return self

    def __exit__(self, *exc):
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def _atomic_write_json(path: Path, obj) -> None:
    """Atomic JSON write: temp in same dir -> fsync -> os.replace (POSIX atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=str(path.parent),
        prefix=f".{path.name}.", suffix=".tmp",
        delete=False, encoding="utf-8",
    )
    try:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, str(path))
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def _state_lock_path() -> Path:
    return STATE_FILE.with_suffix(STATE_FILE.suffix + ".lock")


def load_state():
    if not STATE_FILE.exists():
        return {"passes": {}, "total_words": 0, "total_tokens": 0, "total_calls": 0}
    with _FileLock(_state_lock_path(), exclusive=False):
        with open(STATE_FILE) as f:
            return json.load(f)


def save_state(state):
    with _FileLock(_state_lock_path(), exclusive=True):
        _atomic_write_json(STATE_FILE, state)


# --- W3: pipeline PID lock --------------------------------------------------
# Refuses to start a second pipeline against the same output basename. Killed
# the "watchdog respawn before original is dead" failure mode at the source.
_PID_LOCK_FH = None


def acquire_pipeline_lock() -> None:
    """Acquire an exclusive PID lock on <state>.pid; exit(2) if another holder exists."""
    global _PID_LOCK_FH
    if not _HAVE_FCNTL:
        return  # best-effort on non-POSIX
    pid_path = STATE_FILE.with_suffix(STATE_FILE.suffix + ".pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(pid_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EACCES):
            fh.seek(0)
            existing = fh.read().strip() or "?"
            fh.close()
            print(
                f"[deep_research] REFUSE: another pipeline holds {pid_path} "
                f"(pid={existing}). Kill it first, or use a different --out-name.",
                file=sys.stderr, flush=True,
            )
            sys.exit(2)
        raise
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    os.fsync(fh.fileno())
    _PID_LOCK_FH = fh  # keep open for process lifetime; closes on exit
    atexit.register(_release_pipeline_lock, pid_path)


def _release_pipeline_lock(pid_path: Path) -> None:
    global _PID_LOCK_FH
    if _PID_LOCK_FH is not None:
        try:
            fcntl.flock(_PID_LOCK_FH.fileno(), fcntl.LOCK_UN)
            _PID_LOCK_FH.close()
        except Exception:
            pass
        _PID_LOCK_FH = None
    try:
        pid_path.unlink()
    except OSError:
        pass


# ============================================================================
# Generation
# ============================================================================

# W1: regex strips the legacy "Target: 1800-2500 words." trailer that every
# hardcoded prompt in CHAPTERS carries. We replace it with a dynamic target
# derived from the actual evidence count so the writer is not asked to pad.
_LEGACY_TARGET_RE = re.compile(
    r"\s*Target:\s*\d{2,5}\s*[-–—]\s*\d{2,5}\s*words?[^.]*\.?\s*$",
    re.IGNORECASE,
)


def compute_target_words(n_evidence: int, has_research: bool) -> int:
    """Derive a per-section target from the number of retained evidence sources.

    Why: forcing 1800-2500 words from a 4B writer was the root cause of looping
    and citation gaming -- the model has nothing to say once the topic is
    actually covered, so it pads and drops citations to scrape a verifier score.
    Let length follow content.
    """
    if not has_research:
        return WORD_TARGET_NO_EVIDENCE
    if n_evidence <= 0:
        return WORD_TARGET_FLOOR
    return max(WORD_TARGET_FLOOR, min(WORD_BUDGET, n_evidence * WORD_TARGET_PER_SOURCE))


def _prepare_prompt(prompt: str, target_words: int, has_evidence: bool) -> str:
    """Strip the hardcoded 'Target: NNNN-NNNN words.' trailer and append a
    dynamic, honest length directive that tells the writer to stop when the
    topic is covered."""
    cleaned = _LEGACY_TARGET_RE.sub("", prompt).rstrip()
    if has_evidence:
        suffix = (
            f"\n\nLength: aim for approximately {target_words} words. "
            "If you run out of grounded material from the EVIDENCE block, "
            "stop -- do not pad. A shorter section with accurate citations "
            "is preferred over a long section that drops citations or repeats."
        )
    else:
        suffix = (
            f"\n\nLength: aim for approximately {target_words} words. "
            "Stop when the topic is fully covered -- do not pad."
        )
    return cleaned + suffix


def gen(client, ch_n, ch_t, pp_n, pp_t, prompt, target_words, context_block="", evidence_block=""):
    # W1: num_predict tightly capped to the dynamic target so the writer
    # physically cannot ramble past it. 2.0x is the empirical ratio between
    # tokens and words for gemma3/qwen3.5 English output.
    num_predict = max(800, min(int(target_words * 2.0), 4000))
    has_evidence = bool(evidence_block.strip())
    prepared_prompt = _prepare_prompt(prompt, target_words, has_evidence)
    user_prompt = "%s%sChapter %d: %s -- Section %d: %s\n\n%s" % (
        context_block, evidence_block, ch_n, ch_t, pp_n, pp_t, prepared_prompt,
    )
    # W1: acceptance threshold is now 40% of target (was 25% of 4200 == 1050w
    # regardless of context). 40% gives the writer permission to stop early
    # without making us accept a one-paragraph stub.
    min_accept = max(200, int(target_words * 0.4))
    for attempt in range(3):
        try:
            content, stats = client.generate(
                prompt=user_prompt,
                system=SYS,
                temperature=0.7,
                num_predict=num_predict,
            )
            content = sanitize(content)
            w = wc(content)
            if w >= min_accept or attempt == 2:
                return content, stats, w
        except Exception as e:
            log(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return "", {"tokens": 0, "tps": 0, "elapsed": 0}, 0


# ============================================================================
# Assemble Markdown
# ============================================================================

def assemble(state):
    by_ch = defaultdict(list)
    for k, v in state.get("passes", {}).items():
        by_ch[v["ch"]].append((v["pp"], v))

    book = """---
title: "Large Language Models: A Comprehensive Handbook"
subtitle: "Mathematics, Architecture, Training, Alignment, Deployment, and the Future of AI"
author: Generated by Deep Agent LLM Book Pipeline | %s | %s
lang: en
geometry: "margin=1.5in"
fontsize: 11pt
---

# Large Language Models: A Comprehensive Handbook

_A Comprehensive Handbook: Mathematics, Architecture, Training, Alignment, Deployment, and the Future of AI_

---

""" % (MODEL, datetime.now().strftime("%B %Y"))

    for ch_n in sorted(by_ch.keys()):
        sections = sorted(by_ch[ch_n], key=lambda x: x[0])
        ch_t = sections[0][1]["ch_t"]
        book += "# Chapter %d: %s\n\n" % (ch_n, ch_t)
        for pp_n, result in sections:
            body = sanitize(result["content"])
            book += "## %d. %s\n\n%s\n\n" % (pp_n, result["title"], body)
        book += "---\n\n"

    # Single References page assembled from per-section sources collected by the research layer.
    refs_block = _build_references(state)
    if refs_block:
        book += refs_block

    with open(FINAL_MD, "w", encoding="utf-8") as f:
        f.write(book)
    log(f"[ASSEMBLE] {FINAL_MD} ({len(book):,} chars)")


def _build_references(state) -> str:
    """Collect unique sources across all sections and render a single References page.

    Sources are deduped by URL (then by id as fallback) and emitted in the order
    they first appear in the book.
    """
    seen = {}
    order = []
    for v in state.get("passes", {}).values():
        for s in v.get("sources", []) or []:
            key = s.get("url") or s.get("id")
            if not key or key in seen:
                continue
            seen[key] = s
            order.append(key)
    if not order:
        return ""

    lines = ["# References", ""]
    for n, key in enumerate(order, start=1):
        s = seen[key]
        # neutral_author never prints a search-tool brand (Tavily/DDG/Brave) as author.
        authors = _research.types.neutral_author(
            s.get("authors"), s.get("provider", ""), s.get("url", "")
        ) if RESEARCH_AVAILABLE else ", ".join(s.get("authors") or [])
        year = f" ({s['year']})" if s.get("year") else ""
        title = s.get("title", "Untitled")
        url = s.get("url", "")
        # Build attribution so empty-author + present-year never glues "(2024)_Title_".
        attribution = (authors + year).strip()
        prefix = f"{attribution}. " if attribution else ""
        lines.append(f"[{n}] {prefix}_{title}_. <{url}>")
        lines.append("")
    return "\n".join(lines) + "\n"


# ============================================================================
# Report
# ============================================================================

def make_report(state, total_time):
    all_tps = [v["tps"] for v in state.get("passes", {}).values() if v.get("tps")]
    avg_tps = sum(all_tps) / max(len(all_tps), 1)
    report = {
        "generated_at": datetime.now().isoformat(),
        "model": MODEL,
        "total_time_min": round(total_time / 60, 1),
        "total_calls": state.get("total_calls", 0),
        "total_tokens": state.get("total_tokens", 0),
        "total_words": state.get("total_words", 0),
        "avg_tps": round(avg_tps, 1),
        "pages": state.get("total_words", 0) // 400,
        "passes": {
            k: {
                "ch": v["ch"],
                "pp": v["pp"],
                "title": v["title"],
                "wc": v["wc"],
                "tokens": v["tokens"],
                "tps": v["tps"],
            }
            for k, v in state.get("passes", {}).items()
        },
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    log(f"[REPORT] {REPORT_FILE}")


# ============================================================================
# PDF Render
# ============================================================================

def _ensure_native_libs():
    """On macOS arm64, weasyprint needs brew's pango/gobject libs on the dyld path.

    Without this the framework python3 fails with `OSError: cannot load library
    'libgobject-2.0-0'` even when weasyprint itself is pip-installed.
    """
    if sys.platform != "darwin":
        return
    for brew_lib in ("/opt/homebrew/lib", "/usr/local/lib"):
        if Path(brew_lib).exists():
            cur = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if brew_lib not in cur.split(":"):
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                    f"{brew_lib}:{cur}" if cur else brew_lib
                )


def _prepare_clean_md() -> Path:
    """Strip the YAML front matter from FINAL_MD and rewrite stray `---` HRs
    so pandoc doesn't reinterpret them as section breaks. Returns CLEAN_MD path.
    """
    with open(FINAL_MD) as f:
        content = f.read()
    if content.startswith("---"):
        end = content.find("\n---\n", 4)
        if end >= 0:
            content = content[end + 5:]
    lines = content.split("\n")
    in_code = False
    fixed = []
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            fixed.append(line)
        elif line.strip() == "---" and not in_code:
            fixed.append("* * *")
        else:
            fixed.append(line)
    clean = "\n".join(fixed)
    # Rank10 safety net: repair display math across the WHOLE assembled doc
    # before pandoc->tectonic, so math mangling from any section (or an
    # already-generated book) can't crash tectonic at render time.
    clean = _split_glued_display(clean)
    clean = _balance_display_math(clean)
    clean = _escape_unicode_math(clean)
    CLEAN_MD.write_text(clean)
    return CLEAN_MD


def _render_via_tectonic(clean_md: Path) -> bool:
    """LaTeX-based render via pandoc + tectonic. Paper-quality math output.

    Tectonic auto-fetches missing TeX packages on first use, so the first call
    can be slow (~30s); subsequent calls reuse the cache.
    """
    import shutil
    if not shutil.which("tectonic"):
        return False
    r = subprocess.run(
        [
            "pandoc", str(clean_md), "-o", str(FINAL_PDF),
            "--pdf-engine=tectonic",
            "--toc", "--toc-depth=3",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "linkcolor=blue",
            "-V", "urlcolor=blue",
            "--metadata", "title=Large Language Models Handbook",
        ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"[PDF] tectonic failed ({r.returncode}); stderr tail: "
            + (r.stderr or "")[-400:].replace("\n", " | "))
        return False
    sz = os.path.getsize(FINAL_PDF)
    log(f"[PDF] {FINAL_PDF} ({sz/1024:.0f} KB) via tectonic")
    return True


def _render_via_weasyprint(clean_md: Path) -> bool:
    """Fallback render: pandoc --mathml -> weasyprint. Lower-quality math,
    but works without LaTeX. Used when tectonic is unavailable."""
    _ensure_native_libs()
    import weasyprint, warnings
    warnings.filterwarnings("ignore")
    subprocess.run(
        [
            "pandoc", str(clean_md), "-o", str(FINAL_HTML),
            "--standalone", "--toc", "--toc-depth=3",
            "--mathml",
            "--metadata", "title=Large Language Models Handbook",
        ],
        capture_output=True,
    )
    weasyprint.HTML(filename=str(FINAL_HTML)).write_pdf(str(FINAL_PDF))
    sz = os.path.getsize(FINAL_PDF)
    log(f"[PDF] {FINAL_PDF} ({sz/1024:.0f} KB) via weasyprint+MathML (fallback)")
    return True


def render_pdf():
    """Render the assembled book.md to book.pdf.

    Primary engine: pandoc + tectonic (LaTeX) -- paper-quality math typesetting.
    Fallback:       pandoc --mathml + weasyprint -- lower quality but pure-Python.

    Tectonic is preferred because WeasyPrint's MathML rendering ships BOTH the
    Unicode rendering AND the raw LaTeX source side-by-side in many display
    formulas -- visually broken. LaTeX has no such issue.
    """
    log("[RENDER] Converting Markdown to PDF...")
    try:
        clean_md = _prepare_clean_md()
        if _render_via_tectonic(clean_md):
            return True
        log("[RENDER] tectonic unavailable or failed; trying weasyprint fallback")
        return _render_via_weasyprint(clean_md)
    except Exception as e:
        log(f"[PDF] Failed: {e}")
        return False


# ============================================================================
# Main pipeline
# ============================================================================

def run(batch=2, start_ch=1, start_pp=1, end_ch=None, render=True, review=False,
        research=True, topic=None, n_chapters=None, n_passes=None, out_name=None):
    if out_name:
        _rebind_output_paths(out_name)
        print(f"[deep_research] output paths rebound: prefix={out_name!r}", flush=True)
    migrate_legacy_outputs()

    # W3: refuse to run if another pipeline already owns this output basename.
    # Must happen after _rebind_output_paths so STATE_FILE points at the right file.
    acquire_pipeline_lock()

    # Clear log on fresh start
    if start_ch == 1 and start_pp == 1:
        open(LOG_FILE, "w").close()

    research_enabled = bool(research) and RESEARCH_AVAILABLE
    research_label = "on" if research_enabled else ("requested but unavailable" if research and not RESEARCH_AVAILABLE else "off")

    # Stage 3: planner-generated outline. If --topic was passed AND research is on,
    # ask the planner agent to produce a fresh CHAPTERS list for that topic.
    # Otherwise fall back to the hardcoded CHAPTERS (LLM book) at module level.
    chapters_to_use = CHAPTERS
    outline_source = "hardcoded (Large Language Models)"
    if topic:
        if not research_enabled:
            print(f"[planner] ERROR: --topic requires research layer; falling back to hardcoded CHAPTERS")
        else:
            plan_kwargs = {}
            if n_chapters: plan_kwargs["n_chapters"] = n_chapters
            if n_passes:   plan_kwargs["n_passes"]   = n_passes
            planned = _research.planner.plan_outline(topic, **plan_kwargs)
            if planned:
                chapters_to_use = planned
                outline_source = (f"planner-generated for topic {topic!r} "
                                  f"({len(planned)}x{len(planned[0]['passes'])})")
            else:
                print(f"[planner] WARN: outline generation failed -- falling back to hardcoded CHAPTERS")

    # Outline dedupe: inject 'already-introduced' directives into prompts where a
    # high-traffic concept (attention, scaling laws, RAG, ...) reappears across
    # chapters. Stops the writer re-deriving the same things in Ch1, Ch4, and Ch12.
    chapters_to_use = dedupe_outline(chapters_to_use)
    n_dedupe_directives = sum(
        1 for ch in chapters_to_use for pp in ch["passes"] if "[OUTLINE-DEDUPE]" in pp["pr"]
    )

    total_sections = sum(len(c["passes"]) for c in chapters_to_use)
    print("=" * 70)
    print("  Deep Research Pipeline -- Agentic Book Generator")
    print(f"  Topic: {topic or 'Large Language Models'}")
    print(f"  Outline: {outline_source}")
    print(f"  Sections: %d (%d chapters) | {n_dedupe_directives} prompts auto-augmented with dedupe directives" % (total_sections, len(chapters_to_use)))
    print("  Writer: %s | Batch: %d | Review: %s | Research: %s" % (
        MODEL, batch, "on" if review else "off", research_label,
    ))
    if research_enabled:
        # Rank13: log the EFFECTIVE provider list (after key/session-disable
        # filtering), not just the configured one, so silent degradation
        # (e.g. Tavily disabled with no key) is visible at startup.
        effective = _research.search.available_providers(_research.PROVIDERS_DEFAULT)
        print("  Research:  query/gen=%s  embed=%s  providers=%s (effective: %s)" % (
            _research.QUERY_GEN_MODEL, _research.EMBED_MODEL,
            ",".join(_research.PROVIDERS_DEFAULT),
            ",".join(effective) or "NONE",
        ))
    print("=" * 70)
    print()

    client = OllamaClient()
    if not client.health():
        print("[FATAL] Ollama not reachable at", OLLAMA_BASE)
        print("  Start Ollama: ollama serve")
        sys.exit(1)
    print("[OK] Ollama connected")

    state = load_state()
    all_tasks = [
        (ch["n"], ch["t"], pp["p"], pp["t"], pp["pr"], pp["w"])
        for ch in chapters_to_use
        if ch["n"] >= start_ch and (end_ch is None or ch["n"] <= end_ch)
        for pp in ch["passes"]
        if pp["p"] >= start_pp or ch["n"] > start_ch
    ]
    if end_ch is not None:
        print(f"  [LIMIT] only running chapters {start_ch}..{end_ch} ({len(all_tasks)} sections)")
    total = len(all_tasks)
    done = sum(
        1
        for t in all_tasks
        if "%d.%d" % (t[0], t[2]) in state.get("passes", {})
    )

    print(f"Tasks: {total} ({total - done} remaining, {done} done)")
    print()

    t0 = time.time()
    section_vectors = []  # Rank7: (key, embedding) of finalized sections, for the content-dedup gate
    for i, (ch_n, ch_t, pp_n, pp_t, prompt, _legacy_budget) in enumerate(all_tasks):
        # W1: ignore _legacy_budget (the hardcoded 4200 from CHAPTERS). The
        # real target is computed below from evidence count per section.
        key = "%d.%d" % (ch_n, pp_n)
        if key in state.get("passes", {}):
            print("[SKIP %d/%d] Ch%d.%d: %s -- DONE" % (i + 1, total, ch_n, pp_n, pp_t))
            continue

        print()
        print("[%d/%d] Ch%d.%d: %s" % (i + 1, total, ch_n, pp_n, pp_t))
        print("-" * 60)
        sys.stdout.flush()

        context_block = build_context(state, ch_n, pp_n)
        if context_block:
            print(f"  [CTX] feeding {wc(context_block)}w of prior-section memory")

        # Stage 2 + agentic loop: research -> write -> verify -> if grounding low, re-search + rewrite.
        evidence_block = ""
        ranked = []
        queries = []
        verify_res = None
        rounds_log = []
        content, stats, w = "", {}, 0

        if research_enabled:
            current_hint = None
            best_round = None  # Rank4: keep the highest-grounding round, not the last
            for round_n in range(1, _research.MAX_RESEARCH_ROUNDS + 1):
                t_r = time.time()
                queries = _research.query_gen.queries_for(prompt, ch_t, pp_t, reviewer_hint=current_hint)
                raw_sources = _research.search.gather(
                    queries, providers=_research.PROVIDERS_DEFAULT, per_provider_k=3,
                )
                # Rank5: known-item retrieval. Resolve canonical paper/method names
                # mentioned in this section to arxiv IDs and fetch them directly, so
                # foundations (Vaswani, BERT, GPT-3, ...) are present even when the
                # keyword search never surfaces them. Seeds are protected from the
                # prefilter cosine gate and reserved by rank6's primary_floor.
                seed_ids = _research.canonical_seeds.resolve_seeds(f"{pp_t}\n{prompt}")
                seed_sources = _research.search.arxiv_by_id(seed_ids) if seed_ids else []
                protected = {s.id for s in seed_sources}
                if seed_sources:
                    print(f"  [SEED] injected {len(seed_sources)} canonical paper(s): "
                          + ", ".join(s.id for s in seed_sources))
                raw_sources = seed_sources + raw_sources
                # Prefilter: drop obviously off-topic + noisy-domain results BEFORE rank
                # so the top-8 pool isn't polluted with YouTube/DDG/etc. false matches.
                prefiltered = _research.notes.prefilter(
                    raw_sources, prompt, embed_model=_research.EMBED_MODEL,
                    protected_ids=protected,
                )
                ranked = _research.notes.rank(
                    prefiltered, prompt,
                    top_k=_research.TOP_K_DEFAULT, embed_model=_research.EMBED_MODEL,
                    precomputed=True,            # Rank13: reuse prefilter's cached relevance
                    primary_floor=_research.PRIMARY_FLOOR,  # Rank6: reserve arxiv/wiki slots
                )
                ranked = _research.notes.enrich_top_sources(
                    ranked, top_n=_research.FULL_TEXT_TOP_N, max_words_per=_research.FULL_TEXT_MAX_WORDS,
                )
                evidence_block = _research.notes.format_for_prompt(ranked)
                print(f"  [RESEARCH r{round_n}] {len(queries)} queries -> gathered {len(raw_sources)} "
                      f"-> prefilter {len(prefiltered)} -> ranked {len(ranked)} "
                      f"({wc(evidence_block)}w incl. {min(_research.FULL_TEXT_TOP_N, len(ranked))} full-text) "
                      f"in {time.time()-t_r:.1f}s")

                # W1: target follows evidence -- N sources * 220w, capped at WORD_BUDGET.
                target_words = compute_target_words(len(ranked), has_research=True)
                print(f"  [W1] target={target_words}w (from {len(ranked)} evidence sources)")

                content, stats, w = gen(
                    client, ch_n, ch_t, pp_n, pp_t, prompt, target_words,
                    context_block=context_block, evidence_block=evidence_block,
                )
                if not content:
                    break

                # Drop orphan/out-of-range [N], N-prefixed placeholders ([N1],
                # [N3,N7]), and provider-as-author attributions. Runs BEFORE
                # verify so the judge scores the cleaned text.
                content, n_dropped_cites = _research.notes.clean_citations(content, len(ranked))
                if n_dropped_cites:
                    print(f"  [CITE-FIXUP] dropped {n_dropped_cites} bad citation markers/attributions")
                # Telemetry guard: a citation-shaped [N..] surviving clean_citations
                # means the cleaner regex has a gap -- surface it without eating
                # legit math like [N=512]. (citation-shape = N + digits/commas only)
                _resid = re.findall(r"\[\s*[Nn]\d*(?:\s*,\s*[Nn]?\d+)*\s*\]", content)
                if _resid:
                    print(f"  [CITE-GUARD] WARN residual placeholder survived cleaner: {_resid[:3]}")

                t_v = time.time()
                verify_res = _research.verify.verify_section(
                    content, ranked, model=_research.JUDGE_MODEL,
                )
                rounds_log.append({
                    "round": round_n,
                    "grounding": verify_res["grounding"],
                    "n_citations": verify_res["n_citations"],
                    "n_weak": len(verify_res["weak_citations"]),
                })
                print(f"  [VERIFY r{round_n}] grounding={verify_res['grounding']:.2f} "
                      f"({verify_res['n_citations']} citations, {len(verify_res['weak_citations'])} weak) "
                      f"in {time.time()-t_v:.1f}s")

                # Rank4: record this round as a candidate. "Better" = has citations
                # beats zero-citations; tie-break by grounding. So a 0.16 round-1 with
                # citations is never overwritten by a 0.0 round-2 with none.
                cand = {
                    "content": content, "w": w, "stats": stats, "ranked": ranked,
                    "queries": queries, "verify": verify_res, "evidence": evidence_block,
                }
                def _round_score(c):
                    g = c["verify"].get("grounding") or 0.0
                    has_cites = (c["verify"].get("n_citations") or 0) > 0
                    return (1 if has_cites else 0, g)
                if best_round is None or _round_score(cand) > _round_score(best_round):
                    best_round = cand

                if verify_res["grounding"] >= _research.MIN_GROUNDING:
                    break
                if round_n >= _research.MAX_RESEARCH_ROUNDS:
                    break
                hint_src = verify_res.get("weak_summary") or "previous draft had unsupported citations"
                current_hint = (
                    f"Previous draft's citations did not match their sources: {hint_src}. "
                    "Bias query generation toward canonical / primary sources for the specific claims that failed."
                )
            # Rank4: ship the BEST round, not whatever the loop happened to end on.
            if best_round is not None:
                content = best_round["content"]; w = best_round["w"]
                stats = best_round["stats"]; ranked = best_round["ranked"]
                queries = best_round["queries"]; verify_res = best_round["verify"]
                evidence_block = best_round["evidence"]
                if len(rounds_log) > 1:
                    print(f"  [BEST-ROUND] shipped grounding={verify_res['grounding']:.2f} "
                          f"({verify_res['n_citations']} cites) from the best of {len(rounds_log)} rounds")
        else:
            # No research layer -- straight gen as in legacy mode.
            target_words = compute_target_words(0, has_research=False)
            print(f"  [W1] target={target_words}w (no research)")
            content, stats, w = gen(
                client, ch_n, ch_t, pp_n, pp_t, prompt, target_words,
                context_block=context_block, evidence_block="",
            )

        sources_json = [s.to_dict() for s in ranked] if ranked else []
        queries_used = [q.q for q in queries] if queries else []
        tokens = stats.get("tokens", 0) if stats else 0
        tps = stats.get("tps", 0) if stats else 0
        elapsed_min = (stats.get("elapsed", 0) / 60) if stats else 0

        if not content:
            log(f"  FAILED: no content for Ch{ch_n}.{pp_n}")
            continue

        # Optional prose-quality review (separate axis from citation grounding).
        # The verify loop above already handles re-research; review here only triggers
        # a single in-place regeneration with the same evidence if prose quality is poor.
        review_result = None
        if review and content:
            review_result = review_section(client, content, ch_t, pp_t)
            worst = min(review_result.get("depth", 10),
                        review_result.get("coherence", 10),
                        review_result.get("format", 10))
            issues = review_result.get("issues", "")
            print(f"  [REVIEW] depth={review_result.get('depth')} "
                  f"coherence={review_result.get('coherence')} "
                  f"format={review_result.get('format')} | {issues}")
            if worst < MIN_REVIEW_SCORE and not review_result.get("_skipped"):
                fix_hint = (
                    f"Reviewer feedback on the previous draft: {issues}. "
                    "Rewrite the section addressing this issue while keeping all factual content. "
                )
                regen_prompt = fix_hint + prompt
                print(f"  [REVIEW] score {worst} < {MIN_REVIEW_SCORE} -- regenerating once")
                content2, stats2, w2 = gen(
                    client, ch_n, ch_t, pp_n, pp_t, regen_prompt, target_words,
                    context_block=context_block, evidence_block=evidence_block,
                )
                if content2 and w2 >= max(200, int(target_words * 0.4)):
                    content, w = content2, w2
                    tokens = stats2.get("tokens", tokens)
                    tps = stats2.get("tps", tps)
                    elapsed_min = stats2.get("elapsed", 0) / 60

        # Rank7: runtime content-similarity gate. Embed this section and compare
        # to all prior sections; if cosine >= DUP_THRESHOLD it duplicates earlier
        # material (bookv6 had 65 such pairs, incl a 5-way DPO/RLHF redefinition).
        # Regenerate ONCE from a distinct angle. The repaired concept tracker +
        # ALREADY-DEFINED list should make this fire rarely.
        _DUP_THRESHOLD = 0.85
        cur_vec = None
        if RESEARCH_AVAILABLE and content:
            try:
                _vecs = _research.embeddings.embed([content[:1500]], model=_research.EMBED_MODEL)
                cur_vec = _vecs[0] if _vecs else None
            except Exception:
                cur_vec = None
            if cur_vec is not None and section_vectors:
                _sims = [(_research.embeddings.cosine(cur_vec, pv), pk) for pk, pv in section_vectors]
                max_sim, dup_key = max(_sims, key=lambda x: x[0])
                if max_sim >= _DUP_THRESHOLD:
                    print(f"  [DEDUP] Ch{ch_n}.{pp_n} cosine {max_sim:.2f} vs Ch{dup_key} -- regenerating distinct angle")
                    avoid = (f"IMPORTANT: Chapter section Ch{dup_key} already covers very similar "
                             "material. Write THIS section from a clearly DISTINCT angle: do not "
                             "restate shared definitions -- reference them briefly and focus on what "
                             "is unique to this section's specific title.\n\n")
                    content2, stats2, w2 = gen(
                        client, ch_n, ch_t, pp_n, pp_t, avoid + prompt, target_words,
                        context_block=context_block, evidence_block=evidence_block,
                    )
                    if content2 and w2 >= max(200, int(target_words * 0.4)):
                        if ranked:
                            content2, _ = _research.notes.clean_citations(content2, len(ranked))
                        content, w = content2, w2
                        # Keep stats + verify aligned with the regenerated content
                        # (else tokens/tps and grounding describe the discarded draft).
                        tokens = stats2.get("tokens", tokens)
                        tps = stats2.get("tps", tps)
                        elapsed_min = stats2.get("elapsed", 0) / 60 or elapsed_min
                        if ranked:
                            try:
                                verify_res = _research.verify.verify_section(
                                    content, ranked, model=_research.JUDGE_MODEL)
                            except Exception:
                                pass
                        try:
                            _vecs = _research.embeddings.embed([content[:1500]], model=_research.EMBED_MODEL)
                            cur_vec = _vecs[0] if _vecs else cur_vec
                        except Exception:
                            pass
            if cur_vec is not None:
                section_vectors.append((f"{ch_n}.{pp_n}", cur_vec))

        concepts = extract_concepts(content)
        # Rank4: flag sections that shipped below a hard grounding floor or with
        # zero citations despite having evidence, so downstream (eval/report) can
        # surface them instead of them looking identical to good sections.
        _g = (verify_res or {}).get("grounding")
        _ncit = (verify_res or {}).get("n_citations", 0)
        if sources_json and (_ncit == 0 or (_g is not None and _g < 0.45)):
            quality = "degraded"
        else:
            quality = "ok"
        # Rank8: factual-attribution check (orthogonal to citation grounding).
        # Flags wrong author/year for canonical methods (e.g. Chinchilla credited
        # to Muennighoff). Advisory -- stored for eval/report, not a hard gate.
        attribution_flags = []
        if RESEARCH_AVAILABLE:
            try:
                attribution_flags = _research.canonical_seeds.check_attribution(content)
            except Exception:
                attribution_flags = []
        if attribution_flags:
            print(f"  [FACTCHECK] Ch{ch_n}.{pp_n} {len(attribution_flags)} attribution flag(s): "
                  + "; ".join(f"{f['concept']}: said {f['found']}, expected {f['expected']}"
                              for f in attribution_flags))
        state.setdefault("passes", {})[key] = {
            "ch": ch_n,
            "ch_t": ch_t,
            "pp": pp_n,
            "title": pp_t,
            "content": content,
            "wc": w,
            "tokens": tokens,
            "tps": tps,
            "at": datetime.now().isoformat(),
            "review": review_result,
            "sources": sources_json,
            "queries": queries_used,
            "verify": verify_res,
            "research_rounds": rounds_log,
            "concepts": concepts,
            "quality": quality,
            "attribution_flags": attribution_flags,
        }
        if quality == "degraded":
            print(f"  [QUALITY] Ch{ch_n}.{pp_n} flagged degraded (grounding={_g}, citations={_ncit})")
        if concepts:
            print(f"  [CONCEPTS] extracted {len(concepts)}: " + ", ".join(concepts[:6]) +
                  ("..." if len(concepts) > 6 else ""))
        state["total_words"] = state.get("total_words", 0) + w
        state["total_tokens"] = state.get("total_tokens", 0) + tokens
        state["total_calls"] = state.get("total_calls", 0) + 1

        pages = state["total_words"] // 400
        print(f"  OK: {w}w | {tokens}t | {tps:.1f} tok/s | {elapsed_min:.1f}min | ~{pages}p total")
        sys.stdout.flush()

        # Checkpoint every 2 tasks
        # Rank13: checkpoint after EVERY section. The LLM call (~2-4min) dwarfs the
        # 3.8MB state write (<0.1s), and the old every-2 cadence risked losing a
        # finished section on a crash. Atomic write (W3) makes per-section safe.
        save_state(state)
        if (i + 1) % 2 == 0:
            elapsed = time.time() - t0
            remaining = total - done - (i + 1)
            est = remaining * elapsed / max(i + 1 - done, 1)
            print()
            print("  [CHECKPOINT] %d/%d done | Elapsed: %.1fmin | Est. remaining: %.1fmin" % (
                i + 1 - done, total - done, elapsed / 60, est / 60))
            sys.stdout.flush()

    total_time = time.time() - t0
    save_state(state)
    assemble(state)
    make_report(state, total_time)

    print()
    print("=" * 70)
    print("  PIPELINE COMPLETE!")
    print("  Time: %.1fmin | Words: %d | Pages: ~%d" % (
        total_time / 60, state.get("total_words", 0), state.get("total_words", 0) // 400))
    print("  Output: %s" % FINAL_MD)
    print("=" * 70)
    sys.stdout.flush()

    notify("Deep Agent Done!", "LLM 400+ page book generated!")

    if render:
        render_pdf()

    return state


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Deep Agent LLM Book Pipeline")
    p.add_argument("--batch", "-b", type=int, default=2)
    p.add_argument("--start-ch", type=int, default=1)
    p.add_argument("--start-pp", type=int, default=1)
    p.add_argument("--no-render", action="store_true", help="Skip PDF rendering")
    p.add_argument("--review", action="store_true",
                   help="Enable LLM-as-judge reviewer pass (adds ~10s per section, regenerates "
                        "once if depth/coherence/format scores below %d)" % MIN_REVIEW_SCORE)
    p.add_argument("--no-research", action="store_true",
                   help="Disable Stage 2 agentic research layer (query gen + arxiv/wiki retrieval + "
                        "evidence-grounded writing). Off by default if research layer fails to import. "
                        "Adds ~20s per section when on.")
    p.add_argument("--topic", type=str, default=None,
                   help="Stage 3 planner -- generate a fresh CHAPTERS outline for the given topic "
                        "instead of using the hardcoded LLM outline. Requires research layer to be on. "
                        "Example: --topic \"Diffusion Models for Image Generation\"")
    p.add_argument("--n-chapters", type=int, default=None,
                   help="Number of chapters in planner-generated outline (default 12).")
    p.add_argument("--n-passes", type=int, default=None,
                   help="Sections per chapter in planner-generated outline (default 8). "
                        "Use 10-11 to target >400 pages.")
    p.add_argument("--out-name", type=str, default=None,
                   help="Output basename. Setting to 'book1' produces book1.{md,html,pdf,state.json,...} "
                        "instead of book.{...}. Lets you run multiple variants side-by-side.")
    p.add_argument("--end-ch", type=int, default=None,
                   help="Stop after this chapter (inclusive). Default: run all chapters. "
                        "Use --start-ch 1 --end-ch 1 for a single-chapter smoke test.")
    args = p.parse_args()
    run(batch=args.batch, start_ch=args.start_ch, start_pp=args.start_pp,
        end_ch=args.end_ch,
        render=not args.no_render, review=args.review,
        research=not args.no_research, topic=args.topic,
        n_chapters=args.n_chapters, n_passes=args.n_passes, out_name=args.out_name)
