#!/usr/bin/env python3
"""
Deep Agent Multi-Pass Pipeline -- 400+ Page Book Generator
12 chapters x 6 passes x 5000 words = 360K words ~900 pages
"""
import json, os, re, sys, time, signal, argparse, subprocess, threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict

OLLAMA_BASE = "http://localhost:11434"
MODEL = "gemma3:4b"
DEFAULT_TIMEOUT = 600

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)
STATE_FILE = OUT_DIR / "multipass_state.json"
FINAL_MD = OUT_DIR / "book.md"
FINAL_HTML = OUT_DIR / "book.html"
FINAL_PDF = OUT_DIR / "book.pdf"
REPORT_FILE = OUT_DIR / "benchmark.json"


class OllamaClient:
    def __init__(self, base=OLLAMA_BASE, model=MODEL, timeout=DEFAULT_TIMEOUT):
        self.base = base; self.model = model; self.timeout = timeout; self.lock = threading.Lock()

    def generate(self, prompt, system="", temperature=0.7, num_predict=15000):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        # num_predict is the hard cap on generated tokens.
        # Approx ratio: ~1.3 tokens/word, plus ~30% overhead for markdown/code.
        effective_tokens = min(num_predict, 15000)
        payload = {
            "model": self.model, "stream": False, "messages": msgs,
            "options": {"temperature": temperature, "num_predict": effective_tokens,
                        "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.05},
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
        return content, {"tokens": ec, "tps": round(tps, 1), "elapsed": round(time.time()-t0, 1),
                         "done": data.get("done_reason", "")}

    def health(self):
        try:
            import httpx
            with httpx.Client(timeout=5) as c:
                return c.get(f"{self.base}/api/tags").status_code == 200
        except:
            return False


SYS = "You are a senior technical book writer and university professor. Output ONLY Markdown content — no meta-commentary, no preamble, no closing remarks. Write with the depth and detail of a published technical handbook. Use substantial paragraphs (not bullet lists) for core explanations. Include complete code examples with imports. Target 1500-2500 words per section."



CHAPTERS = [
  {"n":1,"t":"Introduction to Large Language Models","passes":[
    {"p":1,"t":"History and Evolution","w":4200,"pr":"Write a comprehensive section on the HISTORY AND EVOLUTION OF LANGUAGE MODELS. Cover: Statistical language models N-gram and HMM, Neural language models RNN LSTM GRU, The birth of the Transformer 2017, Pre-training revolution ELMo GPT-1 BERT GPT-2, The scaling era GPT-3 PaLM Chinchilla LLaMA, Current state-of-the-art GPT-4 Claude Gemini Mistral. Include key research papers with citations."},
    {"p":2,"t":"Mathematical Foundations","w":4200,"pr":"Write a deep technical section on MATHEMATICAL FOUNDATIONS. Cover: Maximum Likelihood Estimation, Cross-entropy loss formula, Perplexity metric, Chain rule of probability, Information theory entropy and KL divergence, Neural LM formulation, Backpropagation through time. Include Python code for perplexity calculation."},
    {"p":3,"t":"Pre-training Paradigm","w":4200,"pr":"Write a section on THE PRE-TRAINING PARADIGM. Cover: Word embeddings Word2Vec CBOW Skip-gram and GloVe, Contextual embeddings ELMo, Next Sentence Prediction NSP, Masked Language Modeling MLM, Causal Language Modeling CLM, PrefixLM and encoder-decoder T5, Contrastive learning SimCSE SentenceBERT, Scaling laws Chinchilla optimal scaling."},
    {"p":4,"t":"The LLM Ecosystem","w":4200,"pr":"Write an overview section on THE LLM ECOSYSTEM. Cover: Open-source vs closed models, Model families GPT Claude Gemini LLaMA Mistral Gemma Qwen, Commercial API providers, Open-source ecosystem HuggingFace llama.cpp Ollama, Model benchmarks MMLU HELM BIG-bench, Specialized vs general-purpose LLMs."},
    {"p":5,"t":"LLM Capabilities and Limitations","w":4200,"pr":"Write a section on LLM CAPABILITIES AND LIMITATIONS. Cover: What LLMs can do text generation summarization translation reasoning, Emergent capabilities at scale, What LLMs cannot do reliably exact math real-time knowledge, Hallucination causes and mitigation, Context window limitations, Compositionality and systematic generalization, Benchmark saturation, Gap between benchmark and real-world utility."},
    {"p":6,"t":"Societal Impact and Responsible AI","w":4200,"pr":"Write a section on SOCIETAL IMPACT AND RESPONSIBLE AI. Cover: Economic impact on knowledge work, Labor market transformation, Education and research applications, Privacy concerns with training data, Energy consumption and environmental impact, Governance frameworks EU AI Act US Executive Order, International coordination, Role of open-source in AI safety."},
    {"p":7,"t":"Architecture Fundamentals and Attention Mechanism","w":4200,"pr":"Write a section on ARCHITECTURE FUNDAMENTALS. Cover: Encoder-decoder architecture, Attention mechanism intuition, Scaled dot-product attention formula Q K V, Multi-head attention MHA, Why scale by sqrt(d_k), Positional encoding sin/cos, Layer normalization, Residual connections, Feed-forward networks FFN with GELU activation. Include PyTorch attention implementation."},
    {"p":8,"t":"Practical LLM Usage and Prompt Engineering Basics","w":4200,"pr":"Write a section on PRACTICAL LLM USAGE. Cover: How to interact with LLMs via API, Prompt engineering fundamentals zero-shot few-shot, Chain-of-thought prompting, Temperature top-k top-p sampling, Token counting and cost estimation, Handling long contexts, Streaming vs non-streaming responses, Best practices for production deployments, Rate limiting and error handling. Include Python code for API interaction."}
  ]},
  {"n":2,"t":"The Transformer Architecture","passes":[
    {"p":1,"t":"Self-Attention Mechanism","w":4200,"pr":"Write a deep technical section on SELF-ATTENTION MECHANISM. Cover: Scaled dot-product attention formula, Query Key Value projections, Why scaled by sqrt d_k, Multi-head attention, Full MHA formula, Computational complexity O n squared times d, Flash Attention. Include PyTorch code."},
    {"p":2,"t":"Positional Encoding","w":4200,"pr":"Write a technical section on POSITIONAL ENCODING. Cover: Why positional encoding is needed, Sinusoidal PE Vaswani, Learnable positional embeddings, Relative positional encoding Shaw et al., RoPE Rotary Position Embedding, ALiBi Attention with Linear Biases. Include PyTorch code for RoPE."},
    {"p":3,"t":"FFN Residual Connections LayerNorm","w":4200,"pr":"Write a technical section on TRANSFORMER COMPONENTS. Cover: Feed-forward network formula, GELU activation, Residual connections and gradient flow, Layer Normalization formula, Pre-norm vs Post-norm transformers. Include PyTorch module code."},
    {"p":4,"t":"Transformer Variants and Efficiency","w":4200,"pr":"Write a section on TRANSFORMER VARIANTS. Cover: Sparse attention Longformer BigBird, Linear attention Performer Linformer Reformer, Mixture of Experts Switch Transformer Mixtral DBRX, State space models Mamba S4 RWKV, Flash Attention 1 and 2, Grouped Query Attention GQA, Multi-Query Attention MQA, Sliding window attention, Speculative decoding."},
    {"p":5,"t":"Training Dynamics and Optimization","w":4200,"pr":"Write a section on TRANSFORMER TRAINING DYNAMICS. Cover: Learning rate schedules cosine linear warmup, Gradient clipping and norm, Mixed precision training FP16 BF16, Activation checkpointing, Weight decay and AdamW, Training instabilities loss spikes, Distributed training data parallelism ZeRO stages, Practical recipes from LLaMA Mistral Gemma. Include PyTorch training loop."},
    {"p":6,"t":"Inference-Time Computation","w":4200,"pr":"Write a section on INFERENCE-TIME COMPUTATION. Cover: Greedy vs sampling temperature top-k top-p, Beam search tradeoffs, Speculative decoding draft-then-verify, Medusa multi-draft decoding, Lookahead decoding, Early exiting, KV cache management, Cascade and routing, Prefix caching. Include code."},
    {"p":7,"t":"Advanced Attention and Architecture Innovations","w":4200,"pr":"Write a section on ADVANCED ATTENTION VARIANTS. Cover: Flash Attention 2 and 3 detailed algorithm, Grouped Query Attention GQA for efficiency, Multi-Query Attention MQA, Sliding Window Attention for long sequences, Sparse attention patterns mixture of experts, Ring attention for distributed long-context, Flash Decoding optimization, Flash Attention with nested tensors. Include CUDA-like pseudocode for Flash Attention tiling."},
    {"p":8,"t":"Scaling Transformers and Compute Optimal Training","w":4200,"pr":"Write a section on SCALING AND TRAINING AT SCALE. Cover: Scaling laws Kaplan et al, Chinchilla optimal compute allocation, Training compute FLOPs estimation, LLaMA 3 training recipe, Data parallelism and gradient accumulation, Mixed batch size scheduling, Learning rate warmup and decay, Gradient checkpointing for memory, Zero Redundancy Optimizer ZeRO stages, FSDP fully sharded data parallel. Include scaling law formulas."}
  ]},
  {"n":3,"t":"Tokenization and Text Representation","passes":[
    {"p":1,"t":"Tokenization Strategies","w":4200,"pr":"Write a section on TOKENIZATION STRATEGIES. Cover: Character-level, Byte Pair Encoding BPE, WordPiece BERT, Unigram Language Model SentencePiece, Tiktoken fast BPE, Token vocabulary sizes, Multi-language tokenization challenges. Include Python code for BPE."},
    {"p":2,"t":"Embedding Layers and Representation","w":4200,"pr":"Write a section on EMBEDDING LAYERS. Cover: Token embedding matrix, Positional embedding addition, Learned vs fixed embeddings, Contextual vs static embeddings, Representation geometry, Weight tying, Cross-lingual embeddings."},
    {"p":3,"t":"Tokenizer Evaluation","w":4200,"pr":"Write a section on TOKENIZER EVALUATION. Cover: Vocabulary coverage, OOV rates, Compression ratio, BPE dropout, tiktoken vs HuggingFace, Vision tokenization ViT, Image tokenization VQ-VAE VQ-GAN, Byte-level BPE GPT-2, Unicode normalization."},
    {"p":4,"t":"Advanced Representation Learning","w":4200,"pr":"Write a section on ADVANCED REPRESENTATION LEARNING. Cover: Knowledge neurons, Probing classifiers, Information-theoretic analysis, Geometric properties of embedding spaces, Analogical reasoning in embedding space, Cross-lingual alignment."},
    {"p":5,"t":"Unicode Multilingual Tokenization","w":4200,"pr":"Write a section on ADVANCED TOKENIZATION. Cover: Unicode normalization NFC NFD, Challenges with non-Latin scripts CJK Arabic Hindi, SentencePiece language-agnostic approach, Vocabulary size tradeoffs, OOV handling, tiktoken vs HuggingFace, Counting tokens in practice."},
    {"p":6,"t":"Subword Regularization","w":4200,"pr":"Write a section on TOKENIZATION ROBUSTNESS. Cover: BPE dropout for robust training, Subword regularization, Vocabulary pruning, Handling misspellings, Code tokenization special characters identifiers, Domain-specific tokenization math chemical formulas, Choosing a tokenizer practical guide."},
    {"p":7,"t":"Embedding Space Analysis","w":4200,"pr":"Write a section on EMBEDDING SPACE ANALYSIS. Cover: Intrinsic evaluation word similarity analogy, Extrinsic evaluation downstream tasks, Principal Component Analysis PCA visualization, t-SNE and UMAP for embeddings, Anisotropy and representation degeneration, Representation learning theory, Contextualized vs distributional embeddings, Sentence embeddings SBERT SimCSE."},
    {"p":8,"t":"Cross-Lingual and Multimodal Representations","w":4200,"pr":"Write a section on CROSS-LINGUAL AND MULTIMODAL REPRESENTATIONS. Cover: Cross-lingual transfer learning, Multilingual models mBERT XLM-R, Language-agnostic representations, Aligned embedding spaces, CLIP contrastive vision-language, Alignment methods Procrustes SVD, Zero-shot cross-lingual transfer, Evaluation of cross-lingual models XNLI MLQA."}
  ]},
  {"n":4,"t":"Pre-training Objectives and Data","passes":[
    {"p":1,"t":"Causal Language Modeling","w":4200,"pr":"Write a section on CAUSAL LANGUAGE MODELING. Cover: Autoregressive generation, Teacher forcing, Cross-entropy loss, Sequence packing, Efficient training gradient checkpointing, Mixed-precision training, Flash Attention implementation. Include PyTorch training loop."},
    {"p":2,"t":"Masked Language Modeling","w":4200,"pr":"Write a section on MASKED LANGUAGE MODELING. Cover: BERT-style MLM, Whole Word Masking, ELECTRA replaced token detection, DeBERTa with disentangled attention, Span corruption T5, GLM, Denoising autoencoder objectives, Comparison CLM vs MLM vs RTD."},
    {"p":3,"t":"Pre-training Data Curation","w":4200,"pr":"Write a section on PRE-TRAINING DATA ENGINEERING. Cover: Common Crawl extraction and deduplication, The Pile dataset, RedPajama, Data quality filtering perplexity heuristic, Language identification, Toxicity filtering, Decontamination of benchmarks, Data mixing ratios."},
    {"p":4,"t":"Scaling Laws","w":4200,"pr":"Write a section on SCALING LAWS. Cover: Kaplan et al. scaling laws, Chinchilla optimal scaling, LLM training compute FLOPs, Flash Attention, Mixed batch size, Learning rate schedules, Training stability, LLaMA 3 training recipe."},
    {"p":5,"t":"Training Data Quality","w":4200,"pr":"Write a section on TRAINING DATA QUALITY. Cover: Data quality taxonomy text quality deduplication toxicity, Heuristic filters, Classifier-based quality filtering, MinHash deduplication at scale, Near-duplicate detection with SimHash, Web text extraction, Quality vs quantity tradeoff."},
    {"p":6,"t":"Efficient Training","w":4200,"pr":"Write a section on EFFICIENT TRAINING. Cover: Gradient accumulation, ZeRO-1/2/3, FSDP Fully Sharded Data Parallel, Tensor parallelism Megatron-LM, Pipeline parallelism GPipe PipeDream, Sequence parallelism, Flash Attention 2, Memory-efficient fine-tuning. Include PyTorch FSDP code."},
    {"p":7,"t":"Training Recipes and Case Studies","w":4200,"pr":"Write a section on TRAINING RECIPES. Cover: LLaMA training recipe step by step, Mistral training details, Gemma model training, Phi model series from Microsoft, Data cleaning pipeline, Curriculum learning strategies, Learning rate scheduling in practice, Training stability tricks, Common failure modes and debugging."},
    {"p":8,"t":"Datasets and Corpus Engineering","w":4200,"pr":"Write a section on DATASETS AND CORPUS ENGINEERING. Cover: Major pre-training datasets C4 The Pile RedPajama SlimPajama, Dataset composition and mixing ratios, Sampling strategy across sources, Domain-specific pre-training, Continual pre-training, Dataset documentation datasheets, Licensing and copyright considerations, Synthetic data generation for training."}
  ]},
  {"n":5,"t":"Fine-tuning and Task Adaptation","passes":[
    {"p":1,"t":"Fine-tuning vs Parameter-Efficient","w":4200,"pr":"Write a section on FINE-TUNING STRATEGIES. Cover: Full fine-tuning, Catastrophic forgetting, Adapter layers, Prefix tuning, Prompt tuning, LoRA Delta W = BA, QLoRA quantization plus LoRA, Adapter vs LoRA vs full comparison. Include PyTorch LoRA implementation."},
    {"p":2,"t":"Instruction Tuning and SFT","w":4200,"pr":"Write a section on INSTRUCTION TUNING AND SFT. Cover: SFT pipeline, Instruction dataset construction FLAN Alpaca Dolly, SFTTrainer from TRL library, Data formatting chat templates, Curriculum learning, Data quality vs quantity."},
    {"p":3,"t":"Domain Adaptation","w":4200,"pr":"Write a section on DOMAIN ADAPTATION. Cover: Continual pre-training vs fine-tuning, Medical PubMed ClinicalBERT, Code CodeBERT StarCoder CodeLLaMA, Legal ChatLaw, Scientific Galactica SciBERT, Domain vocabulary handling, Mixture-of-adapters."},
    {"p":4,"t":"Dataset Curation","w":4200,"pr":"Write a section on DATASET CURATION. Cover: Data collection strategies, Annotation quality, Synthetic data with LLMs, Self-instruct and evolution, Quality filtering, Dataset balancing, Deduplication, Human preference data."},
    {"p":5,"t":"LoRA Theory Variants Advanced","w":4200,"pr":"Write a section on LoRA VARIANTS. Cover: LoRA theory low-rank factorization, Which layers to adapt, Rank selection r=4 vs r=64, Alpha-rank scaling, QLoRA 4-bit NF4, LoRA+ improved learning rates, DoRA weight-decomposed LoRA, LongLoRA for long context, Merging adapters. Include PyTorch LoRA code."},
    {"p":6,"t":"Prompt Tuning Prefix Tuning Adapter","w":4200,"pr":"Write a section on SOFT PROMPTING AND ADAPTER METHODS. Cover: Prompt tuning learnable soft tokens, Prefix tuning, Adapter modules Houlsby Pfeiffer Compacter, Series vs parallel adapter composition, Efficient Adapter Tuning EAT, AdaptFormer, KronA Kronecker adapters. Include code."},
    {"p":7,"t":"Advanced PEFT Methods","w":4200,"pr":"Write a section on ADVANCED PEFT METHODS. Cover: AdaLoRA adaptive rank allocation, LoftQ quantization-aware LoRA, VeRA very low-rank adapters, LoRA-fa layer-wise rank adaptation, Compacter advanced adapters, Mixtral of Experts fine-tuning, MoE adapter strategies, Expert routing with adapters, Training and merging multiple adapters."},
    {"p":8,"t":"Fine-tuning Best Practices and Troubleshooting","w":4200,"pr":"Write a section on FINE-TUNING BEST PRACTICES. Cover: Catastrophic forgetting prevention, Learning rate selection for LoRA, Weight decay and optimizer choices, Batch size and gradient accumulation, Evaluation during fine-tuning, Hyperparameter tuning systematic approach, Overfitting detection, Model merging techniques TIES Elect, DARE task arithmetic for weight difference."}
  ]},
  {"n":6,"t":"Alignment: RLHF, DPO, and Beyond","passes":[
    {"p":1,"t":"RLHF","w":4200,"pr":"Write a section on RLHF. Cover: Why alignment helpful harmless honest, Three stages of RLHF SFT reward model PPO, Reward model, Bradley-Terry preference model, PPO update with KL penalty, Challenges reward hacking mode collapse. Include PPO pseudocode."},
    {"p":2,"t":"DPO","w":4200,"pr":"Write a section on DPO. Cover: DPO objective formula, Why DPO avoids reward model, GRPO DeepSeek, KTO Kahneman-Tversky Optimization, Comparison RLHF vs DPO vs GRPO. Include PyTorch DPO loss code."},
    {"p":3,"t":"Reward Modeling","w":4200,"pr":"Write a section on REWARD MODELING. Cover: Reward model architecture, Pointwise vs pairwise, Listwise ranking, Reward model scaling laws, Constitutional AI CAI, RLAIF LLM as judge, Self-rewarding, Ensemble reward models."},
    {"p":4,"t":"Safety Red-teaming","w":4200,"pr":"Write a section on AI SAFETY AND RED-TEAMING. Cover: Red-teaming methodologies, Harmful content categories, Safety benchmarks TruthfulQA ToxiGen, Refusal learning, Guardrails, Interpretability for safety, Fairness and bias."},
    {"p":5,"t":"Advanced Alignment","w":4200,"pr":"Write a section on ADVANCED ALIGNMENT. Cover: Constitutional AI CAI principles and training, RLAIF replacing human feedback with LLM feedback, Scalable oversight debate amplification recursive reward modeling, Iterated amplification, Self-play, Addressing reward hacking."},
    {"p":6,"t":"DPO Variants","w":4200,"pr":"Write a section on DPO VARIANTS. Cover: IPO Identity Preference Optimization, KTO Kahneman-Tversky Optimization, CPO Contrastive Preference Optimization, On-policy vs off-policy in DPO, RTF Reinforced Fine-Tuning, Self-rewarding, Iterative DPO, GRPO from DeepSeek. Include PyTorch code."},
    {"p":7,"t":"Alignment in Practice","w":4200,"pr":"Write a section on ALIGNMENT IN PRACTICE. Cover: Collecting preference data human raters, Quality control for preference labels, Inter-annotator disagreement handling, Reward model training pipeline, PPO hyperparameters in practice, KL divergence regularization, KL penalty annealing, Handling distribution shift, Iterative DPO workflow."},
    {"p":8,"t":"AI Safety and Governance","w":4200,"pr":"Write a section on AI SAFETY AND GOVERNANCE. Cover: Current AI safety challenges, Scalable oversight techniques, Interpretability for alignment, Reward model robustness, Handling out-of-distribution inputs, Emerging regulatory frameworks, EU AI Act compliance for LLMs, Safety evaluation methodology, Incident response and reporting."}
  ]},
  {"n":7,"t":"Prompt Engineering and In-Context Learning","passes":[
    {"p":1,"t":"Zero-shot Few-shot Chain-of-Thought","w":4200,"pr":"Write a section on PROMPT ENGINEERING. Cover: Zero-shot, Few-shot in-context learning, Chain-of-Thought CoT, Auto-CoT, Self-consistency, Tree-of-Thought ToT, Best practices clear instructions delimiters, Prompt anti-patterns, Systematic evaluation."},
    {"p":2,"t":"In-Context Learning Theory","w":4200,"pr":"Write a section on IN-CONTEXT LEARNING THEORY. Cover: Inductive bias of attention, Gradient descent in hidden space ICL as meta-learning, Function vectors hypothesis, How demonstrations affect attention, Analogical reasoning, Label semantics, Example ordering."},
    {"p":3,"t":"RAG","w":4200,"pr":"Write a section on RAG. Cover: Knowledge cut-off and hallucination, Vector databases FAISS ChromaDB Milvus Pinecone, Embedding models BGE E5, Chunking strategies, Dense vs sparse retrieval, HyDE, Re-ranking, RAG vs fine-tuning tradeoff."},
    {"p":4,"t":"Agentic Workflows and Tool Use","w":4200,"pr":"Write a section on LLM AGENTS AND TOOL USE. Cover: ReAct Reasoning plus acting, Tool use with function calling, Agent architectures, Planning and decomposition, Memory systems, Code interpreter agents, Multi-agent debate, Error recovery."},
    {"p":5,"t":"Advanced Prompting","w":4200,"pr":"Write a section on ADVANCED PROMPTING. Cover: Reflexion verbal reinforcement learning, Self-refine iterative critique, Prompt evolution, Multi-agent prompting debate consensus role assignment, Generated knowledge prompting, Chain-of-Verification CoV, Automatic prompt engineering APE. Include code."},
    {"p":6,"t":"Advanced RAG Architecture","w":4200,"pr":"Write a section on ADVANCED RAG. Cover: Dense retrieval bi-encoder vs cross-encoder, ANN indexes HNSW IVF-PQ ScaNN, BM25 sparse retrieval, Reciprocal Rank Fusion RRF, Multi-vector ColBERT, Knowledge graphs, Parent document retrieval, Contextual compression, Self-RAG. Include implementation code."},
    {"p":7,"t":"Agent Architectures and Memory Systems","w":4200,"pr":"Write a section on AGENT ARCHITECTURES AND MEMORY. Cover: ReAct and Plan-and-Execute patterns, Reflection agents with self-critique, Memory types episodic semantic procedural, Vector memory and summary memory, Long-term memory for agents, Tool use and API integration, Multi-agent collaboration and communication, Error handling and retry strategies, Agent evaluation frameworks."},
    {"p":8,"t":"Advanced Agent Patterns and Production","w":4200,"pr":"Write a section on ADVANCED AGENT PATTERNS. Cover: LangChain and LlamaIndex for agent development, AutoGen and CrewAI multi-agent frameworks, Planning with LLMs task decomposition, Self-correcting agents with feedback loops, Code generation and execution agents, Autonomous agents BabyAGI AutoGPT, Evaluating agent performance, Safety considerations for autonomous agents, Production deployment patterns."}
  ]},
  {"n":8,"t":"Evaluation and Benchmarking","passes":[
    {"p":1,"t":"Core NLP Benchmarks","w":4200,"pr":"Write a section on CORE LLM BENCHMARKS. Cover: MMLU 57 subjects, GSM8K Grade School Math, HumanEval Code, MATH, HellaSwag, ARC, TruthfulQA, BIG-bench, Evaluation protocols, Limitations and gaming."},
    {"p":2,"t":"LLM Evaluation Frameworks","w":4200,"pr":"Write a section on EVALUATION FRAMEWORKS. Cover: lm-evaluation-harness EleutherAI, LightEval HuggingFace, OpenCompass, HELM, AlpacaEval, MT-Bench, BLEU ROUGE METEOR limitations, Perplexity, Human evaluation best practices."},
    {"p":3,"t":"Code and Reasoning Evaluation","w":4200,"pr":"Write a section on CODE AND REASONING BENCHMARKS. Cover: HumanEval, MBPP, APPS, Codeforces, MultiPL-E, ToolBench, GSM8K, MATH, BIG-Bench Hard, Evaluating CoT quality, Execution-based code evaluation."},
    {"p":4,"t":"Safety Evaluation","w":4200,"pr":"Write a section on SAFETY EVALUATION. Cover: TruthfulQA variants, ToxiGen, BOLD, RealToxicityPrompts, Decoding-time safety, Evaluation of refusal behavior, Fairness BBQ BOLD."},
    {"p":5,"t":"Statistical Evaluation Metrics","w":4200,"pr":"Write a section on STATISTICAL EVALUATION METRICS. Cover: BLEU n-gram precision with brevity penalty, ROUGE recall-oriented metrics, METEOR stem matching, BERTScore contextual embedding similarity, BARTScore, Limitations of surface-level matching, When to use which metric."},
    {"p":6,"t":"Human Evaluation LLM-as-Judge","w":4200,"pr":"Write a section on HUMAN AND LLM-AS-JUDGE EVALUATION. Cover: Human evaluation protocols pairwise Likert scales, Inter-annotator agreement, LLM-as-Judge prompting, Self-evaluation bias, Position bias mitigation, Elo and Bradley-Terry for ranking, Chatbot Arena methodology."},
    {"p":7,"t":"Reasoning and Math Evaluation","w":4200,"pr":"Write a section on REASONING AND MATH EVALUATION. Cover: GSM8K Grade School Math 8K, MATH benchmark, FrontierMath, ARC-C abstract reasoning, GPQA graduate-level science, Competition math evaluation, MATH-ASCII plain text math, Evaluating chain-of-thought quality, Pass@k and sampling-based evaluation, Process reward models for math."},
    {"p":8,"t":"Comprehensive Evaluation Strategy","w":4200,"pr":"Write a section on COMPREHENSIVE EVALUATION STRATEGY. Cover: Designing an evaluation suite, Balancing capability and safety benchmarks, Red-teaming for specific failure modes, Evals for fine-tuned models, Evals for RAG and agentic systems, Continuous evaluation in production, A/B testing with statistical significance, Cost-efficient evaluation, Building custom benchmarks for domain-specific needs."}
  ]},
  {"n":9,"t":"Deployment, Inference, and Optimization","passes":[
    {"p":1,"t":"LLM Inference Optimization","w":4200,"pr":"Write a section on LLM INFERENCE OPTIMIZATION. Cover: KV cache, KV cache memory calculation, Batching static vs dynamic, Continuous batching, PagedAttention vLLM, Tensor parallelism, Pipeline parallelism, Speculative decoding, Beam search vs sampling."},
    {"p":2,"t":"Quantization","w":4200,"pr":"Write a section on QUANTIZATION. Cover: Quantization fundamentals, Post-Training Quantization PTQ, GPTQ, AWQ, SmoothQuant, GGUF formats Q8 Q6 Q5 Q4 Q3 Q2, BitsAndBytes 8-bit 4-bit with NF4, Impact on quality. Include quantization code."},
    {"p":3,"t":"Local Deployment Tools","w":4200,"pr":"Write a section on LOCAL DEPLOYMENT TOOLS. Cover: llama.cpp C C++ Metal CUDA GGUF, Ollama, vLLM PagedAttention, Text Generation Inference TGI, Inference endpoints, OpenAI-compatible APIs, Benchmarking local inference, Memory requirements."},
    {"p":4,"t":"Pruning Distillation","w":4200,"pr":"Write a section on MODEL COMPRESSION. Cover: Structured vs unstructured pruning, Magnitude pruning, Sparse attention, Knowledge distillation teacher-student, MiniLLM, Weight quantization plus pruning combination, Neural Architecture Search."},
    {"p":5,"t":"Distributed Serving","w":4200,"pr":"Write a section on DISTRIBUTED SERVING. Cover: vLLM architecture PagedAttention, Continuous batching, Tensor parallelism for inference, Pipeline parallelism, Expert routing in MoE models, Prefix caching, Latency vs throughput tradeoff, SLA-driven serving P50 P95 P99, Load balancing."},
    {"p":6,"t":"Memory Optimization","w":4200,"pr":"Write a section on MEMORY OPTIMIZATION FOR INFERENCE. Cover: KV cache memory calculation, KV cache quantization INT8 FP8, KIVI 2-bit quantization, Automatic Prefix Caching APC, Chunked prefill, Speculative decoding Medusa EAGLE, Flash Decoding for long-context."},
    {"p":7,"t":"Production Deployment Patterns","w":4200,"pr":"Write a section on PRODUCTION DEPLOYMENT. Cover: Container orchestration Kubernetes for LLMs, API gateway patterns, Rate limiting and autoscaling, Canary releases and rollback, Multi-model serving, Latency optimization techniques, Cost optimization strategies, Monitoring inference quality, Error handling and fallbacks."},
    {"p":8,"t":"Optimization Case Studies and Benchmarks","w":4200,"pr":"Write a section on OPTIMIZATION CASE STUDIES. Cover: Comparing quantization methods on downstream tasks, vLLM vs TGI performance comparison, llama.cpp Metal vs CUDA throughput, Speculative decoding speedup measurements, Batch size tuning for throughput, KV cache compression benchmarks, End-to-end serving benchmarks, Practical guide to choosing optimization methods."}
  ]},
  {"n":10,"t":"Multimodal and Emerging Capabilities","passes":[
    {"p":1,"t":"Vision-Language Models","w":4200,"pr":"Write a section on VISION-LANGUAGE MODELS. Cover: CLIP contrastive learning, LLaVA vision encoder plus LLM, BLIP-2 Q-Former, Flamingo, GPT-4V and Gemini, Vision-language alignment, Instruction tuning for multimodal, Evaluation VQA Captioning Doc understanding."},
    {"p":2,"t":"Audio Reasoning Long Context","w":4200,"pr":"Write a section on AUDIO-LANGUAGE AND EMERGING CAPABILITIES. Cover: Whisper ASR, AudioPaLM speech-to-speech, Emergent abilities, Chain-of-thought emergence, Arithmetic and logical reasoning, Theory of mind, RoPE scaling for long context, YaRN, Landmark attention."},
    {"p":3,"t":"Interpretability","w":4200,"pr":"Write a section on INTERPRETABILITY. Cover: Feature probing and activation patching, Circuit analysis, Attention head roles induction heads, Sparse autoencoders for monosemantic features, Superposition hypothesis, Gradient-based attribution, Probing classifiers."},
    {"p":4,"t":"Future of LLM Research","w":4200,"pr":"Write a section on THE FUTURE OF LLMS. Cover: Open-source vs closed-source gap, MoE scaling, Test-time compute Strawberry o1, Constitutional AI, Mechanistic interpretability, LLM compression, Scientific discovery, Energy efficiency, AGI debates."},
    {"p":5,"t":"Multimodal Training Evolution","w":4200,"pr":"Write a section on MULTIMODAL TRAINING EVOLUTION. Cover: Early fusion vs late fusion, CLIP contrastive learning, ALIGN scaling, Flamingo with Perceiver resampler, BLIP-2 frozen LLM plus Q-Former, LLaVA linear projection, MiniGPT-4, GPT-4V multimodal instruction tuning, Architecture comparison, Document understanding."},
    {"p":6,"t":"Video World Models Embodied AI","w":4200,"pr":"Write a section on VIDEO UNDERSTANDING AND WORLD MODELS. Cover: Video LLMs VideoChat VideoLLaMA, Temporal modeling, Action recognition, World models from video Gaia Genie, Embodied AI instructing robots, Sora, Stable Video Diffusion, LWM long world model."},
    {"p":7,"t":"Multimodal Architectures Deep Dive","w":4200,"pr":"Write a section on MULTIMODAL ARCHITECTURES. Cover: Vision transformer ViT architecture, SigLIP and DINOv2 vision encoders, Cross-attention vs fusion-in-decoder, Perceiver resampler and Q-Former, Gemma multimodal, InternVL architecture, Molmo and NVLM models, Training multimodal models data mixture, Evaluation of multimodal models."},
    {"p":8,"t":"Audio Speech and Video Models","w":4200,"pr":"Write a section on AUDIO SPEECH AND VIDEO MODELS. Cover: Whisper architecture for ASR, AudioPaLM for speech-to-speech, Video generation models Sora Lumiere, World models from video GAIA-1, Temporal video representation, Latent video compression, Text-to-video generation, Understanding long videos, Embodied agents in 3D environments."}
  ]},
  {"n":11,"t":"Practical Applications and Case Studies","passes":[
    {"p":1,"t":"Building RAG Systems","w":4200,"pr":"Write a section on BUILDING PRODUCTION RAG. Cover: Architecture indexer retriever generator, Embedding models BGE-M3 E5-Mistral, Vector databases, Chunking size overlap, Hybrid search, Reranking, Guardrails, RAGAS evaluation, End-to-end RAG with LangChain LlamaIndex."},
    {"p":2,"t":"Building LLM Agents","w":4200,"pr":"Write a section on BUILDING LLM AGENTS. Cover: Agent loop perceive plan act reflect, Tool definition and function calling, ReAct implementation, Multi-agent systems, Memory vector store plus summary, Error handling, Evaluation, LangGraph for complex workflows, Production best practices."},
    {"p":3,"t":"Fine-tuning Case Studies","w":4200,"pr":"Write a section on FINE-TUNING CASE STUDIES. Cover: Medical LLM BioBERT to MedLLaMA, Code LLM Codex to CodeLLaMA, Legal LLM ChatLaw, Finance BloombergGPT, Domain adaptation recipe, LoRA hyperparameter tuning, Dataset size vs quality."},
    {"p":4,"t":"Cost Estimation","w":4200,"pr":"Write a section on LLM COST OPTIMIZATION. Cover: API pricing OpenAI Anthropic open-source, Token estimation, Caching strategies, Batch inference, Model selection by task, Fine-tuning vs RAG vs prompt engineering, Self-hosted vs API."},
    {"p":5,"t":"LLM Security","w":4200,"pr":"Write a section on LLM SECURITY. Cover: Prompt injection direct and indirect attacks, Jailbreaking role-play ASCII art, System prompt extraction, Data leakage from training and retrieval, Membership inference attacks, Adversarial suffixes, Defensive strategies input validation output filtering, OWASP LLM Top 10. Include attack and defense code."},
    {"p":6,"t":"Monitoring Observability","w":4200,"pr":"Write a section on PRODUCTION LLM OPERATIONS. Cover: LLM observability logging latency cost, Tracing frameworks OpenTelemetry LangSmith Arize, Prompt version management, Drift detection, Quality monitoring, Cost attribution, A/B testing LLMs statistical significance, Feature flags for model routing."},
    {"p":7,"t":"Enterprise Application Patterns","w":4200,"pr":"Write a section on ENTERPRISE APPLICATION PATTERNS. Cover: LLM-powered search engines semantic hybrid, Document intelligence extraction classification, Customer service automation, Code generation in IDEs, Data analysis and BI with LLMs, Content moderation systems, Personalized recommendation with LLMs, Compliance and audit trails."},
    {"p":8,"t":"End-to-End Project Case Studies","w":4202,"pr":"Write a section on END-TO-END CASE STUDIES. Cover: Building a medical QA system step by step, Legal document summarization pipeline, Code review automation with multi-agent, Financial report analysis system, Implementing RAG with hybrid search production-ready, Fine-tuning for code generation from scratch, Building a multimodal document processor, Lessons learned and common pitfalls."}
  ]},
  {"n":12,"t":"Research Frontiers and Future Directions","passes":[
    {"p":1,"t":"Current Research Frontiers","w":4200,"pr":"Write a section on CURRENT RESEARCH FRONTIERS. Cover: Open-source vs closed gap, MoE scaling, Test-time compute, Constitutional AI, Mechanistic interpretability circuits features, Superposition hypothesis, Model editing ROME MEMIT, 1-bit LLM, Key papers from 2024-2025."},
    {"p":2,"t":"Tools and Resources","w":4200,"pr":"Write a section on TOOLS AND RESOURCES. Cover: HuggingFace ecosystem, TRL, PEFT, llamafactory, LangChain, LlamaIndex, Weights and Biases, Ollama, PromptLayer, lm-evaluation-harness, Academic resources arxiv Papers with Code, Community resources."},
    {"p":3,"t":"AI Safety and Governance","w":4200,"pr":"Write a section on AI SAFETY AND GOVERNANCE. Cover: AI safety research directions, Scalable oversight, Reward model hacking, Robustness to distribution shift, Privacy-preserving LLM training, Federated learning for LLMs, Regulatory frameworks, EU AI Act compliance."},
    {"p":4,"t":"Building Real-World Applications","w":4200,"pr":"Write a section on BUILDING REAL-WORLD LLM APPLICATIONS. Cover: Architecture patterns RAG agents fine-tuned, System design for production, Latency optimization, Error handling and fallback, Monitoring and observability, A/B testing for LLMs, Cost management, Security prompt injection."},
    {"p":5,"t":"Emergent Capabilities Scaling","w":4200,"pr":"Write a section on EMERGENT CAPABILITIES AND SCALING. Cover: What are emergent capabilities, Phase transitions in model behavior, Predicting capability emergence from scaling laws, Compute-optimal vs performance-optimal training, Data-optimal scaling, Test-time compute Strawberry o1, The Bitter Lesson, Current frontier models."},
    {"p":6,"t":"Ecosystem Competition","w":4200,"pr":"Write a section on THE LLM ECOSYSTEM. Cover: Timeline of major model releases 2020-2026, Open-source leaders LLaMA Mistral Gemma Qwen Phi DeepSeek, Closed-source frontier GPT-4o Claude 3.7 Gemini 2.5, The closing gap, API economics and pricing trends, Fine-tuning ecosystem Axolotl LLaMA-Factory TRL, Custom silicon TPUs Groq Cerebras, What open-source needs to catch up."},
    {"p":7,"t":"1-bit LLMs and Model Efficiency Frontiers","w":4200,"pr":"Write a section on 1-BIT LLMS AND MODEL EFFICIENCY. Cover: 1-bit LLM BitNet b1.58 architecture, BitNet paper analysis, Ternary and binary networks, Sparse models and pruning at scale, Speculative decoding theory and practice, Early exit strategies for adaptive computation, Mixture of Experts routing efficiency, Hardware-algorithm co-design for efficient inference."},
    {"p":8,"t":"The Path Forward: AGI and Long-term AI","w":4202,"pr":"Write a section on AGI AND LONG-TERM AI. Cover: Current definitions of AGI and their limitations, Benchmark saturation and what it means, Reasoning能力的 current state and gaps, World models and situational awareness, LLM benchmarking for general intelligence, What remains unsolved, Timeline debates and expert opinions, The role of open-source in safe AI development, Recommendations for researchers and practitioners."}
  ]},
]



def wc(text):
    return len(re.findall(r'\S+', text))

def gen(client, ch_n, ch_t, pp_n, pp_t, prompt, budget):
    # budget = target word count; num_predict caps generated tokens.
    # ~1.5 tokens/word + 40% markdown overhead for gemma3:4b.
    # 8 passes × 4200w = 33,600w/ch; 12 ch ≈ 403,200w ≈ 400 pages.
    num_predict = min(budget * 2.5, 12000)
    for attempt in range(3):
        content, stats = client.generate(
            prompt="Chapter %d: %s\n\n%s" % (ch_n, ch_t, prompt),
            system=SYS, temperature=0.7,
            num_predict=num_predict,
        )
        w = wc(content)
        if w >= budget * 0.35 or attempt == 2:
            return content, stats, w
    return content, stats, wc(content)

def run(batch=2, start_ch=1, start_pp=1):
    sys.stdout.flush()
    print("=" * 70)
    print("Deep Agent Multi-Pass Pipeline -- 400+ Page Book Generator")
    print("Model: %s | Batch: %d" % (MODEL, batch))
    print("=" * 70)
    print("")

    client = OllamaClient()
    if not client.health():
        print("[FATAL] Ollama not reachable"); sys.exit(1)
    print("[OK] Ollama connected")
    print("")

    state = load_state()
    all_tasks = [(ch["n"], ch["t"], pp["p"], pp["t"], pp["pr"], pp["w"])
                 for ch in CHAPTERS if ch["n"] >= start_ch
                 for pp in ch["passes"] if pp["p"] >= start_pp]
    total = len(all_tasks)
    done = sum(1 for t in all_tasks if "%d.%d" % (t[0], t[2]) in state.get("passes", {}))

    print("Tasks: %d | Done: %d | Remaining: %d" % (total, done, total-done))
    print("")

    t0 = time.time()
    for i, (ch_n, ch_t, pp_n, pp_t, prompt, budget) in enumerate(all_tasks):
        key = "%d.%d" % (ch_n, pp_n)
        if key in state.get("passes", {}):
            print("[SKIP %d/%d] Ch%d Pass %d: done" % (i+1, total, ch_n, pp_n))
            continue

        print("")
        print("[%d/%d] Ch%d Pass %d: %s" % (i+1, total, ch_n, pp_n, pp_t))
        sys.stdout.flush()

        content, stats, w = gen(client, ch_n, ch_t, pp_n, pp_t, prompt, budget)
        tokens = stats.get("tokens", 0)
        tps = stats.get("tps", 0)

        state.setdefault("passes", {})[key] = {
            "ch": ch_n, "ch_t": ch_t, "pp": pp_n, "title": pp_t,
            "content": content, "wc": w, "tokens": tokens, "tps": tps,
            "at": datetime.now().isoformat(),
        }
        state["total_words"] = state.get("total_words", 0) + w
        state["total_tokens"] = state.get("total_tokens", 0) + tokens
        state["total_calls"] = state.get("total_calls", 0) + 1

        print("  OK: %dw, %dt @ %s tok/s" % (w, tokens, tps))
        sys.stdout.flush()

        if (i + 1) % 2 == 0:
            save_state(state)
            elapsed = time.time() - t0
            rem = total - done - (i + 1)
            est = rem * elapsed / max(i + 1 - done, 1)
            print("")
            print("  CHECKPOINT | %d/%d | Elapsed: %.1fmin | Est: %.1fmin" % (
                i+1-done, total-done, elapsed/60, est/60))
            sys.stdout.flush()

    save_state(state)
    total_time = time.time() - t0
    assemble(state)
    make_report(state, total_time)

    print("")
    print("=" * 70)
    print("DONE!")
    print("Time: %.1fmin | Words: %d | Pages: ~%d" % (
        total_time/60, state.get("total_words",0), state.get("total_words",0)//400))
    print("Output: %s" % FINAL_MD)
    print("=" * 70)
    sys.stdout.flush()


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"passes": {}, "total_words": 0, "total_tokens": 0, "total_calls": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def assemble(state):
    by_ch = defaultdict(list)
    for k, v in state.get("passes", {}).items():
        by_ch[v["ch"]].append((v["pp"], v))

    book = '''---
title: "Large Language Model Engineering"
subtitle: "A Comprehensive Handbook: Mathematics, Architecture, Training, Alignment, and Deployment"
author: Generated by Deep Agent Multi-Pass Pipeline | %s | %s
lang: en
---

# Large Language Model Engineering

_A Comprehensive Handbook: Mathematics, Architecture, Training, Alignment, and Deployment_

---

''' % (MODEL, datetime.now().strftime("%B %Y"))

    for ch_n in sorted(by_ch.keys()):
        sections = sorted(by_ch[ch_n], key=lambda x: x[0])
        ch_t = sections[0][1]["ch_t"]
        book += "# Chapter %d: %s\n\n" % (ch_n, ch_t)
        for pp_n, result in sections:
            book += "## Pass %d: %s\n\n%s\n\n" % (pp_n, result["title"], result["content"])
        book += "---\n\n"

    with open(FINAL_MD, "w", encoding="utf-8") as f:
        f.write(book)
    print("")
    print("[ASSEMBLE] %s (%d chars)" % (FINAL_MD, len(book)))


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
        "passes": {k: {"ch": v["ch"], "pp": v["pp"], "title": v["title"],
                       "wc": v["wc"], "tokens": v["tokens"], "tps": v["tps"]}
                   for k, v in state.get("passes", {}).items()},
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print("[REPORT] %s" % REPORT_FILE)


def render_pdf():
    try:
        import weasyprint, warnings
        warnings.filterwarnings("ignore")
        with open(FINAL_MD) as f:
            content = f.read()
        if content.startswith("---"):
            end = content.find("\n---\n", 4)
            if end >= 0:
                content = content[end + 5:]
        lines = content.split('\n')
        in_code = False
        fixed = []
        for line in lines:
            if line.strip().startswith('```'):
                in_code = not in_code
            elif line.strip() == "---" and not in_code:
                fixed.append("* * *")
            else:
                fixed.append(line)
        content = "\n".join(fixed)
        clean_md = OUT_DIR / "book_clean.md"
        with open(clean_md, "w") as f:
            f.write(content)
        subprocess.run(["pandoc", str(clean_md), "-o", str(FINAL_HTML),
                       "--standalone", "--toc", "--toc-depth=3",
                       "--metadata", "title=Large Language Model Engineering"],
                      capture_output=True)
        weasyprint.HTML(filename=str(FINAL_HTML)).write_pdf(str(FINAL_PDF))
        sz = os.path.getsize(FINAL_PDF)
        print("[PDF] %s (%.0f KB)" % (FINAL_PDF, sz/1024))
    except Exception as e:
        print("[PDF] Failed: %s" % e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", "-b", type=int, default=2)
    p.add_argument("--start-ch", type=int, default=1)
    p.add_argument("--start-pp", type=int, default=1)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    run(batch=args.batch, start_ch=args.start_ch, start_pp=args.start_pp)
    if args.render:
        print("")
        print("[RENDER] PDF...")
        render_pdf()
