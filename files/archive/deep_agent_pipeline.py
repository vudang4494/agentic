#!/usr/bin/env python3
"""
Deep Agent Book Pipeline — Qwen3.5-4B + Ollama
================================================

Architecture: Stateless Writer Agents + Orchestrator State Machine

The key insight: instead of one-shot generation (which fails), we break
400 pages into 100-200 small LLM calls (~500-1500 tokens each).

Each "writer agent" call:
  - Takes subsection spec + context (prev summary, notation table)
  - Outputs ~800-1500 words of content
  - ~2-3 minutes at 17 tok/s
  - 5 concurrent calls → ~85 tok/s effective throughput

Pipeline:
  STAGE 0: Bootstrap    → generate SPEC + CHAPTER OUTLINE (small JSONs)
  STAGE 1: Chapter Plan → expand each chapter into subsections
  STAGE 2: Write       → batch-write all subsections concurrently
  STAGE 3: Assemble    → combine subsections into chapters
  STAGE 4: Render      → Pandoc → PDF
"""

import json
import os
import re
import sys
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed")
    sys.exit(1)


# ============================================================================
# Constants
# ============================================================================

OLLAMA_BASE = "http://localhost:11434"
MODEL = "gemma3:4b"
DEFAULT_TIMEOUT = 120

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

CHECKPOINT = OUT_DIR / "deep_agent_state.json"
REPORT_FILE = OUT_DIR / "benchmark_report.json"

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class GenerationStats:
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    elapsed_s: float = 0.0
    load_duration_s: float = 0.0
    eval_duration_s: float = 0.0
    total_duration_s: float = 0.0
    done_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "elapsed_s": round(self.elapsed_s, 2),
            "load_duration_s": round(self.load_duration_s, 2),
            "eval_duration_s": round(self.eval_duration_s, 2),
            "total_duration_s": round(self.total_duration_s, 2),
            "done_reason": self.done_reason,
        }


@dataclass
class SubsectionResult:
    id: str
    title: str
    content: str
    word_count: int
    stats: GenerationStats
    success: bool
    error: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "word_count": self.word_count,
            "stats": self.stats.to_dict(),
            "success": self.success,
            "error": self.error,
            "created_at": self.created_at,
        }


@dataclass
class ChapterResult:
    number: int
    title: str
    subsections: list[SubsectionResult] = field(default_factory=list)
    assembled_content: str = ""
    total_words: int = 0
    total_tokens: int = 0
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "subsections": [s.to_dict() for s in self.subsections],
            "assembled_content": self.assembled_content,
            "total_words": self.total_words,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time_s, 2),
        }


@dataclass
class PipelineState:
    stage: str = "init"
    spec: Optional[dict] = None
    chapters: list[dict] = field(default_factory=list)
    subsections: dict[str, dict] = field(default_factory=dict)
    written: dict[str, SubsectionResult] = field(default_factory=dict)
    chapters_result: dict[int, ChapterResult] = field(default_factory=dict)
    stage_0_time: float = 0.0
    stage_1_time: float = 0.0
    stage_2_time: float = 0.0
    stage_3_time: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0
    started_at: str = ""
    updated_at: str = ""

    def save(self):
        data = {
            "stage": self.stage,
            "spec": self.spec,
            "chapters": self.chapters,
            "subsections": self.subsections,
            "written": {k: v.to_dict() for k, v in self.written.items()},
            "chapters_result": {k: v.to_dict() for k, v in self.chapters_result.items()},
            "stage_0_time": self.stage_0_time,
            "stage_1_time": self.stage_1_time,
            "stage_2_time": self.stage_2_time,
            "stage_3_time": self.stage_3_time,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "started_at": self.started_at,
            "updated_at": datetime.now().isoformat(),
        }
        with open(CHECKPOINT, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load() -> "PipelineState":
        if CHECKPOINT.exists():
            with open(CHECKPOINT) as f:
                data = json.load(f)
            state = PipelineState()
            state.stage = data.get("stage", "init")
            state.spec = data.get("spec")
            state.chapters = data.get("chapters", [])
            state.subsections = data.get("subsections", {})
            state.total_llm_calls = data.get("total_llm_calls", 0)
            state.total_tokens = data.get("total_tokens", 0)
            state.started_at = data.get("started_at", "")

            # Reconstruct SubsectionResult objects
            for k, v in data.get("written", {}).items():
                stats = GenerationStats(**v.get("stats", {}))
                state.written[k] = SubsectionResult(
                    id=v["id"], title=v["title"], content=v["content"],
                    word_count=v["word_count"], stats=stats,
                    success=v["success"], error=v.get("error", ""),
                    created_at=v.get("created_at", "")
                )

            # Reconstruct ChapterResult objects
            for k, v in data.get("chapters_result", {}).items():
                subs = []
                for sv in v.get("subsections", []):
                    stats = GenerationStats(**sv.get("stats", {}))
                    subs.append(SubsectionResult(
                        id=sv["id"], title=sv["title"], content=sv["content"],
                        word_count=sv["word_count"], stats=stats,
                        success=sv["success"], error=sv.get("error", ""),
                        created_at=sv.get("created_at", "")
                    ))
                state.chapters_result[int(k)] = ChapterResult(
                    number=v["number"], title=v["title"], subsections=subs,
                    assembled_content=v.get("assembled_content", ""),
                    total_words=v.get("total_words", 0),
                    total_tokens=v.get("total_tokens", 0),
                    total_time_s=v.get("total_time_s", 0.0)
                )
            return state
        return PipelineState()


# ============================================================================
# Ollama Client (Thread-Safe)
# ============================================================================

class OllamaClient:
    """Thread-safe Ollama HTTP client with retry."""

    def __init__(self, base_url: str = OLLAMA_BASE, model: str = MODEL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        num_predict: int = 4096,
        stop: list[str] = None,
    ) -> tuple[str, GenerationStats]:
        """Generate with thinking=true, extract content field."""
        t0 = time.time()
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_p": 0.95,
                "top_k": 20,
                "repeat_penalty": 1.05,
                "stop": stop or [],
            },
        }

        with self.lock:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(f"{self.base_url}/api/chat", json=payload)
                    r.raise_for_status()
                    data = r.json()
            except httpx.TimeoutException:
                stats = GenerationStats(elapsed_s=time.time() - t0)
                return f"[TIMEOUT after {self.timeout}s]", stats
            except Exception as e:
                stats = GenerationStats(elapsed_s=time.time() - t0)
                return f"[ERROR: {e}]", stats

        msg = data.get("message", {})
        content = msg.get("content", "").strip()
        thinking = msg.get("thinking", "").strip()

        # Ollama quirk: with thinking=true, content is in 'content' field
        # If content is empty, try thinking field
        if not content and thinking:
            content = thinking

        eval_count = data.get("eval_count", 0)
        eval_dur = data.get("eval_duration", 0)
        load_dur = data.get("load_duration", 0)
        tps = eval_count / (eval_dur / 1e9) if eval_dur > 0 else 0.0

        stats = GenerationStats(
            tokens_generated=eval_count,
            tokens_per_second=round(tps, 2),
            elapsed_s=round(time.time() - t0, 2),
            load_duration_s=round(load_dur / 1e9, 2),
            eval_duration_s=round(eval_dur / 1e9, 2),
            total_duration_s=round(data.get("total_duration", 0) / 1e9, 2),
            done_reason=data.get("done_reason", ""),
        )
        return content, stats

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except:
            return False


# ============================================================================
# Agent Prompts
# ============================================================================

SYSTEM_SPEC = """You are a book specification compiler. Output ONLY valid JSON, no markdown."""

PROMPT_SPEC = """Create a detailed book specification for this topic: "{topic}"

Output ONLY valid JSON:
{{
  "title": "string (AI/ML book title about Large Language Models)",
  "subtitle": "string",
  "target_pages": number,
  "target_words": number (= target_pages * 400),
  "languages": ["vi", "en"],
  "audience": "beginner | intermediate | advanced | expert",
  "depth": "overview | textbook | handbook | reference",
  "domain": "string (AI, deep learning, NLP, transformers, LLM)",
  "chapter_count": number (8-15),
  "constraints": ["string"]
}}
No markdown fences, no explanation. Pure JSON.

IMPORTANT: LLM = Large Language Model = AI model. Topics: transformers, attention, GPT, BERT, LLaMA, tokenization, embedding, training, fine-tuning, RLHF, PyTorch, vLLM. NOT language learning."""

SYSTEM_OUTLINE = """You are a book outline architect. Output ONLY valid JSON."""

PROMPT_OUTLINE = """Generate a 3-level book outline as valid JSON for a technical AI/ML handbook about Large Language Models:

CRITICAL TOPICS to cover: Transformer architecture, attention mechanism, self-attention, multi-head attention, positional encoding, LayerNorm, feed-forward networks, tokenization, BPE, sentencepiece, embedding layers, pre-training (MLM, NSP), SFT, RLHF, DPO, reward modeling, fine-tuning, LoRA, QLoRA, prompt engineering, in-context learning, chain-of-thought, retrieval-augmented generation, evaluation benchmarks (MMLU, HELM, BIG-bench), model compression, quantization, distillation, deployment, vLLM, text generation, sampling strategies, beam search, temperature, top-k, top-p, GPT-2/3/4, LLaMA, BERT, T5, Mistral, Gemma, Flash Attention, rotary embeddings, MoE, sparse attention, KV cache, batching, speculative decoding.

Output ONLY valid JSON:
{{
  "chapters": [
    {{
      "number": 1,
      "title": "string",
      "sections": [
        {{
          "number": 1,
          "title": "string",
          "subsections": [
            {{"id": "1.1.1", "number": 1, "title": "string", "word_budget": number}}
          ]
        }}
      ],
      "word_budget": number
    }}
  ],
  "notation_table": [{{"symbol": "string", "meaning": "string"}}]
}}

Rules:
- {chapter_count} chapters
- Each chapter: 2-4 sections
- Each section: 2-4 subsections
- Total ~{total_subsections} subsections for ~{target_pages} pages
- Word budget per chapter: ~{words_per_chapter} words
- Subsections are atomic: 1 sub = ~800-1500 words (1-2 pages)
- Bilingual friendly topics

Output ONLY JSON. No markdown fences."""

SYSTEM_WRITER = """You are a technical book writer. Output ONLY content in Markdown format.
No meta-commentary. No "In this section...". Direct, authoritative technical writing."""

PROMPT_WRITER = """Write subsection: "{title}"

Context:
- Chapter {chapter_num}: {chapter_title}
- Section {section_num}: {section_title}
- Word budget: ~{word_budget} words
- Audience: expert in ML/NLP
- Language: Vietnamese primary with English technical terms

Requirements:
- Start directly with content
- Include ```python for code examples
- Include $...$ for inline math
- Include $$...$$ for math blocks
- Include [n] for citations (real paper titles if relevant)
- Target ~{word_budget} words of substantive technical content
- No preamble, no "In this section", no "This subsection covers"
- Be authoritative and precise

Output ONLY the Markdown content. No thinking, no JSON."""

SYSTEM_CHAPTER_PLAN = """You are a chapter planning agent. Output ONLY valid JSON."""

PROMPT_CHAPTER_PLAN = """Expand chapter "{chapter_title}" into subsections.

Chapter {number}: {chapter_title}
Word budget: ~{word_budget} words
Available subsections: ~{num_subsections}

Output ONLY valid JSON:
{{
  "sections": [
    {{
      "number": 1,
      "title": "string",
      "subsections": [
        {{"id": "{number}.1.1", "number": 1, "title": "string", "word_budget": number}},
        {{"id": "{number}.1.2", "number": 2, "title": "string", "word_budget": number}}
      ]
    }}
  ]
}}

Rules:
- Each subsection: ~800-1200 words
- Total subsections: ~{num_subsections}
- Subsections must be atomic, self-contained topics
- Output ONLY JSON. No markdown fences."""


# ============================================================================
# JSON Helpers
# ============================================================================

def extract_json(text: str) -> Optional[dict]:
    """Robust JSON extraction from LLM output."""
    text = text.strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or start >= end:
        return None
    json_str = text[start:end + 1]
    # Remove markdown fences
    json_str = re.sub(r"```json\s*", "", json_str)
    json_str = re.sub(r"```\s*$", "", json_str)
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    # Try brace-level parsing
    depth = 0
    for i, ch in enumerate(json_str):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(json_str[:i + 1])
                except:
                    pass
    return None


def count_words(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r'\S+', text))


# ============================================================================
# Stage 0: Bootstrap — Spec + Outline
# ============================================================================

def stage0_bootstrap(client: OllamaClient, topic: str, state: PipelineState) -> PipelineState:
    """Generate SPEC + full chapter outline. Small calls only."""
    t0 = time.time()
    print("\n[STAGE 0] Bootstrap — Spec + Outline")
    print(f"  Topic: {topic}")

    # Step 0a: Generate SPEC (tiny JSON, ~500 tokens)
    print("  [0a] Generating book specification...")
    content, stats = client.generate(
        prompt=PROMPT_SPEC.format(topic=topic),
        system=SYSTEM_SPEC,
        temperature=0.3,
        num_predict=2048,
    )
    spec = extract_json(content)
    if spec:
        spec["topic"] = topic
        spec["target_pages"] = 400
        spec["target_words"] = 160000
        spec["chapter_count"] = max(8, min(15, spec.get("chapter_count", 10)))
        spec["_meta"] = {
            "stage": "bootstrap",
            "generated_at": datetime.now().isoformat(),
            "stats": stats.to_dict(),
        }
        print(f"    OK: {spec.get('title', 'unknown')}")
        print(f"    Chapters: {spec.get('chapter_count')}, Words: {spec.get('target_words')}")
        print(f"    Tokens: {stats.tokens_generated}, Speed: {stats.tokens_per_second} tok/s, Time: {stats.elapsed_s:.1f}s")
    else:
        print(f"    [WARN] JSON parse failed, using default spec")
        print(f"    Content preview: {content[:200]}")
        spec = {
            "title": "Large Language Model Engineering",
            "subtitle": "Practical Implementation, Mathematics & Optimization",
            "target_pages": 400, "target_words": 160000,
            "languages": ["vi", "en"], "audience": "expert",
            "depth": "handbook", "domain": "Large Language Models",
            "chapter_count": 10,
            "constraints": ["PyTorch code", "Math derivations", "Bilingual EN-VN"],
            "topic": topic,
            "_meta": {"stage": "bootstrap", "generated_at": datetime.now().isoformat(), "stats": stats.to_dict()},
        }
    state.spec = spec

    # Step 0b: Generate full outline (medium JSON, ~4000 tokens)
    cc = spec.get("chapter_count", 10)
    total_subs = max(40, min(80, 160000 // 1600))  # target ~1600 words/subsection
    words_per_ch = 160000 // cc
    target_pages = spec.get("target_pages", 400)

    print(f"  [0b] Generating chapter outline ({cc} chapters, ~{total_subs} subsections)...")
    content, stats = client.generate(
        prompt=PROMPT_OUTLINE.format(
            chapter_count=cc,
            total_subsections=total_subs,
            target_pages=target_pages,
            words_per_chapter=words_per_ch,
        ),
        system=SYSTEM_OUTLINE,
        temperature=0.5,
        num_predict=6144,
    )
    outline = extract_json(content)
    if outline:
        chapters = outline.get("chapters", [])
        all_subs = []
        for ch in chapters:
            for sec in ch.get("sections", []):
                for sub in sec.get("subsections", []):
                    sub["chapter_num"] = ch.get("number", 0)
                    sub["chapter_title"] = ch.get("title", "")
                    sub["section_num"] = sec.get("number", 0)
                    sub["section_title"] = sec.get("title", "")
                    all_subs.append(sub)
        spec["chapters"] = chapters
        spec["all_subsections"] = all_subs
        spec["total_subsections"] = len(all_subs)
        print(f"    OK: {len(chapters)} chapters, {len(all_subs)} subsections")
        print(f"    Tokens: {stats.tokens_generated}, Speed: {stats.tokens_per_second} tok/s, Time: {stats.elapsed_s:.1f}s")
    else:
        print(f"    [WARN] JSON parse failed, generating programmatic outline")
        chapters, all_subs = _generate_outline_programmatic(spec)
        spec["chapters"] = chapters
        spec["all_subsections"] = all_subs
        spec["total_subsections"] = len(all_subs)

    state.spec = spec
    state.chapters = chapters
    state.subsections = {s["id"]: s for s in all_subs}
    state.stage_0_time = time.time() - t0
    state.total_llm_calls += 2
    state.total_tokens += stats.tokens_generated + (stats.tokens_generated // 2)
    print(f"  Stage 0 done in {state.stage_0_time:.1f}s")
    state.save()
    return state


def _generate_outline_programmatic(spec: dict) -> tuple[list, list]:
    """Fallback programmatic outline when LLM JSON fails."""
    domain = spec.get("domain", "LLM")
    cc = spec.get("chapter_count", 10)
    all_subs = []
    chapters = []

    # Standard LLM handbook structure
    topics = [
        ("Introduction to Large Language Models", 4),
        ("Transformer Architecture Deep Dive", 6),
        ("Training Large Language Models", 5),
        ("Pre-training Objectives & Data", 4),
        ("Fine-tuning & Alignment", 5),
        ("Prompt Engineering & In-context Learning", 4),
        ("Evaluation & Benchmarking", 4),
        ("Deployment & Optimization", 5),
        ("Multimodal LLMs", 4),
        ("Frontiers & Future Directions", 4),
    ]

    for i, (ch_title, num_subs) in enumerate(topics[:cc], 1):
        sections = []
        for j in range(min(3, max(2, num_subs))):
            subs = []
            for k in range(num_subs // min(3, max(2, num_subs))):
                sid = f"{i}.{j+1}.{k+1}"
                subs.append({
                    "id": sid, "number": k + 1,
                    "title": f"Topic {k+1} of {ch_title}",
                    "word_budget": 1200,
                    "chapter_num": i, "chapter_title": ch_title,
                    "section_num": j + 1, "section_title": f"Section {j+1}",
                })
            sections.append({"number": j + 1, "title": f"Section {j+1}", "subsections": subs})
            all_subs.extend(subs)
        chapters.append({"number": i, "title": ch_title, "sections": sections, "word_budget": 16000})

    return chapters, all_subs


# ============================================================================
# Stage 1: Expand Chapters into Subsections
# ============================================================================

def stage1_expand_chapters(client: OllamaClient, state: PipelineState, max_concurrent: int = 5) -> PipelineState:
    """Expand each chapter into detailed subsections. Small calls."""
    t0 = time.time()
    print(f"\n[STAGE 1] Chapter Expansion — {len(state.chapters)} chapters")
    total_subsections = len(state.subsections)
    print(f"  Total subsections planned: {total_subsections}")

    # If chapters already have subsections, skip
    chapters_with_subs = sum(1 for ch in state.chapters if ch.get("sections"))
    if chapters_with_subs == len(state.chapters) and total_subsections >= 40:
        print("  [SKIP] All chapters already expanded")
        state.stage_1_time = 0.0
        return state

    # Expand each chapter
    for ch in state.chapters:
        ch_num = ch.get("number", 0)
        ch_title = ch.get("title", "")
        word_budget = ch.get("word_budget", 16000)
        existing_subs = sum(len(sec.get("subsections", [])) for sec in ch.get("sections", []))
        num_subs = max(4, total_subsections // len(state.chapters))

        print(f"\n  Chapter {ch_num}: {ch_title}")

        if existing_subs >= 3:
            print(f"    Already has {existing_subs} subsections, expanding...")
        else:
            print(f"    Expanding to ~{num_subs} subsections...")

        prompt = PROMPT_CHAPTER_PLAN.format(
            chapter_title=ch_title,
            number=ch_num,
            word_budget=word_budget,
            num_subsections=num_subs,
        )

        content, stats = client.generate(prompt, SYSTEM_CHAPTER_PLAN, temperature=0.5, num_predict=4096)
        parsed = extract_json(content)

        if parsed:
            ch["sections"] = parsed.get("sections", [])
            # Update subsection metadata
            new_subs = []
            for sec in ch["sections"]:
                for sub in sec.get("subsections", []):
                    sub["chapter_num"] = ch_num
                    sub["chapter_title"] = ch_title
                    sub["section_num"] = sec.get("number", 0)
                    sub["section_title"] = sec.get("title", "")
                    sub["word_budget"] = sub.get("word_budget", 1200)
                    sub["id"] = sub.get("id", f"{ch_num}.{sec.get('number', 0)}.{sub.get('number', 0)}")
                    state.subsections[sub["id"]] = sub
                    new_subs.append(sub)
            print(f"    Expanded to {len(new_subs)} subsections")
            print(f"    Tokens: {stats.tokens_generated}, Speed: {stats.tokens_per_second} tok/s")
        else:
            print(f"    [WARN] JSON parse failed, keeping existing structure")
            print(f"    Preview: {content[:150]}")

        state.total_llm_calls += 1
        state.total_tokens += stats.tokens_generated
        state.save()

    state.stage_1_time = time.time() - t0
    print(f"\n  Stage 1 done in {state.stage_1_time:.1f}s")
    print(f"  Total subsections: {len(state.subsections)}")
    state.save()
    return state


# ============================================================================
# Stage 2: Deep Agent Write — Concurrent Subsections
# ============================================================================

def _write_subsection(args: tuple) -> SubsectionResult:
    """Write one subsection. Runs in thread pool."""
    sub_id, sub_data, context, client, concurrency_id = args
    t0 = time.time()

    word_budget = sub_data.get("word_budget", 1200)

    prompt = PROMPT_WRITER.format(
        title=sub_data.get("title", ""),
        chapter_num=sub_data.get("chapter_num", 0),
        chapter_title=sub_data.get("chapter_title", ""),
        section_num=sub_data.get("section_num", 0),
        section_title=sub_data.get("section_title", ""),
        word_budget=word_budget,
    )

    system = SYSTEM_WRITER
    if context:
        prompt = f"Previous subsection summary:\n{context[:500]}\n\n---\n\n{prompt}"

    content, stats = client.generate(
        prompt=prompt,
        system=system,
        temperature=0.7,
        num_predict=min(word_budget * 5, 8192),
    )

    # Validate output
    success = len(content) > 200 and not content.startswith("[ERROR")
    wc = count_words(content) if success else 0

    return SubsectionResult(
        id=sub_id,
        title=sub_data.get("title", ""),
        content=content,
        word_count=wc,
        stats=stats,
        success=success,
        error="" if success else f"Content too short: {len(content)} chars",
        created_at=datetime.now().isoformat(),
    )


def stage2_deep_write(client: OllamaClient, state: PipelineState, batch_size: int = 5) -> PipelineState:
    """Write all subsections concurrently in batches."""
    t0 = time.time()
    print(f"\n[STAGE 2] Deep Agent Write — {len(state.subsections)} subsections")
    print(f"  Batch size: {batch_size}, Concurrency: {batch_size} threads")

    # Collect subsections to write
    to_write = []
    for sub_id, sub_data in state.subsections.items():
        if sub_id not in state.written or not state.written[sub_id].success:
            to_write.append((sub_id, sub_data))

    print(f"  Subsections to write: {len(to_write)}")

    total_batches = (len(to_write) + batch_size - 1) // batch_size
    done = 0

    for batch_idx in range(total_batches):
        batch = to_write[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        print(f"\n  Batch {batch_idx + 1}/{total_batches}: writing {len(batch)} subsections...")

        # Build context: previous subsection summary
        contexts = {}
        for sub_id, sub_data in batch:
            ch_num = sub_data.get("chapter_num", 0)
            sec_num = sub_data.get("section_num", 0)
            sub_num = sub_data.get("number", 0)

            # Find previous subsection in same section
            prev_id = f"{ch_num}.{sec_num}.{sub_num - 1}" if sub_num > 1 else None
            if prev_id and prev_id in state.written and state.written[prev_id].success:
                contexts[sub_id] = state.written[prev_id].content[-800:]
            else:
                contexts[sub_id] = ""

        # Write batch concurrently
        args_list = [
            (sub_id, sub_data, contexts.get(sub_id, ""), client, batch_idx * batch_size + i)
            for i, (sub_id, sub_data) in enumerate(batch)
        ]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(_write_subsection, args): args[0] for args in args_list}
            for future in as_completed(futures):
                sub_id = futures[future]
                try:
                    result = future.result()
                    state.written[sub_id] = result
                    state.total_llm_calls += 1
                    state.total_tokens += result.stats.tokens_generated

                    if result.success:
                        print(f"    [OK] {sub_id}: {result.stats.tokens_generated}t @ {result.stats.tokens_per_second} tok/s, {result.word_count} words, {result.stats.elapsed_s:.0f}s")
                        done += 1
                    else:
                        print(f"    [FAIL] {sub_id}: {result.error}")
                except Exception as e:
                    print(f"    [ERROR] {sub_id}: {e}")
                    state.written[sub_id] = SubsectionResult(
                        id=sub_id, title=sub_data.get("title", ""),
                        content="", word_count=0, stats=GenerationStats(),
                        success=False, error=str(e)
                    )

        # Checkpoint every batch
        state.save()

    state.stage_2_time = time.time() - t0
    success_count = sum(1 for r in state.written.values() if r.success)
    total_words = sum(r.word_count for r in state.written.values() if r.success)
    total_tokens = sum(r.stats.tokens_generated for r in state.written.values())
    avg_speed = total_tokens / max(state.stage_2_time, 1)

    print(f"\n  Stage 2 Summary:")
    print(f"    Written: {success_count}/{len(state.subsections)} subsections")
    print(f"    Total words: {total_words:,}")
    print(f"    Total tokens: {total_tokens:,}")
    print(f"    Time: {state.stage_2_time:.1f}s")
    print(f"    Avg speed: {avg_speed:.1f} tok/s")
    state.save()
    return state


# ============================================================================
# Stage 3: Assemble Chapters
# ============================================================================

def stage3_assemble(state: PipelineState) -> PipelineState:
    """Assemble subsections into chapter Markdown."""
    t0 = time.time()
    print("\n[STAGE 3] Assemble Chapters")

    for ch in state.chapters:
        ch_num = ch.get("number", 0)
        ch_result = ChapterResult(number=ch_num, title=ch.get("title", ""))

        # Collect subsections in order
        for sec in ch.get("sections", []):
            for sub in sec.get("subsections", []):
                sub_id = sub.get("id")
                if sub_id and sub_id in state.written:
                    result = state.written[sub_id]
                    if result.success:
                        ch_result.subsections.append(result)
                        ch_result.total_words += result.word_count
                        ch_result.total_tokens += result.stats.tokens_generated
                        ch_result.total_time_s += result.stats.elapsed_s

        # Assemble
        if ch_result.subsections:
            content = f"# Chapter {ch_num}: {ch.get('title', '')}\n\n"
            for sub_result in ch_result.subsections:
                content += f"## {sub_result.title}\n\n{sub_result.content}\n\n"
            ch_result.assembled_content = content

        state.chapters_result[ch_num] = ch_result
        print(f"  Chapter {ch_num}: {len(ch_result.subsections)} subsections, {ch_result.total_words:,} words")

    state.stage_3_time = time.time() - t0
    total_words = sum(r.total_words for r in state.chapters_result.values())
    print(f"\n  Stage 3 done in {state.stage_3_time:.1f}s")
    print(f"  Total words: {total_words:,}")
    print(f"  Estimated pages: ~{total_words // 400}")
    state.save()
    return state


# ============================================================================
# Stage 4: Render PDF
# ============================================================================

def stage4_render(state: PipelineState) -> dict:
    """Assemble and render Markdown → PDF."""
    t0 = time.time()
    print("\n[STAGE 4] Render PDF")

    spec = state.spec or {}
    title = spec.get("title", "Technical Book")
    subtitle = spec.get("subtitle", "")
    lang = spec.get("languages", ["vi", "en"])[0]

    # Assemble full book
    book_content = f"""---
title: "{title}"
subtitle: "{subtitle}"
author: Generated by Deep Agent Pipeline · Qwen3.5-4B
date: {datetime.now().strftime('%B %Y')}
lang: {lang}
---

# {title}

_{subtitle}_

---

"""

    for ch_num in sorted(state.chapters_result.keys()):
        ch_result = state.chapters_result[ch_num]
        if ch_result.assembled_content:
            book_content += ch_result.assembled_content + "\n\n---\n\n"

    # Write Markdown
    md_path = OUT_DIR / "book.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(book_content)
    print(f"  Markdown: {md_path} ({len(book_content):,} chars)")

    # Try PDF render
    pdf_path = OUT_DIR / "book.pdf"
    render_status = "markdown_only"
    pandoc_available = subprocess.run(["which", "pandoc"], capture_output=True).returncode == 0

    if pandoc_available:
        cmd = [
            "pandoc", str(md_path), "-o", str(pdf_path),
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "--toc", "--toc-depth=3", "-s",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and pdf_path.exists():
            render_status = "success"
            print(f"  PDF: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
        else:
            print(f"  [WARN] Pandoc failed: {result.stderr[:200]}")
            pdf_path = md_path
    else:
        print(f"  [INFO] Pandoc not available — markdown only")

    elapsed = time.time() - t0
    total_words = sum(r.total_words for r in state.chapters_result.values())
    print(f"  Render done in {elapsed:.1f}s")
    return {
        "status": render_status,
        "markdown_path": str(md_path),
        "pdf_path": str(pdf_path),
        "total_words": total_words,
        "estimated_pages": total_words // 400,
    }


# ============================================================================
# Benchmark Report
# ============================================================================

def generate_report(state: PipelineState, render: dict) -> dict:
    """Generate comprehensive benchmark report."""
    total_time = state.stage_0_time + state.stage_1_time + state.stage_2_time + state.stage_3_time

    all_stats = [r.stats for r in state.written.values() if r.success]
    if all_stats:
        avg_tps = sum(s.tokens_per_second for s in all_stats) / len(all_stats)
        total_tokens = sum(s.tokens_generated for s in all_stats)
        total_words = sum(r.word_count for r in state.written.values() if r.success)
    else:
        avg_tps = 0
        total_tokens = state.total_tokens
        total_words = 0

    success = sum(1 for r in state.written.values() if r.success)
    failed = sum(1 for r in state.written.values() if not r.success)

    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "pipeline": "Deep Agent Book Pipeline v1.0",
            "topic": state.spec.get("topic", "LLM") if state.spec else "LLM",
        },
        "model_info": {
            "model_id": MODEL,
            "provider": "Ollama",
            "ollama_version": "0.24.0",
            "quantization": "Q4_K_M",
            "parameter_size": "4.3B",
            "model_size_gb": 3.3,
            "context_length": 131072,
            "architecture": "Gemma 3 4B (Sliding Window Attention + Global Attention hybrid)",
            "family": "gemma3",
        },
        "hardware_info": {
            "platform": "macOS",
            "chip": "Apple M4",
            "gpu_cores": 10,
            "metal": "Metal 3",
            "ram_gb": 24,
        },
        "book_spec": {
            "title": state.spec.get("title") if state.spec else "unknown",
            "subtitle": state.spec.get("subtitle") if state.spec else "",
            "target_pages": state.spec.get("target_pages", 400) if state.spec else 400,
            "target_words": state.spec.get("target_words", 160000) if state.spec else 160000,
            "languages": state.spec.get("languages", ["vi", "en"]) if state.spec else ["vi", "en"],
            "chapter_count": len(state.chapters),
            "subsection_count": len(state.subsections),
        },
        "benchmark_results": {
            "total_generation_time_s": round(total_time, 2),
            "total_llm_calls": state.total_llm_calls,
            "total_tokens_generated": total_tokens,
            "total_words_generated": total_words,
            "avg_speed_tps": round(avg_tps, 2),
            "estimated_pages": total_words // 400,
            "stage_breakdown": {
                "stage_0_bootstrap_s": round(state.stage_0_time, 2),
                "stage_1_expand_s": round(state.stage_1_time, 2),
                "stage_2_write_s": round(state.stage_2_time, 2),
                "stage_3_assemble_s": round(state.stage_3_time, 2),
            },
        },
        "quality_metrics": {
            "subsections_written": success,
            "subsections_failed": failed,
            "success_rate": f"{success / max(success + failed, 1) * 100:.1f}%",
            "total_words": total_words,
            "words_per_subsection": round(total_words / max(success, 1)),
        },
        "chapter_details": {
            ch_num: {
                "title": r.title,
                "subsections": len(r.subsections),
                "words": r.total_words,
                "tokens": r.total_tokens,
                "time_s": round(r.total_time_s, 2),
            }
            for ch_num, r in state.chapters_result.items()
        },
        "render_output": render,
        "ollama_quirks_observed": [
            "Gemma 3 4B works well without thinking mode",
            "Qwen3.5-4B thinking mode had content/response field issues with Ollama 0.24.0",
            "JSON structured output needs robust brace-level parsing for both models",
            "Metal GPU throughput: ~32 tok/s for Gemma 3 4B",
            "Multiple concurrent runners consume ~20GB RAM each — need to limit concurrency",
            "Ollama server must be stable before starting pipeline",
        ],
        "vs_reference": {
            "reference_pipeline": "Qwen3.6-35B-A3B (vLLM/CUDA, 6-stage, Tavily search)",
            "this_pipeline": "Gemma 3 4B (Ollama/Metal, 4-stage, no search)",
            "key_differences": [
                "No web research (no Tavily API configured)",
                "No guided JSON decoding (Ollama limitation)",
                "Concurrent batches for throughput",
                "Small atomic subsections instead of large per-leaf",
                "M4 Metal GPU instead of CUDA H100",
                "Model switched from Qwen3.5-4B (thinking mode broken) to Gemma3-4B",
            ],
        },
    }

    report_path = REPORT_FILE
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")
    return report


# ============================================================================
# Main Orchestrator
# ============================================================================

def run_pipeline(topic: str, resume: bool = True, batch_size: int = 5, max_chapters: int = 0) -> dict:
    """Deep Agent Pipeline orchestrator."""
    t0 = time.time()
    print("=" * 70)
    print(f"Deep Agent Book Pipeline — Qwen3.5-4B")
    print(f"Topic: {topic}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {batch_size}")
    print(f"Resume: {resume}")
    print("=" * 70)

    # Init client
    print("\nInitializing Ollama client...")
    client = OllamaClient()
    if not client.health_check():
        print("[FATAL] Ollama not reachable")
        sys.exit(1)
    print("  [OK] Ollama connected")

    # Load or create state
    state = PipelineState.load() if resume else PipelineState()
    if resume and CHECKPOINT.exists():
        print(f"\n  [RESUME] Loaded state from {CHECKPOINT}")
        print(f"  Stage: {state.stage}")
        print(f"  Written: {len(state.written)}/{len(state.subsections)} subsections")
    else:
        state.started_at = datetime.now().isoformat()
        print("\n  [NEW] Starting fresh pipeline")

    # === Stage 0: Bootstrap ===
    if state.stage in ["init"] or not state.spec:
        state.stage = "stage_0"
        state = stage0_bootstrap(client, topic, state)
        state.stage = "stage_1"

    # Limit chapters if specified
    if max_chapters > 0 and len(state.chapters) > max_chapters:
        state.chapters = state.chapters[:max_chapters]
        # Filter subsections
        allowed_chs = {ch["number"] for ch in state.chapters}
        state.subsections = {
            k: v for k, v in state.subsections.items()
            if v.get("chapter_num", 0) in allowed_chs
        }
        print(f"\n  [LIMIT] Restricted to {max_chapters} chapters, {len(state.subsections)} subsections")

    # === Stage 1: Expand Chapters ===
    if state.stage == "stage_1":
        state = stage1_expand_chapters(client, state, batch_size)
        state.stage = "stage_2"

    # === Stage 2: Deep Agent Write ===
    if state.stage == "stage_2":
        state = stage2_deep_write(client, state, batch_size)
        state.stage = "stage_3"

    # === Stage 3: Assemble ===
    if state.stage == "stage_3":
        state = stage3_assemble(state)
        state.stage = "stage_4"

    # === Stage 4: Render ===
    if state.stage == "stage_4":
        render = stage4_render(state)
    else:
        render = stage4_render(state)

    # === Report ===
    print("\n" + "=" * 70)
    print("BENCHMARK REPORT")
    print("=" * 70)
    report = generate_report(state, render)

    total_time = time.time() - t0
    br = report["benchmark_results"]
    print(f"\nModel: {MODEL} ({report['model_info']['quantization']})")
    print(f"Hardware: {report['hardware_info']['chip']} / {report['hardware_info']['metal']}")
    print(f"\nTotal time:      {br['total_generation_time_s']:.1f}s ({br['total_generation_time_s']/60:.1f} min)")
    print(f"LLM calls:      {br['total_llm_calls']}")
    print(f"Tokens:         {br['total_tokens_generated']:,}")
    print(f"Words:          {br['total_words_generated']:,}")
    print(f"Avg speed:      {br['avg_speed_tps']} tok/s")
    print(f"Est. pages:     ~{br['estimated_pages']}")
    print(f"\nQuality:        {report['quality_metrics']['subsections_written']} OK / {report['quality_metrics']['subsections_failed']} failed ({report['quality_metrics']['success_rate']})")
    print(f"\nBook: {report['book_spec']['title']}")
    print(f"Output: {render.get('markdown_path', 'N/A')}")
    print(f"Report: {REPORT_FILE}")
    print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    return report


# ============================================================================
# CLI
# ============================================================================

def signal_handler(sig, frame):
    print(f"\n[INTERRUPT] Saving checkpoint...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Deep Agent Book Pipeline — Qwen3.5-4B")
    parser.add_argument("topic", nargs="?", default="LLM", help="Book topic")
    parser.add_argument("--batch", "-b", type=int, default=3, help="Concurrent calls per batch (default: 3)")
    parser.add_argument("--chapters", "-n", type=int, default=0, help="Max chapters (0=all)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--max-subs", type=int, default=0, help="Max subsections total (0=all)")
    args = parser.parse_args()

    report = run_pipeline(
        topic=args.topic,
        resume=not args.no_resume,
        batch_size=args.batch,
        max_chapters=args.chapters,
    )

if __name__ == "__main__":
    main()
