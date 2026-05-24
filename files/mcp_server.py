#!/usr/bin/env python3
"""
MCP server for Qwen Book Pipeline.
Provides tools to interact with the Qwen3.6-35B-A3B book generation pipeline.

Install: pip install mcp httpx pydantic
Run:    python mcp_server.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_ROOT = Path(__file__).parent.parent if Path(__file__).name == "mcp_server.py" else Path.cwd()
CHECKPOINT_DIR = PIPELINE_ROOT / "output" / ".checkpoints"

STAGES = {
    "0": {"name": "Spec Compiler", "llm_calls": 1, "time": "5s"},
    "1": {"name": "STORM Outline", "llm_calls": "~30", "time": "~5 min"},
    "2": {"name": "Research", "llm_calls": "~3000", "time": "~30 min"},
    "3": {"name": "Writer", "llm_calls": "~500", "time": "2-4 hours"},
    "4": {"name": "Citations", "llm_calls": "0", "time": "~5 min"},
    "5": {"name": "Quality Gates", "llm_calls": "~500", "time": "~15 min"},
    "6": {"name": "Render PDF", "llm_calls": "0", "time": "~2 min"},
}

CHECKPOINT_FILES = {
    "00-spec.json": "Stage 0 — Spec Compiler",
    "01-outline.json": "Stage 1 — STORM Outline",
    "02-research.json": "Stage 2 — Research",
    "03-content.json": "Stage 3 — Writer",
    "04-bib.json": "Stage 4 — Citations (bib)",
    "04-content-cited.json": "Stage 4 — Citations (content)",
    "05-quality.json": "Stage 5 — Quality Gates",
}

HARDWARE_PRESETS = {
    "24gb": {
        "gpu": "RTX 4090 / RTX 5090",
        "quant": "AWQ Q4",
        "context": "128K",
        "throughput": "~50 tok/s",
        "time_400p": "~6 hours",
        "vram": "24GB",
    },
    "48gb": {
        "gpu": "RTX 6000 Ada / 2x4090",
        "quant": "FP8",
        "context": "256K",
        "throughput": "~70 tok/s",
        "time_400p": "~4 hours",
        "vram": "48GB",
    },
    "96gb": {
        "gpu": "RTX PRO 6000",
        "quant": "BF16",
        "context": "256K",
        "throughput": "~150 tok/s",
        "time_400p": "~2 hours",
        "vram": "96GB",
    },
    "h100x8": {
        "gpu": "8x H100 80GB",
        "quant": "BF16 TP=8",
        "context": "1M",
        "throughput": "~500 tok/s",
        "time_400p": "~30 min",
        "vram": "640GB",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def count_leaves(data: dict) -> int:
    """Count leaf nodes in an outline tree."""
    count = 0
    chapters = data.get("chapters", [])
    for ch in chapters:
        for sec in ch.get("sections", []):
            count += len(sec.get("subsections", []))
    return count


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("qwen-book-pipeline")


@mcp.tool()
def get_pipeline_status() -> dict[str, Any]:
    """Check the current status of the book pipeline, including which stages have completed."""
    status = {"completed": [], "in_progress": None, "next_stage": None, "resume_available": False}

    completed_stages = []
    for ckpt_file, stage_name in CHECKPOINT_FILES.items():
        path = CHECKPOINT_DIR / ckpt_file
        if path.exists():
            completed_stages.append(stage_name)
        else:
            status["next_stage"] = stage_name
            break

    status["completed"] = completed_stages
    status["in_progress"] = None
    status["resume_available"] = len(completed_stages) > 0 and len(completed_stages) < len(CHECKPOINT_FILES)

    if len(completed_stages) == len(CHECKPOINT_FILES):
        status["in_progress"] = "ALL COMPLETE — book.pdf ready"
        status["next_stage"] = None
        status["resume_available"] = False

        pdf_path = PIPELINE_ROOT / "output" / "book.pdf"
        if pdf_path.exists():
            status["pdf_path"] = str(pdf_path)
            status["pdf_size_mb"] = round(pdf_path.stat().st_size / 1e6, 1)

    return status


@mcp.tool()
def get_stage_info(stage_number: int) -> dict[str, Any]:
    """Get detailed information about a specific pipeline stage (0-6)."""
    if stage_number < 0 or stage_number > 6:
        return {"error": "Stage number must be between 0 and 6"}

    stage_key = str(stage_number)
    info = STAGES.get(stage_key, {})

    # Add checkpoint info
    ckpt_file = None
    for f, name in CHECKPOINT_FILES.items():
        if name.startswith(f"Stage {stage_number}"):
            ckpt_file = f
            break

    result = {
        "stage": stage_number,
        "name": info.get("name"),
        "llm_calls": info.get("llm_calls"),
        "estimated_time": info.get("time"),
        "checkpoint_file": ckpt_file,
        "checkpoint_exists": (CHECKPOINT_DIR / ckpt_file).exists() if ckpt_file else False,
    }

    # Add checkpoint content preview if exists
    if ckpt_file:
        ckpt_data = load_json(CHECKPOINT_DIR / ckpt_file)
        if ckpt_data:
            if stage_number == 0:
                result["spec_preview"] = {
                    "title": ckpt_data.get("title", "unknown"),
                    "target_pages": ckpt_data.get("target_pages"),
                    "languages": ckpt_data.get("languages"),
                    "audience": ckpt_data.get("audience"),
                    "depth": ckpt_data.get("depth"),
                }
            elif stage_number == 1:
                leaf_count = count_leaves(ckpt_data)
                result["outline_preview"] = {
                    "title": ckpt_data.get("title"),
                    "chapter_count": len(ckpt_data.get("chapters", [])),
                    "leaf_count": leaf_count,
                }
            elif stage_number == 2:
                result["research_preview"] = {
                    "leaf_count": len(ckpt_data) if isinstance(ckpt_data, dict) else 0,
                }
            elif stage_number == 3:
                result["content_preview"] = {
                    "leaf_count": len(ckpt_data) if isinstance(ckpt_data, dict) else 0,
                }
            elif stage_number == 5:
                if isinstance(ckpt_data, dict):
                    reports = ckpt_data.get("reports", list(ckpt_data.values())[:5])
                    result["quality_preview"] = {
                        "total_leaves": len(ckpt_data),
                        "sample_reports": [
                            {"leaf_id": r.get("leaf_id"), "overall": r.get("overall")}
                            for r in reports if isinstance(r, dict)
                        ],
                    }

    return result


@mcp.tool()
def list_hardware_presets() -> dict[str, dict]:
    """List all available hardware presets for running the pipeline with vLLM."""
    return HARDWARE_PRESETS


@mcp.tool()
def estimate_pipeline(
    target_pages: int = 400,
    hardware_preset: str = "24gb",
    languages: str = "vi,en",
) -> dict[str, Any]:
    """
    Estimate pipeline cost and time for a given book specification.

    Args:
        target_pages: Target number of pages (50-2000, default 400)
        hardware_preset: Hardware preset (24gb, 48gb, 96gb, h100x8)
        languages: Comma-separated language codes (default: vi,en)
    """
    if hardware_preset not in HARDWARE_PRESETS:
        return {"error": f"Unknown preset. Choose from: {list(HARDWARE_PRESETS.keys())}"}

    preset = HARDWARE_PRESETS[hardware_preset]
    langs = [l.strip() for l in languages.split(",")]

    # Stage time estimates scaled by page count
    scale = target_pages / 400
    stage_times = {
        "0_Spec": "5s",
        "1_Outline": f"~{int(5 * scale)} min",
        "2_Research": f"~{int(30 * scale)} min",
        "3_Writing": f"~{int(180 * scale)}–{int(240 * scale)} min",
        "4_Citations": "~5 min",
        "5_Quality": f"~{int(15 * scale)} min",
        "6_Render": "~2 min",
    }

    # Approximate token count for context
    leaf_count = max(50, int(target_pages / 0.8))

    return {
        "target_pages": target_pages,
        "hardware_preset": hardware_preset,
        "gpu": preset["gpu"],
        "estimated_total_time": f"~{int(3.5 * scale)}–{int(5.5 * scale)} hours",
        "stage_times": stage_times,
        "estimated_leaf_count": leaf_count,
        "llm_cost": "$0 (local GPU)",
        "estimated_search_cost_usd": f"~${12.50 * scale:.2f}" if "basic" else f"~${25 * scale:.2f}",
        "languages": langs,
        "note": "Search cost depends on Tavily plan. LLM cost is $0 if running on local GPU.",
    }


@mcp.tool()
def generate_run_command(
    prompt: str,
    hardware_preset: str = "24gb",
    resume: bool = True,
) -> dict[str, str]:
    """
    Generate the command to run the book pipeline.

    Args:
        prompt: User prompt describing the book to generate
        hardware_preset: Hardware preset (24gb, 48gb, 96gb, h100x8)
        resume: Whether to resume from last checkpoint (default True)
    """
    if hardware_preset not in HARDWARE_PRESETS:
        return {"error": f"Unknown preset. Choose from: {list(HARDWARE_PRESETS.keys())}"}

    preset = HARDWARE_PRESETS[hardware_preset]

    commands = {
        "setup": "# 1. Install dependencies\npython3.11 -m venv .venv && source .venv/bin/activate\npip install -r requirements.txt",
        "env": "# 2. Set environment variables\nexport TAVILY_API_KEY='tvly-...'\nexport VLLM_MODEL_PATH='Qwen/Qwen3.6-35B-A3B-AWQ'  # for 24gb, or use Qwen/Qwen3.6-35B-A3B for bf16",
        "vllm_start": f"# 3. Start vLLM server\n./scripts/start_vllm.sh {hardware_preset}\n# or manually:\n# vllm serve {preset.get('gpu', 'Qwen/Qwen3.6-35B-A3B-AWQ' if hardware_preset == '24gb' else 'Qwen/Qwen3.6-35B-A3B')}",
        "run": f"# 4. Run the pipeline\npython -m src.pipeline \"{prompt}\""
        + ("" if resume else " --no-resume"),
        "run_resume_note": "# Resume from last checkpoint (default behavior)" if resume else "# Force re-run (ignores checkpoints)",
    }

    return {
        "commands": commands,
        "hardware": preset,
        "pipeline_prompt": prompt,
        "resume": resume,
        "note": "Run commands sequentially. Check pipeline status with get_pipeline_status after starting.",
    }


@mcp.tool()
def get_checkpoint_content(stage: int) -> dict[str, Any] | str:
    """
    Read and return the content of a pipeline checkpoint for a given stage.

    Args:
        stage: Stage number (0-5)
    """
    if stage < 0 or stage > 5:
        return {"error": "Stage must be between 0 and 5"}

    ckpt_map = {
        0: "00-spec.json",
        1: "01-outline.json",
        2: "02-research.json",
        3: "03-content.json",
        4: "04-content-cited.json",
        5: "05-quality.json",
    }

    ckpt_file = ckpt_map[stage]
    ckpt_path = CHECKPOINT_DIR / ckpt_file

    data = load_json(ckpt_path)
    if data is None:
        return f"Checkpoint not found: {ckpt_file}. Stage may not have completed yet."

    # Return a summary based on stage
    if stage == 0:
        return {
            "stage": 0,
            "checkpoint": ckpt_file,
            "data": {
                "title": data.get("title"),
                "target_pages": data.get("target_pages"),
                "target_words": data.get("target_words"),
                "languages": data.get("languages"),
                "audience": data.get("audience"),
                "depth": data.get("depth"),
                "chapter_count_target": data.get("chapter_count_target"),
                "constraints": data.get("constraints", []),
            },
        }
    elif stage == 1:
        return {
            "stage": 1,
            "checkpoint": ckpt_file,
            "data": {
                "title": data.get("title"),
                "chapter_count": len(data.get("chapters", [])),
                "leaf_count": count_leaves(data),
                "perspectives": [p.get("name") for p in data.get("perspectives", [])],
            },
        }
    elif stage == 2:
        if isinstance(data, dict):
            return {
                "stage": 2,
                "checkpoint": ckpt_file,
                "leaf_count": len(data),
                "sample_leaves": list(data.keys())[:5],
            }
        return {"stage": 2, "checkpoint": ckpt_file, "data": data}
    elif stage == 3:
        if isinstance(data, dict):
            return {
                "stage": 3,
                "checkpoint": ckpt_file,
                "leaf_count": len(data),
                "sample_leaves": list(data.keys())[:5],
            }
        return {"stage": 3, "checkpoint": ckpt_file, "data": data}
    elif stage == 4:
        return {
            "stage": 4,
            "checkpoint": ckpt_file,
            "bib_entries": len(data) if isinstance(data, list) else "unknown",
            "sample": data[0] if isinstance(data, list) and data else None,
        }
    elif stage == 5:
        if isinstance(data, dict):
            total = len(data)
            pass_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("overall") == "pass")
            warn_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("overall") == "warn")
            fail_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("overall") == "fail")
            return {
                "stage": 5,
                "checkpoint": ckpt_file,
                "total_leaves": total,
                "pass": pass_count,
                "warn": warn_count,
                "fail": fail_count,
                "pass_rate": f"{pass_count / total * 100:.1f}%" if total > 0 else "N/A",
            }
        return {"stage": 5, "checkpoint": ckpt_file, "data": data}
    return {"error": "Unknown stage"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
