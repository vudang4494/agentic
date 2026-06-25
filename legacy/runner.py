#!/usr/bin/env python3
"""
Autonomous Runner for the Deep Research Pipeline
================================================
Wraps deep_research.py with crash recovery, Ollama health monitoring, stall detection,
and end-of-run PDF rendering. Total section count is derived from the CHAPTERS list in
deep_research.py so adding/removing sections does not require touching this file.
"""
import json, os, sys, time, signal, subprocess
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent
OUT_DIR = HERE / "output"

# Output prefix: env-controlled so a single runner can drive book.pdf, book1.pdf, etc.
# Must agree with the --out-name passed to deep_research.py.
# When unset, fall back to the legacy filenames (no prefix on state/report/log files).
_OUT_NAME = os.environ.get("DEEP_RESEARCH_OUT_NAME", "").strip()

SCRIPT = HERE / "deep_research.py"
# Each run lives in output/runs/<name>/ with neutral filenames. Must mirror
# deep_research._rebind_output_paths. Default (no DEEP_RESEARCH_OUT_NAME) is the
# run named "book".
RUN_DIR         = OUT_DIR / "runs" / (_OUT_NAME or "book")
RUN_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE      = RUN_DIR / "state.json"
LOG_FILE        = RUN_DIR / "runner.log"
PIPELINE_STDOUT = RUN_DIR / "pipeline.stdout.log"
REPORT_FILE     = RUN_DIR / "report.json"
FINAL_MD        = RUN_DIR / "book.md"
FINAL_HTML      = RUN_DIR / "book.html"
FINAL_PDF       = RUN_DIR / "book.pdf"
CLEAN_MD        = RUN_DIR / "book.clean.md"

MAX_HOURS   = 15.0
BATCH       = 2

OLLAMA_BASE = "http://localhost:11434"
MODEL       = "batiai/qwen3.6-35b:iq3"
PIPELINE_PATTERN = "legacy/deep_research.py"  # used for pgrep -- specific enough to avoid false matches

POLL_HEALTH  = 60   # seconds between health checks
POLL_LOG     = 30   # seconds between log reads
STALL_MAX    = 1800 # 30 min before considering stalled
RESTART_DELAY = 15  # seconds after kill
KILL_TIMEOUT  = 15  # seconds before SIGKILL


def _derive_total_tasks() -> int:
    """Derive section count from deep_research.CHAPTERS so runner stays in sync with the pipeline."""
    try:
        sys.path.insert(0, str(HERE))
        import deep_research  # noqa: WPS433 (intentional dynamic import)
        return sum(len(c["passes"]) for c in deep_research.CHAPTERS)
    except Exception as e:
        print(f"[runner] WARN: could not derive total tasks from deep_research ({e}); using 96", flush=True)
        return 96


TOTAL_TASKS = _derive_total_tasks()


# === LOGGING ===
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# === NOTIFICATION ===
def notify(title: str, body: str):
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{body}" with title "{title}"'],
            timeout=5, capture_output=True,
        )
    except:
        pass


# === OLLAMA ===
def is_ollama_healthy() -> bool:
    try:
        import httpx
        with httpx.Client(timeout=5) as c:
            r = c.get(f"{OLLAMA_BASE}/api/tags")
            return r.status_code == 200
    except:
        return False


def is_model_responsive() -> bool:
    try:
        import httpx
        with httpx.Client(timeout=15) as c:
            r = c.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": MODEL, "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {"num_predict": 10},
                },
            )
            return r.status_code == 200 and r.json().get("message", {}).get("content")
    except:
        return False


def restart_ollama():
    log("Restarting Ollama...")
    subprocess.run(["pkill", "-9", "-f", "ollama runner"], capture_output=True)
    time.sleep(2)
    subprocess.Popen(
        ["/Applications/Ollama.app/Contents/MacOS/Ollama", "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for i in range(20):
        time.sleep(3)
        if is_ollama_healthy():
            log("Ollama is healthy again")
            return True
        log(f"  Waiting for Ollama... {i+1}/20")
    return is_ollama_healthy()


# === PIPELINE PROCESS ===
def is_pipeline_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "legacy/deep_research.py"], capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def kill_pipeline() -> bool:
    """SIGTERM -> wait -> SIGKILL holdouts -> verify gone. Return True only when
    pgrep confirms no surviving PIDs. The runner's respawn logic must trust this
    return value before calling start_pipeline (W3: blocks the two-writers race
    on state.json + Ollama)."""
    result = subprocess.run(
        ["pgrep", "-f", "legacy/deep_research.py"], capture_output=True, text=True
    )
    pids = [int(l) for l in result.stdout.strip().split("\n") if l.strip()]
    if not pids:
        return True
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Graceful wait up to KILL_TIMEOUT, polling every 0.5s.
    deadline = time.time() + KILL_TIMEOUT
    while time.time() < deadline and is_pipeline_running():
        time.sleep(0.5)
    if not is_pipeline_running():
        return True
    # SIGKILL the holdouts.
    for pid in pids:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    # Confirm dead -- up to 5s for the kernel to reap.
    deadline = time.time() + 5
    while time.time() < deadline and is_pipeline_running():
        time.sleep(0.5)
    if is_pipeline_running():
        log("WARN: kill_pipeline could not confirm all PIDs dead; refusing respawn this cycle")
        return False
    return True


def start_pipeline(start_ch: int = 1, start_pp: int = 1) -> bool:
    """Refuse to start if a pipeline is already running. Returns True iff Popen ran.
    Belt-and-suspenders with deep_research.acquire_pipeline_lock() which also
    refuses double-start at the child side."""
    if is_pipeline_running():
        log("REFUSE start_pipeline: an existing legacy/deep_research.py is already running")
        return False
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--batch", str(BATCH),
        "--start-ch", str(start_ch),
        "--start-pp", str(start_pp),
        "--no-render",
    ]
    if os.environ.get("DEEP_RESEARCH_REVIEW") == "1":
        cmd.append("--review")
    if t := os.environ.get("DEEP_RESEARCH_TOPIC"):
        cmd.extend(["--topic", t])
    if n := os.environ.get("DEEP_RESEARCH_N_PASSES"):
        cmd.extend(["--n-passes", n])
    if n := os.environ.get("DEEP_RESEARCH_N_CHAPTERS"):
        cmd.extend(["--n-chapters", n])
    if o := os.environ.get("DEEP_RESEARCH_OUT_NAME"):
        cmd.extend(["--out-name", o])
    if e := os.environ.get("DEEP_RESEARCH_END_CH"):
        cmd.extend(["--end-ch", e])
    log(f"Starting pipeline: {' '.join(cmd[2:])}")
    subprocess.Popen(
        cmd,
        stdout=open(PIPELINE_STDOUT, "a"),
        stderr=subprocess.STDOUT,
    )
    return True


# === PROGRESS ===
def get_progress() -> dict:
    if not STATE_FILE.exists():
        return {"passes": 0, "words": 0, "tokens": 0, "calls": 0}
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        return {
            "passes": len(d.get("passes", {})),
            "words": d.get("total_words", 0),
            "tokens": d.get("total_tokens", 0),
            "calls": d.get("total_calls", 0),
            "age": time.time() - os.path.getmtime(STATE_FILE),
        }
    except:
        return {"passes": 0, "words": 0, "tokens": 0, "error": True}


def get_resume_point() -> tuple:
    """Find the first (chapter, pass) not yet in state, using the live CHAPTERS structure."""
    if not STATE_FILE.exists():
        return 1, 1
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        passes = state.get("passes", {})
        if not passes:
            return 1, 1
        sys.path.insert(0, str(HERE))
        import deep_research
        for ch in deep_research.CHAPTERS:
            for pp in ch["passes"]:
                key = f"{ch['n']}.{pp['p']}"
                if key not in passes:
                    return ch["n"], pp["p"]
        return TOTAL_TASKS, 1
    except Exception as e:
        log(f"Resume point error: {e}")
        return 1, 1


def is_pipeline_done() -> bool:
    """Pipeline finished only when REPORT_FILE exists -- it's written at the very end of
    deep_research.run() after assemble() + make_report(). Relying on a hardcoded passes-count
    threshold would fire prematurely when the planner generates more sections than the
    runner knew about at import time."""
    return REPORT_FILE.exists()


def get_last_log() -> str:
    try:
        if PIPELINE_STDOUT.exists():
            with open(PIPELINE_STDOUT) as f:
                lines = f.readlines()
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("/"):
                    return stripped[-120:]
    except:
        pass
    return ""


# === RENDER PDF ===
def _ensure_native_libs():
    """On macOS arm64, weasyprint needs brew's pango/gobject libs on the dyld path."""
    if sys.platform != "darwin":
        return
    from pathlib import Path as _P
    for brew_lib in ("/opt/homebrew/lib", "/usr/local/lib"):
        if _P(brew_lib).exists():
            cur = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if brew_lib not in cur.split(":"):
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                    f"{brew_lib}:{cur}" if cur else brew_lib
                )


def render_pdf():
    """Delegate to deep_research.render_pdf so we only maintain one renderer.

    Critical: if DEEP_RESEARCH_OUT_NAME is set, we MUST call _rebind_output_paths
    before invoking the renderer -- otherwise deep_research's module-level
    FINAL_MD/FINAL_PDF stay at the default 'book.md'/'book.pdf' and the render
    silently no-ops because the input file doesn't exist under that name.
    """
    log("Rendering PDF (delegating to deep_research.render_pdf)...")
    try:
        sys.path.insert(0, str(HERE))
        import deep_research
        if _OUT_NAME:
            deep_research._rebind_output_paths(_OUT_NAME)
            log(f"  rebound output paths to prefix={_OUT_NAME!r}: {deep_research.FINAL_PDF.name}")
        return deep_research.render_pdf()
    except Exception as e:
        log(f"[PDF FAIL] {e}")
        return False


def _render_pdf_legacy_unused():
    """Old inline weasyprint renderer -- preserved here for reference / rollback only.

    No longer called; render_pdf() above delegates to deep_research.
    """
    try:
        _ensure_native_libs()
        import weasyprint, warnings
        warnings.filterwarnings("ignore")
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
            elif line.strip() == "---" and not in_code:
                fixed.append("* * *")
            else:
                fixed.append(line)
        content = "\n".join(fixed)
        clean_md = CLEAN_MD
        with open(clean_md, "w") as f:
            f.write(content)
        subprocess.run(
            ["pandoc", str(clean_md), "-o", str(FINAL_HTML),
             "--standalone", "--toc", "--toc-depth=3",
             "--mathml",
             "--metadata", "title=Large Language Models Handbook"],
            capture_output=True,
        )
        weasyprint.HTML(filename=str(FINAL_HTML)).write_pdf(str(FINAL_PDF))
        sz = os.path.getsize(FINAL_PDF)
        log(f"[PDF OK] {FINAL_PDF} ({sz/1024:.0f} KB)")
        return True
    except Exception as e:
        log(f"[PDF FAIL] {e}")
        return False


# === MAIN ===
def main():
    # Clear log on fresh start
    if not STATE_FILE.exists():
        open(LOG_FILE, "w").close()

    log("=" * 62)
    log("AUTONOMOUS RUNNER -- Deep Research Pipeline")
    topic_env = os.environ.get("DEEP_RESEARCH_TOPIC", "Large Language Models (hardcoded)")
    n_ch = os.environ.get("DEEP_RESEARCH_N_CHAPTERS", "12")
    n_pp = os.environ.get("DEEP_RESEARCH_N_PASSES", "8 (default)")
    log(f"Topic:      {topic_env}")
    log(f"Outline:    {n_ch} chapters x {n_pp} passes (planner if --topic, else hardcoded)")
    log(f"Max runtime: {MAX_HOURS}h")
    log(f"Done check: REPORT_FILE existence ({REPORT_FILE.name})")
    log("=" * 62)

    start_time = time.time()

    # Initial Ollama check
    if not is_ollama_healthy():
        log("Ollama not healthy -- restarting...")
        restart_ollama()
    else:
        log("Ollama is healthy")

    # Determine resume point
    if STATE_FILE.exists():
        resume_ch, resume_pp = get_resume_point()
        if resume_ch < TOTAL_TASKS:
            log(f"Resuming from Ch{resume_ch}, Pass {resume_pp}")
        else:
            log("All tasks already complete!")
    else:
        resume_ch, resume_pp = 1, 1
        log("Starting fresh")

    # Start pipeline
    start_pipeline(start_ch=resume_ch, start_pp=resume_pp)
    time.sleep(5)

    last_progress = 0
    last_log_ts = 0
    ollama_check_counter = 0
    pipeline_restarts = 0
    last_log_content = ""
    last_progress_time = time.time()
    last_progress_check = 0
    consecutive_stalls = 0

    while True:
        elapsed = time.time() - start_time

        # === TIMEOUT ===
        if elapsed > MAX_HOURS * 3600:
            log(f"TIMEOUT after {MAX_HOURS}h")
            notify("Pipeline Timeout", f"Max {MAX_HOURS}h reached")
            break

        # === DONE ===
        if is_pipeline_done():
            log("Pipeline COMPLETED!")
            notify("Deep Research Done", "Book generation finished")
            render_pdf()
            break

        # === PIPELINE DIED ===
        if not is_pipeline_running():
            log("Pipeline died -- restarting...")
            pipeline_restarts += 1
            if pipeline_restarts > 10:
                log("TOO MANY RESTARTS -- giving up")
                notify("Pipeline Error", "Too many restarts, check manually")
                break
            restart_ollama()
            time.sleep(RESTART_DELAY)
            # W3: pgrep can briefly miss a process that is exiting; re-confirm and
            # hard-kill any survivor before respawn so we never run two writers.
            if not kill_pipeline():
                log("Skipping respawn this cycle -- prior PID still alive")
                time.sleep(POLL_HEALTH)
                continue
            resume_ch, resume_pp = get_resume_point()
            if not start_pipeline(start_ch=resume_ch, start_pp=resume_pp):
                time.sleep(POLL_HEALTH)
                continue
            time.sleep(10)

        # === OLLAMA HEALTH CHECK ===
        ollama_check_counter += 1
        if ollama_check_counter >= 3:
            ollama_check_counter = 0
            if not is_ollama_healthy():
                log("Ollama unhealthy -- restarting...")
                restart_ollama()
                consecutive_stalls = 0

        # === PROGRESS CHECK ===
        p = get_progress()
        passes = p.get("passes", 0)
        words = p.get("words", 0)
        tokens = p.get("tokens", 0)
        pages = words // 400

        # === STALL RECOVERY ===
        if passes > 0:
            if passes != last_progress_check:
                last_progress_check = passes
                last_progress_time = time.time()
                consecutive_stalls = 0
            else:
                stall_elapsed = time.time() - last_progress_time
                if stall_elapsed >= STALL_MAX:
                    log(f"STALLED (no progress for {stall_elapsed/60:.0f}min) -- restarting...")
                    consecutive_stalls += 1
                    last_progress_time = time.time()
                    # W3: must confirm kill before respawn -- otherwise two writers
                    # race on state.json + Ollama and we corrupt the resume point.
                    if not kill_pipeline():
                        log("Stall recovery aborted -- old PID survived SIGKILL; will retry next cycle")
                        time.sleep(POLL_HEALTH)
                        continue
                    time.sleep(RESTART_DELAY)
                    resume_ch, resume_pp = get_resume_point()
                    log(f"Resuming from Ch{resume_ch}, Pass {resume_pp}")
                    if not start_pipeline(start_ch=resume_ch, start_pp=resume_pp):
                        time.sleep(POLL_HEALTH)
                        continue
                    time.sleep(10)

        # === PERIODIC PROGRESS LOG ===
        now = time.time()
        if passes > 0 and (passes != last_progress or now - last_log_ts > POLL_LOG):
            last_log_ts = now
            est_rem = (TOTAL_TASKS - passes) * (elapsed / passes) if passes > 0 else 0
            log(f"Progress: {passes}/{TOTAL_TASKS} ({100*passes/TOTAL_TASKS:.0f}%) | "
                f"Words: {words:,} (~{pages}p) | Tokens: {tokens:,} | "
                f"Elapsed: {elapsed/60:.1f}min | ETA: {est_rem/60:.1f}min | "
                f"Stalls: {consecutive_stalls}")


        # === LOG PIPELINE OUTPUT ===
        log_line = get_last_log()
        if log_line and log_line != last_log_content and passes > 0:
            last_log_content = log_line
            if "OK:" in log_line or "CHECKPOINT" in log_line or "PASS" in log_line or "ERROR" in log_line:
                log(f"  >> {log_line[:100]}")

        time.sleep(POLL_HEALTH)

    # === FINAL ===
    p = get_progress()
    log(f"\n{'='*62}")
    log(f"AUTONOMOUS RUNNER FINISHED")
    log(f"  Passes: {p.get('passes', 0)}/{TOTAL_TASKS}")
    log(f"  Words:  {p.get('words', 0):,}")
    log(f"  Pages:  ~{p.get('words', 0)//400}")
    log(f"  Tokens: {p.get('tokens', 0):,}")
    log(f"  Runtime: {(time.time()-start_time)/60:.1f}min")
    log(f"  Output: {FINAL_PDF}")
    log(f"{'='*62}")
    print("\nAll done.")

    # Final PDF render if not done
    if not is_pipeline_done():
        render_pdf()


if __name__ == "__main__":
    main()
