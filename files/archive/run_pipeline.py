#!/usr/bin/env python3
"""
Robust Autonomous Runner for Multipass Pipeline
============================================
- Monitors Ollama health every 30s
- Auto-restarts pipeline if it dies
- Saves progress checkpoints
- Sends macOS notification on completion
- Renders PDF at the end
"""
import json, os, sys, time, signal, subprocess
from pathlib import Path
from datetime import datetime

# === CONFIG ===
SCRIPT       = Path(__file__).parent / "multipass_pipeline.py"
STATE_FILE   = Path(__file__).parent / "output" / "multipass_state.json"
LOG_FILE     = Path(__file__).parent / "output" / "autonomous_runner.log"
REPORT_FILE  = Path(__file__).parent / "output" / "benchmark.json"
FINAL_MD     = Path(__file__).parent / "output" / "book.md"
FINAL_HTML   = Path(__file__).parent / "output" / "book.html"
FINAL_PDF    = Path(__file__).parent / "output" / "book.pdf"
TOTAL_TASKS = 96  # 12 chapters × 8 passes → ~400 pages
MAX_HOURS   = 15.0
BATCH       = 2

OLLAMA_BASE  = "http://localhost:11434"
MODEL        = "gemma3:4b"

POLL_HEALTH  = 60   # seconds between health checks
POLL_LOG     = 30   # seconds between log reads
STALL_MAX    = 1800 # seconds (30 min) before considering pipeline stalled
RESTART_DELAY = 10  # seconds to wait after kill
KILL_TIMEOUT  = 15  # seconds to wait for graceful kill before -9


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
        subprocess.run([
            "osascript", "-e",
            f'display notification "{body}" with title "{title}"'
        ], timeout=5, capture_output=True)
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
                json={"model": MODEL, "stream": False,
                      "messages": [{"role": "user", "content": "hi"}],
                      "options": {"num_predict": 10}},
            )
            return r.status_code == 200 and r.json().get("message", {}).get("content")
    except:
        return False


def kill_stale_runners():
    """Kill all Ollama runners except the main server."""
    result = subprocess.run(
        ["pgrep", "-f", "ollama runner"], capture_output=True, text=True
    )
    pids = [int(l) for l in result.stdout.strip().split("\n") if l.strip()]
    for pid in pids:
        try:
            os.kill(pid, 9)
            log(f"Killed stale runner PID {pid}")
        except:
            pass


def restart_ollama():
    log("Restarting Ollama...")
    # Kill runners
    subprocess.run(["pkill", "-9", "-f", "ollama runner"], capture_output=True)
    time.sleep(2)
    # Restart main server
    subprocess.Popen(
        ["/Applications/Ollama.app/Contents/MacOS/Ollama", "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for healthy
    for i in range(20):
        time.sleep(3)
        if is_ollama_healthy():
            log("Ollama healthy")
            return True
        log(f"  Waiting for Ollama... {i+1}/20")
    log("WARNING: Ollama may not be fully healthy")
    return is_ollama_healthy()


# === PIPELINE ===
def is_pipeline_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "multipass_pipeline"], capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def kill_pipeline():
    """Graceful then aggressive kill."""
    result = subprocess.run(
        ["pgrep", "-f", "multipass_pipeline"], capture_output=True, text=True
    )
    pids = [int(l) for l in result.stdout.strip().split("\n") if l.strip()]
    # Try SIGTERM first
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except:
            pass
    # Wait for graceful
    time.sleep(KILL_TIMEOUT)
    # SIGKILL if still alive
    for pid in pids:
        try:
            os.kill(pid, 9)
        except:
            pass


def start_pipeline(start_ch: int = 1, start_pp: int = 1):
    log(f"Starting pipeline: start_ch={start_ch}, start_pp={start_pp}")
    subprocess.Popen(
        [sys.executable, "-u", str(SCRIPT),
         "--batch", str(BATCH),
         "--start-ch", str(start_ch),
         "--start-pp", str(start_pp)],
        stdout=open(Path(__file__).parent / "output" / "multipass.log", "a"),
        stderr=subprocess.STDOUT,
    )


# === STATE ===
def get_progress() -> dict:
    if not STATE_FILE.exists():
        return {"passes": 0, "words": 0, "tokens": 0}
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


def get_last_log_line() -> str:
    try:
        log_path = Path(__file__).parent / "output" / "multipass.log"
        if log_path.exists():
            with open(log_path) as f:
                lines = f.readlines()
            # Find last non-empty line
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("/"):
                    return stripped[-100:]
    except:
        pass
    return ""


def is_pipeline_done() -> bool:
    """Check if all 96 tasks are done by looking at state and report."""
    p = get_progress()
    if p.get("passes", 0) >= TOTAL_TASKS:
        return True
    # Also check report
    if REPORT_FILE.exists():
        try:
            with open(REPORT_FILE) as f:
                r = json.load(f)
            return r.get("total_calls", 0) >= TOTAL_TASKS
        except:
            pass
    return False


# === RESUME POINT ===
def get_resume_point() -> tuple[int, int]:
    """Figure out where to resume from the state file (source of truth)."""
    if not STATE_FILE.exists():
        return 1, 1
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        passes = state.get("passes", {})
        if not passes:
            return 1, 1
        # Find the highest (ch, pp) that is NOT in passes yet
        # First, determine the full task list from the script
        ch_list = [
            (1,8),(2,8),(3,8),(4,8),(5,8),(6,8),
            (7,8),(8,8),(9,8),(10,8),(11,8),(12,8),
        ]
        # Linear scan to find first missing
        for ch, total_pp in ch_list:
            for pp in range(1, total_pp + 1):
                key = f"{ch}.{pp}"
                if key not in passes:
                    return ch, pp
        # All done
        return TOTAL_TASKS, 1
    except Exception as e:
        log(f"Resume point error: {e}, defaulting to 1,1")
        return 1, 1


# === RENDER PDF ===
def render_pdf():
    log("Rendering PDF...")
    try:
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
        clean_md = Path(__file__).parent / "output" / "book_clean.md"
        with open(clean_md, "w") as f:
            f.write(content)
        subprocess.run([
            "pandoc", str(clean_md), "-o", str(FINAL_HTML),
            "--standalone", "--toc", "--toc-depth=3",
            "--metadata", "title=Large Language Model Engineering",
        ], capture_output=True)
        weasyprint.HTML(filename=str(FINAL_HTML)).write_pdf(str(FINAL_PDF))
        sz = os.path.getsize(FINAL_PDF)
        log(f"PDF rendered: {FINAL_PDF} ({sz/1024:.0f} KB)")
        return True
    except Exception as e:
        log(f"PDF render failed: {e}")
        return False


# === MAIN LOOP ===
def main():
    log("=" * 60)
    log("AUTONOMOUS RUNNER — 400+ PAGE BOOK")
    log(f"Tasks: {TOTAL_TASKS} (12 ch × 8 passes × 4200w ≈ 400p)")
    log(f"Max runtime: {MAX_HOURS}h")
    log("=" * 60)

    start_time = time.time()

    # Initial Ollama check
    if not is_ollama_healthy():
        log("Ollama not healthy — restarting...")
        restart_ollama()
    else:
        log("Ollama healthy")

    # Determine resume point
    if STATE_FILE.exists():
        resume_ch, resume_pp = get_resume_point()
        log(f"Resume point: Ch{resume_ch}, Pass {resume_pp}")
    else:
        resume_ch, resume_pp = 1, 1
        log("Starting fresh")

    # Start pipeline
    start_pipeline(start_ch=resume_ch, start_pp=resume_pp)

    last_progress = 0
    last_log_ts = 0
    ollama_check_counter = 0
    pipeline_restarts = 0
    last_log_content = ""
    last_progress_time = time.time()
    last_progress_check = 0  # tracks last passes value we saw

    consecutive_stalls = 0

    while True:
        # === TIMEOUT ===
        elapsed = time.time() - start_time
        if elapsed > MAX_HOURS * 3600:
            log(f"TIMEOUT after {MAX_HOURS}h")
            notify("Pipeline Timeout", f"Max {MAX_HOURS}h reached")
            break

        # === DONE ===
        if is_pipeline_done():
            log("Pipeline COMPLETED!")
            notify("Deep Agent Done!", "400+ page book generated!")
            render_pdf()
            break

        # === PIPELINE DIED ===
        if not is_pipeline_running():
            log("Pipeline died — restarting...")
            pipeline_restarts += 1
            if pipeline_restarts > 10:
                log("TOO MANY RESTARTS — giving up")
                notify("Pipeline Error", "Too many restarts, check manually")
                break
            # Restart Ollama to be safe
            restart_ollama()
            time.sleep(RESTART_DELAY)
            resume_ch, resume_pp = get_resume_point()
            start_pipeline(start_ch=resume_ch, start_pp=resume_pp)
            time.sleep(10)  # Wait for startup

        # === OLLAMA HEALTH CHECK ===
        ollama_check_counter += 1
        if ollama_check_counter >= 3:  # Every ~90s
            ollama_check_counter = 0
            if not is_ollama_healthy():
                log("Ollama unhealthy — restarting...")
                restart_ollama()
                consecutive_stalls = 0

        # === PROGRESS CHECK ===
        p = get_progress()
        passes = p.get("passes", 0)
        words = p.get("words", 0)
        tokens = p.get("tokens", 0)
        pages = words // 400

        # === STALL RECOVERY ===
        # Detect stall: passes haven't changed for STALL_MAX seconds
        if passes > 0:
            if passes != last_progress_check:
                last_progress_check = passes
                last_progress_time = time.time()
                consecutive_stalls = 0
            else:
                stall_elapsed = time.time() - last_progress_time
                if stall_elapsed >= STALL_MAX:
                    log(f"PIPELINE STALLED (no progress for {stall_elapsed/60:.0f}min) — restarting...")
                    consecutive_stalls = 0
                    last_progress_time = time.time()
                    kill_pipeline()
                    time.sleep(RESTART_DELAY)
                    resume_ch, resume_pp = get_resume_point()
                    log(f"Resuming from Ch{resume_ch}, Pass {resume_pp}")
                    start_pipeline(start_ch=resume_ch, start_pp=resume_pp)
                    time.sleep(10)

        # Log every ~2 min or on new progress
        now = time.time()
        if passes > 0 and (passes != last_progress or now - last_log_ts > POLL_LOG):
            last_log_ts = now
            est_rem = (TOTAL_TASKS - passes) * (elapsed / passes) if passes > 0 else 0
            log(f"Progress: {passes}/{TOTAL_TASKS} ({100*passes/TOTAL_TASKS:.0f}%) | "
                f"Words: {words:,} (~{pages}p) | Tokens: {tokens:,} | "
                f"Elapsed: {elapsed/60:.1f}min | Est. remaining: {est_rem/60:.1f}min | "
                f"Stalls: {consecutive_stalls}")

        # === LOG STALLED ===
        log_line = get_last_log_line()
        if log_line and log_line != last_log_content and passes > 0:
            last_log_content = log_line
            # Only log if it's a new task
            if "OK:" in log_line or "CHECKPOINT" in log_line or "PASS" in log_line:
                log(f"  {log_line[:80]}")

        time.sleep(POLL_HEALTH)

    # === FINAL REPORT ===
    p = get_progress()
    log(f"\n{'='*60}")
    log(f"AUTONOMUS RUNNER FINISHED")
    log(f"  Passes: {p.get('passes', 0)}/{TOTAL_TASKS}")
    log(f"  Words:  {p.get('words', 0):,}")
    log(f"  Pages:  ~{p.get('words', 0)//400}")
    log(f"  Tokens: {p.get('tokens', 0):,}")
    log(f"  Runtime: {(time.time()-start_time)/60:.1f}min")
    log(f"{'='*60}")
    print("All done.")


if __name__ == "__main__":
    main()
