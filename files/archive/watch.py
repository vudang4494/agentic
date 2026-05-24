#!/usr/bin/env python3
"""Lightweight pipeline progress monitor. Run: python3 watch.py"""
import json, time, sys

STATE = "/Users/vudang/PythonLab/AgentDeepLearning/files/output/deep_agent_state.json"
REPORT = "/Users/vudang/PythonLab/AgentDeepLearning/files/output/benchmark_report.json"
CHECKPOINT_AGE_MAX = 300  # 5 min

def get_progress():
    import os
    if not os.path.exists(STATE):
        return None
    with open(STATE) as f:
        d = json.load(f)
    written = len(d.get("written", {}))
    total = len(d.get("subsections", {}))
    stage = d.get("stage", "?")
    tokens = d.get("total_tokens", 0)
    words = sum(r.get("word_count", 0) for r in d.get("written", {}).values())
    age = time.time() - os.path.getmtime(STATE)
    return {"written": written, "total": total, "stage": stage,
            "tokens": tokens, "words": words, "age": age}

def is_done():
    import os
    return os.path.exists(REPORT)

print("Watching pipeline... Ctrl+C to stop")
last = None
while True:
    p = get_progress()
    if p:
        pct = f"{100*p['written']/max(p['total'],1):.0f}%"
        line = (f"[{time.strftime('%H:%M:%S')}] "
                f"{p['written']}/{p['total']} ({pct}) | "
                f"Stage: {p['stage']} | "
                f"Tokens: {p['tokens']:,} | "
                f"Words: {p['words']:,} (~{p['words']//400}p) | "
                f"Age: {p['age']:.0f}s")
        if line != last:
            print(line)
            last = line
    else:
        print(f"[{time.strftime('%H:%M:%S')}] No state file yet...")
    if is_done():
        print("DONE! Check report.")
        sys.exit(0)
    time.sleep(30)
