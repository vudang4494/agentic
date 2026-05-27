# HANDOFF -- agentic (last updated 2026-05-25 08:42)

## TL;DR -- standardization just landed

Project is now **Claude-native**. Cursor / Codex artifacts removed:
- Deleted: `.cursor/`, `.cursor-plugin/`, `.codex-plugin/`, `files/.cursor-plugin/`, `files2/.cursor-plugin/`
- The Cursor "qwen-book-pipeline" skill was for a DIFFERENT project (Qwen3+vLLM+STORM) -- not applicable here, deleted
- MCP config migrated to root `.mcp.json` (Claude convention); MCP server name renamed to `agentic-deep-research`
- `.gitignore` updated to keep those other-tool dirs out

The 4-pillar Claude pipeline structure (Plan / Memory / Skill / MCP):
| Pillar | Where |
|---|---|
| Plan   | `WORKPLAN.md` (root) |
| Memory | `~/.claude/projects/-Users-vudang-PythonLab-AgentDeepLearning/memory/` + `MEMORY.md` index |
| Skill / agent context | `CLAUDE.md` (root, auto-loaded by Claude Code) |
| MCP    | `.mcp.json` (root) -> `files/mcp_server.py` |



> Quick context-restoration document. Read this when picking up the project after a break.

---

## 1. Where things stand right now

| Aspect | State |
|---|---|
| Pipelines running | **None** -- all killed by user request 2026-05-25 |
| GitHub | https://github.com/vudang4494/agentic (public, MIT). Only Stage 3+ code (`files/`) pushed. **`files2/` (Stage 4) is local-only**, not yet committed. |
| Last finished full run | `book2.pdf` (Stage 2++) -- 282 pages, 96/96 sections, 384 min runtime. Finished 2026-05-24 04:29. |
| Stage 4 partial run | `book_v2` -- 9/96 sections, 10 KB facts. State + KB intact for resume. |
| Active web-search keys | Tavily quota exhausted (auto-disables on 432). Brave free tier returns 402 (needs CC on file) -- **dropped from PROVIDERS_DEFAULT**. Pipeline now runs arxiv + wiki + ddg only. |

---

## 2. Two parallel pipelines on disk

### `files/` -- Stage 3+ production pipeline (committed to GitHub)

The version that produced `book.pdf`, `book1.pdf`, `book2.pdf`. Stable.
- `deep_research.py` -- writer + assemble + render
- `runner.py` -- autonomous watchdog
- `research/{search,query_gen,notes,fetch,embeddings,verify,planner,types}.py`
- Outputs in `files/output/`

### `files2/` -- Stage 4 Understanding-Layer fork (local-only, experimental)

Fork of files/ + 4 new modules that shift the pipeline from "prompting" to "understanding":
- `research/questions.py` -- section prompt → 5-8 must-answer questions JSON
- `research/facts.py` -- evidence block → structured atomic facts (claim/type/source/confidence)
- `research/kb.py` -- project Knowledge Base (JSON-persisted, bge-m3 cosine queryable)
- `research/comprehension.py` -- re-asks the section's questions of the written section, returns coverage %

Integrated into `files2/deep_research.py run()` loop with this flow per section:
```
questions = decompose(prompt)                          # NEW
kb_facts  = kb.query(prompt, exclude=this_section)     # NEW (cross-section memory)
sources   = search.gather() -> rank -> enrich
write     = gen(prompt, ctx, kb_block, evidence, qs_block)
clean_citations(content)
grounding = verify_section(content, sources)
comprehension = verify_comprehension(content, questions)  # NEW
if grounding < 0.55 OR coverage < 0.65: re-research with hints
extract facts -> kb.add_facts(section_key, facts) -> kb.save()  # NEW
```

Outputs go to `files2/output/book_v2.*` (separate prefix from `files/output/book{,1,2}.*`).

---

## 3. Run history & artifacts

| File | Pipeline | Sections | Pages | Notes |
|---|---|---|---|---|
| `files/output/baseline_stage0/book.pdf` | Stage 0/1 (hardcoded prompts, no research) | 96 | ~310 | Original baseline, no citations |
| `files/output/book1.pdf` | Stage 2+ (Tavily on first half) | 96 | 303 | Mean grounding 0.79 BUT 49/96 zero-cite (gaming inflated) |
| `files/output/book2.pdf` | Stage 2++ (zero-cite penalty, no Tavily after exhaustion) | 96 | 282 | Mean grounding 0.29 (honest), 88% sections went round-2 |
| `files2/output/book_v2.*` | **Stage 4 (partial 9/96)** | 9 (gaps at 1.1, 1.3, 2.1) | n/a | KB has 10 facts. Resume any time. |

Detailed 3-way audit is captured in chat history (search "3-way comparison" or "audit"). Key insight: book1 vs book2 shows **transparency tradeoff** -- book1 looks better metrically (0.79) but is gaming; book2 is honest (0.29) and knows where it's weak.

---

## 4. Open issues / recent debugging

### Brave Search HTTP 402 (root cause: free tier requires CC)
- Key `BSA...` user supplied works briefly, then returns 402 Payment Required
- Patched both `files/research/search.py` and `files2/research/search.py` with `_BRAVE_DISABLED_THIS_SESSION` (auto-disable after 3 fails, mirroring Tavily 432 logic)
- Dropped `brave` from both `PROVIDERS_DEFAULT` lists -- pipeline now runs `("arxiv", "wikipedia", "ddg", "tavily")` by default
- Tavily key still in env; auto-disables on 432 since quota is exhausted

### Runner watchdog cascading zombies on Stage 4
- Stage 4 per-section time (~5-10 min with questions + comprehension + facts) often hits the old `STALL_MAX=1800s` runner threshold
- Runner kills + respawns pipeline → respawn doesn't kill old subprocess cleanly → cascading concurrent deep_research.py instances → all compete for Ollama → 500s → more respawns
- Patched `files2/runner.py STALL_MAX 1800 → 5400` (90 min)
- **For Stage 4 prefer direct launch** (no runner watchdog) -- see resume commands below

### Multiple processes writing same state.json
- When cascading happens, multiple writers corrupt state
- Mitigation: always `pkill -9 -f files2/deep_research.py && pkill -9 -f files2/runner.py` before relaunch

---

## 5. How to resume Stage 4 book_v2 run

```bash
cd /Users/vudang/PythonLab/AgentDeepLearning

# 1. Sanity -- nothing running
ps aux | grep -E "files2/(runner|deep_research)" | grep -v grep
# (must print nothing; otherwise pkill -9 first)

# 2. Warm Ollama models (prevents cold-start 500s on the first section)
python3 -c "
import httpx
with httpx.Client(timeout=120) as c:
    for m in ['gemma3:4b', 'qwen3.5:4b']:
        c.post('http://localhost:11434/api/chat',
            json={'model':m,'stream':False,'messages':[{'role':'user','content':'hi'}],'options':{'num_predict':2}})
    c.post('http://localhost:11434/api/embed', json={'model':'bge-m3:latest','input':['hi']})
print('Ollama warm')"

# 3. Launch DIRECT (no runner watchdog -- Stage 4 sections are too slow for it)
nohup python3 -u files2/deep_research.py --review --out-name book_v2 --no-render \
    > files2/output/book_v2.direct.stdout.log 2>&1 &
disown

# 4. Monitor (no polling, just snapshot)
python3 -c "
import json, os, time
d = json.load(open('files2/output/book_v2.state.json'))
print(f'{len(d[\"passes\"])}/96 sections, {d[\"total_words\"]:,} words')
print(f'KB facts: {json.load(open(\"files2/output/book_v2.kb.json\"))[\"n_facts\"]}')"
tail -30 files2/output/book_v2.direct.stdout.log
```

ETA from current 9/96 state with arxiv+wiki+ddg only: ~6h (87 sections × ~250s each, no Tavily/Brave overhead).

---

## 6. Compare runs after Stage 4 completes

When `book_v2.report.json` exists, run this audit (4-way: baseline / book1 / book2 / book_v2):

```bash
python3 -c "
import json, re, statistics, os
runs = {
    'baseline (Stage 0/1)': 'files/output/baseline_stage0/state.json',
    'book1 (Stage 2+)':     'files/output/book1.state.json',
    'book2 (Stage 2++)':    'files/output/book2.state.json',
    'book_v2 (Stage 4)':    'files2/output/book_v2.state.json',
}
for label, path in runs.items():
    if not os.path.exists(path): continue
    d = json.load(open(path))
    p = d['passes']
    cites = [len(re.findall(r'\[\d+\]', v['content'])) for v in p.values()]
    grounds = [(v.get('verify') or {}).get('grounding') for v in p.values() if (v.get('verify') or {}).get('grounding') is not None]
    covs = [(v.get('comprehension') or {}).get('coverage') for v in p.values() if (v.get('comprehension') or {}).get('coverage') is not None]
    mg = f'{statistics.mean(grounds):.2f}' if grounds else 'n/a'
    mc = f'{statistics.mean(covs):.2f}' if covs else 'n/a'
    print(f'{label:30s} {len(p):3d}/96 sec  cites~{statistics.mean(cites):.1f}  ground={mg}  coverage={mc}')"
```

Stage 4 introduces a NEW axis: **coverage** (= % questions the section actually answered). book/book1/book2 don't have this. book_v2 should be the first with measurable coverage.

---

## 7. Files & locations cheat-sheet

| Need | Path |
|---|---|
| Stage 3+ pipeline source | `files/deep_research.py`, `files/research/` |
| Stage 4 pipeline source | `files2/deep_research.py`, `files2/research/` |
| Production launchers | `run.sh`, `scripts/launch_book2.sh`, `scripts/launch_book_v2.sh` |
| Public README | `README.md` |
| Architecture / roadmap | `WORKPLAN.md`, `CLAUDE.md` |
| GitHub repo | https://github.com/vudang4494/agentic |
| Memory (Claude auto-loads) | `~/.claude/projects/-Users-vudang-PythonLab-AgentDeepLearning/memory/` |

---

## 8. Decisions taken (don't relitigate unless something changed)

- **Brave is dropped** -- user asked "vậy bỏ brave" after 402s. Removed from PROVIDERS_DEFAULT in both files/ and files2/. Don't add it back without a working CC-attached account.
- **Direct launch preferred over runner watchdog for Stage 4** -- watchdog's stall-respawn cascades into Ollama-overloading zombies. Use the runner ONLY for Stage 3+ (files/) pipelines.
- **gemma3:4b stays as writer** -- user has not pulled qwen2.5:7b. Recommended upgrade still in queue.
- **No --topic for book_v2** -- using hardcoded LLM outline for apples-to-apples comparison with book2. Planner can be invoked separately if user wants a different topic later.
- **Stage 4 files in files2/, not files/** -- user explicitly asked to fork so the experimental code doesn't disturb the production pipeline.

---

## 9. Open questions for the user next time

1. Resume book_v2 to completion (~6h)? Or abandon and move on?
2. Pull `qwen2.5:7b` for the writer? Mini test showed Stage 4 question-decomposition + comprehension works well, but the writer still drops citations under verify pressure. A 7B writer would likely lift cite count and grounding both.
3. Should `files2/` be committed to GitHub (as an experimental branch or in the main repo)? Right now it's local-only.
4. Should Brave be removed entirely from `files2/research/search.py` (delete the function) or just gated (current state: still defined, auto-disabled in default flow)?
