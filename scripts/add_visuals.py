#!/usr/bin/env python3
"""
Inject structured visuals (tables, code blocks, comparison lists) into the book.

Each visual is injected after the section's ## heading, before the first ### subheading.
Matching is done via state.json section titles.

Usage:
  python3 scripts/add_visuals.py --run llm_trends_2026_2027 [--dry-run]
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_book_index(book_md: str, state: dict) -> dict:
    """Build a mapping: section_key -> (line_num, ch_n) for ## headings in book."""
    lines = book_md.splitlines(keepends=True)

    # ## heading pattern (section-level headings in the assembled book)
    section_re = re.compile(r"^##\s+(.+?)\s*$")
    chapter_re = re.compile(r"^#\s+([IVX\d]+)\.\s+(.+?)\s*$")

    # Build list of (line_idx, heading_text, chapter_n)
    section_map = []  # [(line_num, title, ch_n)]
    current_ch = 0
    for i, line in enumerate(lines):
        cm = chapter_re.match(line)
        if cm:
            # This is a chapter heading like "# 1. Introduction..."
            # Extract chapter number from first line
            pass
        sm = section_re.match(line)
        if sm:
            title = sm.group(1).strip()
            section_map.append((i, title, current_ch))

    # Also track current chapter from ## headings (since ## is the section level)
    # The chapter is determined by the preceding # heading
    chapters_by_line = {}
    current_ch_n = 0
    for i, line in enumerate(lines):
        if line.startswith("# ") and not line.startswith("##"):
            # This is a chapter heading like "# Chapter N: Title"
            m = re.match(r"^#\s+(\d+)\.", line)
            if m:
                current_ch_n = int(m.group(1))

    # Now map section titles to state keys
    # State uses key like "2.1" and title "Self-Attention Mechanism"
    title_to_key = {}
    ch_to_keys = {}
    for key, sec in state["passes"].items():
        title = sec.get("title", "")
        title_to_key[title.lower()] = key
        ch_n = int(key.split(".")[0])
        ch_to_keys.setdefault(ch_n, []).append((key, title))

    # For each section in book, find its state key
    section_info = {}  # key -> line_num
    for line_num, title, _ in section_map:
        # Try exact match
        key = title_to_key.get(title.lower())
        if key:
            section_info[key] = (line_num, title)
        else:
            # Try fuzzy match
            for t, k in title_to_key.items():
                if title.lower() in t or t in title.lower():
                    section_info[k] = (line_num, title)
                    break

    return {
        "lines": lines,
        "section_info": section_info,  # state_key -> (line_num, book_title)
        "title_to_key": title_to_key,
    }


def find_insert_line(lines: list, section_start: int) -> int:
    """Find the line after which to insert a visual.

    Insert BEFORE the first ### subheading within the section,
    or at the first blank line after section start if no subheading.
    """
    section_end_limit = section_start + 200  # Sanity limit

    for i in range(section_start + 1, min(len(lines), section_end_limit + 1)):
        line = lines[i]

        # Hit next ## section (another section at same level)
        if re.match(r"^##\s", line):
            return i

        # Hit next # chapter heading
        if re.match(r"^#\s", line) and not re.match(r"^##\s", line):
            return i

        # Found a subheading
        if re.match(r"^###\s", line):
            return i

        # Found a code block marker (start of content)
        if re.match(r"^```", line):
            return i

    return section_start + 2


def build_visual_text(suggestion: dict) -> str:
    """Build the visual block as markdown."""
    vtype = suggestion["visual_type"]
    content = suggestion.get("visual_content", "")

    if vtype == "table":
        # content IS the table markdown
        return content

    elif vtype == "code":
        # content IS the code block
        return content

    elif vtype == "list":
        # content IS the structured list
        return content

    return content


def main():
    p = argparse.ArgumentParser(description="Inject visuals into book")
    p.add_argument("--run", required=True, help="Run name")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = p.parse_args()

    run_dir = ROOT / "output/runs" / args.run
    book_path = run_dir / "book.md"
    state_path = run_dir / "state.json"
    suggestions_path = run_dir / "visual_suggestions.json"

    with open(state_path) as f:
        state = json.load(f)
    with open(book_path, encoding="utf-8") as f:
        book_md = f.read()
    with open(suggestions_path) as f:
        suggestions = json.load(f)

    print(f"Suggestions: {len(suggestions)} | Book: {len(book_md.split())}w")

    # Build book index
    index = build_book_index(book_md, state)
    lines = index["lines"]

    # Collect all insertions
    insertions = []  # [(line_num, visual_md, section_key)]

    for suggestion in suggestions:
        key = suggestion["key"]
        section_title = suggestion["title"]
        visual_md = build_visual_text(suggestion)

        if key not in index["section_info"]:
            # Try to find by title
            matched = False
            for k, (line_num, book_title) in index["section_info"].items():
                if section_title.lower() in book_title.lower() or book_title.lower() in section_title.lower():
                    key = k
                    matched = True
                    break
            if not matched:
                print(f"  [SKIP] {key} '{section_title}': not found in book")
                continue

        line_num, book_title = index["section_info"][key]
        insert_line = find_insert_line(lines, line_num)

        insertions.append((insert_line, visual_md, key, section_title))
        print(f"  {key}: '{book_title}' → line {insert_line}")

    if not insertions:
        print("No sections matched!")
        return

    # Apply insertions (highest line first to preserve indices).
    # Secondary key: visual index so same-line inserts keep original order.
    # (Later in list = first to insert = appears FIRST in the book.)
    insertions_with_idx = [(insert_line, -i, visual_md, key, title)
                           for i, (insert_line, visual_md, key, title) in enumerate(insertions)]
    insertions_with_idx.sort(key=lambda x: (x[0], x[1]), reverse=True)

    injected_lines = 0
    for line_num, _, visual_md, key, title in insertions_with_idx:
        lines.insert(line_num, "\n" + visual_md + "\n")
        injected_lines += 1

    updated = "".join(lines)

    print(f"\nInjected: {injected_lines}/{len(suggestions)} visuals")
    print(f"Book: {len(updated.split())}w ({len(updated.split()) - len(book_md.split()):+d})")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        backup_path = book_path.with_suffix(".md.visual_bak")
        book_path.rename(backup_path)
        book_path.write_text(updated, encoding="utf-8")
        print(f"Backup: {backup_path}")
        print(f"Written: {book_path}")


if __name__ == "__main__":
    main()
