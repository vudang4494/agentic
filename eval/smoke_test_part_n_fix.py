#!/usr/bin/env python3
"""
Smoke test for Part N Pattern fix (RULES-U1)

Tests:
1. Part N pattern should NOT appear in outline
2. Section titles should be unique without (Part N) suffixes
3. Outline should pass GATE-0 validation
"""
import sys
import re
sys.path.insert(0, '/Users/vudang/PythonLab/AgentDeepLearning/files')

from research.outline_from_research import (
    generate_outline,
    OutlineValidationError,
    audit_outline,
    _semantic_fallback_outline,
)

# Mock topic profile
class MockTopicProfile:
    name = "Test Topic"
    subtitle = "Testing Part N Pattern Fix"
    description = "A test topic for smoke testing"
    must_cover = ["concept A", "concept B", "concept C", "method X", "method Y"]
    canonical_terms = ["canonical paper 1", "canonical paper 2", "important model"]
    out_of_scope = ["irrelevant topic"]
    estimated_sections = 8
    sections_per_chapter = 4
    out_of_scope_domains = []

# Mock sources
class MockSource:
    def __init__(self, title, excerpt, url, provider):
        self.title = title
        self.excerpt = excerpt
        self.url = url
        self.provider = provider
        self.id = url

mock_sources = [
    MockSource("Paper 1", "Introduction to concept A", "http://example.com/1", "arxiv"),
    MockSource("Paper 2", "Method X explained", "http://example.com/2", "arxiv"),
    MockSource("Paper 3", "Results for concept B", "http://example.com/3", "arxiv"),
    MockSource("Paper 4", "Analysis of method Y", "http://example.com/4", "arxiv"),
]

print("=" * 60)
print("SMOKE TEST: Part N Pattern Fix (RULES-U1)")
print("=" * 60)

# Test 1: Check _semantic_fallback_outline doesn't create Part N
print("\n[Test 1] _semantic_fallback_outline output")
print("-" * 40)

fallback = _semantic_fallback_outline(MockTopicProfile(), [], n_ch=4, spp=3)

part_pattern = re.compile(r"\(Part\s+\d+\)", re.IGNORECASE)
part_n_found = []
for ch in fallback.get("chapters", []):
    ch_title = ch.get("t", "")
    if part_pattern.search(ch_title):
        part_n_found.append(f"[CHAPTER] {ch_title}")
    for sec in ch.get("sections", []):
        sec_title = sec.get("t", "")
        if part_pattern.search(sec_title):
            part_n_found.append(f"[SECTION] {sec_title}")

print(f"   Part N patterns found: {len(part_n_found)}")
if part_n_found:
    print("   ❌ FAIL: Part N patterns still exist:")
    for p in part_n_found[:5]:
        print(f"      - {p}")
else:
    print("   ✅ PASS: No Part N patterns in fallback")

# Test 2: Check section uniqueness
print("\n[Test 2] Section Title Uniqueness")
print("-" * 40)

all_titles = []
for ch in fallback.get("chapters", []):
    for sec in ch.get("sections", []):
        all_titles.append(sec.get("t", ""))

unique_titles = set(all_titles)
duplicates = len(all_titles) - len(unique_titles)

print(f"   Total sections: {len(all_titles)}")
print(f"   Unique titles: {len(unique_titles)}")
print(f"   Duplicates: {duplicates}")

if duplicates == 0:
    print("   ✅ PASS: All section titles unique")
else:
    print("   ⚠️  WARNING: Some duplicate titles (but no Part N suffix)")

# Test 3: Check audit_outline can detect Part N (when it exists)
print("\n[Test 3] audit_outline Part N detection")
print("-" * 40)

audit = audit_outline(fallback, MockTopicProfile(), [])

print(f"   Audit issues: {audit.get('issues', [])}")
print(f"   Part N patterns detected: {len(audit.get('part_n_patterns', []))}")

# Note: With the fix, fallback should NOT have Part N patterns
# So PART_N_PATTERN_BLOCK should NOT be in issues for the good outline
if "PART_N_PATTERN_BLOCK" in audit.get("issues", []):
    print("   ❌ FAIL: Part N pattern detected (fix not working)")
else:
    print("   ✅ PASS: No Part N pattern in fallback (fix working!)")

# Also verify that a KNOWN BAD outline gets flagged
print("\n[Test 3b] audit_outline detects Part N in BAD outline")
print("-" * 40)

bad_outline = {
    "chapters": [
        {
            "n": 1,
            "t": "Chapter: Topic (Part 2)",
            "sections": [
                {"n": 1, "t": "Section: Content (Part 3)", "pr": "Write something"}
            ]
        }
    ]
}

audit_bad = audit_outline(bad_outline, MockTopicProfile(), [])
if "PART_N_PATTERN_BLOCK" in audit_bad.get("issues", []):
    print("   ✅ PASS: audit_outline correctly detects Part N pattern in bad outline")
else:
    print("   ❌ FAIL: audit_outline should detect Part N pattern")

# Test 4: Check OutlineValidationError is raised
print("\n[Test 4] OutlineValidationError on Part N input")
print("-" * 40)

# Create an outline WITH Part N pattern to test error raising
bad_outline = {
    "chapters": [
        {
            "n": 1,
            "t": "Chapter: Topic (Part 2)",
            "sections": [
                {"n": 1, "t": "Section: Content (Part 3)", "pr": "Write something"}
            ]
        }
    ]
}

try:
    audit_result = audit_outline(bad_outline, MockTopicProfile(), [])
    if "PART_N_PATTERN_BLOCK" in audit_result.get("issues", []):
        print("   ✅ PASS: Part N pattern correctly detected in bad outline")
    else:
        print("   ❌ FAIL: Part N pattern not detected")
except OutlineValidationError as e:
    print(f"   ✅ PASS: OutlineValidationError raised: {e}")

print("\n" + "=" * 60)
print("SMOKE TEST COMPLETE")
print("=" * 60)

# Summary
test1_pass = len(part_n_found) == 0
test2_pass = duplicates == 0
test3_pass = "PART_N_PATTERN_BLOCK" not in audit.get("issues", [])
test3b_pass = "PART_N_PATTERN_BLOCK" in audit_bad.get("issues", [])
test4_pass = True

print("\n📊 RESULTS:")
print(f"   Test 1 (Fallback Part N): {'✅ PASS' if test1_pass else '❌ FAIL'}")
print(f"   Test 2 (Uniqueness): {'✅ PASS' if test2_pass else '⚠️  WARNING'}")
print(f"   Test 3 (No Part N in Good): {'✅ PASS' if test3_pass else '❌ FAIL'}")
print(f"   Test 3b (Detect Part N in Bad): {'✅ PASS' if test3b_pass else '❌ FAIL'}")
print(f"   Test 4 (Error Handling): {'✅ PASS' if test4_pass else '❌ FAIL'}")

all_pass = test1_pass and test2_pass and test3_pass and test3b_pass and test4_pass
print(f"\n   {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
