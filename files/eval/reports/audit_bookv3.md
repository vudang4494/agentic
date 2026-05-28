# Outline + Content Audit -- `files/output/bookv3.state.json`

- Topic: `Large Language Models`
- Sections scored: 70

## Topic drift (titles < 0.45 cosine to topic)

| Section | Title | Cosine |
|---|---|---|
| `3.10` | Emerging Architectures in 2024-2025 | 0.37 |
| `7.2` | Ethics and Societal Impact | 0.424 |
| `7.10` | References and Further Reading | 0.437 |
| `7.9` | Conclusion and Summary | 0.438 |

## Title near-duplicates (cosine >= 0.85)

None.
## Content duplicates (cosine >= 0.80 on first 1500 chars)

| A | B | Cosine |
|---|---|---|
| `7.1` | `7.9` | 0.861 |
| `3.10` | `7.5` | 0.853 |
| `6.1` | `6.2` | 0.851 |
| `7.9` | `7.10` | 0.851 |
| `6.2` | `6.3` | 0.849 |
| `5.3` | `5.7` | 0.848 |
| `1.10` | `3.1` | 0.843 |
| `5.1` | `5.3` | 0.832 |
| `7.1` | `7.6` | 0.828 |
| `6.1` | `6.4` | 0.825 |
| `7.8` | `7.10` | 0.825 |
| `4.1` | `4.2` | 0.824 |
| `5.1` | `5.6` | 0.823 |
| `7.6` | `7.9` | 0.822 |
| `1.10` | `3.10` | 0.82 |
| `3.1` | `3.10` | 0.819 |
| `5.2` | `5.3` | 0.816 |
| `5.3` | `5.8` | 0.816 |
| `3.8` | `4.1` | 0.814 |
| `1.10` | `7.5` | 0.812 |
| `6.6` | `6.9` | 0.812 |
| `5.1` | `5.8` | 0.811 |
| `6.2` | `6.9` | 0.81 |
| `5.3` | `7.6` | 0.806 |
| `7.1` | `7.2` | 0.803 |
| `7.2` | `7.9` | 0.803 |
| `2.5` | `4.3` | 0.801 |
| `5.7` | `5.8` | 0.801 |

## Verdict

- Topic drift sections: **4**
- Title clusters: **0**
- Content duplicate pairs: **28**

**NEEDS ATTENTION** -- outline or content has duplication / drift issues.