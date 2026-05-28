# Outline + Content Audit -- `files/output/bookv4.state.json`

- Topic: `Large Language Models`
- Sections scored: 96

## Topic drift (titles < 0.45 cosine to topic)

| Section | Title | Cosine |
|---|---|---|
| `6.6` | DPO Variants and On-Policy Issues | 0.428 |

## Title near-duplicates (cosine >= 0.85)

None.
## Content duplicates (cosine >= 0.80 on first 1500 chars)

| A | B | Cosine |
|---|---|---|
| `10.2` | `10.8` | 0.889 |
| `6.2` | `6.6` | 0.885 |
| `2.8` | `4.4` | 0.86 |
| `7.4` | `11.2` | 0.854 |
| `7.4` | `7.7` | 0.85 |
| `10.4` | `12.6` | 0.841 |
| `1.4` | `10.4` | 0.84 |
| `4.3` | `4.8` | 0.835 |
| `1.4` | `12.6` | 0.82 |
| `9.1` | `9.6` | 0.819 |
| `3.4` | `10.3` | 0.812 |
| `3.4` | `3.7` | 0.811 |
| `5.1` | `5.8` | 0.811 |
| `5.1` | `5.7` | 0.808 |
| `7.3` | `12.4` | 0.806 |
| `7.7` | `11.2` | 0.805 |
| `10.4` | `12.1` | 0.802 |

## Verdict

- Topic drift sections: **1**
- Title clusters: **0**
- Content duplicate pairs: **17**

**NEEDS ATTENTION** -- outline or content has duplication / drift issues.