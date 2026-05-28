# Outline + Content Audit -- `files/output/bookv5.state.json`

- Topic: `Large Language Models`
- Sections scored: 150

## Topic drift (titles < 0.45 cosine to topic)

| Section | Title | Cosine |
|---|---|---|
| `15.7` | Energy Efficiency and Carbon Footprint of AGI | 0.336 |
| `5.10` | Practical Implementation: Frameworks and Deployment Considerations | 0.409 |
| `3.1` | Meta's Release Strategy and Model Lineage | 0.425 |
| `14.6` | Edge Deployment Constraints: Memory Footprint and Compute Limits | 0.427 |
| `9.10` | Safety and Alignment in Autonomous Execution: Preventing Harmful Actions | 0.435 |
| `8.1` | Early Fusion vs. Late Fusion Architectures | 0.441 |
| `1.1` | The Transformer Revolution: From RNNs to Self-Attention | 0.442 |
| `10.3` | Patient Interaction and Triage: Ethical Constraints and Safety Filters | 0.442 |
| `9.9` | Evaluation Metrics for Agents: Beyond Perplexity to Task Success | 0.444 |
| `2.5` | Compute Efficiency and Hardware Evolution | 0.446 |

## Title near-duplicates (cosine >= 0.85)

### Cluster 1 (max pair cosine 0.853)

- `2.2` -- The Emergence of Scaling Laws
- `15.3` -- Emergent Abilities in Scaling Laws

### Cluster 2 (max pair cosine 0.872)

- `12.7` -- Adversarial Training and Data Augmentation for Robustness
- `13.9` -- Adversarial Training and Robust Fine-Tuning

## Content duplicates (cosine >= 0.80 on first 1500 chars)

| A | B | Cosine |
|---|---|---|
| `3.9` | `10.1` | 0.895 |
| `8.7` | `12.3` | 0.894 |
| `10.1` | `12.3` | 0.891 |
| `8.7` | `10.1` | 0.89 |
| `7.4` | `7.6` | 0.875 |
| `2.2` | `2.3` | 0.87 |
| `6.5` | `6.6` | 0.87 |
| `3.9` | `12.3` | 0.867 |
| `12.4` | `12.5` | 0.867 |
| `3.9` | `8.7` | 0.859 |
| `2.8` | `15.3` | 0.844 |
| `7.2` | `10.5` | 0.83 |
| `4.5` | `4.6` | 0.828 |
| `13.7` | `13.8` | 0.827 |
| `5.2` | `5.6` | 0.819 |
| `12.6` | `12.7` | 0.817 |
| `2.3` | `15.3` | 0.816 |
| `7.2` | `7.4` | 0.816 |
| `5.2` | `5.5` | 0.814 |
| `7.4` | `7.5` | 0.814 |
| `1.10` | `3.3` | 0.813 |
| `5.8` | `5.10` | 0.813 |
| `1.10` | `5.8` | 0.807 |
| `10.1` | `11.6` | 0.807 |
| `6.6` | `6.7` | 0.806 |
| `1.1` | `1.4` | 0.804 |
| `3.1` | `3.10` | 0.804 |
| `6.1` | `6.9` | 0.803 |
| `8.7` | `11.6` | 0.803 |
| `14.9` | `14.10` | 0.803 |
| `5.2` | `5.9` | 0.801 |
| `12.1` | `12.9` | 0.801 |

## Verdict

- Topic drift sections: **10**
- Title clusters: **2**
- Content duplicate pairs: **32**

**NEEDS ATTENTION** -- outline or content has duplication / drift issues.