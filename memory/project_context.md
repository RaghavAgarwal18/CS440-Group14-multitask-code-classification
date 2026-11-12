---
name: project-context
description: NLBSE'27 competition — task, dataset, baseline scores, scoring formula, and what FinalModel.ipynb contains
metadata:
  type: project
---

# CS440 Group 14 — NLBSE'27 Multitask Code Classification

## Task
Build ONE model that jointly classifies source-code functions for two binary labels:
- `weakness` (index 0): vulnerability/weakness present
- `MAT` (index 1): self-admitted technical debt (SATD) present

Input: formatted C/C++ source function. Output: `[weakness, MAT]` binary vector.

## Dataset (HuggingFace: `NLBSE/nlbse27-code-comment-classification`)
- Train: 214,586 instances
- Public test: 30,655 instances
- Hidden test: 61,311 instances (organisers only, used for final ranking)
- Columns: `Projectname`, `Filepath`, `Function`, `Label`
- Max tokens: 500 (cl100k_base), functions already clang-formatted

## Scoring Formula
```
score = 0.6 * avg_F1
      + 0.2 * max((1000 - avg_runtime_s) / 1000, 0)
      + 0.2 * max((3_000_000 - avg_GFLOPs) / 3_000_000, 0)
```
Primary metric is **mean F1** across both tasks (60%). Execution time and GFLOPs each worth 20%.

## Baseline (CodeBERT — `microsoft/codebert-base`)
HuggingFace: `NLBSE/nlbse27-code-comment-classification-baseline`

| Output | Precision | Recall | F1 |
|---|---|---|---|
| Vulnerability | 0.943 | 0.935 | 0.939 |
| MAT | 0.978 | 0.802 | 0.881 |
| **Average** | **0.961** | **0.869** | **0.910** |

- Avg runtime: ~199s
- GFLOPs: ~2,666,509
- **Baseline score: 0.71**

## Key Strategic Insight
Baseline F1 is already 0.91 — hard to push much higher. Max possible F1 gain = +0.09 → +0.054 score.
Time/compute both nearly maxed out (199s of 1000s, 2.67M of 3M GFLOPs) — large gains available there.
Smaller/faster model that preserves F1 can beat 0.71 easily.

## Team & Timeline
- 3 people total (Jacob, Raghav, ian)
- Due ~2026-11-25
- Compute: **Colab free only** (T4 GPU, 16 GB VRAM, sessions can die)

## Current State (as of 2026-11-11)
- `CodeBERT_baseline.ipynb`: baseline evaluation notebook (complete, run on Colab GPU)
- `FinalModel.ipynb`: empty — work not yet started

**Why:** This is a CS440 (NLP/ML course) final project submission to a real competition.
**How to apply:** When suggesting approaches, weight efficiency improvements heavily — there's more room there than on F1.
