# Design Decisions — NLBSE'27 Final Model

Every non-obvious choice made in `FinalModel.ipynb`, with the reasoning.

---

## 1. Model: `distilroberta-base` instead of `microsoft/codebert-base`

**What**: Swap the 12-layer, 125M-param CodeBERT encoder for 6-layer, 82M-param DistilRoBERTa.

**Why (scoring math)**:  
The competition score is `0.6×F1 + 0.2×time_score + 0.2×flops_score`.  
The CodeBERT baseline scores 0.71 and uses **2,666,509 GFLOPs** (89% of the 3M cap).  
FLOPs are roughly proportional to the number of transformer layers.  
DistilRoBERTa has 6 layers vs 12 → ~50% the FLOPs → ~1,333,000 GFLOPs estimated.

| Component | Baseline (CodeBERT) | Our model (DistilRoBERTa, estimate) |
|---|---|---|
| F1 component (0.6 × 0.91) | 0.546 | ~0.54 (assume slight drop) |
| Runtime component | 0.160 | ~0.18 (faster) |
| FLOPs component | 0.022 | ~0.111 |
| **Total** | **0.71** | **~0.83+** |

Even if F1 drops slightly (say 0.89 instead of 0.91), we still win on total score.  
DistilRoBERTa uses the same tokenizer vocabulary as RoBERTa, so it's a drop-in swap.

**Why not CodeBERTa-small?** `huggingface/CodeBERTa-small-v1` is a code-specific 6-layer model — also worth trying if DistilRoBERTa F1 is too low. Same architecture, trained on CodeSearchNet. Try it by changing `CONFIG['model_name']`.

**Why not CodeT5-small?** T5 is encoder-decoder; adapting it to classification requires treating the encoder only or modifying the head, adding complexity. Not worth it for 2 weeks.

---

## 2. Focal Loss instead of plain BCE

**What**: `FocalLoss(gamma=2.0)` with per-class `pos_weight`.

**Why**: The baseline's MAT recall is only **0.802** (vulnerability recall is 0.935). MAT is an imbalanced class. Standard BCE treats every example equally — the model learns to predict the majority class (non-MAT) to minimise loss.

Focal loss formula: `FL = -(1 - p_t)^γ · log(p_t)`  
The `(1 - p_t)^γ` term downweights easy, correctly-classified examples.  
When γ=2, an example predicted at 90% confidence contributes only 1% of its full loss.  
This forces the model to focus on hard, misclassified examples — exactly what MAT needs.

**pos_weight on top**: Even focal loss benefits from the initial class weight signal.  
`pos_weight[i] = neg_count[i] / pos_count[i]` — computed from training set distribution.  
This scales the gradient of positive examples proportionally to their scarcity.

**Why γ=2?** Standard value from the original RetinaNet paper; shown effective across many binary classification tasks. Could tune to [1.0, 2.0, 3.0] if time allows.

---

## 3. Validation Split for Threshold Tuning

**What**: 5% of training data held out as validation; thresholds tuned on this split only.

**Why**: The public test set labels are available, but tuning thresholds on the test set is data leakage — it won't generalise to the hidden test set used for final ranking. Using a held-out validation split simulates real-world behaviour.

**5% = ~10,730 samples** — large enough for stable threshold estimates while only slightly reducing training data (95% = ~203,857 still for training).

**Stratified split**: `stratify` on the concatenated label string (`"00"`, `"01"`, `"10"`, `"11"`) to maintain the same label-combination ratios in both splits. Prevents the val set from being dominated by the most common class.

---

## 4. Per-Task Threshold (not fixed 0.5)

**What**: Independently sweep thresholds from 0.05 to 0.95 for each output head; pick the threshold that maximises F1 on the validation set.

**Why**: Sigmoid outputs are probabilities, but the optimal decision boundary depends on class frequency. For MAT (imbalanced), the optimal threshold is likely below 0.5 — a lower bar means more positives predicted, increasing recall at some precision cost. The F1-maximising threshold naturally balances this tradeoff.

**Why independent per task?** The two tasks have different class distributions and different baseline F1s. Vulnerability F1 = 0.939 at threshold 0.5; MAT F1 = 0.881. They need different thresholds.

---

## 5. Training Hyperparameters

| Param | Value | Reasoning |
|---|---|---|
| `learning_rate` | 2e-5 | Standard fine-tuning LR for BERT-family models. Too high → catastrophic forgetting. Too low → slow convergence. |
| `batch_size_train` | 32 | Fits T4 (16 GB VRAM) with 512-token sequences. Larger batches would OOM. |
| `batch_size_eval` | 512 | Inference uses no gradient storage → more VRAM available. Larger batches = faster. |
| `num_epochs` | 3 | Empirically, fine-tuned transformers on large datasets plateau after 3 epochs. More risks overfitting. |
| `weight_decay` | 0.01 | AdamW default; L2 regularisation on weights (not biases). |
| `warmup_ratio` | 0.1 | 10% of steps as linear warmup. Prevents large gradient updates at start before the model has calibrated. |
| `dropout` | 0.1 | Matches pre-trained model defaults; classification head regularisation. |
| `max_length` | 512 | DistilRoBERTa max context. Dataset already filtered to ≤500 cl100k tokens. |
| `val_split` | 0.05 | 5% = ~10k samples; enough for threshold tuning without sacrificing too much training data. |

---

## 6. Dynamic Padding in Training vs Pre-tokenised Eval

**Training** uses a custom `collate_fn` that pads to the longest sequence *in each batch*. This is more memory-efficient than padding all sequences to 512 — most code functions are shorter.

**Evaluation** pre-tokenises the full test set once with `padding=True` (pads to longest in the whole test set). This matches the baseline's approach and ensures the FLOPs/time measurements are comparable.

---

## 7. Gradient Clipping at 1.0

`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — clips global gradient norm to 1.0.  
Prevents gradient explosions during early training, especially with focal loss where loss can spike on hard examples.

---

## 8. `torch.amp.autocast('cuda')` in Evaluation

Using FP16 autocasting during the profiled evaluation loop:
- Reduces memory bandwidth → faster inference
- Does NOT change the FLOPs count (profiler counts multiply-accumulate ops, same regardless of precision)
- Matches the spirit of the baseline (which used the deprecated `torch.cuda.amp.autocast()`)

---

## 9. Scoring Formula Reference

```
score = 0.6 × avg_F1
      + 0.2 × max((1000 - avg_runtime_s) / 1000, 0)
      + 0.2 × max((3,000,000 - avg_GFLOPs) / 3,000,000, 0)
```

Baseline breakdown:
- F1 = 0.910 → 0.6 × 0.910 = **0.546**
- Runtime = 199s → 0.2 × (801/1000) = **0.160**
- GFLOPs = 2,666,509 → 0.2 × (333,491/3,000,000) = **0.022**
- **Total = 0.728 ≈ 0.71**

The FLOPs component is nearly zero — the baseline essentially gives away 18% of the available score.  
This is the primary reason a smaller model wins even if F1 is slightly lower.
