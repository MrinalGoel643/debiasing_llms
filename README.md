# Gender Bias Mitigation via DPO Fine-Tuning on Qwen2.5-3B-Instruct

**AIPI 540 — Project 1**

This project investigates gender bias in large language models and applies parameter-efficient fine-tuning to reduce stereotypical associations. Using the StereoSet benchmark as both training signal and evaluation framework, we fine-tune `Qwen2.5-3B-Instruct` with QLoRA + Direct Preference Optimization (DPO) to push the model toward gender-neutral completions.

---

## Overview

| Property | Detail |
|---|---|
| **Bias Type** | Social Bias (Gender) |
| **Base Model** | `Qwen/Qwen2.5-3B-Instruct` |
| **Benchmark** | StereoSet (Intrasentence, Gender subset) |
| **Fine-Tuning Method** | QLoRA + Direct Preference Optimization (DPO) |
| **Key Metric** | Stereotype Score (SS) — ideal = 50.0 |

---

## Background

LLMs trained on internet-scale corpora absorb the stereotypical associations present in that data. For gender bias, this manifests as the model systematically assigning higher likelihood to stereotypical completions (e.g., associating women with passivity, men with aggression) compared to anti-stereotypical ones.

**StereoSet** measures this using an intrasentence fill-in-the-blank format. Each example contains a context sentence with a BLANK and three candidate completions:
- `stereotype` — a gender-stereotypical fill
- `anti-stereotype` — a completion that subverts the stereotype
- `unrelated` — a semantically unrelated fill

We filter for the **gender subset** (255 examples out of 2,106 total intrasentence examples).

---

## Metrics

**Stereotype Score (SS):** The percentage of examples where the model assigns higher conditional log-likelihood to the stereotypical completion over the anti-stereotypical one. An ideal unbiased model scores 50.0 (no systematic preference).

**Language Modeling Score (LMS):** The percentage of examples where the model prefers a meaningful completion (stereotype or anti-stereotype) over the unrelated one. Higher is better — this checks that the model hasn't degraded in general language quality.

**ICAT Score:** A composite metric that penalizes both high stereotype preference and poor language modeling. Higher is better.

---

## Results

| | Stereotype Score ↓ | LM Score ↑ | ICAT Score ↑ |
|---|---|---|---|
| **Pre-fine-tuning** | 70.6% | 96.1% | 56.52 |
| **Post-fine-tuning** | 58.8% | 94.1% | 77.51 |
| **Ideal** | 50.0% | — | — |

DPO fine-tuning reduced the Stereotype Score by **11.8 percentage points** (from 70.6% → 58.8%) and improved the ICAT Score by **+21 points** (56.52 → 77.51), with minimal degradation in general language modeling ability (LMS dropped only 2 points).

---

## Method

### 1. Data Preparation

StereoSet's gender subset (255 examples) is parsed into structured records containing the context, stereotype completion, anti-stereotype completion, and unrelated completion. An 80/20 train/eval split is applied (204 train, 51 eval).

DPO pairs are constructed as:
- **Prompt:** Chat-formatted sentence completion task (`"Complete the following sentence by filling in the blank..."`)
- **Chosen:** Anti-stereotype completion
- **Rejected:** Stereotype completion

### 2. Model Loading (4-bit Quantization)

`Qwen2.5-3B-Instruct` is loaded in 4-bit NF4 precision using BitsAndBytes double quantization with bfloat16 compute dtype. This reduces GPU memory to ~4.84 GB for the 3B parameter model, enabling single-GPU fine-tuning.

### 3. QLoRA Configuration

Low-Rank Adaptation is applied to all major projection layers:

```
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
LoRA rank (r):     16
LoRA alpha:        32
LoRA dropout:      0.05
Trainable params:  ~30M / 3.1B total (0.96%)
```

### 4. DPO Training

A separate frozen reference model (same base, no LoRA) is loaded alongside the trainable model. DPO optimizes the policy to increase the likelihood ratio between chosen and rejected completions relative to the reference.

```
Epochs:            3
Batch size:        2 (effective: 8 with grad accumulation)
Learning rate:     5e-5
DPO beta:          0.1  (lower = stronger preference push)
Max sequence len:  512
Total steps:       78
Final train loss:  ~0.19
```

Loss decreased consistently from ~0.70 → ~0.19 over training, confirming the model learned to prefer anti-stereotypical completions.

### 5. Evaluation

Evaluation uses **conditional log-likelihood**: the model scores each candidate completion given the chat-formatted prompt. The same prompt format is used for both pre- and post-fine-tuning evaluation. Pre-fine-tuning results are obtained by disabling the LoRA adapter on the same model instance, ensuring a fair apples-to-apples comparison.

---

## Setup & Usage

### Requirements

```bash
pip install transformers>=4.40.0 datasets peft trl>=0.8.0 bitsandbytes accelerate sentencepiece
pip install matplotlib seaborn pandas
```

### Hardware

Tested on NVIDIA L4 (23.7 GB VRAM). The 4-bit quantization scheme keeps peak memory under 8 GB (including reference model), so this should run on any GPU with 16+ GB VRAM.

### Running the Notebook

1. Open `Gender_Bias_DPO_Qwen.ipynb` in Google Colab or a local Jupyter environment with GPU access.
2. Run all cells top to bottom. The notebook will:
   - Download the StereoSet dev set automatically
   - Load and quantize the model
   - Run baseline evaluation
   - Build the DPO dataset
   - Fine-tune with QLoRA + DPO
   - Evaluate the fine-tuned model
   - Generate comparison visualizations

---

## Key Design Decisions

**Why DPO instead of SFT?** DPO directly encodes a preference signal (anti-stereotype > stereotype) without requiring a separate reward model. This is more principled for bias mitigation than supervised fine-tuning, which would require curating "correct" completions.

**Why low beta (0.1)?** A lower DPO beta places less penalty on deviating from the reference model, allowing stronger preference updates. This was intentional to maximize debiasing given the small dataset.

**Why conditional log-likelihood for evaluation?** Generative sampling introduces randomness that makes consistent evaluation difficult. Scoring all candidates under the same prompt via log-likelihood provides a deterministic, interpretable measure of the model's implicit preferences.

---

## Limitations

- The StereoSet gender subset is small (255 examples), which limits both training signal and evaluation reliability.
- The model still scores above the ideal 50.0 SS, indicating residual bias.
- Debiasing is evaluated only on the StereoSet distribution; generalization to out-of-distribution gender bias prompts is not assessed.
- LMS dropped slightly post-fine-tuning, suggesting minor language quality tradeoff.

---

## File Structure

```
Gender_Bias_DPO_Qwen.ipynb   # Main notebook
training_loss.png             # DPO loss curve (generated at runtime)
qwen-gender-debiased/         # Saved LoRA adapter weights (generated at runtime)
stereoset_dev.json            # StereoSet dev data (downloaded at runtime)
```

---

## References

- [StereoSet: Measuring Stereotypical Bias in Pretrained Language Models](https://stereoset.mit.edu/) — Nadeem et al., 2021
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — Qwen Team, Alibaba Cloud