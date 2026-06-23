# Physics as a Hard Reward Floor: reproduction package

> Anonymized copy for double-blind review (MOSS @ COLM 2026). Author and citation
> details are omitted. Run the code from the uploaded supplementary files; there is
> no public repository to clone here.

## Overview
An unbounded hard penalty on physical-law violations gives a reinforcement-learning
post-training reward a floor that no bounded preference signal can lift a violating
completion above. The same reward plugs into PPO, DPO, and GRPO. This package
reproduces the two headline results at small scale, under 10^15 FLOPs, in a free
Google Colab notebook:

- Table 1: a model-free reward-hacking probe (CPU, seconds).
- Table 2: a small-scale GRPO run on Qwen2.5-0.5B-Instruct (one GPU, a few minutes).

## Requirements
Python (>= 3.9) with `torch`, `transformers`, `accelerate`, and `numpy`. The probe
needs only NumPy and runs on CPU.

## Reproduce in Google Colab
1. Go to https://colab.research.google.com and choose **File -> Upload notebook**,
   selecting `moss_pigrpo_probe.ipynb`.
2. Open the **Files** sidebar and upload `run_probe.py` and `run_grpo_local.py` next
   to the notebook (or unzip the supplement).
3. **Runtime -> Change runtime type -> T4 GPU**, then **Save** (Table 1 is CPU-only).
4. Run the first cell, "ONE-CLICK: reproduce BOTH tables". It installs the
   dependencies and runs both scripts end to end.

## Reproduce locally
```
pip install torch transformers accelerate
python run_probe.py                              # Table 1 (CPU, seconds)
STEPS=40 WHARDS=0.0,5.0 python run_grpo_local.py # Table 2 (uses a GPU if present)
```

## Results
Table 1, reward-hacking probe:
```
preference-only  (w_hard=0):  +10.0    0/5 caught
physics-grounded (w_hard=5):  -490.5   5/5 caught
```
Table 2, small-scale GRPO (hard-violation rate, before -> after training):
```
preference-only  (w_hard=0):  0.58 -> 1.00
physics-grounded (w_hard=5):  0.50 -> 0.00
```

## Repository structure
```text
.
|-- moss_pigrpo_probe.ipynb   # Colab notebook: first cell reproduces both tables
|-- run_probe.py              # Table 1, model-free reward-hacking probe (CPU)
|-- run_grpo_local.py         # Table 2, small-scale GRPO on Qwen2.5-0.5B
|-- probe_results.json        # saved Table 1 output
|-- grpo_results.json         # saved Table 2 output
`-- COLAB_GUIDE.md            # detailed Colab walkthrough
```

## Citation
Omitted for double-blind review.
