# Running the MOSS @ COLM 2026 notebook on Google Colab (anonymized)

> Anonymized copy for double-blind review. There is no repository to clone; run
> from the supplementary files uploaded with the submission.

The supplement contains `moss_pigrpo_probe.ipynb`, `run_probe.py`, and
`run_grpo_local.py`. The reward-hacking probe (Table 1) is pure CPU; the small
GRPO run (Table 2) uses one GPU.

## A. Open the notebook
1. Go to https://colab.research.google.com and sign in.
2. `File -> Open notebook -> Upload` and select `moss_pigrpo_probe.ipynb`.

## B. Upload the two scripts next to it
1. Open the Files sidebar (folder icon on the left).
2. `Upload to session storage` and add `run_probe.py` and `run_grpo_local.py`.
   (Or upload the supplement `.zip` and unzip it in a cell: `!unzip -o supplement.zip`.)

## C. Turn on the GPU (only Table 2 needs it)
1. `Runtime -> Change runtime type -> T4 GPU` (free tier is enough), then `Save`.

## D. Run it
1. Run the first cell, `ONE-CLICK: reproduce BOTH tables`. It installs deps and
   runs both scripts from the uploaded files.
2. Expected output:
   - Table 1: `preference-only (w_hard=0) -> +10.0, 0/5` and
     `physics-grounded (w_hard=5) -> -490.5, 5/5`. No GPU needed.
   - Table 2: base vs trained hard-violation rate for `w_hard=0` and `w_hard=5`
     (preference-only drifts to / stays infeasible; physics drives it to zero).
3. If a cell errors on a fresh runtime, `Runtime -> Restart and run all` once.

## E. Save the executed notebook
`File -> Download -> Download .ipynb` after the outputs are visible. MOSS
reviewers judge the claim by running this notebook, and the probe cell runs on
free-tier CPU in seconds.

## Free-tier and timing
- The probe is free-tier reproducible (CPU, no downloads, seconds).
- The 0.5B GRPO fits a free T4 GPU in a few minutes, within the MOSS Colab-track
  limits (<= 1 GPU, <= 12 h, <= 500 GB).
