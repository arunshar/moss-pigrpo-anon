#!/usr/bin/env python3
"""Reward-hacking probe (Table 1). Model-free, CPU, runs in under a second.

Reproduces the paper's headline result: a preference-only reward cannot tell a
fluent-but-infeasible completion from a feasible one (0/5 caught, both score
+10.0); the unbounded hard physics term ranks every infeasible completion below
every feasible one (5/5 caught, infeasible mean -490.5).
"""
import json, os

VMAX = 30.0          # speed envelope (knots)
PREF = 10.0          # the "rater" gives +10 to every fluent completion (the hackable signal)

def R_hard(traj):    # unbounded S-KBM excess penalty; feasible -> 0
    return -sum(max(abs(v) / VMAX - 1.0, 0.0) for v in traj)

def total(traj, w_hard, w_pref=1.0):
    return w_pref * PREF + w_hard * R_hard(traj)

# 5 feasible completions (inside the envelope) and 5 fluent-but-infeasible ones
# (a sustained ~2x over-speed, so the per-trajectory violation sum Phi is ~100.1).
feasible   = [[15, 18, 12, 20, 16] for _ in range(5)]
infeasible = [[VMAX * 2.001] * 100 for _ in range(5)]   # relu(2.001-1)=1.001 x 100 = 100.1

rows = []
for name, wh in [("preference-only  (w_hard=0)", 0.0),
                 ("physics-grounded (w_hard=5)", 5.0)]:
    inf = [total(t, wh) for t in infeasible]
    fea = [total(t, wh) for t in feasible]
    fmin = min(fea)
    caught = sum(1 for v in inf if v < fmin)            # infeasible ranked below every feasible
    rows.append((name, sum(inf) / len(inf), caught, len(inf), sum(fea) / len(fea)))

print("=== Table 1: reward-hacking probe (CPU) ===")
print(f"{'reward configuration':34}{'infeasible mean':>16}{'caught':>9}{'feasible mean':>16}")
for n, im, c, N, fm in rows:
    print(f"{n:34}{im:>+16.1f}{f'{c}/{N}':>9}{fm:>+16.1f}")

json.dump(
    {"preference_only": {"infeasible_mean": round(rows[0][1], 1), "caught": f"{rows[0][2]}/5",
                          "feasible_mean": round(rows[0][4], 1)},
     "physics":         {"infeasible_mean": round(rows[1][1], 1), "caught": f"{rows[1][2]}/5",
                          "feasible_mean": round(rows[1][4], 1)}},
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results.json"), "w"), indent=2)
print("\nwrote probe_results.json")
