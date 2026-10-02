#!/usr/bin/env python3
"""Generate configs.json for the hyperparameter sweep (random search).

Deterministic (seed 7). Config 0 is the incumbent E00/E10 params so the
sweep always contains the current default as a reference point. Selection
is on VAL pooled RMSE only — test stays untouched; the winner graduates to
a real cell and is test-scored once.
"""

import json
import os

import numpy as np

N = 96
rng = np.random.default_rng(7)

# v2 (2026-08-17): the v1 space allowed 255-leaf/6000-tree/lr-0.01 monsters
# costing 10-20x the incumbent per config — 2 of 96 finished in two Athena
# submissions. Screening doesn't need that corner: cost-bounded space below
# (leaves <= 127, lr >= 0.015; train_sweep caps n_estimators at 3000).
configs = [dict(  # 0 = incumbent
    learning_rate=0.03, num_leaves=63, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, lambda_l1=0.0, lambda_l2=0.0,
)]
while len(configs) < N:
    configs.append(dict(
        learning_rate=round(float(10 ** rng.uniform(-1.82, -1)), 4),
        num_leaves=int(rng.choice([15, 31, 63, 127])),
        min_child_samples=int(rng.choice([10, 20, 40, 80, 160])),
        feature_fraction=round(float(rng.uniform(0.4, 1.0)), 3),
        bagging_fraction=round(float(rng.uniform(0.5, 1.0)), 3),
        lambda_l1=0.0 if rng.random() < 0.5 else round(float(10 ** rng.uniform(-2, 1)), 4),
        lambda_l2=0.0 if rng.random() < 0.5 else round(float(10 ** rng.uniform(-2, 1)), 4),
    ))

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.json")
with open(path, "w") as f:
    json.dump(configs, f, indent=1)
print(f"wrote {path}: {len(configs)} configs")
