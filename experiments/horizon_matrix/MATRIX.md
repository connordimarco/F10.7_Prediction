# F10.7 horizon matrix

*Created 2026-08-12, patterned on LAUREN's extreme_matrix harness. Goal:
predict daily observed F10.7 at leads 1–30 days from 60-day windows of
F10.7 + SSN + SRS active-region aggregates (model spec in ../../CLAUDE.md).*

## Design (LAUREN rules)

- **One folder = one cell = one change** vs `F107_E00_control`. Baselines are
  `C*` cells sharing the same prediction contract so every row of the
  leaderboard is scored identically.
- **Fixed scorer** `shared/score_cell.py` — never edited per cell. Truth is
  OBSERVED F10.7 from `data/daily.csv`; every cell predicts the canonical
  test-origin set (identical n).
- **Splits by forecast origin**: train 1996–2021 · val 2022–2023 (early
  stopping/tuning only) · **test 2024→ (untouched)** — spans the cycle-25
  maximum, deliberately hard.
- **Modeling space**: adjusted (1-AU) F10.7; the ±3.3% Earth–Sun distance
  cycle is restored analytically at predict time (`common.adj_to_obs`).
- **Guards:** pooled test r/RMSE — a candidate may not lose pooled RMSE to
  the best baseline while claiming a target win.
  **Targets:** skill vs persistence in lead bands 1–7 / 8–14 / 15–30, and
  RMSE in the high-activity bin (true F10.7 ≥ 150).

## Cells

| Cell | One-line description | Mechanism probed |
|---|---|---|
| C00_persist | pred(t+h) = obs(t) | scorer's own reference — skill must be exactly 0 (harness check) |
| C01_recur27 | pred(t+h) = adj(t+h−27) | 27-day rotation recurrence alone |
| C02_mean81 | pred(t+h) = trailing 81-day mean | hold the F10.7a envelope |
| E00_control | LightGBM ×30 leads, full 720-feature windows | the model to beat / diff base |
| E01_train05 | E00 trained 2005–2021 only | span cost of the MIDL constraint (E02's clean reference) |
| E02_sw | E01 + 8 daily MIDL solar-wind series (+480 feats) | does the solar wind add F10.7 information? |
| E03_swnan | E00 + SW as optional feats (NaN pre-2005, native missing) | solar wind decoupled from the span cost |
| E04_recuralign | E00 + per-lead rotation-aligned f107 (lags 27−h, 54−h) | is the long-lead persistence crossover a feature-discovery problem? |
| E05_deepspan | E00 trained 1947–2021, ar_* optional (NaN pre-1996) | three more solar maxima (incl. cycle 19) vs the hi-act shrinkage |
| E06_stack | per-lead nonneg blend of E00+recur+mean81+persist, val-fitted | classic stacking RMSE shave (val fit mildly optimistic — see cell) |
| E07_seedens | E00 × 5 seeds, predictions averaged | pure variance reduction |
| E08_tailcal | E00 + val-fitted quantile map, tail-extrapolating | LAUREN-E08 un-shrinkage aimed at the hi-act bin |

Solar-wind cells (2026-08-12): MIDL @ 14 Re is authorized **2005-on only**
(owner decision; files exist from 1999 but are not trusted). SW cells train
on `train05` = 2005–2021 and diff against E01, not E00. The canonical test
set requires valid SW windows for ALL cells (identical n everywhere), so it
ends where MIDL ends (last posted month; June 2026 at build time).

## Running

```
shared/run_local.sh F107_E00_control      # local (default — data is MB-scale)
python shared/leaderboard.py              # -> LEADERBOARD.md
```

`shared/run_cell.pbs` is the Athena template (tur_ath, adapted from LAUREN),
prepared for when the matrix outgrows the Mac — venv + project sync on
/nobackupp28 required first; owner submits all jobs.

## Results / decision log

- **2026-08-12 — matrix stood up, all four cells run locally** (round-1
  test set: 925 origins, 2024-01-01→2026-07-13; all cells were later
  rescored on the 882-origin canonical set once solar wind joined — the
  numbers in this entry are the originals, current ones are in
  LEADERBOARD.md). C00 skill exactly 0.000 at every band —
  harness verified. Baselines split the horizon as physics predicts:
  persistence wins leads 1–7, recurrence/envelope win 8–30. **E00 beats all
  baselines everywhere** — pooled RMSE 30.79 vs 33.16 (C02), skill vs
  persistence +0.226/+0.303/+0.161 across the three bands — the only cell
  positive in all bands. High-activity RMSE 35.96 vs 37.38 (C02).
  See `LEADERBOARD.md`. Next: variant cells (feature ablations — does SRS
  actually pay? —, target transforms, seq models) diffing against E00.
- **2026-08-12 — solar-wind round (E01–E03), all cells rescored on the new
  canonical test set** (882 origins, 2024-01-01→2026-05-31; requires valid
  SW windows for every cell; MIDL's posted June 2026 file is ~empty, so SW
  ends 2026-05-31). Verdicts:
  - **E01 span cost is severe**: dropping 1996–2004 (cycle-23 max) collapses
    high-activity skill — pooled RMSE 30.74→36.48, hi-act RMSE 35.36→44.49.
    The model can't extrapolate above what it trained on (same shrinkage
    phenomenon LAUREN fought). Cycle-23 years are load-bearing.
  - **E02 solar wind on the 2005 span: negative** (36.48→37.78 vs E01).
  - **E03 decouples SW from span** (full span, SW NaN pre-2005, LightGBM
    native missing): 31.79 — still worse than E00's 30.74 at every band.
  - **Verdict: MIDL solar wind does not add F10.7 information in this
    framework; it subtracts (noise features).** E00 remains champion.
    Physics-consistent: the wind is downstream of solar activity, not a
    precursor of EUV/radio emission. Revisit only with a mechanism-specific
    encoding (e.g. high-speed-stream recurrence phase), not raw covariates.
- **2026-08-12 — E04 (rotation-aligned features): no change** (30.70 vs
  30.74 ≈ noise; r@27 0.605→0.607). Context: persistence beats the model on
  *correlation* at leads 24–29 (r 0.65 vs 0.61 at the 27-day echo) while the
  model still edges it on RMSE. E04 proves this is NOT a feature-discovery
  problem — the trees already had the recurrence signal and the aligned
  columns added nothing. It's an objective problem: MSE training hedges
  toward the conditional mean and doesn't reward phase-tracking, which is
  what r measures. If long-lead r matters operationally, the lever is the
  predictor combination or the loss (e.g. per-lead val-tuned blend of E00
  with C01 recurrence, or predicting the deviation-from-recurrence), not
  more features. **Decision (owner, 2026-08-12): the application is
  satellite drag → RMSE is the metric, the long-lead r crossover is
  cosmetic, blend/deviation cells stay parked.** Drag implications: the
  high-activity guard (RMSE @ F10.7 ≥ 150) is a first-class target (density
  error compounds when activity is high), and downstream density models
  consume daily F10.7 *and* 81-day F10.7a — so 30-day-out errors partially
  average away in F10.7a, further favoring the hedged MSE model. TODO when
  a consumer is wired up: confirm whether it wants observed or adjusted
  F10.7 (we predict adjusted and convert, so both are always available).
- **2026-08-12 — round 2 (E05–E08) scored.** Verdicts:
  - **E06 stacking = new champion**: 29.75 pooled (−1.0 vs E00), best on
    every pooled metric and the hi-act bin (34.07). The val-fit optimism
    caveat doesn't taint the result (test is untouched); it only means the
    weights could be even better with OOF bases.
  - **E05 deepspan: real but lead-dependent** — pooled ≈ E00 but L1–7
    21.29 vs 22.83 and L8–14 31.34 vs 32.03 (best single-model numbers),
    while L15–30 degrades (33.70) and hi-act unexpectedly worsens (36.12).
    1947-era data helps where the signal is strong, dilutes where it isn't.
  - **E07 seed ensemble: +0.1 — real, tiny, free.** Fold into future cells.
  - **E08 tailcal as implemented: FAILED HARD** (38.50; L1–7 skill −0.34).
    Diagnosis: the map was fitted on VAL (2022–23, rising phase) and
    applied to TEST (2024+, maximum) — distribution shift poisoned the
    whole upper range. LAUREN fit theirs out-of-fold *within* the training
    years. Not evidence against calibration itself; redo OOF-on-train if
    revisited. (The scorer's guard did its job.)
  - Obvious round 3: E09 = stack with E05-deepspan + seed-ensembled bases
    and OOF blend weights — combine everything that won.
