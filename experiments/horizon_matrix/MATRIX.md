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
- **Data version** (added 2026-09-02): `common.DATA_VERSION` is stamped
  into every scorecard and `leaderboard.py` pools only cards from the
  current version; older cards drop to a frozen table (their numbers are
  also snapshotted in `LEADERBOARD_v1.md`). `build_samples(..., flux=)`
  selects the F10.7 series used for the flux feature block + target
  (`f107_adj` canonical noon value, or `f107_adj_rob` flare-robust);
  origin validity and scoring truth always use the canonical column.
- **Probabilistic scorer** `shared/score_prob.py` (separate file; the
  deterministic scorer is untouched): CRPS, 80/50% coverage + width by
  year/band/activity, spike capture, on `<cell>/out/ensemble.csv`.
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
| E09_combo | per-lead OOF-weighted blend: E10-feature LGBM (96 + 47 spans, seed-ens) + 3 baselines | everything that won, combined; E06's val double-dip removed |
| E10_farside | E00 + 14 far-side AR catalog aggregates (GONG f6x + JSOC SARD), NaN pre-2010 | does far-side info fix the L8-30 blind spot? (see FARSIDE_SCOPING.md) |
| E12_f107ssn | E00 restricted to f107_adj + ssn windows (120 feats) | SRS ablation: what do the ar_* features actually pay? |
| E13_qbounds | p10/p50/p90 quantile-objective LGBMs, E10 features | calibrated uncertainty bounds (owner request 2026-08-17) |
| E14_calbounds | E13 bounds + OOF-on-train conformal widening, activity-binned, cfg-44 params | can conformal calibration fix E13's undercoverage? |
| E15_combo_tuned | E09 recipe with sweep-winning cfg-44 params | does the val-selected config survive the test set? |
| **— v2 data era (2026-09-02) —** | all cells below run on the corrected daily.csv; nothing above is comparable (see log) | |
| E20_base2 | single LGBM, E10 features, cfg-44 params, canonical flux, raw target | the v2-data control (train 1996–2021) |
| E21_robust | E20 with the flare-robust flux series as feature block + target | do flare spikes in the targets cost skill? |
| E22_ratio | E21 with target = flux(t+h) / env81(t), multiplied back at predict | level factored out — can trees stop capping the tail? |
| E23_ratio_mcs40 | E22 with min_child_samples 40 (vs 160) | does the ratio target want less regularization? |
| E24_ratio_deep | E22 trained 1947–2021 (ar_*/fs_* optional-NaN) | three more maxima for the ratio model |
| E25_ratio_wt | E22 with sample weights ∝ origin envelope (≤3×) | owner's "high activity weighs heaviest", in the loss |
| E26_ratio_pers | E21 with target = flux(t+h) / flux(t) | persistence-relative instead of envelope-relative target |
| E30_combo_v2 | E15 recipe (two LGBM spans, 3 seeds, OOF nonneg blend + 3 baselines) with the winning E2x options; saves train-origin OOF blend predictions | the v2 champion candidate |
| E31_ensemble | trajectory ensemble: E30 test path × exp(OOF log-residual trajectories from the same activity tercile) | calibrated probabilistic product + scenario members (scored by score_prob.py) |
| E32_ensemble_adapt | E31 with the residual spread rescaled online from the trailing 90 days of realized errors | adaptive calibration for regimes the training record lacks |
| E33_combo_v2_raw | E30 recipe with the level (raw) target | control for the ratio-target combo |
| E34_combo_gated | four LGBM bases (level + ratio, 96 + 47 spans) + baselines, blend weights per activity tercile | regime-dependent stacking aimed at the tail |
| E35_stack | level-2 nonneg blend over every cell's per-lead OOF record | what is additive across families? |
| E40_nn / E41_mlp | GRU / MLP, joint 30-lead output in envelope-ratio space, flux-space MSE, 5 seeds, train 1947→ | first non-tree family |
| E50_diskfwd | disk forward model: SRS regions + GONG far-side detections rotated forward, decay + limb weighting, 35 fitted numbers | does explicit rotation geometry beat the window of totals? |
| E51_geolin | OLS per lead on [flux, env81, recur27, S_0, S_h] (E50's geometric series) | geometry with an almost-nonexistent learner |
| E60_mgii | E21 + Bremen Mg II core-to-wing index (plage proxy, 1978→, optional-NaN 60-day window) | does a plage measure carry what SRS sunspot data lacks? |
| E61_mgii_deep | E24 (deep-span ratio model) + Mg II window | same, on the best single-model recipe |

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
- **2026-08-17 — E10 far-side: REAL — best single model in the matrix.**
  E00 + 14 `fs_*`/`sard_*` daily aggregates from two independent far-side
  AR catalogs (GONG helioseismic `f6x` + JSOC/HMI SARD, 2010→, optional-NaN;
  scouting report in FARSIDE_SCOPING.md, endpoints in data/SOURCES.md).
  Pooled 30.74→30.37, and the FIRST feature family ever to add band skill:
  L1–7 +0.217→+0.234, L8–14 +0.298→+0.316 (matrix-best, tied w/ E05),
  hi-act 35.36→34.72 (single-model best). L15–30 ≈ unchanged
  (+0.156→+0.159) — the gain sits at L1–14, consistent with the physics:
  a far-side region returns to the disk within ≤ half a rotation
  (~13.6 d), so detections inform the imminent-return window, not leads
  that outlive region evolution. Literature expectation (Lei+2019 ~12.5%
  at 27 d with full maps) was optimistic for scalar aggregates; what
  survived is exactly where the E06 stack was already strongest, so the
  blend must be refit to see what remains additive.
  Data notes: GONG WAF hard-blocks fast crawls (fetch serially ≤1 req/s;
  had to run the backfill from solsticedisk after home IP got 403'd);
  f6x has rare glitch rows (per-region flux up to 8e37) — builder
  winsorizes at ~10× p95. Next: E09 should now be the combination cell
  with the E10 feature set as the model base (+ deepspan + seed ensemble,
  OOF blend weights).
- **2026-08-17 — E09 combo: NEW CHAMPION, big margin.** Per-lead nonneg
  blend of two E10-feature LGBMs (train96 + deepspan train47, both 3-seed
  ensembles) with recur/mean81/persist; weights fitted on OOF train
  predictions (5 chronological folds) — E06's val double-dip removed.
  Pooled 29.75→**28.62**, r 0.714, skill +0.262; bands
  +0.277/+0.350/+0.211 and hi-act **32.49** — every column matrix-best.
  Weight story: deepspan LGBM dominates L1–7 (w≈0.65–0.82 — E05's
  short-lead strength, now safe because the blend contains it), the two
  LGBMs+mean81 share the mid band, and long leads go ~half LGBM ~0.4
  mean81 with small recurrence — MSE-hedging delegated to the envelope
  exactly where the trees flatten. All three E05 caveats (L15–30
  degradation, hi-act worsening) are absorbed by the blend. Runtime ~5 h
  local (480 LGBM fits) — future combo iterations are Athena candidates.
- **2026-08-17 — E12 (SRS ablation): the ar_* features PAY.** f107+ssn
  only: 31.33 pooled / hi-act 36.41 / L1–7 +0.167 — vs E00's 30.74 /
  35.36 / +0.217. So sunspot-group data buys ~0.6 sfu pooled, most of it
  at short leads; far-side adds another ~0.4 on top (E10). Data-story
  figure: `plots/data_story.png` (same LGBM, families added one at a
  time). All three model lines converge to the baselines past lead ~21 —
  beyond three weeks the input families stop mattering and the envelope
  is all anyone knows (hence E09's blend weights there).
- **2026-08-17 — E13 (quantile bounds): right shape, too narrow.**
  p10/p50/p90 quantile-objective LGBMs (E10 features, train96, val for
  early stopping only). Test coverage of the 80%-target band: **59.2%**
  overall — and activity-dependent: 46% in 2024 (max), 53% hi-act bin,
  71% in 2026. Same val→test era shift E08 hit, now measured instead of
  fatal. The band still transforms the example plots (`forecast_examples.png`):
  July/April observations live inside it; November's 220 spike exceeds
  the ~185 band top — the residual undercoverage in one picture.
  **Next (parked until the sweep lands, then build with winning params):
  E14 = conformalized quantile calibration, OOF on train (never val),
  widening conditioned on activity level (CQR-style), success = ~80%
  coverage in every year and the hi-act bin.**
- **2026-08-18 — hyperparameter sweep (96 configs, Athena): incumbent
  params were mediocre — rank 57/96 on val.** Winner cfg 44
  (lr 0.016, 31 leaves, **min_child_samples 160**, ff 0.73, bf 0.69,
  λ1 1.46): val 22.99 vs incumbent 23.33 (−1.4%). Top 8 form a plateau
  within 0.06 and 7/8 share mcs=160 — the real signal is MUCH stronger
  regularization than the hand-picked defaults (mcs 40). Selection on
  val only; the winner graduates via E15 (E09 recipe, cfg-44 params) and
  meets test exactly once.
  Ops lessons, earned over three failed submissions: (1) LightGBM's
  sklearn wrapper IGNORES OMP_NUM_THREADS (defaults to all 512 threads)
  — with 8 workers/node that's ~100× oversubscription and a
  near-standstill that looks like "busy but producing nothing"; pin
  `n_jobs` explicitly for any multi-process-per-node job. (2) Time a
  single config before fanning out; both walltime failures were sized by
  assumption. With the fix: 2,880 fits, 12 nodes, ~30 min.
- **2026-08-18 — E15: NEW CHAMPION — the sweep's val gain survived test.**
  E09 recipe rebuilt with cfg-44 params, run as a 30-lead fan-out on
  Athena (4 nodes, ~3 min wall; the regularized config also trains ~4×
  faster). Pooled 28.62→**28.32**, r 0.722, bands +0.290/+0.356/+0.218,
  hi-act 32.49→**32.19** — every column improves. The full stack from the
  original ask: f107+ssn model 31.33 → all-data 30.37 → +blend 28.62 →
  +tuning 28.32 (−9.6%). Owner direction after this: methodological
  changes over knob-turning — next real cell is the survival-gated
  regime blend (dynamic recurrence weight from a region-survival
  classifier); E14 conformal bounds close out the current paradigm.
- **2026-08-18 — E14 conformal bounds: 59% → 75%, not yet 80%.** Quantile
  models with cfg-44 params + widening = 80th pct of OOF-on-train
  conformity scores, binned by f107a terciles at the origin (busy bins
  widen ~7× more than quiet at lead 14). Test coverage: overall 75.1%,
  **2025 84.6% / 2026 86.8% (passing), 2024 60.9% (failing)** — the
  cycle-25 peak exceeds the training record's support, which no
  train-fitted calibration can anticipate. Product ships as "80% nominal,
  ~75% realized, degrades in extreme maxima" OR gets the known fix:
  **adaptive conformal** (widths updated online from realized residuals —
  legitimate operationally, would need prequential evaluation; queued).
  Bounds live in `F107_E14_calbounds/out/bounds.csv` alongside the E15
  point forecast; example plots + PDF page 3 show the pair.
- **2026-09-02 — two data-layer bugs found on return; everything above is
  re-based.** Revisiting the tail failure (champion never predicts above
  273 while 2024 truth reaches 401; bias −56 sfu when truth > 240; 78% of
  E14 band misses are *above* the band) exposed:
  1. **Every F10.7 date in daily.csv was one day late.** `build_dataset.py`
     added +0.5 to the LISIRD Julian dates before `pd.to_datetime(...,
     origin="julian")`, which already counts from noon — so the 20 UT
     reading of day D was filed under D+1. Independent check: CelesTrak's
     NOAA series has the X28-flare spike (560.9) on 2003-11-04, ours had it
     on 11-05; after the fix all 25,090 overlapping days agree within
     5 sfu (before: 541 days differed by >20). Effect on the matrix: the
     f107 block was misaligned by a day against SSN/SRS/far-side features,
     and the scoring truth carried the wrong date label; baselines are
     unaffected (a uniform shift), so C00–C02 reproduce exactly on v2.
  2. **Flare-contaminated readings in the targets.** The daily value is the
     20 UT reading; when a flare hits it, the day reads 400–940 sfu
     (2011-03-07: 939 vs 152/162 at 17/23 UT; 2024-07-30: 401 vs 222/222).
     These are the official/consumer values (SWPC and CelesTrak keep them),
     so they stay in `f107_obs`/`f107_adj` and in the scoring truth — but
     they are noise as *training targets*. New columns `f107_obs_rob` /
     `f107_adj_rob`: on days whose three readings disagree by >10% the
     reading nearest the previous day's robust value is taken (causal —
     flares push a reading up, instrument glitches push one down, so
     neither min nor median is safe: 2001-12-01's odd reading was a low
     one); residual one-day spikes/dips vs both neighbours are replaced by
     the neighbour mean. 191 days differ from the canonical value
     (max robust adj: train96 286, train47 458, test 314).
  The ~8-year record shows the tail problem is real, not flare noise: the
  test-era days above 240 are mostly sustained multi-day runs (Aug 2024
  240→336 over 13 days; Oct/Nov/Dec 2024 runs of 6–11 days), and the
  model's 273 ceiling sits below all of them. Hypothesis for the v2 round:
  trees predicting a *level* cannot exceed their training leaves, and the
  sweep's min_child_samples=160 (chosen on the rising-phase val years)
  forbids leaves that isolate the ~1% of train days above 240 — hence the
  ratio-to-envelope target (E22–E26) as the main bet, then the combo (E30)
  and a residual-trajectory ensemble (E31/E32) as the probabilistic
  product the 2026-08-18 group poll asked about (no replies received as of
  today; the ensemble serves all three answers: its mean is the RMSE
  forecast, its members are structure-preserving scenarios, and CRPS /
  coverage / spike capture score the distribution).
- **2026-09-02 — v2 round scored (all on the corrected data; Athena fan-outs
  + local screens). Champion `F107_E33_combo_v2_raw`: pooled 28.09, r 0.726,
  bands +0.267/+0.359/+0.232, hi-act 31.97.** Verdicts, in order of
  importance:
  1. **Benchmark vs the operational product (new `benchmarks/swpc_27day.py`,
     SWPC weekly-PDF 27-day outlooks, 125 issues in the test period, 3,375
     matched (origin, target) pairs, our origin = issue − 1 day):** SWPC
     32.1 pooled / 25.1 L1–7 / 35.2 L8–14 / 33.7 L15–27 / 37.3 hi-act;
     E33 27.7 / 21.1 / 29.4 / 29.8 / 31.5. **−14% pooled, −16% hi-act**;
     the SWPC outlook is no better than the 81-day mean beyond a week.
  2. **A linear model of three baselines gets most of the way.** E51 without
     the geometric columns — OLS per lead on [flux(t), env81(t),
     recur27(t+h)] — scores **28.75**. The whole ML stack (two LightGBM
     spans × seeds, far-side features, blending, plus a GRU) reaches 28.09:
     **0.66 sfu (2.3%) over a 4-number-per-lead linear model.** The
     inputs, not the learner, are the limit.
  3. **Single-model screens (E20–E26):** level control 30.09; robust
     targets neutral (30.27); ratio-to-envelope target 30.02 pooled but the
     ceiling rises 246→280 sfu and the >240 bias −70→−52 at a mid-range
     cost (RMSE 150–180 bin 17.5→24.4); less regularization does nothing
     (30.15); deep-span ratio model 29.10 = best single LightGBM; envelope
     weights and persistence-ratio target neutral.
  4. **Combos:** E33 (level) 28.09 · E34 (gated, level+ratio bases) 28.15 ·
     E30 (ratio) 28.32 with ceiling 301 and >240 bias −48 but mid-range
     cost; E35 stack over all seven cells 28.10 (weights go to E34 +
     E51 + a little GRU) — nothing is additive beyond E33.
  5. **Neural family:** GRU (E40) 28.92 — best *single* model in the
     matrix, beating every single LightGBM (29.10–30.27); MLP 30.27. Early
     stopping at epochs 4–11 says it is capacity- not data-limited; in the
     stack it takes ≤0.28 weight at L1–7 and ~0 elsewhere.
  6. **Disk forward model (E50), the step-1 test of the geometry
     hypothesis:** standalone 35.58 — beats persistence (38.79) at every
     band but loses to 27-day recurrence (34.99). Fitted limb law
     mu^1.5, tau 40 d, far-side scale 1. In the stack it adds **0.00**
     (28.15 vs 28.14); in the linear model it adds 0.26 pooled, all at
     L1–7 (24.11→23.15). **Verdict: rotation geometry from sunspot AREA is
     information the trees already extract from the window; it is not the
     missing ingredient.** Whether magnetic flux (plage) would be is
     untested — step 2 (ADAPT maps) is now a bigger bet than the step-1
     framing implied, because the geometric machinery itself bought
     nothing.
  7. **Probabilistic product (E31/E32 on the stack):** static residual
     ensemble CRPS 15.32, 80%-band coverage 0.794 (2024 0.734, 2025 0.843,
     2026 0.814), width 63; adaptive (trailing-90-day rescale, s_t
     0.88–1.72) CRPS 15.62, coverage **0.883 (2024 0.850)**, width 80,
     catches 59% of ≥50 sfu rises inside q90 (static 38%). The ensemble
     mean reproduces the point forecast's RMSE exactly (28.10). Adaptive
     fixes the 2024 undercoverage that E14 could not; it over-covers in
     quiet years and could use a mild target below 80% when rescaling.
  Operational notes: Athena fan-outs (`run_e30/e33/e34.pbs`,
  `shared/run_cells.pbs`) all ran in 3–20 min per node after a ~15-min
  queue wait; the pull is `shared/sync_athena.sh pull`. Torch (CPU) is in
  the Athena venv now.
- **2026-09-02 — Mg II plage proxy (E60/E61): NULL.** Bremen composite Mg II
  index (LISIRD `bremen_composite_mgii`, daily 1978-11→yesterday, so
  realtime-viable; `scripts/build_proxy_daily.py` → `data/proxy_daily.csv`,
  joined as the optional `PROXY_COLS` family). E60 30.26 vs E21 30.27;
  E61 29.15 vs E24 29.10; band skills and the tail bins identical within
  noise. A disk-integrated plage measure adds nothing to a model that
  already sees the F10.7 history — the two are contemporaneous proxies of
  the same thing. Together with the E50 null (rotation geometry from
  sunspot areas), two of the three premises behind a magnetogram/ADAPT
  program have now failed on the test set; the only untested one is
  *spatially resolved* magnetic flux (which regions, where, and the far
  side, all as flux rather than area). Verdict: with scalar daily inputs
  the ceiling is genuine — a 3-baseline linear model (28.75) sits 0.66 sfu
  from the full stack (28.09); productionizing the point forecast +
  calibrated ensemble is the recommended next step, and a magnetic-map
  program is a research bet the group should choose deliberately, not a
  known lever.
