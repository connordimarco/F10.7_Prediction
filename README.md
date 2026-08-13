# F10.7_Prediction

Forecast daily F10.7 solar radio flux 1–30 days ahead, for satellite drag.

Inputs: 60-day histories of F10.7, sunspot number, and SWPC active-region
summaries. 

- `scripts/` — build `data/daily.csv` from the raw archives (`data/SOURCES.md`)
- `experiments/horizon_matrix/` — one-change-per-cell experiment harness;
  design and verdicts in `MATRIX.md`, results in `LEADERBOARD.