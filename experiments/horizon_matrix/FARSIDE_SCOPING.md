# Far-side / multi-vantage scoping (2026-08-16)

*Motivation: E04 proved the Earth-view features are exhausted — long-lead
skill (L15–30, the weakest band) is limited by what happens on the far side
before it rotates into view, which is exactly what breaks 27-day recurrence.
Idea sparked by MSWIM2D's use of STEREO-A / Solar Orbiter as off-Earth-line
inner-boundary inputs. All URLs below fetch-verified 2026-08-16/17.*

## Verdict

**Two viable data tracks, one clear priority.**

1. **E10 — GONG helioseismic far-side AR catalog (`f6x`)** — do this first.
   Ready-made, parseable, whole-far-side, realtime, zero image processing.
2. **E11 — STEREO-A EUVI disk-integrated EUV flux** — second round.
   Physically closer to F10.7 (EUV brightness vs seismic proxy) but needs a
   4–30 GB image download + a DN-summing pipeline, and only sees a slice of
   the far side (vantage drifts +22°/yr).
3. **Solar Orbiter EUI — parked.** Science data lags 2–8 months (bulk
   releases); only fast product is twice-daily 8-bit quicklook JP2s that
   would need DIY photometry. Geometry is good (>30° off Earth–Sun line
   ~62% of days since 2022); revisit only if E10/E11 pay.
4. **Multi-vantage in-situ solar wind — not pursued.** That's the data
   family MIDL already falsified (MATRIX.md E02/E03); a different longitude
   doesn't change the physics objection.

## Literature support (why this isn't a coin flip)

- **Henney et al. 2012** (ADAPT flux-transport → F10.7; now operational as
  AFRL/NSO **SIFT**): r ≈ 0.97/0.95/0.93 at 1/3/7 d, **near-side only**;
  the 2022 decadal white paper (Jain et al., arXiv:2210.01291) states
  far-side input both improves values and **extends skill beyond 7 days**.
- **Lei et al. 2019** (full-Sun 304 Å maps from STEREO A+B + SDO, rotated
  forward → F10.7 at 1–27 d): **~12.5 % error reduction vs a 54-order AR
  model at rotation-scale leads** — the best available anchor for expected
  gain. Relied on 2011–14 two-STEREO geometry; modern replication needs
  helioseismic far-side (→ E10).
- **No published ML F10.7 model uses far-side features** (2018–2026 survey:
  all time-series-only; the one multimodal paper, Zhao 2025 Mamba, stops at
  3-d leads with ~1–3 % gains). Open niche.
- **SWPC/USAF 45-day product is pure 27-day recurrence beyond day 7** — the
  operational bar at long leads is exactly our C01 baseline.
- Expected shape of the win: concentrated at **~7–20 d leads** and
  rising/max phases (flags newly emerged far-side flux; drops decayed flux
  that recurrence falsely carries over). Plausible 5–15 % RMSE in-band.
- Calibration caveat (Liewer et al. 2017): 95 % of strong seismic far-side
  detections are real regions, but seismic strength is only weakly
  correlated with EUV magnitude → treat `fs_*` as presence/size signals,
  weight by detection probability, don't over-trust amplitude.

## Track 1 (E10): GONG far-side AR catalog `f6x`

- **URL pattern** (HTTPS, tiny whitespace-delimited text, 2 files/day):
  `https://gong2.nso.edu/oQR/f6x/YYYYMM/mrf6xYYMMDD/mrf6xYYMMDDtHHMM.txt`
  (HH = 0000/1200). Verified **2010-01 → present**, latency **< 24 h**.
- **Per region**: Carrington lon/lat, phase strength, detection probability
  (%), effective area, estimated magnetic flux, magnetic max, **predicted
  date-of-return to front side**, NOAA cross-ID when it matches a returning
  region.
- Maps are built from 24 h of GONG Dopplergrams every 12 h; maps with
  network duty cycle < 0.8 are **discarded** → real gaps to handle (treat
  missing day as NaN, not zero). Files can be recomputed within 48 h of
  first posting — minor look-ahead risk for training files; document, don't
  fight.
- Portal: https://farside.nso.edu/ (new calibrated pipeline; legacy
  pipeline retires 2027-01-01 — build the parser against the new one).
- **Second, independent catalog for cross-check / second family**:
  Stanford/JSOC HMI SARD lists,
  `http://jsoc.stanford.edu/data/farside/AR_Lists/` (plain **HTTP only**,
  HTTPS 403s), verified 2010-04-25 → present, 12 h cadence, includes
  **ETA-at-east-limb** column (ready-made lead-time feature). Independent
  instrument (HMI vs GONG) *and* detector.
- **Offline validation set**: Hamada et al. 2024 catalog (Harvard Dataverse
  doi:10.7910/DVN/8ZKY8Z), GONG seismic vs ground-truth STEREO+SDO 304 Å
  regions, 2010-05→2016-05 — use to sanity-check strength→size mapping,
  never as a production input.
- **Archive floor is 2010** (f6x). GONG `fqo` FITS phase maps extend to
  2006-05 if we ever want to compute our own index, but 2007–09 is deep
  minimum — essentially nothing on the far side anyway. `fs_*` = NaN
  pre-2010, the exact optional-NaN pattern E05 validated.

### Proposed feature family (`fs_*`, daily, from the t0000 + t1200 files)

Aggregates analogous to the SRS `ar_*` family:
- `fs_n` — region count (prob-thresholded, e.g. ≥ 60 %)
- `fs_area_sum`, `fs_area_max` — effective area
- `fs_flux_sum`, `fs_flux_max` — estimated magnetic flux
- `fs_strength_sum` — phase strength (prob-weighted)
- `fs_ret7_flux`, `fs_ret14_flux` — flux due to return within 7 / 14 days
  (from the date-of-return column — this encodes the lead alignment for
  the trees directly)
- `fs_lon_centroid` — flux-weighted Carrington distance from east limb
- (optional second family `sard_*` from JSOC: count, strength sum, min
  days-from-east-limb)

### Cell design

- `F107_E10_farside` = E00 control + `fs_*` columns in the same 60-day
  windowing, NaN pre-2010, train span unchanged (1996–2021). One change vs
  control, per matrix rules; canonical 882-origin test set unchanged (fs
  valid everywhere 2024→).
- Fold in seed-ensemble (E07, "tiny but free") only if run as an E06-style
  combination later — keep the screening cell pure.
- **Success criterion**: pooled RMSE ≥ E00 (guard), skill gain concentrated
  in L8–14 / L15–30; watch the hi-act bin. If fs_* moves L15–30 at all,
  it's the first feature family ever to do so in this matrix.

### Operational lesson (2026-08-16)

GONG's WAF **rate-limits hard**: a 12-thread crawl got the IP 403-blocked
(site-wide, both gong2 and nispdata mirrors) after ~2k requests. The block
outlasted several minutes; `fetch_farside.py` now fetches GONG serially at
≤ 1 req/s with long backoffs — do not speed it up. JSOC tolerated a small
thread pool fine.

### Build steps

1. `scripts/fetch_farside.py` — idempotent crawl of f6x (2 files/day,
   2010-01→present; ~12k tiny files) + JSOC AR_Lists → `data/farside/`.
2. `scripts/build_farside_daily.py` → `data/farside_daily.csv`
   (one row/day, `fs_*` + `sard_*`, NaN where maps missing/discarded).
   Follow the sw_daily.csv pattern (separate file, joined in-cell).
3. Cell folder diffing E00 (copy E05's optional-NaN handling).
4. Realtime: same URL patterns serve today's files sub-day — slots directly
   into the future realtime path with the same parser.

## Track 2 (E11): STEREO-A EUVI disk-integrated flux

- **No ready-made series exists anywhere** (LISIRD, CDAWeb, NCEI, ROB, SSC
  all checked — images only). Must be computed from images.
- **Key find**: the SSC *beacon* FITS tree is full-mission, not just
  realtime — `https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/secchi/img/euvi/YYYYMMDD/`
  spans **2007-02-05 → today** (6,972 day-dirs), 512×512 decompressed ICER
  FITS, ~0.55 MB each, ~130/day now (195 Å every ~10 min, 304 Å ~30 min).
  Usable headers: `WAVELNTH`, `EXPTIME`, `DATE-OBS`, `NMISSING`.
- **Series recipe**: daily median of DN-sum ÷ `EXPTIME` over `NMISSING==0`
  frames, per passband. 4–8 frames/day ≈ **15–30 GB one-time download**;
  1/day ≈ 4 GB. Same tree updates ~1–2 h behind realtime → history and
  operational feed are one homogeneous product.
- Gaps: one big one, **2015-03-19 → 2015-07-11** (superior conjunction);
  thin (~7 files/day) Aug 2014–Nov 2015 sidelobe era. STEREO-B lost 2014 —
  Ahead only.
- Calibration: EUVI has no published long-term degradation series (CMAD
  says vents were added to avoid EIT-style decay, drift believed small;
  EUVI↔AIA cross-cal due "later in 2026"). For an ML feature, normalize
  against a slow baseline or let the trees absorb drift. Do **not** use the
  wavelet-enhanced browse JPEGs (non-photometric).
- **Geometry features required**: Earth–STEREO-A separation angle (feature,
  not constant — drifts +22°/yr). Sources: JPL Horizons API
  (`COMMAND='-234'`, `CENTER='500@10'`, `QUANTITIES='31'`, minus Earth
  `399`) — verified working 2007→present+predicts — or the MSWIM2D hourly
  HGI trajectory files on solsticedisk
  (`/data/tuija/cdimarco/MSWIM2D/spice/trajectories/{STEREO-A,Earth}.dat`,
  2006-10→present, refreshed monthly via SSC kernels). Cross-checked
  2026-08-16: **STEREO-A is 67° ahead of Earth** (Horizons + STEREO SSC
  agree) → currently previews the disk Earth faces in ~22 days; preview
  lead shrinks ~1.8 d/yr.
- Feature shape: `stereo_euv195`, `stereo_euv304` daily series + separation
  angle; optionally a phase-aligned variant (flux lagged by
  days-until-vantage-longitude-faces-Earth). Screen raw first, engineer
  later.

## Reusable scaffolding (MSWIM2D repo, solsticedisk)

- `Production_Scripts/data_download/` — SPDF/CDAWeb fetch patterns.
- `Production_Scripts/data_download/refresh_stereo_kernel.py` — solves
  STEREO-A ephemeris currency (NAIF frozen at 2018; merges SSC deliveries).
- `spice/trajectories/*.dat` — hourly HGI positions, all bodies, kept
  current by the monthly cron.

## Open questions for the owner

- Priority order confirmed as E10 → E11? (E10 is ~a day of work; E11 adds
  a multi-GB fetch + FITS pipeline.)
- JSOC SARD is plain-HTTP only — acceptable for fetch_raw.sh? (We already
  tolerate plain FTP for the SWPC warehouse.)
