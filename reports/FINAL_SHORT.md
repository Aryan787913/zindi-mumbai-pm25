# FINAL (10 Nov 2026, 15:40 IST) - supersedes FINAL_PRIVATE_PLAN.md for the 10 Nov release
Data: Train.csv 10 Nov release (sha256 8adbcf55..., 48,400 rows, last obs 10 Nov 09:00 IST). Private 11-12 Nov = h16-63.
Hark's frozen pair assumed h97-192 (data ending 5-7 Nov); that assumption no longer holds.

Files (python src/final_short.py final1110):
- final1110_fade24.csv  level = m28 + (m24 - m28) * exp(-h/24), x Fourier-3 28-day hour shape. UPLOAD 1st, SELECT.
- final1110_fade48.csv  same with tau = 48 h.                                                  UPLOAD LAST, SELECT.
Private means (B/Malad/Byculla/Shivaji): fade24 51.9/73.7/41.5/66.9, fade48 52.7/87.7/41.0/74.6. Last obs 50/190/44/77.

Evidence (identical geometry: cutoff 09:00, targets h16-63):
| model | bt_short all (65 origins) | high-ratio rows | Diwali 2025 origins | bt_hist 5 seasons mean-case |
| dec24 | 29.87 | 32.64 | 30.93 | 35.3 |
| dec36 | 29.83 | 33.13 | 36.23 | 36.2 |
| dec48 | 30.00 | 34.31 | 41.26 | 37.4 |
| m28f3 | 30.71 | 35.12 | 29.62 | 37.2 |
| LightGBM (cal/nocal/avg) | 44-45 | 62-74 | 131-145 | n/a (blends worse) |
| HARK_PRIMARY | 33.82 | 44.45 | 79.30 | 59.0 |
Every LGB blend (w 0.2-0.5) is worse than the pure decay on all, high, stress and Diwali-2025 subsets.
Weather (Open-Meteo, user-supplied 10 Nov): 11-12 Nov dry, wind ~7 km/h, BLH 660-716 m vs ~880 m on 7-9 Nov -> slower fade plausible -> fade48 last.
Malad spike check: count = 4 quarter-hours, values vary (not flatline); past Malad days >120 fell to 20-110 next day (2025-10-21 167 -> 62).

Red team (10) and status:
1 Horizon mismatch (Hark h97-192 vs real h16-63) - FIXED: everything re-validated at h16-63.
2 LightGBM extrapolates spikes - TESTED: dropped from both files.
3 x1.35 November ramp - TESTED: rejected at this geometry (HARK_PRIMARY 79 on Diwali-2025).
4 Malad 200+ could be a glitch - CHECKED: valid readings; decay model does not carry it forward.
5 Only best-of-2 vs latest-only rule unknown - HANDLED: both files are good alone; last upload = fade48, both selected.
6 Selection after seeing results (mild optimism) - ACCEPTED: only 2 simple parameters (tau 24/48) chosen from a coarse grid; same ranking in 2 independent tests.
7 Weather not modelled - PARTIAL: used only as tie-breaker for upload order (no training on weather; earlier tests rejected raw weather).
8 Hour shape from 28 days may miss post-Diwali night pattern - OPEN (not tested; low impact vs level).
9 History stations differ from target stations - KNOWN LIMIT; 2025 and Train-based bt_short use the real 4 stations.
10 Reproducibility for code review - DONE: deterministic (no randomness in final files), independent recompute max diff 0.0.

## Weather test (10 Nov 15:50 IST, src/bt_weather.py, Open-Meteo historical-forecast archive Jan 2025-Oct 2026, 549 daily origins)
Correction of fade level by forecast BLH / wind / rain anomalies: out-of-sample gain <=0.07 RMSE, improves <60% of months,
and makes Oct/Nov WORSE (54.10 -> 54.29). REJECTED. 2026 forecast (BLH ~22% below prior days) would have added only ~+7%.
Weather stays a tie-breaker only (fade48 = latest upload).

## v2 (10 Nov 15:55 IST) - RED TEAM 2: pair chosen for BEST-OF-2, supersedes v1 (fade24 + fade48)
Finding: v1 files were too similar (fail together). Best-of-2 search over 171 pairs on 3 test sets:
| pair | ordinary (65) | 5 post-Diwali seasons | Diwali/Nov stress |
| v1 fade24 + fade48 | 19.53 | 33.69 | 51.58 |
| v2 fade12 + slow72 x1.1 | 19.25 | 31.85 | 49.40 |
Top-12 pairs are all the same family (tau 12-18 + tau 48-72 x1.1-1.2) -> robust, not a lucky pick.
Files: final1110v2_slow72x11.csv (upload 1st, select), final1110v2_fade12.csv (upload LAST, select; best single on both Diwali tests).
Private means B/Malad/Byculla/Shivaji: fade12 51.3/63.8/41.8/61.6; slow72x11 58.4/105.1/44.8/86.9.
Independent recompute: max diff 0.0 for both. Format checks pass.
