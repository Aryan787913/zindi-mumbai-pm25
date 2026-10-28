# NEXT_ACTIONS (ranked, after red team + LB feedback 2026-10-28)
0. Level calibration: public 22-23 Oct showed our level too HIGH (x1.1 hurt by 3 RMSE). Test robust level estimators (median/trimmed 7d, exp-decay, shrink toward station-month climatology) at h1-48 and h97+. Use LB only as sanity check.
1. (DONE E009) hmax=192.
2. Remove doy calendar dependence; add season-level + post-Diwali analog multiplier from 2018–2020/2024 history (RED_TEAM #2).
3. Weather retry as anomalies (deviation from trailing 30-day mean) and rain/wind only.
4. Cross-station features from Other_Mumbai_Stations_2025_2026 (Sewri etc.).
5. CatBoost comparison + 3-seed averaging; per-horizon blend weight.
6. Retrain on 31 Oct update; candidate for public window 27–28 Oct.
