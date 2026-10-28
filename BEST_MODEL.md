# BEST_MODEL (2026-10-28 22:00 IST)
blend_v1 = 0.5 * LightGBM direct multi-horizon (raw target, all features, 600 rounds, seed 42) + 0.5 * lvl7xshape28 (7-day station mean × 28-day hour-of-day shape). h>144: baseline only.
Validation (rolling origin, 72 origins, values >1000 masked): all 38.19 | h1-48 34.92 | h49-96 40.58 | h97-144 38.77.
Best simple baseline h97-144: 40.89 (mean7d). Gain ≈ 5%.
Weakness: autumn25 fold RMSE ~54, dominated by Shivaji Nagar spikes (to 929 µg/m³) around Diwali 2025.
Files: submissions/submission_002_blend_v1.csv (sha256 9dc6ed3e…), submission_001 baseline.
