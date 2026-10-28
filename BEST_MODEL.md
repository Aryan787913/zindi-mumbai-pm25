# BEST_MODEL (2026-10-28 22:20 IST)
blend_v2 = 0.5 * LightGBM direct (h 1-192, raw target, seed 42, 600 rounds) + 0.5 * lvl7xshape28 baseline.
Validation: all 38.35 | h1-48 35.08 | h97-144 38.75 | h145-192 38.69 (best baseline there 41.51-41.86).
File: submissions/submission_003_blend_v2_h192.csv sha256 0edeee4f… (candidate, NOT submitted).

## SUBMISSION GATE (Aryan wants every try to move the rank up; adopted 2026-10-28)
Submit a new file only if ALL hold: (1) beats current best offline by >=0.3 RMSE on BOTH h1-48 and h97-192, overall and in recent26 fold; (2) no tuning on public LB windows; (3) no blanket post-hoc multipliers (E010 lesson); (4) format asserts pass. Data-update retrains of the current best are always allowed (fresher data). Otherwise keep improving offline.
