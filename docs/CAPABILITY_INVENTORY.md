# CAPABILITY_INVENTORY (verified 2026-10-28 22:15 IST)
| Capability | Status | Evidence |
|---|---|---|
| Python 3.12 + shell | Available, verified | scripts ran |
| Persistent venv /workspace/.venv | Available, verified | system site-packages were wiped once mid-session; venv on persistent disk survives |
| pandas/numpy/sklearn/LightGBM/CatBoost/pyarrow/pytest/scipy | Verified | requirements.lock.txt; pytest 2 passed |
| XGBoost, statsmodels | Not in venv yet (were in lost system env) | import fails; install only if an experiment needs them |
| Git | Verified | repo initialised in /workspace/competition, first commit |
| GPU | Unavailable | no nvidia-smi |
| CPU/RAM | 2 cores / ~7 GB | nproc, free |
| Web search / fetch / curl | Verified | Open-Meteo APIs fetched |
| Browser with saved Zindi login | Verified | 7 files downloaded on zindi.world |
| Scheduled execution | Verified | daily 10:00 task + 10 Nov 19:00 final check created |
| Background jobs / parallel subagents | Verified | used for experiment batches |
| Persistent memory + custom specialty | Verified | aryan-autonomous-protocol saved |
| Experiment tracking | Lightweight CSV (EXPERIMENT_LOG.csv) + parquet preds | MLflow not needed |
| Zindi submission | Requires Aryan's approval | protocol |
| AutoML (Optuna-auto search, AutoGluon, H2O, FLAML) | Prohibited | competition rules |
| Hark specialties relevant | aryan-autonomous-protocol, scheduled-tasks, sign-in, build-charts (optional plots) | catalog list |
