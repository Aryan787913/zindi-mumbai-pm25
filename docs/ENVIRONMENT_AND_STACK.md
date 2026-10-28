# ENVIRONMENT, ML STACK, WORKFLOW
- Hardware: 2 CPU, 7 GB RAM, no GPU, 1 PB JuiceFS workspace. LightGBM num_threads=2. Full blocked-CV run ≈ 35-45 s.
- Stack (pinned, requirements.lock.txt): pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, lightgbm 4.7.0, catboost 1.2.10, pyarrow 26.0.0, scipy 1.18.1, pytest 9.2.0.
- Deliberately NOT added: MLflow, Polars, deep TS libraries (data ~47k rows; overhead > value), any AutoML.
- Workflow: Zindi download (browser) → data/raw + SHA manifest → src/data.py (hourly grid, >1000 masked) → src/features.py (origin-based trailing features, optional weather) → src/validate.py (rolling origins, 3 folds, horizon buckets) → src/train_lgb.py / baselines.py → EXPERIMENT_LOG.csv → src/predict.py (asserts format) → submissions/ + SUBMISSION_LEDGER.csv → Aryan approves → upload.
- Tests: tests/test_leakage.py (future-corruption invariance, RMSE correctness).
- Gaps: no git remote/backup; no independent second-implementation check of metric beyond unit test; XGBoost absent.
