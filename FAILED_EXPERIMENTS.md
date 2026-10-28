# FAILED_EXPERIMENTS
- E002 dropping calendar feats: worse (41.87 vs 39.26 h97+).
- E004 log1p target: worse everywhere (44.5 h97+) — RMSE rewards predicting spikes in mean, log shrinks.
- E003 ratio target: NaN predictions in winter fold (m168 missing) — bug, rerun after fix.
- E007/E008 Open-Meteo 5-day-lead weather as raw levels: worse (h97+ 40.5–40.7 vs 39.3; recent fold 12.8→15.6–20.9). Retry as anomalies / rain+wind only.
