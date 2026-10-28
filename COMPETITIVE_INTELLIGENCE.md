# COMPETITIVE_INTELLIGENCE (facts vs hypotheses)

## Verified (public pages, 2026-10-28)
- 86 joined / 33 active at day 3. Starter notebook = Random Forest baseline (Zindi data page).
- Public LB = rolling 2-day window that changes on 31 Oct, 5 Nov, 9 Nov; private = 11–12 Nov, latest submission.

## Hypotheses (unverified, medium confidence)
- Many participants will fit the starter RF/GBM once and stop updating -> stale models when private window arrives. Edge: automated retrain on 10 Nov 09:00 data.
- Many will train on random splits / use reanalysis weather -> overfit or leakage. Edge: origin-based validation at 15–63 h horizons, historical-forecast weather.
- Few will model the post-Diwali regime explicitly. Edge: analog features from past Diwali periods in AdditionalHistory (if it covers them).
- Public LB chasing on a 2-day window is noisy; strong players will rely on CV.

## Experiments to test edges
1. Does weather forecast data reduce 15–63 h RMSE vs pure history?
2. Diwali-offset features on past years: gain on post-Diwali folds?
3. Shrink-to-climatology blend at long horizons.
