# RULES_AUDIT — Google Mumbai Data Challenge (Zindi)

Audited: 2026-10-28 21:30 IST. Sources fetched live:
- [INFO] https://zindi.africa/competitions/google-mumbai-data-challenge (overview + "Full Challenge Rules")
- [DATA] https://zindi.africa/competitions/google-mumbai-data-challenge/data
- [CV]   https://india.climateverse.net/dataverse/google-air-quality-forecasting-challenge
- Discussions page: not readable without sign-in (returned the info page). **Unchecked.**

## 1. Task & evaluation
| Rule | Source | Notes |
|---|---|---|
| Predict hourly PM2.5 (µg/m³) at **4** Mumbai CPCB CAAQMS stations | DATA | |
| Metric: RMSE = sqrt(mean((y - ŷ)²)), lower is better | INFO | Exact implementation not published; assume standard RMSE over all scored rows. |
| Test.csv / SampleSubmission.csv: **2,592 fixed IDs**, fixed row order, columns `ID, Target` | DATA | 2,592 = 4 stations × 648 h = 27 days. To confirm from file. |
| Target must be **non-negative** for every ID | DATA | Clip at 0. |
| Train.csv updated 31 Oct, 5 Nov, 9 Nov, final **10 Nov 09:00 IST** | DATA | |
| Public LB windows: 22–23 Oct (to 31 Oct) → 27–28 Oct → 1–2 Nov → 5–6 Nov; only current window scores | DATA | Public LB is a rolling 2-day window, not a fixed % split. |
| **Private = 11–12 Nov 2026**, scored on "latest valid submission" | INFO | Effectively a ~15–63 h ahead forecast from the 10 Nov 09:00 data. |
| Final deadline **10 Nov 2026 23:59:59 IST** | INFO | ~15 h between final data drop and deadline. |
| Dates/times in IST (UTC+05:30) | DATA | |

## 2. Contradictions
| # | Topic | Statement A | Statement B | Conservative interpretation | To resolve |
|---|---|---|---|---|---|
| C1 | Total submissions | Overview: **140 overall** | Full rules: **300** | Cap at **140 total, 10/day** | Ask on discussion board (needs user approval) |
| C2 | Public/private split | Overview: 30% / 70% | Full rules: 20% / 80%; DATA page: rolling 2-day windows | Treat public LB as a small, shifting sample; DATA page describes the actual mechanism | None needed for strategy |
| C3 | External data | Overview: core data **+ ClimateVerse challenge-page datasets** allowed | Full rules: "You may use only the datasets provided for this challenge" | Use only Zindi files + datasets listed on the ClimateVerse challenge page; nothing else. Treat ClimateVerse as "provided" since the overview names it explicitly. Keep a no-external-data model as a fallback. | Ask organisers to confirm |
| C4 | Which submission is scored privately | Overview: "**latest valid submission** will be evaluated on the private period" | Full rules: choose 2 submissions; else 2 best public | Make the **final 10 Nov post-09:00 retrained submission both the latest AND one of the 2 selected**. | Ask organisers |
| C5 | Who can compete | Overview: residents of India | Full rules: "open to all" | India residents; prizes Maharashtra only | — |

## 3. Other binding rules
- Open-source languages/tools only; no paid services or credit-card free trials.
- **No AutoML.** Manual model design only (LightGBM/XGBoost/CatBoost/sklearn are fine as libraries).
- Pretrained models allowed only if openly available.
- Fixed seeds; code must reproduce the LB score. No custom packages.
- Top 10: code within 48 h. Zindi can request code anytime (24 h).
- One account per person; no private code sharing outside a team; team max 4; no new members in final 5 days (from ~5 Nov).
- Top 3 assign copyright of solution to Zindi.
- Data licence CC-BY-SA 4.0.
- No leakage: external data must not include info unavailable at prediction time (e.g. reanalysis weather for future hours → use *forecast* products for the private window).

## 4. Implications for strategy
1. Private window starts ~15 h after the final data drop → horizons 15–63 h. Validation should mimic **"train up to T, predict T+15h … T+63h"**.
2. Retrain + resubmit on **10 Nov after 09:00 IST** is mandatory. Pipeline must run end-to-end in minutes.
3. **Diwali: Lakshmi Puja Sun 8 Nov 2026; Bhai Dooj 11 Nov** (drikpanchang.com). Private window sits right after the festival — post-Diwali fireworks/stagnation effects matter. Check past Diwali periods in AdditionalHistory.
4. Weather predictors (Open-Meteo is listed on ClimateVerse) must use **historical forecast** archives for training to match what's available at prediction time.

## 5. Actions needing user authorisation
- Posting C1/C3/C4 clarifications on the discussion board.
- Any submission to Zindi.
