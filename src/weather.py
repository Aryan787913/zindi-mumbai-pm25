"""Open-Meteo previous-runs API, forecasts issued 5 days before valid time (no look-ahead for a 4-6 day horizon).
Source listed on ClimateVerse challenge page (see RULES_AUDIT C3)."""
import json, os, numpy as np, pandas as pd
P = os.path.join(os.path.dirname(__file__), '..', 'data', 'external', 'openmeteo_prev5.json')
ORDER = [11611, 3409511, 11606, 3409510]  # request order of coordinates
def load_weather(path=P):
    d = json.load(open(path)); d = d if isinstance(d, list) else [d]
    out = {}
    for loc, blk in zip(ORDER, d):
        h = pd.DataFrame(blk['hourly'])
        h.columns = [c.replace('_previous_day5', '') for c in h.columns]
        h['ts'] = pd.to_datetime(h.time).dt.tz_localize('Asia/Kolkata') + pd.Timedelta(hours=1)  # align to hour-end
        h = h.drop(columns='time').set_index('ts')
        ws, wd = h.wind_speed_10m, np.deg2rad(h.wind_direction_10m)
        h['wu'] = -ws * np.sin(wd); h['wv'] = -ws * np.cos(wd)
        h['rain24'] = h.precipitation.rolling(24, min_periods=1).sum()

        h['ws24'] = ws.rolling(24, min_periods=1).mean()
        h = h.drop(columns=['wind_direction_10m', 'boundary_layer_height'], errors='ignore')
        out[loc] = h.add_prefix('w_')
    return out
