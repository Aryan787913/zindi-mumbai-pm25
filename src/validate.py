"""Rolling-origin validation framework.
Origin T = timestamp (hour end) of the last available observation. Forecast T+h, h=1..H.
Only data with ts <= T is visible. Horizon buckets: 1-48 (public-like), 49-96, 97-144 (private-like,
since source data arrive ~72-115 h late, the final 10 Nov 09:00 data likely end ~6-7 Nov)."""
import numpy as np, pandas as pd
from data import STATIONS
from metrics import rmse
H = 192
BUCKETS = [(1, 48), (49, 96), (97, 144), (145, 192)]

def origins(wide, start, end, step_h=72, hour=0):
    idx = wide.index
    o = idx[(idx >= pd.Timestamp(start, tz='Asia/Kolkata')) & (idx <= pd.Timestamp(end, tz='Asia/Kolkata'))]
    o = o[o.hour == hour]
    o = o[::max(1, step_h // 24)]
    return [t for t in o if t + pd.Timedelta(hours=H) <= idx[-1]]

FOLDS = {  # season-matched + recent
    'autumn25': ('2025-10-01', '2025-12-31'),
    'winter26': ('2026-01-01', '2026-02-28'),
    'recent26': ('2026-08-15', '2026-10-15'),
}

def targets(wide, T):
    fut = wide.loc[T + pd.Timedelta(hours=1): T + pd.Timedelta(hours=H)]
    rows = []
    for s in STATIONS:
        for i, (ts, v) in enumerate(fut[s].items(), 1):
            rows.append((T, s, i, ts, v))
    return pd.DataFrame(rows, columns=['origin', 'station', 'h', 'ts', 'y'])

def evaluate(pred_df, name=''):
    """pred_df: columns origin, station, h, ts, y, p, fold"""
    d = pred_df.dropna(subset=['y'])
    out = {'name': name, 'all': rmse(d.y, d.p)}
    for a, b in BUCKETS:
        out[f'h{a}-{b}'] = rmse(d[(d.h >= a) & (d.h <= b)].y, d[(d.h >= a) & (d.h <= b)].p)
    for f, g in d.groupby('fold'):
        out[f] = rmse(g.y, g.p)
        g2 = g[g.h >= 97]; out[f + '_h97+'] = rmse(g2.y, g2.p)
    return out
