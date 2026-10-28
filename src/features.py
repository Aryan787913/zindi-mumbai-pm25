"""Origin-based direct multi-horizon features. All features use data with ts <= T only (trailing windows)."""
import numpy as np, pandas as pd
from data import STATIONS
H = 192
DIWALI = {2025: '2025-10-20', 2026: '2026-11-08'}  # Lakshmi Puja (drikpanchang)

def origin_table(wide):
    """Per (T, station) trailing statistics, computed for every hour T."""
    feats = {}
    city = wide.mean(axis=1)
    for s in STATIONS:
        x = wide[s]
        f = pd.DataFrame(index=wide.index)
        f['last'] = x.ffill()
        f['age_last'] = (x.notna().cumsum().groupby(x.notna().cumsum()).cumcount()).astype(float)
        for w in [3, 6, 12, 24, 72, 168, 336, 720]:
            f[f'm{w}'] = x.rolling(w, min_periods=max(1, w // 4)).mean()
        f['s24'] = x.rolling(24, min_periods=6).std()
        f['mx24'] = x.rolling(24, min_periods=6).max()
        f['q90_168'] = x.rolling(168, min_periods=40).quantile(0.9)
        f['miss24'] = x.isna().rolling(24).mean()
        f['trend'] = f['m24'] / (f['m168'] + 1)
        f['city24'] = city.rolling(24, min_periods=6).mean()
        f['city168'] = city.rolling(168, min_periods=40).mean()
        f['rel168'] = f['m168'] / (f['city168'] + 1)
        feats[s] = f
    return feats

def hour_profiles(wide, days):
    """For each T and each hour-of-day k: trailing mean of obs at hour k over last `days` days. Returns dict s-> array [len, 24]"""
    out = {}
    hrs = wide.index.hour.values
    for s in STATIONS:
        x = wide[s].values
        arr = np.full((len(x), 24), np.nan)
        for k in range(24):
            xs = pd.Series(np.where(hrs == k, x, np.nan), index=wide.index)
            arr[:, k] = xs.rolling(days * 24, min_periods=24).mean().values
        out[s] = arr
    return out

def build(wide, origin_idx, with_target=True, hmax=H, weather=None):
    ot = origin_table(wide)
    p7 = hour_profiles(wide, 7); p28 = hour_profiles(wide, 28)
    pos = {t: i for i, t in enumerate(wide.index)}
    hs = np.arange(1, hmax + 1)
    rows = []
    for s_i, s in enumerate(STATIONS):
        f = ot[s].loc[origin_idx]
        ii = np.array([pos[t] for t in origin_idx])
        n = len(origin_idx)
        rep = np.repeat(np.arange(n), hmax)
        hh = np.tile(hs, n)
        D = f.iloc[rep].reset_index(drop=True)
        T = pd.DatetimeIndex(origin_idx)[rep]
        ts = T + pd.to_timedelta(hh, unit='h')
        thr = ts.hour.values
        D['origin'] = T; D['ts'] = ts; D['station'] = s_i; D['h'] = hh
        D['hour'] = thr; D['dow'] = ts.dayofweek.values
        doy = ts.dayofyear.values
        D['doy_s'] = np.sin(2 * np.pi * doy / 365.25); D['doy_c'] = np.cos(2 * np.pi * doy / 365.25)
        D['prof7'] = p7[s][ii[rep], thr]; D['prof28'] = p28[s][ii[rep], thr]
        D['shape28'] = D['prof28'] / (np.nanmean(p28[s][ii[rep]], axis=1) + 1)
        D['lvl_x_shape'] = D['m168'] * D['shape28']
        dd = np.full(len(D), np.nan)
        for y, d in DIWALI.items():
            off = (ts.normalize().tz_localize(None) - pd.Timestamp(d)).days.values
            dd = np.where(np.abs(off) <= 20, off, dd)
        D['diwali_off'] = dd
        if with_target:
            tpos = ii[rep] + hh
            ok = tpos < len(wide)
            y = np.full(len(D), np.nan); y[ok] = wide[s].values[tpos[ok]]
            D['y'] = y
        if weather is not None:
            W = weather[s]
            D = D.join(W, on='ts')
        rows.append(D)
    X = pd.concat(rows, ignore_index=True)
    return X

FEATS = None
def feat_cols(X, drop=()):
    return [c for c in X.columns if c not in ('origin', 'ts', 'y', 'fold') and c not in drop]
