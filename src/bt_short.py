"""Backtest at the REAL final geometry: cutoff T = 09:00, target = hour-ends T+16 .. T+63 (11-12 Nov from 10 Nov 09:00).
Everything trained/fitted on data <= T only. Saves per-row predictions to reports/bt_short.parquet."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
import features
from data import load_train, STATIONS
from features import build, feat_cols
from train_lgb import params
from final_submit import f3_shape

H0, H1, HM = 16, 63, 63
CAL = ['doy_s', 'doy_c', 'diwali_off']
wide, _ = load_train()
step = int(sys.argv[1]) if len(sys.argv) > 1 else 9
origins = list(pd.date_range('2025-06-02 09:00', '2026-11-07 09:00', freq=f'{step}D', tz='Asia/Kolkata'))
extra = ['2025-10-21 09:00', '2025-10-22 09:00', '2025-10-23 09:00', '2025-11-05 09:00', '2025-11-12 09:00',
         '2026-11-05 09:00', '2026-11-06 09:00', '2026-11-07 09:00']
origins = sorted(set(origins) | {pd.Timestamp(x, tz='Asia/Kolkata') for x in extra})

rows = []
for T in origins:
    t0 = time.time()
    W = wide.loc[:T]
    idx = W.index
    tro = idx[(idx.hour % 12 == 9) & (idx >= idx[0] + pd.Timedelta(days=30)) & (idx <= T - pd.Timedelta(hours=HM))]
    X = build(W, tro, hmax=HM); X = X[X.y.notna()]
    Xt = build(W, pd.DatetimeIndex([T]), with_target=False, hmax=HM)
    Xt = Xt[(Xt.h >= H0) & (Xt.h <= H1)].copy()
    for nm, drop in (('lgb_cal', ()), ('lgb_nocal', CAL)):
        fc = feat_cols(X, drop)
        m = lgb.train(params(), lgb.Dataset(X[fc], X.y.values, categorical_feature=['station']), num_boost_round=600)
        Xt[nm] = m.predict(Xt[fc])
    Xt['location_id'] = [STATIONS[i] for i in Xt.station]
    for s in STATIONS:
        x = W[s]; h28 = x.iloc[-672:]
        shp = f3_shape(h28.groupby(h28.index.hour).mean().reindex(range(24)).values)
        lv = {'m28': h28.mean(), 'm7': x.iloc[-168:].mean(), 'm72': x.iloc[-72:].mean(), 'm24': x.iloc[-24:].mean()}
        last = x.dropna().iloc[-1]
        r = Xt.location_id == s
        hr = Xt.loc[r, 'ts'].dt.hour.values; hh = Xt.loc[r, 'h'].values
        for k, v in lv.items():
            Xt.loc[r, f'{k}f3'] = (v if np.isfinite(v) else lv['m28']) * shp[hr]
        for tau in (24, 48):
            lvl = lv['m28'] + (lv['m24'] - lv['m28']) * np.exp(-hh / tau)
            Xt.loc[r, f'dec{tau}f3'] = lvl * shp[hr]
        Xt.loc[r, 'persist'] = last
    y = wide.stack(future_stack=True).rename('y').reset_index(); y.columns = ['ts', 'location_id', 'y']
    Xt = Xt.merge(y, on=['ts', 'location_id'], how='left')
    Xt['origin'] = T
    rows.append(Xt[['origin', 'ts', 'h', 'location_id', 'y', 'lgb_cal', 'lgb_nocal', 'persist'] + [c for c in Xt.columns if c.endswith('f3')]])
    print(T, len(X), f'{time.time() - t0:.0f}s', flush=True)
P = pd.concat(rows, ignore_index=True)
os.makedirs('reports', exist_ok=True); P.to_parquet('reports/bt_short.parquet')
print('saved', len(P))
