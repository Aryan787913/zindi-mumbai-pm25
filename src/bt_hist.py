"""Post-Diwali backtest at the REAL final geometry on every usable past season (2018, 2019, 2021, 2024, 2025).
Cutoff T = Diwali + d days at 09:00 (d = 1, 2, 3; 2026 actual = D+2), targets = hour-ends T+16..T+63.
LightGBM is retrained per season on that season's own stations, data <= T only (same features/params as final).
Usage: python src/bt_hist.py <AdditionalHistory dir>  -> reports/bt_hist.parquet"""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
import features
from features import build, feat_cols
from train_lgb import params
from final_submit import f3_shape
from data import load_train

AH = sys.argv[1]
DIW = {2018: '2018-11-07', 2019: '2019-10-27', 2021: '2021-11-04', 2024: '2024-10-31', 2025: '2025-10-20'}
CAL = ['doy_s', 'doy_c', 'diwali_off']
HM, H0, H1 = 63, 16, 63


def wide_hist(year):
    if year in (2018, 2019):
        a = pd.read_csv(os.path.join(AH, 'History_Mumbai_2018_2020.csv.gz'), usecols=['Datetime', 'StationName', 'PM2.5'])
        a['ts'] = pd.to_datetime(a.Datetime) + pd.Timedelta(hours=1)
        w = a.pivot_table(index='ts', columns='StationName', values='PM2.5')
    elif year == 2021:
        b = pd.read_csv(os.path.join(AH, 'Legacy_Observations_2020_2022.csv.gz'), usecols=['location_id', 'datetime', 'value'])
        b['ts'] = pd.to_datetime(b.datetime.str[:19]).dt.ceil('h')
        w = b.pivot_table(index='ts', columns='location_id', values='value')
    elif year == 2024:
        b = pd.read_csv(os.path.join(AH, 'History_Mumbai_2024.csv.gz'), usecols=['station_name', 'timestamp_ist', 'value'])
        b['ts'] = pd.to_datetime(b.timestamp_ist.str[:19]).dt.ceil('h')
        w = b.pivot_table(index='ts', columns='station_name', values='value')
    else:
        w, _ = load_train(); w.index = w.index.tz_localize(None)
    w = w.where((w > 0) & (w <= 1000)).asfreq('h')
    w.index = w.index.tz_localize('Asia/Kolkata')
    return w


features.DIWALI = {int(k): v for k, v in {**{y: d for y, d in DIW.items()}, 2026: '2026-11-08'}.items()}
rows = []
for y, d in DIW.items():
    Wall = wide_hist(y)
    for dd in (1, 2, 3):
        T = pd.Timestamp(d, tz='Asia/Kolkata') + pd.Timedelta(days=dd, hours=9)
        h28 = Wall.loc[T - pd.Timedelta(hours=671):T]; fut = Wall.loc[T + pd.Timedelta(hours=H0):T + pd.Timedelta(hours=H1)]
        st = [s for s in Wall.columns if h28[s].notna().mean() > .6 and fut[s].notna().mean() > .5 and h28[s].iloc[-24:].notna().sum() >= 12]
        if not st:
            print(y, dd, 'no stations'); continue
        W = Wall.loc[:T, st].copy(); W.columns = range(len(st))
        features.STATIONS = list(range(len(st)))
        idx = W.index
        tro = idx[(idx.hour % 12 == 9) & (idx >= idx[0] + pd.Timedelta(days=30)) & (idx <= T - pd.Timedelta(hours=HM))]
        X = build(W, tro, hmax=HM); X = X[X.y.notna()]
        Xt = build(W, pd.DatetimeIndex([T]), with_target=False, hmax=HM); Xt = Xt[(Xt.h >= H0) & (Xt.h <= H1)].copy()
        for nm, drop in (('lgb_cal', ()), ('lgb_nocal', CAL)):
            fc = feat_cols(X, drop)
            m = lgb.train(params(), lgb.Dataset(X[fc], X.y.values, categorical_feature=['station']), num_boost_round=600)
            Xt[nm] = m.predict(Xt[fc])
        for i, s in enumerate(st):
            x = W[i]; hh28 = x.iloc[-672:]
            shp = f3_shape(hh28.groupby(hh28.index.hour).mean().reindex(range(24)).values)
            m28, m24 = hh28.mean(), x.iloc[-24:].mean()
            r = Xt.station == i; hr = Xt.ts[r].dt.hour.values; hh = Xt.h[r].values
            Xt.loc[r, 'm28f3'] = m28 * shp[hr]; Xt.loc[r, 'm24f3'] = m24 * shp[hr]
            for tau in (12, 24, 48):
                Xt.loc[r, f'dec{tau}f3'] = (m28 + (m24 - m28) * np.exp(-hh / tau)) * shp[hr]
            Xt.loc[r, 'persist'] = x.dropna().iloc[-1]
            Xt.loc[r, 'y'] = Wall[s].reindex(Xt.ts[r]).values
            Xt.loc[r, 'st_name'] = str(s)
        Xt['year'] = y; Xt['cut'] = dd; Xt['n_train'] = len(X)
        rows.append(Xt.drop(columns=[c for c in Xt.columns if c not in ('year', 'cut', 'ts', 'h', 'st_name', 'y', 'n_train', 'persist', 'lgb_cal', 'lgb_nocal') and not c.endswith('f3')]))
        print(y, f'D+{dd}', 'stations', len(st), 'train rows', len(X), flush=True)
P = pd.concat(rows, ignore_index=True); P.to_parquet('reports/bt_hist.parquet'); print('saved', len(P))
