"""Backtest base (28d level x Fourier-3 shape) with November/post-Diwali factor k on past Diwalis.
Window = observations of Diwali+3 and Diwali+4 (the 11-12 Nov analog for Diwali 8 Nov 2026).
Cutoff = window start - gap hours (gap 96/120/144). Usage: python src/bt_diwali.py <AdditionalHistory dir>"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from final_submit import f3_shape
from data import load_train

AH = sys.argv[1]
DIW = {2018: '2018-11-07', 2019: '2019-10-27', 2024: '2024-10-31', 2025: '2025-10-20'}


def wide_hist(year):
    if year in (2018, 2019):
        a = pd.read_csv(os.path.join(AH, 'History_Mumbai_2018_2020.csv.gz'), usecols=['Datetime', 'StationName', 'PM2.5'])
        a['ts'] = pd.to_datetime(a.Datetime) + pd.Timedelta(hours=1)  # hour start -> hour end
        w = a.pivot_table(index='ts', columns='StationName', values='PM2.5')
    elif year == 2024:
        b = pd.read_csv(os.path.join(AH, 'History_Mumbai_2024.csv.gz'), usecols=['station_name', 'timestamp_ist', 'value'])
        b['ts'] = pd.to_datetime(b.timestamp_ist.str[:19]).dt.ceil('h')
        w = b.pivot_table(index='ts', columns='station_name', values='value')
    else:
        w, _ = load_train(); w.index = w.index.tz_localize(None)
    w = w.where((w > 0) & (w <= 1000))
    return w.asfreq('h')


rows = []
for y, d in DIW.items():
    W = wide_hist(y)
    w0 = pd.Timestamp(d) + pd.Timedelta(days=3, hours=1); w1 = w0 + pd.Timedelta(hours=47)
    for gap in (96, 120, 144):
        T = w0 - pd.Timedelta(hours=gap + 1)
        h28 = W.loc[T - pd.Timedelta(hours=671):T]; fut = W.loc[w0:w1]
        st = [s for s in W.columns if h28[s].notna().mean() > 0.6 and fut[s].notna().mean() > 0.6]
        for s in st:
            lvl = h28[s].mean(); lvl7 = h28[s].iloc[-168:].mean()
            shp = f3_shape(h28[s].groupby(h28[s].index.hour).mean().reindex(range(24)).values)
            f = fut[s].dropna()
            b = lvl * shp[f.index.hour]; b7 = lvl7 * shp[f.index.hour]
            for i, (t, v) in enumerate(f.items()):
                rows.append(dict(year=y, gap=gap, st=s, y=v, base=b[i], base7=b7[i]))
D = pd.DataFrame(rows)
r = lambda g, p: float(np.sqrt(np.mean((g.y - p) ** 2)))
out = []
for (y, gap), g in D.groupby(['year', 'gap']):
    o = dict(year=y, gap=gap, n_st=g.st.nunique(), y_mean=g.y.mean(), base_mean=g.base.mean(), kopt=(g.y * g.base).sum() / (g.base ** 2).sum())
    for k in (1.0, 1.2, 1.35, 1.5, 1.7):
        o[f'k{k}'] = r(g, k * g.base)
    o['m7_k1.0'] = r(g, g.base7); o['m7_k1.35'] = r(g, 1.35 * g.base7)
    out.append(o)
O = pd.DataFrame(out).round(2)
pd.set_option('display.width', 250); print(O.to_string(index=False))
print('\nmean over years/gaps:'); print(O.drop(columns=['year', 'gap', 'n_st']).mean().round(2).to_string())
