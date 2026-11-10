"""Does the weather forecast fix the fade model's level error? (Open-Meteo historical-forecast archive, ClimateVerse-listed.)
Base = dec36 (middle of the pair). Unit = origin (09:00, daily) x station x target day (D1 = h16-39, D2 = h40-63).
Response: log(actual day mean / predicted day mean). Weather anomalies use the same point's forecast:
  dblh = log(mean BLH target day / mean BLH over the 3 days before the origin), dws likewise, rain = target-day rain (mm).
Out-of-sample: expanding window by time (fit on origins before Jun 2026 -> test Jun-Oct 2026), plus leave-one-month-out.
Usage: python src/bt_weather.py <openmeteo_histfc.json>"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, STATIONS
from final_submit import f3_shape
ORDER = [11611, 3409511, 11606, 3409510]
j = json.load(open(sys.argv[1]))
wx = {}
for loc, blk in zip(ORDER, j):
    h = pd.DataFrame(blk['hourly']); h['ts'] = pd.to_datetime(h.time).dt.tz_localize('Asia/Kolkata') + pd.Timedelta(hours=1)
    wx[loc] = h.set_index('ts')[['boundary_layer_height', 'wind_speed_10m', 'precipitation']]
w, _ = load_train()
rows = []
for T in pd.date_range('2025-02-20 09:00', '2026-10-25 09:00', freq='D', tz='Asia/Kolkata'):
    for s in STATIONS:
        x = w[s].loc[:T]; h28 = x.iloc[-672:]
        if h28.notna().mean() < .6 or x.iloc[-24:].notna().sum() < 12: continue
        shp = f3_shape(h28.groupby(h28.index.hour).mean().reindex(range(24)).values)
        m28, m24 = h28.mean(), x.iloc[-24:].mean()
        W = wx[s]; pre = W.loc[T - pd.Timedelta(hours=71):T]
        for day, (a, b) in (('D1', (16, 39)), ('D2', (40, 63))):
            ts = pd.date_range(T + pd.Timedelta(hours=a), T + pd.Timedelta(hours=b), freq='h')
            y = w[s].reindex(ts)
            if y.notna().sum() < 16: continue
            hh = np.arange(a, b + 1)
            p = (m28 + (m24 - m28) * np.exp(-hh / 36)) * shp[ts.hour]
            tw = W.reindex(ts)
            if tw.isna().any().any() or len(pre) < 48: continue
            ok = y.notna().values
            rows.append(dict(T=T, s=s, day=day, y=y[ok].mean(), p=p[ok].mean(), se_base=np.mean((y[ok].values - p[ok]) ** 2),
                             dblh=np.log(tw.boundary_layer_height.mean() / pre.boundary_layer_height.mean()),
                             dws=np.log((tw.wind_speed_10m.mean() + 1) / (pre.wind_speed_10m.mean() + 1)),
                             rain=np.log1p(tw.precipitation.sum()), yv=y[ok].values, pv=p[ok]))
D = pd.DataFrame(rows); D['lr'] = np.log(D.y / D.p); D = D[np.isfinite(D.lr)].reset_index(drop=True)
print('units', len(D), 'origins', D['T'].nunique())
print('corr with log(actual/pred):', D[['lr', 'dblh', 'dws', 'rain']].corr().lr.round(3).to_dict())
X = lambda d, cols: np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
def evaluate(train, test, cols, shrink=1.0):
    b = np.linalg.lstsq(X(train, cols), train.lr.values, rcond=None)[0]
    adj = np.exp(shrink * (X(test, cols) @ b - b[0]))   # weather part only (no intercept / level shift)
    se_new = np.mean(np.concatenate([(yv - pv * a) ** 2 for yv, pv, a in zip(test.yv, test.pv, adj)]))
    se_old = np.mean(np.concatenate([(yv - pv) ** 2 for yv, pv in zip(test.yv, test.pv)]))
    return np.sqrt(se_old), np.sqrt(se_new), b
cut = pd.Timestamp('2026-06-01', tz='Asia/Kolkata')
for cols in (['dblh'], ['dws'], ['rain'], ['dblh', 'dws'], ['dblh', 'dws', 'rain']):
    o, n, b = evaluate(D[D['T'] < cut], D[D['T'] >= cut], cols)
    mo = D['T'].dt.to_period('M'); res = []
    for m in mo.unique():
        o2, n2, _ = evaluate(D[mo != m], D[mo == m], cols); res.append((o2, n2))
    wins = np.mean([n2 < o2 for o2, n2 in res])
    oo = np.sqrt(np.mean([x[0] ** 2 for x in res])); nn = np.sqrt(np.mean([x[1] ** 2 for x in res]))
    print(f'{"+".join(cols):16s} coef {np.round(b[1:], 3)} | time-split base {o:.2f} -> {n:.2f} | leave-month-out base {oo:.2f} -> {nn:.2f}, months improved {wins:.0%}')
oct_nov = D[D['T'].dt.month.isin([10, 11])]
for cols in (['dblh'], ['dblh', 'dws']):
    o, n, b = evaluate(D[~D['T'].dt.month.isin([10, 11])], oct_nov, cols)
    print('fit non-Oct/Nov -> test Oct/Nov', cols, f'{o:.2f} -> {n:.2f}')
D.drop(columns=['yv', 'pv']).to_csv('reports/bt_weather_units.csv', index=False)
