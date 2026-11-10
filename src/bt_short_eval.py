"""Score reports/bt_short.parquet: overall, recent (2026-08+), Diwali/Nov stress origins, high-regime origins; per-origin win rate."""
import numpy as np, pandas as pd
P = pd.read_parquet('reports/bt_short.parquet'); P = P[P.y.notna() & (P.y <= 1000)].copy()
nov = P.ts.dt.month == 11
C = {}
for b in ('m28f3', 'm7f3', 'm72f3', 'm24f3', 'dec24f3', 'dec48f3', 'persist', 'lgb_cal', 'lgb_nocal'):
    C[b] = P[b]
for L in ('lgb_cal', 'lgb_nocal'):
    for b in ('m28f3', 'm72f3', 'm24f3', 'dec24f3', 'dec48f3'):
        for w in (0.3, 0.5, 0.7):
            C[f'{w}{L[3:]}+{b}'] = w * P[L] + (1 - w) * P[b]
    C[f'avg{L[3:]}+m24+dec48'] = (P[L] + P.m24f3 + P.dec48f3) / 3
C['HARK_PRIMARY'] = np.where(nov, 1.35, 1) * (0.5 * P.lgb_cal + 0.5 * P.m28f3)
C['HARK_BH'] = np.where(nov, 1.35, 1) * (0.2 * P.lgb_cal + 0.8 * P.m28f3)
C['lgbavg'] = 0.5 * (P.lgb_cal + P.lgb_nocal)
for b in ('m24f3', 'dec48f3'):
    C[f'0.5lgbavg+{b}'] = 0.5 * C['lgbavg'] + 0.5 * P[b]
o = P.origin
stress = o.dt.strftime('%Y-%m-%d').isin(['2025-10-21', '2025-10-22', '2025-10-23', '2025-11-05', '2025-11-12', '2026-11-05', '2026-11-06', '2026-11-07'])
recent = o >= pd.Timestamp('2026-08-01', tz='Asia/Kolkata')
# high regime: last-24h mean > 1.3 x 28-day mean at the origin (m24f3/m28f3 ratio of level)
ratio = P.groupby(['origin', 'location_id']).apply(lambda g: (g.m24f3 / g.m28f3).mean(), include_groups=False).rename('ratio')
P = P.join(ratio, on=['origin', 'location_id']); high = P.ratio > 1.3
r = lambda m, p: float(np.sqrt(np.mean((P.y[m] - p[m]) ** 2)))
allm = pd.Series(True, index=P.index)
out = []
for k, p in C.items():
    p = pd.Series(np.clip(p, 0, None), index=P.index)
    po = ((P.y - p) ** 2).groupby(P.origin).mean() ** .5
    out.append(dict(model=k, all=r(allm, p), recent=r(recent, p), stress=r(stress, p), high=r(high, p), oct_nov=r(P.ts.dt.month.isin([10, 11]), p), _po=po))
R = pd.DataFrame(out)
ref = R.set_index('model').loc['0.5cal+m28f3', '_po']
R['win_vs_0.5cal+m28'] = [float((x < ref).mean()) for x in R._po]
R = R.drop(columns='_po').sort_values('all').round(2)
pd.set_option('display.width', 200); print(R.to_string(index=False))
print('rows', len(P), 'origins', P.origin.nunique(), 'stress rows', int(stress.sum()), 'high rows', int(high.sum()))
