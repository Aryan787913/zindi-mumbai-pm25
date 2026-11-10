"""Cheap grid on the decay model (no LightGBM): shape window (7/14/28 d), shape smoothing (F3/raw), long-run level (m28/m14/median28),
tau (24/36/48). Same geometry as final (cutoff 09:00, h16-63). Two independent test sets:
  A = Train origins (same as bt_short: every 9 days Jun 2025 - 7 Nov 2026 + stress dates), B = 5 post-Diwali seasons D+1..D+3.
Usage: python src/bt_level_grid.py <AdditionalHistory dir>"""
import sys, os, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from final_submit import f3_shape
from data import load_train
sys.argv += [] ; AH = sys.argv[1]
exec(open(os.path.join(os.path.dirname(__file__), 'bt_hist.py')).read().split("features.DIWALI =")[0].split("AH = sys.argv[1]")[1])  # DIW, wide_hist

def preds(x, T, ts, cfg):
    sw, sm, lv, tau = cfg
    hs = x.loc[:T].iloc[-sw * 24:]; prof = hs.groupby(hs.index.hour).mean().reindex(range(24))
    if sm == 'f3':
        shp = f3_shape(prof.values)
    else:
        p = prof.interpolate(limit_direction='both').values; shp = p / p.mean()
    h28 = x.loc[:T].iloc[-672:]
    L = {'m28': h28.mean(), 'm14': h28.iloc[-336:].mean(), 'med28': h28.median()}[lv]
    m24 = x.loc[:T].iloc[-24:].mean()
    hh = ((ts - T) / pd.Timedelta(hours=1)).values
    return (L + (m24 - L) * np.exp(-hh / tau)) * shp[ts.hour]

GRID = list(itertools.product((7, 14, 28), ('f3', 'raw'), ('m28', 'm14', 'med28'), (24, 36, 48)))
def run(cases):
    E = {c: [] for c in GRID}; case_rmse = {c: [] for c in GRID}
    for x, T in cases:
        f = x.loc[T + pd.Timedelta(hours=16):T + pd.Timedelta(hours=63)].dropna()
        if len(f) < 24 or x.loc[:T].iloc[-24:].notna().sum() < 12 or x.loc[:T].iloc[-672:].notna().mean() < .6: continue
        for c in GRID:
            e = (f.values - preds(x, T, f.index, c)) ** 2; E[c].append(e); case_rmse[c].append(np.sqrt(e.mean()))
    return {c: (np.sqrt(np.concatenate(E[c]).mean()), np.mean(case_rmse[c])) for c in GRID}, len(case_rmse[GRID[0]])

w, _ = load_train()
orig = list(pd.date_range('2025-06-02 09:00', '2026-11-07 09:00', freq='9D', tz='Asia/Kolkata'))
orig += [pd.Timestamp(d, tz='Asia/Kolkata') for d in ('2025-10-21 09:00', '2025-10-22 09:00', '2025-10-23 09:00', '2025-11-05 09:00', '2025-11-12 09:00', '2026-11-05 09:00', '2026-11-06 09:00', '2026-11-07 09:00')]
A, nA = run([(w[s], T) for T in orig for s in w.columns])
casesB = []
for y, d in DIW.items():
    W = wide_hist(y)
    for dd in (1, 2, 3):
        T = pd.Timestamp(d, tz='Asia/Kolkata') + pd.Timedelta(days=dd, hours=9)
        casesB += [(W[s], T) for s in W.columns]
B, nB = run(casesB)
R = pd.DataFrame([dict(shape_days=c[0], smooth=c[1], level=c[2], tau=c[3], A_all=A[c][0], A_case=A[c][1], B_all=B[c][0], B_case=B[c][1]) for c in GRID])
R['rankA'] = R.A_all.rank(); R['rankB'] = R.B_all.rank(); R['rank_sum'] = R.rankA + R.rankB
pd.set_option('display.width', 200); print(R.sort_values('rank_sum').round(2).head(15).to_string(index=False))
cur = R[(R.shape_days == 28) & (R.smooth == 'f3') & (R.level == 'm28')]; print('\nCURRENT (28/f3/m28):'); print(cur.round(2).to_string(index=False))
print('station-cases A', nA, 'B', nB)
R.to_csv('reports/level_grid.csv', index=False)
