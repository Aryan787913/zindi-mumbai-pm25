"""E011: robust level estimators x 28d hour shape; blended 50/50 with LGB h192."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, STATIONS
from validate import origins, targets, evaluate, FOLDS
wide, _ = load_train()
L = pd.read_parquet('reports/preds_lgb_raw_h192.parquet')
fr = []
for fold, (a, b) in FOLDS.items():
    for T in origins(wide, a, b, step_h=72):
        tg = targets(wide, T)
        h28 = wide.loc[T - pd.Timedelta(hours=671):T]; h7 = wide.loc[T - pd.Timedelta(hours=167):T]; h14 = wide.loc[T - pd.Timedelta(hours=335):T]
        shape = h28.groupby(h28.index.hour).mean() / h28.mean()
        lv = {'mean7': h7.mean(), 'med7': h7.median(), 'med14': h14.median(), 'mean14': h14.mean(),
              'shrink7_28': 0.5 * h7.mean() + 0.5 * h28.mean(), 'med7_28': 0.5 * h7.median() + 0.5 * h28.median()}
        hr = tg.ts.dt.hour.values
        for k, l in lv.items():
            p = np.array([l[s] * shape.loc[hh, s] for s, hh in zip(tg.station, hr)])
            d = tg.copy(); d['p'] = np.nan_to_num(p, nan=np.nanmean(p)); d['model'] = k; d['fold'] = fold; fr.append(d)
P = pd.concat(fr); P['station'] = P.station.map({s: i for i, s in enumerate(STATIONS)})
rows = []
for k, g in P.groupby('model'):
    rows.append(evaluate(g, k + '_alone'))
    m = L.merge(g[['origin', 'station', 'h', 'p']].rename(columns={'p': 'pb'}), on=['origin', 'station', 'h'])
    m['p'] = 0.5 * m.p + 0.5 * m.pb; rows.append(evaluate(m, k + '_blend'))
R = pd.DataFrame(rows).set_index('name')[['all', 'h1-48', 'h97-144', 'h145-192', 'recent26', 'recent26_h97+']].round(2)
print(R.sort_values('h97-144').to_string())
