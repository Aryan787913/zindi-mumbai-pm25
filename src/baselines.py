import numpy as np, pandas as pd, sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, STATIONS
from validate import origins, targets, evaluate, FOLDS, H

def hist(wide, T, hours):
    return wide.loc[T - pd.Timedelta(hours=hours - 1): T]

def predict_baselines(wide, T, tg):
    out = {}
    h7 = hist(wide, T, 168); h14 = hist(wide, T, 336); h28 = hist(wide, T, 672); h1 = hist(wide, T, 24)
    hour = tg.ts.dt.hour.values
    st = tg.station.values
    last = {s: wide.loc[:T, s].dropna().iloc[-1] if wide.loc[:T, s].notna().any() else np.nan for s in STATIONS}
    m24 = h1.mean(); m7 = h7.mean(); m14 = h14.mean()
    prof7 = h7.groupby(h7.index.hour).mean(); prof14 = h14.groupby(h14.index.hour).median()
    shape28 = h28.groupby(h28.index.hour).mean() / h28.mean()
    city7 = np.nanmean(h7.values)
    out['persist'] = np.array([last[s] for s in st])
    out['mean24'] = np.array([m24[s] for s in st])
    out['mean7d'] = np.array([m7[s] for s in st])
    out['prof7d'] = np.array([prof7.loc[hh, s] for s, hh in zip(st, hour)])
    out['med14_hour'] = np.array([prof14.loc[hh, s] for s, hh in zip(st, hour)])
    out['lvl7xshape28'] = np.array([m7[s] * shape28.loc[hh, s] for s, hh in zip(st, hour)])
    for k, v in out.items():  # fallback for NaN
        out[k] = np.where(np.isnan(v), np.array([m14[s] for s in st]), v)
        out[k] = np.where(np.isnan(out[k]), city7, out[k])
    return out

if __name__ == '__main__':
    t0 = time.time()
    wide, raw = load_train()
    frames = []
    for fold, (a, b) in FOLDS.items():
        for T in origins(wide, a, b, step_h=72):
            tg = targets(wide, T)
            pr = predict_baselines(wide, T, tg)
            for k, v in pr.items():
                d = tg.copy(); d['p'] = np.clip(v, 0, None); d['model'] = k; d['fold'] = fold
                frames.append(d)
    P = pd.concat(frames)
    os.makedirs('reports', exist_ok=True)
    P.to_parquet('reports/baseline_preds.parquet')
    res = [evaluate(g, k) for k, g in P.groupby('model')]
    R = pd.DataFrame(res).set_index('name').round(2).sort_values('h97-144')
    print(R.to_string()); print('n origins', P.groupby('fold').origin.nunique().to_dict(), 'time %.1fs' % (time.time() - t0))
    R.to_csv('reports/baselines.csv')
