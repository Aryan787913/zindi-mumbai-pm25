import numpy as np, pandas as pd, lightgbm as lgb, sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train
from features import build, feat_cols
from validate import FOLDS, evaluate, origins
from metrics import rmse
SEED = 42
def params(obj='l2'):
    return dict(objective=obj, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED, deterministic=True, num_threads=2, verbose=-1)

def run(drop=(), target='raw', scheme='blocked', rounds=600, tag='lgb', step_train=12, use_weather=False):
    wide, _ = load_train()
    idx = wide.index
    tr_orig = idx[(idx.hour % step_train == 0) & (idx >= idx[0] + pd.Timedelta(days=30)) & (idx <= idx[-1] - pd.Timedelta(hours=1))]
    W = None
    if use_weather:
        from weather import load_weather
        W = load_weather()
    X = build(wide, tr_orig, weather=W)
    X = X[X.y.notna()].reset_index(drop=True)
    fc = feat_cols(X, drop)
    res = []
    for fold, (a, b) in FOLDS.items():
        vo = origins(wide, a, b, step_h=72)
        A = pd.Timestamp(a, tz='Asia/Kolkata'); B = pd.Timestamp(b, tz='Asia/Kolkata') + pd.Timedelta(days=7)
        tgt_end = X.origin + pd.to_timedelta(X.h, unit='h')
        if scheme == 'expanding':
            trm = tgt_end < A
        else:  # blocked: exclude any row whose origin..target overlaps [A-7d, B]
            trm = (tgt_end < A - pd.Timedelta(days=1)) | (X.origin > B)
        va = X[X.origin.isin(vo)]
        tr = X[trm]
        ytr = tr.y.values
        if target == 'log': ytr = np.log1p(ytr)
        if target == 'ratio': ytr = tr.y.values / (tr.m168.values + 1)
        m = lgb.train(params(), lgb.Dataset(tr[fc], ytr, categorical_feature=['station']), num_boost_round=rounds)
        p = m.predict(va[fc])
        if target == 'log': p = np.expm1(p)
        if target == 'ratio': p = p * (va.m168.values + 1)
        d = va[['origin', 'station', 'h', 'ts', 'y']].copy(); d['p'] = np.clip(p, 0, None); d['fold'] = fold
        res.append(d)
    P = pd.concat(res)
    return P, m, fc

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--target', default='raw'); ap.add_argument('--scheme', default='blocked')
    ap.add_argument('--drop', default=''); ap.add_argument('--tag', default='lgb'); ap.add_argument('--rounds', type=int, default=600); ap.add_argument('--weather', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    P, m, fc = run(drop=[d for d in a.drop.split(',') if d], target=a.target, scheme=a.scheme, rounds=a.rounds, use_weather=a.weather)
    P.to_parquet(f'reports/preds_{a.tag}.parquet')
    r = evaluate(P, a.tag); r['runtime_s'] = round(time.time() - t0, 1)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
    imp = pd.Series(m.feature_importance('gain'), index=fc).sort_values(ascending=False)
    print(imp.head(15).round(0).to_dict())
