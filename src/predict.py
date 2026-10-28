"""Train on all available data and write a validated submission.
Model v1: 0.5*LightGBM(direct, h<=144, raw target, all features) + 0.5*baseline lvl7xshape28; h>144 baseline only."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, hashlib, json
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, load_test, STATIONS
from features import build, feat_cols
from train_lgb import params
from baselines import predict_baselines

def main(out, w=0.5, rounds=600, season_mult=True):
    wide, _ = load_train(); te = load_test()
    T = wide.index[-1]
    idx = wide.index
    tr_orig = idx[(idx.hour % 12 == 0) & (idx >= idx[0] + pd.Timedelta(days=30))]
    X = build(wide, tr_orig); X = X[X.y.notna()]
    fc = feat_cols(X)
    m = lgb.train(params(), lgb.Dataset(X[fc], X.y.values, categorical_feature=['station']), num_boost_round=rounds)
    Xt = build(wide, pd.DatetimeIndex([T]), with_target=False, hmax=192)
    Xt['p_lgb'] = m.predict(Xt[fc]); Xt['location_id'] = [STATIONS[i] for i in Xt.station]
    te['h'] = ((te.ts - T) / pd.Timedelta(hours=1)).round().astype(int)
    assert te.h.min() >= 1, 'test overlaps train'
    tg = pd.DataFrame({'station': te.location_id, 'ts': te.ts})
    # baseline for all rows
    base = predict_baselines(wide, T, tg.assign(ts=te.ts))['lvl7xshape28']
    te['p_base'] = base
    te = te.merge(Xt[['location_id', 'h', 'p_lgb']], on=['location_id', 'h'], how='left')
    te['Target'] = np.where(te.p_lgb.notna(), w * te.p_lgb + (1 - w) * te.p_base, te.p_base)
    if season_mult:  # E010: Oct-Nov rising-season under-prediction correction (validated Oct-Nov 2025 & Oct 2026)
        mo = te.ts.dt.month.isin([10, 11])
        te.loc[mo, 'Target'] *= np.where(te.loc[mo, 'h'] <= 96, 1.10, 1.15)
    te['Target'] = te.Target.clip(lower=0)
    sub = te[['ID', 'Target']]
    ss = pd.read_csv('data/raw/SampleSubmission.csv')
    assert list(sub.columns) == ['ID', 'Target'] and len(sub) == len(ss) == 2592
    assert (sub.ID.values == ss.ID.values).all() and sub.ID.is_unique
    assert np.isfinite(sub.Target).all() and (sub.Target >= 0).all()
    sub.to_csv(out, index=False)
    h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    rep = dict(file=out, sha256=h, origin=str(T), rows=len(sub), lgb_rows=int(te.p_lgb.notna().sum()),
               target_stats=sub.Target.describe().round(2).to_dict())
    print(json.dumps(rep, indent=1))
    return rep

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'submissions/submission_002_blend_v1.csv')
