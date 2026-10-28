"""Generate several validated candidate submissions from one training pass (2026-10-28 batch)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, load_test, STATIONS
from features import build, feat_cols
from train_lgb import params
from baselines import predict_baselines
wide, _ = load_train(); te = load_test(); T = wide.index[-1]; idx = wide.index
tr_orig = idx[(idx.hour % 12 == 0) & (idx >= idx[0] + pd.Timedelta(days=30))]
X = build(wide, tr_orig); X = X[X.y.notna()]
Xt = build(wide, pd.DatetimeIndex([T]), with_target=False, hmax=192); Xt['location_id'] = [STATIONS[i] for i in Xt.station]
def fit_pred(drop=(), seed=42):
    fc = feat_cols(X, drop); p = params(); p['seed'] = seed
    m = lgb.train(p, lgb.Dataset(X[fc], X.y.values, categorical_feature=['station']), num_boost_round=600)
    return m.predict(Xt[fc])
Xt['lgb42'] = fit_pred()
Xt['lgb3s'] = (Xt.lgb42 + fit_pred(seed=7) + fit_pred(seed=2026)) / 3
Xt['lgbnocal'] = fit_pred(drop=('doy_s', 'doy_c', 'diwali_off'))
te['h'] = ((te.ts - T) / pd.Timedelta(hours=1)).round().astype(int)
tg = pd.DataFrame({'station': te.location_id, 'ts': te.ts})
te['base'] = predict_baselines(wide, T, tg)['lvl7xshape28']
h14 = wide.loc[T - pd.Timedelta(hours=335):T]; h28 = wide.loc[T - pd.Timedelta(hours=671):T]
shape = h28.groupby(h28.index.hour).mean() / h28.mean(); lv = h14.mean()
te['base14'] = [lv[s] * shape.loc[hh, s] for s, hh in zip(te.location_id, te.ts.dt.hour)]
te = te.merge(Xt[['location_id', 'h', 'lgb42', 'lgb3s', 'lgbnocal']], on=['location_id', 'h'], how='left')
ss = pd.read_csv('data/raw/SampleSubmission.csv')
def out(name, lgbcol, w, basecol='base'):
    t = np.where(te[lgbcol].notna(), w * te[lgbcol] + (1 - w) * te[basecol], te[basecol]); t = np.clip(t, 0, None)
    sub = pd.DataFrame({'ID': te.ID, 'Target': t})
    assert len(sub) == 2592 and (sub.ID.values == ss.ID.values).all() and np.isfinite(sub.Target).all() and (sub.Target >= 0).all()
    f = f'submissions/{name}.csv'; sub.to_csv(f, index=False)
    print(name, hashlib.sha256(open(f, 'rb').read()).hexdigest()[:12], 'mean %.2f' % sub.Target.mean(), 'pub-window mean %.2f' % sub.Target[te.ts < '2026-10-24'].mean())




Xt['lgbnc3']=(Xt.lgbnocal+fit_pred(drop=('doy_s','doy_c','diwali_off'),seed=7)+fit_pred(drop=('doy_s','doy_c','diwali_off'),seed=2026))/3; te=te.merge(Xt[['location_id','h','lgbnc3']],on=['location_id','h'],how='left'); out('s011_blend_nocal_3seed','lgbnc3',0.5)

