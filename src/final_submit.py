"""Final 2-file submission (see WHAT_TO_DO.md, 'Winning approach').
base    = 28-day mean level per station x smooth 24-h shape (Fourier, 3 harmonics, from last 28 days)
LGB     = direct forecast h1-192, all features (incl. calendar), seed 42, 600 rounds
File A  = 0.2*LGB + 0.8*base; Nov rows x k_A, k_A = 0.5*1.35 + 0.5*(CLIM / city 28-day mean).
          CLIM = 62.6 = mean city PM2.5 on 11-12 Nov in 2018/2019/2024/2025 (59.9/64.2/63.0/62.9, AdditionalHistory + Train).
          Leave-one-year-out RMSE: k1.35 37.47, clim 36.00, this 50/50 36.43 (bt_diwali.py + reports/loo_level.txt).
File B  = same, Nov rows x1.60 (hedge for a late, 8-Nov Diwali; past 11-12 Nov / prior-28-day ratios were 1.23-1.85).
Zindi scores the BEST of the 2 selected files, so the two files are deliberately different bets.
Test hours already present in Train.csv are filled with the observed value (unscored / public only).
Hours beyond h192 use base only. Usage: python src/final_submit.py [outdir]"""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, hashlib, json
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, load_test, STATIONS
from features import build, feat_cols
from train_lgb import params

HMAX, ROUNDS = 192, 600
CLIM = 62.6


def f3_shape(prof):
    """prof: 24 values (mean by hour-of-day). Returns 24 values, mean 1, smoothed by 3 Fourier harmonics."""
    p = pd.Series(prof).interpolate(limit_direction='both').values
    p = p / np.nanmean(p)
    t = 2 * np.pi * np.arange(24) / 24
    A = np.column_stack([np.ones(24)] + [f(k * t) for k in (1, 2, 3) for f in (np.cos, np.sin)])
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    s = A @ coef
    return s / s.mean()


def base_components(wide, T):
    h28 = wide.loc[T - pd.Timedelta(hours=671): T]
    lvl, shp = {}, {}
    for s in STATIONS:
        x = h28[s]
        lvl[s] = x.mean() if x.notna().any() else np.nanmean(h28.values)
        prof = x.groupby(x.index.hour).mean().reindex(range(24))
        shp[s] = f3_shape(prof.values) if prof.notna().sum() >= 12 else np.ones(24)
    return lvl, shp


def main(outdir='submissions'):
    wide, _ = load_train(); te = load_test()
    T = wide.index[-1]
    idx = wide.index
    tr_orig = idx[(idx.hour % 12 == 0) & (idx >= idx[0] + pd.Timedelta(days=30)) & (idx <= T - pd.Timedelta(hours=1))]
    X = build(wide, tr_orig, hmax=HMAX); X = X[X.y.notna()]
    fc = feat_cols(X)
    m = lgb.train(params(), lgb.Dataset(X[fc], X.y.values, categorical_feature=['station']), num_boost_round=ROUNDS)
    Xt = build(wide, pd.DatetimeIndex([T]), with_target=False, hmax=HMAX)
    Xt['p_lgb'] = m.predict(Xt[fc]); Xt['location_id'] = [STATIONS[i] for i in Xt.station]

    te['h'] = ((te.ts - T) / pd.Timedelta(hours=1)).round().astype(int)
    te = te.merge(Xt[['location_id', 'h', 'p_lgb']], on=['location_id', 'h'], how='left')
    lvl, shp = base_components(wide, T)
    te['p_base'] = [lvl[s] * shp[s][t.hour] for s, t in zip(te.location_id, te.ts)]
    obs = wide.stack(future_stack=True).rename('obs').reset_index(); obs.columns = ['ts', 'location_id', 'obs']
    te = te.merge(obs, on=['ts', 'location_id'], how='left')
    fut = te.h >= 1
    nov = fut & (te.ts.dt.month == 11)
    ss = pd.read_csv('data/raw/SampleSubmission.csv')
    city28 = float(np.mean(list(lvl.values())))
    kA = 0.5 * 1.35 + 0.5 * CLIM / city28
    reps = {}
    for name, w, k in (('final_A_clim', 0.2, kA), ('final_B_k160', 0.2, 1.60)):
        p = np.where(te.p_lgb.notna(), w * te.p_lgb + (1 - w) * te.p_base, te.p_base)
        p = np.where(nov, p * k, p)
        p = np.where(~fut & te.obs.notna(), te.obs, p)        # already-observed hours
        p = np.clip(p, 0, None)
        sub = pd.DataFrame({'ID': te.ID, 'Target': p})
        assert len(sub) == len(ss) == 2592 and (sub.ID.values == ss.ID.values).all() and sub.ID.is_unique
        assert np.isfinite(sub.Target).all() and (sub.Target >= 0).all()
        os.makedirs(outdir, exist_ok=True); out = os.path.join(outdir, f'{name}.csv')
        sub.to_csv(out, index=False)
        priv = te.ts.between(pd.Timestamp('2026-11-11 01:00', tz='Asia/Kolkata'), pd.Timestamp('2026-11-13 00:00', tz='Asia/Kolkata'))
        reps[name] = dict(file=out, sha256=hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16],
                          private_mean=round(float(sub.Target[priv].mean()), 2),
                          private_h=f'{te.h[priv].min()}-{te.h[priv].max()}', lgb_rows_private=int(te.p_lgb[priv].notna().sum()))
    print(json.dumps(dict(origin=str(T), city28=round(city28, 1), kA=round(kA, 3), level28={str(k): round(v, 1) for k, v in lvl.items()}, files=reps), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
