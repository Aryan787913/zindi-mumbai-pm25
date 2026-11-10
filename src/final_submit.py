"""Rebuild of Hark's FROZEN final pair (FINAL_PRIVATE_PLAN.md, UPDATE 2026-11-02 RN5, authoritative):
PRIMARY_F3 = 1.35 (Nov target rows) x [0.5 * cal LGB + 0.5 * m28 level x F3(28 d shape)]  -> upload 1st, select
BH         = 1.35 (Nov target rows) x [0.2 * cal LGB + 0.8 * m28 level x F3(28 d shape)]  -> upload LAST, select
LGB: direct h1-192, all features incl. calendar, seed 42, 600 rounds. F3 = Fourier order-3 LSQ fit of the 28-day hour profile.
NOTE: reconstructed on 10 Nov from the 28 Oct repo + Hark notes (Hark's final src was not in the handoff); not byte-verified
against Hark's u1031/u1102 files. Usage: python src/final_submit.py [outdir] [tag]"""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, hashlib, json
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, load_test, STATIONS
from features import build, feat_cols
from train_lgb import params

HMAX, ROUNDS, K = 192, 600, 1.35


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


def main(outdir='submissions', tag='final1110'):
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
    reps = {}
    for name, w, k in ((f'{tag}_primary_m28nov135_f3', 0.5, K), (f'{tag}_bh_w02_m28nov135_f3', 0.2, K)):
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
    print(json.dumps(dict(origin=str(T), level28={str(k): round(v, 1) for k, v in lvl.items()}, files=reps), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
