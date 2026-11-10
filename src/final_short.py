"""FINAL build for the 10 Nov 09:00 release (private 11-12 Nov = h16-63 from the last observation).
Components (all from data <= T only): lgb_cal / lgb_nocal = LightGBM direct h1-63 (seed 42, 600 rounds, origins at 09:00/21:00),
m28f3 = 28-day mean x Fourier-3 hour shape, m24f3, dec{24,48}f3 = level decaying from last-24h mean to 28-day mean (tau hours).
The two file formulas are set in FILES after the bt_short.py / bt_short_eval.py backtest (see reports/FINAL_SHORT.md).
Observed test hours (h<=0) are filled with Train values (unscored); h>63 (unscored) use dec48f3.
Usage: python src/final_short.py <tag>"""
import sys, os, json, hashlib, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from data import load_train, load_test, STATIONS
from features import build, feat_cols
from train_lgb import params
from final_submit import f3_shape

HM = 63
CAL = ['doy_s', 'doy_c', 'diwali_off']
# Chosen 10 Nov 15:35 IST from bt_short (65 origins) + bt_hist (5 post-Diwali seasons): LightGBM hurts at h16-63 (spikes),
# simple level decay wins. Pair = the two fade speeds (best-of-2 selected); dec48 is uploaded LAST (dry/calm forecast).
FILES = {'fade24': lambda t: t.dec24f3, 'fade48': lambda t: t.dec48f3}


def components():
    wide, _ = load_train(); te = load_test()
    T = wide.index[-1]; idx = wide.index
    tro = idx[(idx.hour % 12 == 9) & (idx >= idx[0] + pd.Timedelta(days=30)) & (idx <= T - pd.Timedelta(hours=HM))]
    X = build(wide, tro, hmax=HM); X = X[X.y.notna()]
    Xt = build(wide, pd.DatetimeIndex([T]), with_target=False, hmax=HM)
    for nm, drop in (('lgb_cal', ()), ('lgb_nocal', CAL)):
        fc = feat_cols(X, drop)
        m = lgb.train(params(), lgb.Dataset(X[fc], X.y.values, categorical_feature=['station']), num_boost_round=600)
        Xt[nm] = m.predict(Xt[fc])
    Xt['location_id'] = [STATIONS[i] for i in Xt.station]
    te['h'] = ((te.ts - T) / pd.Timedelta(hours=1)).round().astype(int)
    te = te.merge(Xt[['location_id', 'h', 'lgb_cal', 'lgb_nocal']], on=['location_id', 'h'], how='left')
    for s in STATIONS:
        x = wide[s]; h28 = x.iloc[-672:]
        shp = f3_shape(h28.groupby(h28.index.hour).mean().reindex(range(24)).values)
        m28, m24 = h28.mean(), x.iloc[-24:].mean()
        m24 = m24 if np.isfinite(m24) else m28
        r = te.location_id == s; hr = te.ts[r].dt.hour.values; hh = np.clip(te.h[r].values, 0, None)
        te.loc[r, 'm28f3'] = m28 * shp[hr]; te.loc[r, 'm24f3'] = m24 * shp[hr]
        for tau in (24, 48):
            te.loc[r, f'dec{tau}f3'] = (m28 + (m24 - m28) * np.exp(-hh / tau)) * shp[hr]
    obs = wide.stack(future_stack=True).rename('obs').reset_index(); obs.columns = ['ts', 'location_id', 'obs']
    te = te.merge(obs, on=['ts', 'location_id'], how='left')
    return te, T


def main(tag):
    te, T = components()
    ss = pd.read_csv('data/raw/SampleSubmission.csv')
    priv = te.ts.between(pd.Timestamp('2026-11-11 01:00', tz='Asia/Kolkata'), pd.Timestamp('2026-11-13 00:00', tz='Asia/Kolkata'))
    rep = dict(origin=str(T), private_h=f'{te.h[priv].min()}-{te.h[priv].max()}', files={})
    for name, f in FILES.items():
        p = np.asarray(f(te), dtype=float)
        p = np.where(te.h > HM, te.dec48f3, p)
        p = np.where((te.h <= 0) & te.obs.notna(), te.obs, p)
        p = np.where(np.isfinite(p), p, te.dec48f3)
        p = np.clip(p, 0, None)
        sub = pd.DataFrame({'ID': te.ID.values, 'Target': p})
        assert len(sub) == len(ss) == 2592 and (sub.ID.values == ss.ID.values).all() and sub.ID.is_unique
        assert np.isfinite(sub.Target).all() and (sub.Target >= 0).all()
        assert te.lgb_cal[priv].notna().all(), 'LGB missing in private window'
        os.makedirs('submissions', exist_ok=True); out = f'submissions/{tag}_{name}.csv'
        sub.to_csv(out, index=False)
        st = sub[priv.values].assign(loc=te.location_id[priv].values).groupby('loc').Target.mean().round(1).to_dict()
        rep['files'][name] = dict(file=out, sha256=hashlib.sha256(open(out, 'rb').read()).hexdigest(),
                                  private_mean=round(float(sub.Target[priv.values].mean()), 1), private_by_station=st)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'final1110')
