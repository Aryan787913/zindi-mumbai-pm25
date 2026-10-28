import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data import load_train
from features import build, feat_cols
from metrics import rmse

def test_rmse():
    assert abs(rmse([1, 2, 3], [1, 2, 5]) - np.sqrt(4 / 3)) < 1e-12
    assert rmse([1, np.nan], [1, 100]) == 0.0

def test_no_future_leak():
    wide, _ = load_train()
    T = wide.index[-2000]
    X1 = build(wide, pd.DatetimeIndex([T]), with_target=False)
    w2 = wide.copy(); w2.loc[w2.index > T] = 999.0   # corrupt the future
    X2 = build(w2, pd.DatetimeIndex([T]), with_target=False)
    fc = feat_cols(X1)
    pd.testing.assert_frame_equal(X1[fc], X2[fc])
