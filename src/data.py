"""Load Train.csv onto a full hourly grid per station (timestamps = hour END, IST)."""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import parse_ist
R = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
STATIONS = [11606, 11611, 3409510, 3409511]
CAP = 1000.0  # values above are treated as sensor errors for training/eval (flag_high_1000)

def load_train(path=None, cap=CAP):
    tr = pd.read_csv(path or os.path.join(R, 'Train.csv'))
    tr['ts'] = parse_ist(tr.timestamp_ist)
    tr = tr[['ts', 'location_id', 'pm25']]
    wide = tr.pivot(index='ts', columns='location_id', values='pm25')
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq='h', tz='Asia/Kolkata')
    wide = wide.reindex(grid)[STATIONS]
    raw = wide.copy()
    if cap is not None:
        wide = wide.where(wide <= cap)
    return wide, raw

def load_test():
    te = pd.read_csv(os.path.join(R, 'Test.csv'))
    te['ts'] = parse_ist(te.target_end_ist)
    return te
