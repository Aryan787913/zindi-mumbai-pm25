import numpy as np
def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = ~np.isnan(y)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
