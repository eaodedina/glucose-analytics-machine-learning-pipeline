import pickle
import numpy as np
import pandas as pd

# Feature extraction
def series_feature(x: pd.Series) -> np.ndarray:
    vec = x.values.astype(float)
    n = len(vec)
    if n == 0 or np.all(np.isnan(vec)):
        return np.full(10, np.nan)
    vmax = np.nanmax(vec)
    vmin = np.nanmin(vec)
    delta = vmax - vmin
    mean = np.nanmean(vec)
    std = np.nanstd(vec)
    time_to_peak = (int(np.nanargmax(vec)) / max(n - 1, 1)) if np.isfinite(vmax) else 0.0
    d1 = np.diff(vec, n=1)
    abs_mean = np.nanmean(np.abs(d1)) if d1.size else 0.0
    abs_max = np.nanmax(np.abs(d1)) if d1.size else 0.0
    idx = np.arange(n, dtype=float)
    if np.any(np.isfinite(vec)) and np.sum(np.isfinite(vec)) > 1:
        slope = np.polyfit(idx[np.isfinite(vec)], vec[np.isfinite(vec)], 1)[0]
    else:
        slope = 0.0
    auc_norm = np.nansum(vec) / n
    d2 = np.diff(vec, n=2)
    abs_mean2 = np.nanmean(np.abs(d2)) if d2.size else 0.0
    return np.array([vmax, vmin, delta, mean, std, time_to_peak, abs_mean, abs_max, slope, auc_norm + 0.0*abs_mean2])

# Turn matrix of test rows into the N×10 feature matrix

def featurize_matrix(matrix: np.ndarray) -> np.ndarray:
    X = []
    for row in matrix:
        s = pd.Series(row.astype(float)).replace([np.inf, -np.inf], np.nan)
        s = s.interpolate(limit_direction="both").bfill().ffill()
        X.append(series_feature(s))
    return np.vstack(X) if len(X) else np.empty((0, 10))

# Main

if __name__ == "__main__":
    with open("model.pkl", "rb") as f: model = pickle.load(f)
    test = pd.read_csv("test.csv", header=None)
    if test.shape[1] != 24:
        num_cols = test.select_dtypes(include=[np.number]).columns
        test = test[num_cols[-24:]]
    matrix = test.to_numpy(dtype=float)
    X_test = featurize_matrix(matrix)
    y_pred = model.predict(X_test).astype(int)
    pd.DataFrame(y_pred).to_csv("Result.csv", header=False, index=False)
