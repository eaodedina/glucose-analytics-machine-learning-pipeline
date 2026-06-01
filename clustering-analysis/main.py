import numpy as np
import pandas as pd

from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Hyperparameters
ANCHOR_GAP   = "10min"   # CGM anchor tolerance
N_POINTS     = 30        # CGM samples that go into each meal window
BIN_SIZE     = 20        # gram bin width for labels
ALLOW_NAN_K  = 2         # Allow up to K NaNs per window (interpolate)

EXCLUDE_DBSCAN_NOISE_FOR_METRICS = True

# Kmeans tuning
PCA_COMPS_FOR_KMEANS = 2  
WEIGHT_GLOBAL_SCALE   = 1.1

# DBSCAN tuning
DBSCAN_SPACE_SCALE = 1.2
DBSCAN_EPS         = 0.15
DBSCAN_MIN_SAMPLES = 10

# Data processing function
def normalize(s):
    s = str(s).strip().lower().replace("_", " ")
    return " ".join(s.split())

# Function to extract a timestamp column from the DataFrame 
def make_timestamp(df):
    normalize_origin = {normalize(c): c for c in df.columns}
    def get_col(*canidates):
        for canidate in canidates:
            if canidate in normalize_origin:
                return normalize_origin[canidate]
        return None
    # Date + Time
    for dk, tk in [("date","time"), ("device date","device time")]:
        dcol, tcol = get_col(dk), get_col(tk)
        if dcol and tcol:
            dt_col = df[dcol].astype(str).str.strip() + " " + df[tcol].astype(str).str.strip()
            ts = pd.to_datetime(dt_col, format="%m/%d/%Y %H:%M:%S", errors="coerce")
            if ts.notna().sum() == 0:
                ts = pd.to_datetime(dt_col, errors="coerce")
            return ts
  
    for key in ("date time stamp","timestamp","datetime","new device time","device time"):
        col = normalize_origin.get(key)
        if col:
            s = df[col]
            if not str(s.dtype).startswith("datetime64"):
                s = pd.to_datetime(s, errors="coerce")
            return s
    raise ValueError("Could not find Date/Time or Timestamp columns.")

# Function to extract a glucose column from the CGM DataFrame
def get_glucose_col(df):
    for c in ["Glucose","Sensor Glucose (mg/dL)","Glucose Value (mg/dL)","CGM"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if "glucose" in c.lower():
            return pd.to_numeric(df[c], errors="coerce")
    raise ValueError("Could not find CGM glucose column in CGM DataFrame.")

# Function to extract a carb input column from the insulin DataFrame
def get_carb_col(df):
    for c in ["BWZ Carb Input (grams)","Carb Input (grams)","Carb Input (g)"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        n = normalize(c)
        if "carb" in n and ("gram" in n or "(g)" in n) and "ratio" not in n:
            return pd.to_numeric(df[c], errors="coerce")
    raise ValueError("Carb grams column not found (expected 'BWZ Carb Input (grams)').")

# Load datasets
def load_dataset(cgm_path, insulin_path):
    return pd.read_csv(cgm_path, low_memory=False), pd.read_csv(insulin_path, low_memory=False)

# Extract a CGM window aligned to a meal time
def meal_window(cgm, start, n_points=N_POINTS, ts_col="timestamp", max_anchor_gap=ANCHOR_GAP, allow_nan_k=ALLOW_NAN_K):
    g = cgm.copy()
    if ts_col not in g.columns:
        raise ValueError(f"CGM is missing '{ts_col}' column.")
    g[ts_col] = pd.to_datetime(g[ts_col], errors="coerce")
    g = g.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    glucose = get_glucose_col(g)

    start = pd.to_datetime(start)
    idx = g[ts_col].searchsorted(start, side="left")
    if idx == len(g): return pd.Series(dtype=float)
    canidate = idx
    if idx > 0:
        prev_diff = abs((start - g.at[idx-1, ts_col]).total_seconds())
        next_diff = abs((g.at[idx, ts_col] - start).total_seconds())
        canidate = idx-1 if prev_diff <= next_diff else idx

    tolerance = pd.to_timedelta(max_anchor_gap)
    if abs((g.at[canidate, ts_col] - start)) > tolerance: return pd.Series(dtype=float)
    end = canidate + n_points
    if end > len(g): return pd.Series(dtype=float)

    seg = glucose.iloc[canidate:end].reset_index(drop=True)
    if seg.isna().sum() > 0:
        if allow_nan_k and seg.isna().sum() <= allow_nan_k:
            seg = seg.interpolate(limit=allow_nan_k, limit_direction="both")
        else:
            return pd.Series(dtype=float)
    return seg

# Extract multiple CGM windows aligned to meal times
def get_meal(cgm, meal_times, n_points=N_POINTS, ts_col="timestamp", max_anchor_gap=ANCHOR_GAP, allow_nan_k=ALLOW_NAN_K, return_times=False):
    kept = dropped_far = dropped_tail = dropped_nan = 0
    rows, used_times = [], []
    g = cgm.copy()
    g[ts_col] = pd.to_datetime(g[ts_col], errors="coerce")
    g = g.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    for t in pd.to_datetime(pd.Series(meal_times)).dropna().sort_values():
        seg = meal_window(g, t, n_points=n_points, ts_col=ts_col, max_anchor_gap=max_anchor_gap, allow_nan_k=allow_nan_k)
        if seg.empty:
            idx = g[ts_col].searchsorted(t, side="left")
            if idx == len(g): dropped_tail += 1; continue
            canidate = idx if idx == 0 else (idx-1 if abs((t - g.at[idx-1, ts_col]).total_seconds()) <= abs((g.at[idx, ts_col] - t).total_seconds()) else idx)
            if abs((g.at[canidate, ts_col] - t)) > pd.to_timedelta(max_anchor_gap): dropped_far += 1; continue
            if canidate + n_points > len(g): dropped_tail += 1; continue
            dropped_nan += 1; continue
        kept += 1; rows.append(seg.values); used_times.append(pd.to_datetime(t))
    X = pd.DataFrame(rows) if rows else pd.DataFrame()
    return (X, pd.Series(used_times, name="timestamp")) if return_times else X

# Label meals into carb bins
def label_bins(meals_df, carb_col, bin_size=BIN_SIZE):
    carbs = pd.to_numeric(meals_df[carb_col], errors="coerce")
    lo, hi = carbs.min(), carbs.max()
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(dtype="Int64"), []
    n_bins = max(int(np.ceil((hi - lo) / bin_size)), 1)
    edges = [lo + i*bin_size for i in range(n_bins+1)]
    if edges[-1] < hi: edges.append(hi)
    labels = pd.cut(carbs, bins=edges, include_lowest=True, right=True, labels=False)
    return labels.astype("Int64"), edges

# Featurization of CGM segments
def series_feature(x: pd.Series) -> np.ndarray:
    vec = x.values.astype(float)
    n = len(vec)
    if n == 0 or np.all(np.isnan(vec)): return np.full(10, np.nan)
    vmax = np.nanmax(vec)
    vmin = np.nanmin(vec)
    delta = vmax - vmin
    mean = np.nanmean(vec)
    std = np.nanstd(vec)
    time_to_peak = (int(np.nanargmax(vec)) / max(n - 1, 1)) if np.isfinite(vmax) else 0.0
    d1 = np.diff(vec, n=1); abs_mean = np.nanmean(np.abs(d1)) if d1.size else 0.0
    abs_max = np.nanmax(np.abs(d1)) if d1.size else 0.0
    idx = np.arange(n, dtype=float)
    if np.any(np.isfinite(vec)) and np.sum(np.isfinite(vec)) > 1:
        slope = np.polyfit(idx[np.isfinite(vec)], vec[np.isfinite(vec)], 1)[0]
    else:
        slope = 0.0
    auc_norm = np.nansum(vec) / n
    d2 = np.diff(vec, n=2)
    return np.array([vmax, vmin, delta, mean, std, time_to_peak, abs_mean, abs_max, slope, auc_norm])

# Featurize multiple segments
def featurize(segments):
    feature = [series_feature(pd.Series(s) if not isinstance(s, pd.Series) else s) for s in segments]
    X = np.vstack([f for f in feature if np.all(np.isfinite(f))]) if feature else np.empty((0, 10))
    return X

# Function to compute the Sum of Squared Errors for clustering labels
def compute_sse(X, labels, exclude_noise=False):
    X = np.asarray(X)
    labels = np.asarray(labels)

    if exclude_noise:
        mask = labels != -1
        if not mask.any():
            return float("inf")
        X = X[mask]
        labels = labels[mask]

    sse = 0.0
    for l in np.unique(labels):
        idx = labels == l
        if not idx.any():
            continue
        centroid = X[idx].mean(axis=0, keepdims=True)
        diffs = X[idx] - centroid
        sse += float((diffs * diffs).sum())
    return sse

# Function to create a contingency matrix from true bins and cluster labels
def contingency_matrix(y_bins, labels):
    y_bins = np.asarray(y_bins, dtype=int)
    labels = np.asarray(labels)
    unique_bins = np.unique(y_bins)
    unique_cluster = np.unique(labels)
    bin_to_col = {b:i for i,b in enumerate(unique_bins)}
    cluster_to_row = {c:i for i,c in enumerate(unique_cluster)}
    M = np.zeros((len(unique_cluster), len(unique_bins)), dtype=int)
    for b, c in zip(y_bins, labels):
        M[cluster_to_row[c], bin_to_col[b]] += 1
    return M, unique_cluster, unique_bins

# Function to compute entropy from a contingency matrix
def entropy_calc(M):
    N = M.sum()
    if N == 0: return 0.0
    ent = 0.0
    for i in range(M.shape[0]):
        row = M[i]; n_i = row.sum()
        if n_i == 0: continue
        p = row / n_i
        ent_i = -np.sum([pi*np.log2(pi) for pi in p if pi > 0])
        ent += (n_i / N) * ent_i
    return float(ent)

# Function to compute purity from a contingency matrix
def purity_calc(M):
    N = M.sum()
    if N == 0: return 0.0
    return float(M.max(axis=1).sum() / N)

# Main 
def main():
    cgm_path, insulin_path = "CGMData.csv", "InsulinData.csv"

    cgm_raw, insulin_raw = load_dataset(cgm_path, insulin_path)

    # CGM timestamps
    cgm = cgm_raw.copy()
    cgm["timestamp"] = make_timestamp(cgm)

    # Insulin carbs + timestamps
    insulin = insulin_raw.copy()
    carb = get_carb_col(insulin)
    insulin["Carbs"] = carb
    print("Carb non-null count:", int(carb.notna().sum()))
    print("Raw meal events >0g:", int((carb > 0).sum()))
    insulin["timestamp"] = make_timestamp(insulin)

    # Ground truth bins with 2hour policy
    meals_raw = insulin[(insulin["Carbs"] > 0) & insulin["timestamp"].notna()].copy()
    meals_dedup = meals_raw.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    keep_mask, last_t = [], None
    for t in meals_dedup["timestamp"]:
        if last_t is None or (t - last_t) >= timedelta(hours=2):
            keep_mask.append(True); last_t = t
        else:
            keep_mask.append(False)
    meals_kept = meals_dedup.loc[keep_mask].copy()
    print("After 2-hour policy:", len(meals_kept))
    y, edges = label_bins(meals_kept, carb_col="Carbs", bin_size=BIN_SIZE)
    edges_print = [float(x) for x in edges]
    counts_print = {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}
    print("bin edges:", edges_print)
    print("class counts:", counts_print)

    # CGM windows aligned to meals_kept
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    cgm = cgm.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    Xwindow, used_times = get_meal(cgm, meals_kept["timestamp"], n_points=N_POINTS, ts_col="timestamp", max_anchor_gap=ANCHOR_GAP, allow_nan_k=ALLOW_NAN_K, return_times=True)
    print("window matrix:", Xwindow.shape)

    # Align labels to used_times
    y_by_ts = pd.Series(y.to_numpy(), index=pd.to_datetime(meals_kept["timestamp"].values))
    y_aligned = y_by_ts.reindex(pd.to_datetime(used_times.values))
    mask = ~y_aligned.isna()
    Xwindow = Xwindow.loc[mask.values].reset_index(drop=True)
    y_aligned = y_aligned.loc[mask.values].astype(int).reset_index(drop=True)
    print("aligned windows:", Xwindow.shape, "| labels:", y_aligned.shape)

    segments = [pd.Series(row) for _, row in Xwindow.iterrows()]
    Xfeature = featurize(segments)

    present_bins = np.unique(y_aligned.to_numpy())
    n_clusters = max(len(present_bins), 2)
    print(f"KMeans on PCA space (n_clusters={n_clusters})")

    # Kmeans path: MinMaxScaler + PCA + feature weights + KMeans
    MMScaler = MinMaxScaler()
    X_scaled = MMScaler.fit_transform(Xfeature)

    # order = [vmax, vmin, delta, mean, std, time_to_peak, abs_mean, abs_max, slope, auc_norm]
    base_weights = np.array([0.5, 0.4, 3.6, 0.6, 0.5, 0.6, 0.6, 0.6, 0.5, 3.2], dtype=float)
    weights = base_weights * WEIGHT_GLOBAL_SCALE
    X_weighted = X_scaled * weights

    pca_km = PCA(n_components=PCA_COMPS_FOR_KMEANS, random_state=42)
    X_kmeans = pca_km.fit_transform(X_weighted)
   
    km = KMeans(n_clusters=n_clusters, n_init=50, max_iter=500, random_state=42)
    km_labels = km.fit_predict(X_kmeans)

    # SSE for KMeans measured in the PCA space used
    centroids = km.cluster_centers_[km_labels]
    sse_kmeans = float(((X_kmeans - centroids) ** 2).sum())

    # DBSCAN path: PCA + with fixed hyperparams
    X_db = X_kmeans * DBSCAN_SPACE_SCALE
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    db_labels = db.fit_predict(X_db)

    sse_dbscan = compute_sse(X_db, db_labels, exclude_noise=True)

    # Entropy and Purity from bin/cluster matrices
    mk, rk, ck = contingency_matrix(y_aligned.to_numpy(), km_labels)
    md_raw, rd_raw, cd = contingency_matrix(y_aligned.to_numpy(), db_labels)

    # Exclude DBSCAN noise row (-1) for metrics
    if EXCLUDE_DBSCAN_NOISE_FOR_METRICS and (-1 in rd_raw):
        keep_rows = [i for i, r in enumerate(rd_raw) if r != -1]
        md = md_raw[keep_rows, :]
        rd = rd_raw[keep_rows]
    else:
        md, rd = md_raw, rd_raw

    entropy_kmeans = entropy_calc(mk)
    entropy_dbscan = entropy_calc(md)
    purity_kmeans = purity_calc(mk)
    purity_dbscan = purity_calc(md)

    # Results output
    result_matrix = np.array([[sse_kmeans, sse_dbscan, entropy_kmeans, entropy_dbscan, purity_kmeans, purity_dbscan]], dtype=float)
    pd.DataFrame(result_matrix).to_csv("Result.csv", header=False, index=False)
   
if __name__ == "__main__":
    main()