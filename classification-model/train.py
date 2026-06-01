# Machine Model Training Project
# Enoch Odedina - 1221347699

# Necessary imports

import numpy as np
import pandas as pd
import pickle

from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Helper function for CGM and Insulin CSVs to get one canonical timestamp column

def make_timestamp(df: pd.DataFrame) -> pd.Series:
    date_cols = [c for c in df.columns if c.strip().lower() in ("date", "date..mm.dd.yy.", "cgmdate")]
    time_cols = [c for c in df.columns if c.strip().lower() in ("time", "time..hh.mm.ss.", "cgmtime")]
    dt_cols   = [c for c in df.columns if ("datetime" in c.strip().lower()) or ("timestamp" in c.strip().lower())]

    if dt_cols:
        time_stamp = pd.to_datetime(df[dt_cols[0]], errors="coerce",)
    elif date_cols and time_cols:
        time_stamp = pd.to_datetime(df[date_cols[0]].astype(str).str.strip() + " " + df[time_cols[0]].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        if len(df.columns) >= 2:
            time_stamp = pd.to_datetime(df.iloc[:, 0].astype(str).str.strip() + " " + df.iloc[:, 1].astype(str).str.strip(),
                errors="coerce",
            )
        else:
            time_stamp = pd.to_datetime(pd.Series([np.nan] * len(df)))
    return time_stamp

# Helper functions to identify CGM glucose column

def get_glucose_col(df: pd.DataFrame) -> pd.Series:
    choices = [
        "Sensor Glucose (mg/dL)", "Sensor Glucose (mg/dl)",
    ]
    for c in df.columns:
        if c.strip().lower() in [cc.lower() for cc in choices]:
            g = pd.to_numeric(df[c], errors="coerce")
            g.name = "Glucose"
            return g
        
    # Fallback to last numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        g = pd.to_numeric(df[num_cols[-1]], errors="coerce")
        g.name = "Glucose"
        return g
    return pd.Series(np.nan, name="Glucose", index=df.index)

# Helper functions to identify Insulin carb column

def get_carb_col(df: pd.DataFrame) -> pd.Series:
    for c in df.columns:
        cl = c.strip().lower()
        if "carb" in cl and (("gram" in cl) or "(g" in cl):
            s = pd.to_numeric(df[c], errors="coerce")
            s.name = c
            return s
        
    # Fallback to first numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        return pd.to_numeric(df[num_cols[0]], errors="coerce")
    return pd.Series(np.nan, index=df.index)

# Load datasets into time-sorted DataFrames for pipeline

def load_dataset(cgm_data: str, insulin_data: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cgm_raw = pd.read_csv(cgm_data, low_memory=False)
    insulin_raw = pd.read_csv(insulin_data, low_memory=False)

    cgm = pd.DataFrame({
        "date_time_stamp": make_timestamp(cgm_raw),
        "Glucose": get_glucose_col(cgm_raw),
    }).dropna(subset=["date_time_stamp"]).sort_values("date_time_stamp").reset_index(drop=True)

    insulin = pd.DataFrame({
        "date_time_stamp": make_timestamp(insulin_raw),
        "Carbs": get_carb_col(insulin_raw),
    }).dropna(subset=["date_time_stamp"]).sort_values("date_time_stamp").reset_index(drop=True)

    return cgm, insulin

# Meal and No-Meal extraction time rules

FIVE_MIN = pd.Timedelta(minutes=5)
THIRTY_MIN = pd.Timedelta(minutes=30)
ONE_HALF_HR = pd.Timedelta(hours=1, minutes=30)
TWO_HR = pd.Timedelta(hours=2)
FOUR_HR = pd.Timedelta(hours=4)

# Helper function to find meal start times from insulin data

def meal_start_time(insulin: pd.DataFrame) -> List[pd.Timestamp]:
    meal_rows = insulin.loc[insulin["Carbs"].fillna(0) > 0, ["date_time_stamp"]].copy()
    meal_rows = meal_rows.drop_duplicates(subset=["date_time_stamp"])
    return list(meal_rows["date_time_stamp"].sort_values())

# Function to extract a CGM segment of n_points starting from start time, aligned to 5 min grid

def meal_window(cgm: pd.DataFrame, start: pd.Timestamp, n_points: int) -> pd.Series:
    five_min_idx = pd.Timedelta(minutes=5)
    # end = start + (n_points - 1) * five_min_idx

    # Exact 5-min grid
    full_index = pd.date_range(start=start, periods=n_points, freq="5min")

    # Slice a slightly bigger window to capture neighbors near the edges
    get_data = cgm.loc[
        (cgm["date_time_stamp"] >= (full_index[0] - five_min_idx)) &
        (cgm["date_time_stamp"] <= (full_index[-1] + five_min_idx)),
        ["date_time_stamp", "Glucose"]
    ].copy()
    if get_data.empty:
        return pd.Series(dtype=float)

    get_data = get_data.sort_values("date_time_stamp")

    # Nearest-neighbor alignment with 5min grid
    grid = pd.DataFrame({"date_time_stamp": full_index})
    aligned = pd.merge_asof(grid, get_data, on="date_time_stamp", tolerance=pd.Timedelta(minutes=2.5), direction="nearest")

    vec = aligned["Glucose"].astype(float)

    # Discard if too many missing values
    max_missing = int(np.floor(0.05 * len(vec)))
    if vec.isna().sum() > max_missing:
        return pd.Series(dtype=float)

    # Interpolate remaining gaps
    vec = vec.interpolate(limit_direction="both").bfill().ffill()

    # Guarantee the output is exactly n_points long with a clean 0 to N-1 index. 
    if len(vec) != n_points:
        return pd.Series(dtype=float)
    return vec.reset_index(drop=True)

# Function to turn each detected meal time into a 2.5 hour CGM segment, 30 points

def get_meal(cgm: pd.DataFrame, meal_times: List[pd.Timestamp]) -> List[pd.Series]:
    meal_times = sorted(meal_times)
    out = []
    for i, tm in enumerate(meal_times):
        # If the immediate next meal is strictly within 2h, skip tm
        if i + 1 < len(meal_times):
            t_next = meal_times[i + 1]
            if (t_next > tm) and (t_next < tm + TWO_HR):
                continue

        exact = False
        if i + 1 < len(meal_times):
            exact = abs((meal_times[i + 1] - tm) - TWO_HR) <= pd.Timedelta(seconds=60)

        start = (tm + ONE_HALF_HR) if exact else (tm - THIRTY_MIN)
        seg = meal_window(cgm, start, n_points=30)
        if not seg.empty:
            out.append(seg)
    return out

# Function to turn dettected no-meal times into 2 hour CGM segments, 24 points

def get_no_meal(cgm: pd.DataFrame, meal_times: List[pd.Timestamp]) -> List[pd.Series]:
    meal_times = sorted(meal_times)
    out = []
    for i, tm in enumerate(meal_times):
        start = tm + TWO_HR
        end   = tm + FOUR_HR
        t_next = meal_times[i + 1] if i + 1 < len(meal_times) else None
        if t_next is not None and t_next < end:
            continue
        seg = meal_window(cgm, start, n_points=24)
        if not seg.empty:
            out.append(seg)
    return out

# Feature extraction for classification

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

    return np.array([vmax, vmin, delta, mean, std, time_to_peak, abs_mean, abs_max, slope, auc_norm])

# Convert a list of time series segments into a single 2D feature matrix for the model

def featurize(segments: List[pd.Series]) -> np.ndarray:
    features = [series_feature(s) for s in segments]
    X = np.vstack([f for f in features if np.all(np.isfinite(f))]) if features else np.empty((0, 10))
    return X

# Train classifier with scaling and cross-validation, then save model

def train_and_save(X: np.ndarray, y: np.ndarray, model_path: str = "model.pkl") -> None:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=False, random_state=0)),
    ])

    # Determine safe number of CV splits based on the smallest class count
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    min_class = counts.min() if counts.size else 0

    if min_class >= 2:
        n_splits = min(10, int(min_class))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        accuracy = cross_val_score(pipeline, X, y, scoring="accuracy", cv=skf)
        f1  = cross_val_score(pipeline, X, y, scoring="f1", cv=skf)
        print(f"CV ({n_splits}-fold) Accuracy: {accuracy.mean():.3f} ± {accuracy.std():.3f}")
        print(f"CV ({n_splits}-fold) F1: {f1.mean():.3f} ± {f1.std():.3f}")
    else:
        print("Not enough samples per class for CV; fitting without cross-validation. Class counts:", class_counts)

    pipeline.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved trained pipeline to {model_path}")

# Main

if __name__ == "__main__":
    # Load both datasets
    cgm1, insulin1 = load_dataset("CGMData.csv", "InsulinData.csv")
    cgm2, insulin2 = load_dataset("CGM_patient2.csv", "Insulin_patient2.csv")

    # Meal start times
    meals1 = meal_start_time(insulin1)
    meals2 = meal_start_time(insulin2)

    # Extract segments from both datasets
    meal_seg_1 = get_meal(cgm1, meals1)
    nomeal_seg_1 = get_no_meal(cgm1, meals1)

    meal_seg_2 = get_meal(cgm2, meals2)
    nomeal_seg_2 = get_no_meal(cgm2, meals2)

    # Combine and featurize
    meal_segments = meal_seg_1 + meal_seg_2
    nomeal_segments = nomeal_seg_1 + nomeal_seg_2

    X_meal = featurize(meal_segments)
    X_no_meal = featurize(nomeal_segments)

    # Labels
    y_meal = np.ones(len(X_meal), dtype=int)
    y_no_meal = np.zeros(len(X_no_meal), dtype=int)

    # Aggregate
    X = np.vstack([X_meal, X_no_meal])
    y = np.concatenate([y_meal, y_no_meal])

    if X.size == 0:
        raise RuntimeError("No training data produced. Verify CSV columns and meal times.")

    # Train and save model
    train_and_save(X, y, model_path="model.pkl")