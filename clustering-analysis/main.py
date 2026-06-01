# Extracting Time Series Properties of Glucose Levels in Artificial Pancreas Project
# Enoch Odedina - 1221347699

# Sources: docs.python.org
#          pandas.pydata.org
#          numpy.org

# Necessary imports

import pandas as pd
import numpy as np

# Load datasets

cgm_data = pd.read_csv("CGMData.csv", low_memory=False)
insulin_data = pd.read_csv("InsulinData.csv", low_memory=False)

# Data Preprocessing (parsing date and time stamps)

cgm_data["date_time_stamp"] = pd.to_datetime(cgm_data["Date"].astype(str) + " " + cgm_data["Time"].astype(str), errors="coerce")
insulin_data["date_time_stamp"] = pd.to_datetime(insulin_data["Date"].astype(str) + " " + insulin_data["Time"].astype(str), errors="coerce")

cgm_data["Glucose"] = pd.to_numeric(cgm_data["Sensor Glucose (mg/dL)"], errors="coerce")

# Find the column that contains the "AUTO MODE ACTIVE PLGM OFF" text

q_col = None
for col in insulin_data.columns:
    if insulin_data[col].astype(str).str.contains("AUTO MODE ACTIVE PLGM OFF", case=False, na=False).any():
        q_col = col
        break
if q_col is None:
    raise RuntimeError("Could not find AUTO MODE ACTIVE PLGM OFF in any InsulinData column")

# Pinpoint the earliest start time in the Insulin dataset

start_automode = insulin_data.loc[
    insulin_data[q_col].astype(str).str.contains("AUTO MODE ACTIVE PLGM OFF", case=False, na=False),
    "date_time_stamp"
].min()

# Align split to the first CGM timestamp at/after the insulin start time

event_time = cgm_data.loc[cgm_data["date_time_stamp"] >= start_automode, "date_time_stamp"]
start_time = event_time.min() if not event_time.empty else None

# Split data into Manual Mode and Auto Mode

if start_time is not None:
    get_manual_data = cgm_data.loc[cgm_data["date_time_stamp"] <  start_time, ["date_time_stamp", "Glucose"]].copy()
    get_auto_data = cgm_data.loc[cgm_data["date_time_stamp"] >= start_time, ["date_time_stamp", "Glucose"]].copy()
else:
    get_manual_data = cgm_data.loc[:, ["date_time_stamp", "Glucose"]].copy()
    get_auto_data = cgm_data.iloc[0:0][["date_time_stamp", "Glucose"]].copy()

# Resample to 5min and interpolate since glucose measurements are collected every 5 mins
# Returns a frame indexed by date_time_stamp on a strict 5min grid,'Sensor Glucose (mg/dL)'
# time-interpolated, with Date/Time columns, and ONLY days that have exactly 288 rows

def resample_func(df_day: pd.DataFrame) -> pd.Series:
    g = (
        df_day.dropna(subset=["date_time_stamp"])
              .set_index("date_time_stamp")
              .sort_index()
              [["Glucose"]]
    )
    # Determine calendar daytime start
    start_day = (g.index.min() if not g.empty else df_day["date_time_stamp"].iloc[0]).normalize()
    idx = pd.date_range(start_day, periods=288, freq="5min")

    # Resample to 5 min then reindex to ensure exactly 288 slots
    sample = g["Glucose"].resample("5min").mean()
    sample = sample.reindex(idx)

    # Interpolate gaps
    sample = sample.interpolate(method="time", limit=6, limit_area="inside")

    return sample

# Boolean masks for a DatetimeIndex to slice per-day CGM series by time of day.

def time_interval_func(df_idx: pd.DatetimeIndex):
    t = pd.Series(df_idx.time)
    wholeday = np.ones(len(df_idx), dtype=bool)
    daytime   = ((t >= pd.to_datetime("06:00").time()) & (t <= pd.to_datetime("23:59").time())).to_numpy()
    overnight = ((t >= pd.to_datetime("00:00").time()) & (t <  pd.to_datetime("06:00").time())).to_numpy()
    return wholeday, daytime, overnight

# The metric vector for a single day CGM series using a pd.Series indexed by date_time_stamp

def metrics_func(df_sample: pd.Series) -> np.ndarray:
    wd, dt, on = time_interval_func(df_sample.index)
    val = df_sample.values
    def counts(mask):
        vector_mask = val[mask]
        return np.array([
            np.sum(vector_mask > 180),
            np.sum(vector_mask > 250),
            np.sum((vector_mask >= 70) & (vector_mask <= 180)),
            np.sum((vector_mask >= 70) & (vector_mask <= 150)),
            np.sum(vector_mask < 70),
            np.sum(vector_mask < 54),
        ], dtype=float)

    denom = 288.0
    overnight = counts(on) / denom * 100.0
    daytime   = counts(dt) / denom * 100.0
    wholeday = counts(wd) / denom * 100.0
    return np.concatenate([overnight, daytime, wholeday])

# Final 18-metric row by averaging per-day metrics.

def mean_func(df_mean: pd.DataFrame) -> np.ndarray:
    if df_mean.empty:
        return np.zeros(18, dtype=float)
    df_mean = df_mean.copy()
    df_mean["DateOnly"] = df_mean["date_time_stamp"].dt.date
    vec = []
    for _, df_day in df_mean.groupby("DateOnly", sort=True):
        sample = resample_func(df_day)
        vec.append(metrics_func(sample))
    return np.nanmean(np.vstack(vec), axis=0) if vec else np.zeros(18, dtype=float)

# The mean metric vectors for Manual and Auto modes using the mean_func function

row1 = mean_func(get_manual_data)
row2 = mean_func(get_auto_data)
matrix = np.vstack([row1, row2])

# Write to CSV
pd.DataFrame(matrix).to_csv("Result.csv", header=False, index=False, float_format="%.1f")
