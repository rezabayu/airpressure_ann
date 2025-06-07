from pathlib import Path
import pandas as pd, numpy as np, json, click
from sklearn.preprocessing import MinMaxScaler

RAIN      = "RAINFALL_LAST_MM"        # change if your file uses a different name
TARGETS   = ("PRESSURE_QFF_MB_DERIVED", "PRESSURE_QFE_MB_DERIVED")            # unscaled but must be gap-free
K_MAD     = 3.5                       # outlier threshold

# ───────────────── column helpers ─────────────────
def split_columns(df):
    meta_cols = [c for c in df.columns
                 if c.startswith(("WMO", "STATION")) or df[c].dtype == "object"]

    numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()

    # Predictors to be scaled (everything numeric except the targets)
    input_cols = [c for c in numeric_all if c not in TARGETS]

    return meta_cols, numeric_all, input_cols

# ─────────────── preprocessing helpers ───────────
def interpolate_numeric(df, numeric_cols):
    """Linear interpolation in time for every numeric column passed."""
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit_direction="both")
    return df

def fill_rain(df):
    """Apply rule-3 filling for RAIN column."""
    month_med = (df[RAIN]
                 .groupby(df.index.month)
                 .transform("median"))

    mask = df[RAIN].isna()
    for ts in df.index[mask]:
        if not df.loc[ts].drop(RAIN).isna().any():
            df.at[ts, RAIN] = 0.0
        else:
            val = month_med.loc[ts]
            df.at[ts, RAIN] = 0.0 if np.isnan(val) else val
    return df

def month_median_cap(df, numeric_cols, k=K_MAD):
    for col in numeric_cols:
        for m, sub in df[col].groupby(df.index.month):
            med = sub.median()
            mad = np.median(np.abs(sub - med))
            if mad == 0:
                continue
            hi, lo = med + k*mad, med - k*mad
            mask = (df.index.month == m) & ((df[col] > hi) | (df[col] < lo))
            df.loc[mask, col] = med
    return df

def scale_inputs(df, input_cols, feature_range=(-1, 1)):
    scaler = MinMaxScaler(feature_range)
    df_scaled = df.copy()
    df_scaled[input_cols] = scaler.fit_transform(df[input_cols])
    return df_scaled, scaler

# ───────────────────── CLI ────────────────────────
@click.command()
@click.option("--infile",  type=click.Path(exists=True), required=True)
@click.option("--outfile", type=click.Path(),            required=True)
@click.option("--meta",    type=click.Path(),            required=True)
def main(infile, outfile, meta):
    df = pd.read_parquet(infile)

    # Ensure timestamp index
    if "ts" in df.columns:
        df = df.set_index(pd.to_datetime(df["ts"]))
    elif "DATE_OBS" in df.columns:
        df = df.set_index(pd.to_datetime(df["DATE_OBS"]))
    df = df.sort_index()

    meta_cols, numeric_cols, input_cols = split_columns(df)

    # 1  Interpolate *all* numeric cols except rainfall (targets included)
    non_rain = [c for c in numeric_cols if c != RAIN]
    df = interpolate_numeric(df, non_rain)

    # 2  Rainfall rule
    df = fill_rain(df)

    # 3  Month-aware outlier capping on every numeric col (incl. rainfall & targets)
    df = month_median_cap(df, numeric_cols)

    # 4  Final NaN guard (numeric only)
    still_missing = df[numeric_cols].isna().sum()
    if still_missing.any():
        print("Columns still containing NaNs:")
        print(still_missing[still_missing > 0])
        raise ValueError("Missing values remain after cleaning!")

    # 5  Min-Max scaling (predictor inputs only)
    df_scaled, scaler = scale_inputs(df, input_cols)

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    df_scaled.reset_index().to_parquet(outfile, index=False)

    Path(meta).parent.mkdir(parents=True, exist_ok=True)
    with open(meta, "w") as f:
        json.dump({
            "feature_range": scaler.feature_range,
            "min_": scaler.data_min_.tolist(),
            "max_": scaler.data_max_.tolist(),
            "input_cols": input_cols}, f, indent=2)

    print("✓  Cleaned & scaled table saved →", outfile)

if __name__ == "__main__":
    main()
