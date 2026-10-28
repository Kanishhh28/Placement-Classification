"""Score a list of registered students with the saved placement model.

    python predict.py --input some_students.csv --output predictions.csv

This script only LOADS the model trained in analysis.ipynb (model.joblib, expected
next to this file) and scores new rows. It never fits anything.

Design notes (see analysis.ipynb for the reasoning):
  * `status_code` and `student_id` are never used as features.
  * Input may have empty cells, missing columns, odd casing, impossible values or
    unseen departments. Every input row still gets a 0/1 prediction.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")

NUMERIC = ["cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops"]
CATEGORICAL = ["department"]
FEATURES = NUMERIC + CATEGORICAL

# Values outside these ranges are impossible -> treated as missing (then imputed).
VALID_RANGE = {
    "cgpa": (0.0, 10.0),
    "backlogs": (0.0, np.inf),
    "internship_months": (0.0, np.inf),
    "coding_score": (0.0, 100.0),
    "projects": (0.0, np.inf),
    "workshops": (0.0, np.inf),
}


def prepare(df, bounds=None):
    """Clean raw student rows into the model's feature frame.

    Used identically at training time (notebook) and prediction time.
    `bounds` = {col: (lo, hi)} learned on the training data; if given, values are
    winsorised to that range so one wild value cannot dominate the linear model.
    """
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC:
        s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        s = s.astype(float)
        lo, hi = VALID_RANGE[col]
        s = s.where((s >= lo) & (s <= hi))          # impossible -> NaN
        s = s.replace([np.inf, -np.inf], np.nan)
        if bounds is not None:
            s = s.clip(*bounds[col])
        out[col] = s
    for col in CATEGORICAL:
        if col in df.columns:
            s = df[col].astype("string").str.strip().str.upper()
            s = s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
            out[col] = s.astype(object).where(s.notna(), np.nan)
        else:
            out[col] = np.nan
    return out[FEATURES]


def load_artifact(path=MODEL_PATH):
    import joblib
    if not os.path.exists(path):
        sys.exit(f"Model file not found: {path}. Run analysis.ipynb first to train and save it.")
    return joblib.load(path)


def predict_frame(df, artifact):
    X = prepare(df, artifact["bounds"])
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    return (proba >= artifact["threshold"]).astype(int), proba


def main():
    ap = argparse.ArgumentParser(description="Predict placement for a list of students.")
    ap.add_argument("--input", required=True, help="CSV of registered students")
    ap.add_argument("--output", required=True, help="where to write predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"student_id": str})
    artifact = load_artifact()

    if len(df):
        pred, _ = predict_frame(df, artifact)
    else:
        pred = np.array([], dtype=int)

    if "student_id" in df.columns:
        ids = df["student_id"]
    else:
        ids = pd.Series([f"ROW{i}" for i in range(len(df))])
    out = pd.DataFrame({"student_id": ids.values, "placed": pred.astype(int)})
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output} "
          f"({int(out['placed'].sum())} predicted placed).")


if __name__ == "__main__":
    main()