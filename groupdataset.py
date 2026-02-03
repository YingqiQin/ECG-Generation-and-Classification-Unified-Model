import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def infer_subject_id(npy_path: str) -> str:
    p = Path(str(npy_path))
    sid = p.parent.name
    return sid

def load_patient_table(csv_path: str, thr: float, positive_if_ge: bool = True):
    df = pd.read_csv(csv_path)
    if "npy_path" not in df.columns or "LVEF" not in df.columns:
        df = df.iloc[:, :2].copy()
        df.columns = ["npy_path", "LVEF"]

    df["npy_path"] = df["npy_path"].astype(str)
    df["LVEF"] = pd.to_numeric(df["LVEF"], errors="coerce")
    df = df.dropna(subset=["LVEF"]).reset_index(drop=True)

    df["subject_id"] = df["npy_path"].apply(infer_subject_id)

    if positive_if_ge:
        df["y"] = (df["LVEF"] >= thr).astype(int)
    else:
        df["y"] = (df["LVEF"] <= thr).astype(int)

    # patient-level label (you said one LVEF per patient, but keep robust)
    pat = df.groupby("subject_id", as_index=False)["y"].max()
    return df, pat

def make_holdout_test_subjects(pat_df: pd.DataFrame,
                               n_test_minority: int = 6,
                               test_majority_ratio: int = 10,
                               seed: int = 42):
    """
    pat_df has columns: subject_id, y  (y=1 is majority if positive=ge40 in your original task)
    We define minority as the less frequent class in pat_df.
    """
    rng = np.random.default_rng(seed)
    counts = pat_df["y"].value_counts().to_dict()
    minority_label = 0 if counts.get(0, 0) < counts.get(1, 0) else 1
    majority_label = 1 - minority_label

    minority_subj = pat_df.loc[pat_df["y"] == minority_label, "subject_id"].tolist()
    majority_subj = pat_df.loc[pat_df["y"] == majority_label, "subject_id"].tolist()

    if len(minority_subj) < n_test_minority:
        raise ValueError(f"Not enough minority patients for test: have {len(minority_subj)}, need {n_test_minority}")

    test_minority = rng.choice(minority_subj, size=n_test_minority, replace=False).tolist()

    n_test_majority = min(len(majority_subj), n_test_minority * test_majority_ratio)
    test_majority = rng.choice(majority_subj, size=n_test_majority, replace=False).tolist()

    test_subjects = set(test_minority) | set(test_majority)
    return test_subjects, minority_label

def make_cv_folds(pat_df_train: pd.DataFrame, n_splits: int = 3, seed: int = 42):
    """
    Stratified CV at patient level.
    """
    subjects = pat_df_train["subject_id"].values
    y = pat_df_train["y"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(subjects, y)):
        tr_sub = set(subjects[tr_idx])
        va_sub = set(subjects[va_idx])
        folds.append((tr_sub, va_sub))
    return folds
