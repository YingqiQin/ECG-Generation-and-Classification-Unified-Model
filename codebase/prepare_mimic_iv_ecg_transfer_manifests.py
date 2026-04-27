import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _first_present_column(df: pd.DataFrame, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_signal_format(path_value: str, explicit_format: Optional[str]) -> str:
    if explicit_format is not None and str(explicit_format).strip():
        fmt = str(explicit_format).strip().lower()
        if fmt not in {"wfdb", "npy"}:
            raise ValueError("Unsupported path format '{}'. Use 'wfdb' or 'npy'.".format(explicit_format))
        return fmt

    suffix = Path(str(path_value)).suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix in {".hea", ".dat"} or suffix == "":
        return "wfdb"
    raise ValueError("Could not infer path format from '{}'.".format(path_value))


def _normalize_wfdb_path(path_value: str, file_name_value: Optional[str], study_id_value: Optional[str]) -> str:
    path = Path(str(path_value).strip())

    if path.suffix.lower() in {".hea", ".dat"}:
        return path.with_suffix("").as_posix()

    if path.suffix == "":
        if path.name.startswith("s") and path.name[1:].isdigit():
            if file_name_value is not None and str(file_name_value).strip():
                return (path / str(file_name_value).strip()).as_posix()
            if study_id_value is not None and str(study_id_value).strip():
                return (path / str(study_id_value).strip()).as_posix()
        return path.as_posix()

    return path.as_posix()


def _build_manifest(df: pd.DataFrame, args) -> pd.DataFrame:
    path_col = args.path_column or _first_present_column(
        df,
        ["path", "waveform_path", "record_path", "signal_path", "file_path", "npy_dst", "npy_path"],
    )
    file_name_col = args.file_name_column or _first_present_column(df, ["file_name", "record_name"])
    subject_col = args.subject_column or _first_present_column(df, ["subject_id", "patient_id"])
    study_col = args.study_column or _first_present_column(df, ["study_id", "record_id", "ecg_id", "xml_file"])
    fs_col = args.fs_column or _first_present_column(df, ["sampling_rate_hz", "fs", "sample_rate", "sampling_frequency"])
    n_samples_col = args.n_samples_column or _first_present_column(df, ["n_samples", "num_samples"])
    format_col = args.format_column or _first_present_column(df, ["path_format", "signal_format", "waveform_format"])

    if path_col is None:
        raise ValueError("Could not find waveform path column. For official MIMIC-IV-ECG use --path-column path.")
    if subject_col is None:
        raise ValueError("Could not find subject id column. For official MIMIC-IV-ECG use --subject-column subject_id.")
    if study_col is None:
        raise ValueError("Could not find study id column. For official MIMIC-IV-ECG use --study-column study_id.")

    keep_rows = []
    for _, row in df.iterrows():
        raw_path = row[path_col]
        if pd.isna(raw_path):
            continue

        explicit_format = args.path_format
        if explicit_format is None and format_col is not None and not pd.isna(row[format_col]):
            explicit_format = str(row[format_col])

        signal_format = _normalize_signal_format(str(raw_path), explicit_format)
        file_name_value = None if file_name_col is None or pd.isna(row[file_name_col]) else str(row[file_name_col])
        study_id_value = None if pd.isna(row[study_col]) else str(row[study_col])
        if signal_format == "wfdb":
            waveform_path = _normalize_wfdb_path(str(raw_path), file_name_value, study_id_value)
        else:
            waveform_path = str(raw_path).strip()

        fs_value = args.default_fs
        if fs_col is not None and not pd.isna(row[fs_col]):
            fs_value = int(float(row[fs_col]))

        if fs_value not in {500, 1000}:
            continue

        keep_row = {
            "waveform_path": waveform_path,
            "path_format": signal_format,
            "subject_id": str(row[subject_col]),
            "study_id": study_id_value,
            "sampling_rate_hz": int(fs_value),
        }
        if n_samples_col is not None and not pd.isna(row[n_samples_col]):
            keep_row["n_samples"] = int(float(row[n_samples_col]))
        if file_name_value is not None:
            keep_row["file_name"] = file_name_value
        keep_rows.append(keep_row)

    manifest = pd.DataFrame(keep_rows)
    manifest = manifest.dropna(subset=["waveform_path", "subject_id", "study_id", "sampling_rate_hz"]).copy()
    return manifest.reset_index(drop=True)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Prepare patient-split transfer manifests from official MIMIC-IV-ECG metadata such as record_list.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input metadata CSV. For the original PhysioNet release this is usually record_list.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write transfer_train.csv and transfer_valid.csv.")
    parser.add_argument("--path-column", type=str, default=None, help="Waveform path column. For official MIMIC-IV-ECG this is usually 'path'.")
    parser.add_argument("--file-name-column", type=str, default=None, help="Optional file-name column if the CSV stores directory path and record name separately.")
    parser.add_argument("--subject-column", type=str, default=None, help="Subject id column. For official MIMIC-IV-ECG this is usually 'subject_id'.")
    parser.add_argument("--study-column", type=str, default=None, help="Study id column. For official MIMIC-IV-ECG this is usually 'study_id'.")
    parser.add_argument("--fs-column", type=str, default=None, help="Optional sampling-rate column.")
    parser.add_argument("--n-samples-column", type=str, default=None, help="Optional signal-length column.")
    parser.add_argument("--format-column", type=str, default=None, help="Optional format column if the metadata mixes wfdb and npy paths.")
    parser.add_argument("--path-format", type=str, choices=["wfdb", "npy"], default=None, help="Force a single path format for all rows.")
    parser.add_argument("--default-fs", type=int, default=500, help="Sampling rate used when the CSV has no fs column. Official MIMIC-IV-ECG is 500 Hz.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Patient-level validation ratio.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for subject split.")
    return parser


def main():
    args = build_argparser().parse_args()

    df = pd.read_csv(args.input_csv)
    df.columns = [str(c).strip() for c in df.columns]
    manifest = _build_manifest(df, args)

    subjects = manifest["subject_id"].drop_duplicates().to_numpy()
    if len(subjects) < 2:
        raise ValueError("Need at least two unique subjects to create train/valid splits.")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    n_valid = max(1, int(round(len(subjects) * args.valid_ratio)))
    n_valid = min(n_valid, len(subjects) - 1)
    valid_subjects = set(subjects[:n_valid].tolist())

    train_df = manifest[~manifest["subject_id"].isin(valid_subjects)].reset_index(drop=True)
    valid_df = manifest[manifest["subject_id"].isin(valid_subjects)].reset_index(drop=True)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "transfer_train.csv"
    valid_path = out_dir / "transfer_valid.csv"
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "path_format_counts": manifest["path_format"].value_counts().to_dict(),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "train_subjects": int(train_df["subject_id"].nunique()),
                "valid_subjects": int(valid_df["subject_id"].nunique()),
                "train_csv": str(train_path),
                "valid_csv": str(valid_path),
                "example_train_path": None if train_df.empty else str(train_df.iloc[0]["waveform_path"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
