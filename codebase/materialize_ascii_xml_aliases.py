import argparse
from pathlib import Path

import pandas as pd


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Create ASCII-safe XML alias files from an HF manifest mapping."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory that contains the original XML files.")
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("data/manifests/hf_manifest_ascii_mapping.csv"),
        help="CSV produced by prepare_local_hf_xml_manifest.py.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="Whether to create symlinks or physical copies for the ASCII aliases.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    df = pd.read_csv(args.mapping_csv)

    created = 0
    skipped = 0
    for row in df.to_dict(orient="records"):
        src = args.data_root / str(row["xml_path"])
        dst = args.data_root / str(row["ascii_xml_path"])
        if not src.exists():
            skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            skipped += 1
            continue

        if args.mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            dst.write_bytes(src.read_bytes())
        created += 1

    print(
        {
            "mapping_csv": str(args.mapping_csv),
            "mode": args.mode,
            "created": int(created),
            "skipped": int(skipped),
        }
    )


if __name__ == "__main__":
    main()
