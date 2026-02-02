import re
import ast
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ======================
# 你只需要改这三个路径
# ======================
XLSX_PATH = r"/path/to/patient_table.xlsx"
TXT_DIR   = r"/path/to/raw_txt_folder"
OUT_DIR   = r"/path/to/processed_out"

FS = 100  # PPG sampling rate

FNAME_RE = re.compile(r"^(?P<name>.+?)_(?P<ts>\d+?)_(?P<exam>.+?)\.txt$", re.IGNORECASE)

def open_text_robust(path: Path):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.open("r", encoding=enc, errors="strict")
        except Exception:
            continue
    return path.open("r", encoding="utf-8", errors="ignore")

def parse_filename(fname: str):
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return m.group("name").strip(), m.group("ts").strip(), m.group("exam").strip()

def safe_ascii(s: str) -> str:
    # 仅保留字母数字/下划线/短横线，其余全部替换为 _
    s = str(s)
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.ASCII)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def parse_line(line: str):
    s = line.strip()
    if not s:
        return None
    if s.endswith(","):
        s = s[:-1]
    return ast.literal_eval(s)

def main():
    out_root = Path(OUT_DIR)
    npy_root = out_root / "npy"
    log_root = out_root / "logs"
    npy_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    # 1) 读 xlsx
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    need_cols = {"Name", "LVEF", "ID"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"xlsx 缺列: {miss}, 现有列: {list(df.columns)}")

    df["Name"] = df["Name"].astype(str).str.strip()
    df["ID"]   = df["ID"].astype(str).str.strip()

    name_groups = df.groupby("Name", dropna=False)

    txt_dir = Path(TXT_DIR)
    txt_files = sorted(txt_dir.glob("*.txt"))

    records = []
    bad_filename_count = 0

    for p in tqdm(txt_files, desc="Parsing txt"):
        meta = parse_filename(p.name)
        if meta is None:
            bad_filename_count += 1
            continue
        name, ts, exam = meta

        # 2) 用 Name 在 xlsx 找 ID/LVEF（同名可能多条）
        if name in name_groups.groups:
            sub = name_groups.get_group(name)
            candidate_ids = list(sub["ID"].astype(str).unique())
            id_ambiguous = 1 if len(candidate_ids) != 1 else 0
            patient_id = candidate_ids[0] if len(candidate_ids) == 1 else ""
            lvef_candidates = list(sub["LVEF"].tolist())
            lvef = lvef_candidates[0] if len(lvef_candidates) > 0 else np.nan
        else:
            candidate_ids = []
            id_ambiguous = 1
            patient_id = ""
            lvef = np.nan
            lvef_candidates = []

        parse_error_lines = 0
        missing_key_lines = 0

        g1_list, g2_list, ir_list, red_list = [], [], [], []

        with open_text_robust(p) as f:
            # 固定跳前 25 行
            for _ in range(25):
                _ = f.readline()

            for line_idx, line in enumerate(f, start=26):
                try:
                    obj = parse_line(line)
                    if obj is None:
                        continue
                except Exception:
                    parse_error_lines += 1
                    continue

                # 你指定的路径：sensorData -> ppgDataArray
                sensor = obj.get("sensorData", {})
                ppg_arr = sensor.get("ppgDataArray", {})
                if not isinstance(ppg_arr, dict):
                    missing_key_lines += 1
                    continue

                try:
                    g1  = np.asarray(ppg_arr["ppgG1Values"],  dtype=np.float32).reshape(-1)
                    g2  = np.asarray(ppg_arr["ppgG2Values"],  dtype=np.float32).reshape(-1)
                    ir  = np.asarray(ppg_arr["ppgIRValues"],  dtype=np.float32).reshape(-1)
                    red = np.asarray(ppg_arr["ppgRedValues"], dtype=np.float32).reshape(-1)
                except Exception:
                    missing_key_lines += 1
                    continue

                # 对齐长度（医院数据偶尔某一路少1-2个点，直接裁最短）
                L = min(len(g1), len(g2), len(ir), len(red))
                if L <= 0:
                    missing_key_lines += 1
                    continue

                g1_list.append(g1[:L])
                g2_list.append(g2[:L])
                ir_list.append(ir[:L])
                red_list.append(red[:L])

        out_path = ""
        n_samples = 0
        duration_sec = 0.0

        if len(g1_list) > 0:
            g1_all  = np.concatenate(g1_list, axis=0)
            g2_all  = np.concatenate(g2_list, axis=0)
            ir_all  = np.concatenate(ir_list, axis=0)
            red_all = np.concatenate(red_list, axis=0)

            X = np.stack([g1_all, g2_all, ir_all, red_all], axis=1)  # [T,4]
            n_samples = int(X.shape[0])
            duration_sec = n_samples / FS

            # 3) npy 存储：不含中文名
            pid = patient_id if patient_id else "unknown"
            sub_dir = npy_root / pid
            sub_dir.mkdir(parents=True, exist_ok=True)

            base = f"{pid}_{ts}_{exam}"
            base = safe_ascii(base)  # 去掉 exam 里可能的中文/特殊符号
            out_path = sub_dir / f"{base}.npy"

            # 防止重名覆盖：存在则追加序号
            if out_path.exists():
                k = 1
                while True:
                    cand = sub_dir / f"{base}_{k}.npy"
                    if not cand.exists():
                        out_path = cand
                        break
                    k += 1

            np.save(out_path, X)

        records.append({
            "npy_path": str(out_path) if out_path else "",
            "patient_id": patient_id,
            "candidate_ids": "|".join(candidate_ids),
            "id_ambiguous": id_ambiguous,
            "name": name,  # csv 里保留，方便你排查；npy 文件名不含中文
            "lvef": lvef,
            "lvef_candidates": "|".join([str(x) for x in lvef_candidates]),
            "exam_ts": ts,
            "exam_no": exam,
            "fs": FS,
            "n_samples": n_samples,
            "duration_sec": duration_sec,
            "parse_error_lines": parse_error_lines,
            "missing_key_lines": missing_key_lines,
            "src_txt_path": str(p),
        })

    manifest = pd.DataFrame(records)
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    summary = {
        "bad_filename_count": bad_filename_count,
        "n_record_in_manifest": len(records),
        "n_saved_npy": int((manifest["npy_path"].fillna("") != "").sum()),
        "n_ambiguous": int((manifest["id_ambiguous"] == 1).sum()),
        "xlsx_path": XLSX_PATH,
        "txt_dir": TXT_DIR,
        "out_dir": OUT_DIR,
    }
    with (log_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("manifest:", manifest_path)
    print("summary :", log_root / "summary.json")

if __name__ == "__main__":
    main()
