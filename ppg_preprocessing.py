import os
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
OUT_DIR   = r"/path/to/processed_out"  # 会生成 npy/manifest/logs

FS = 100  # PPG sampling rate

# ----------------------
# 工具：更稳健地读中文 txt
# ----------------------
def open_text_robust(path: Path):
    # 常见医院数据：utf-8 / utf-8-sig / gbk
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.open("r", encoding=enc, errors="strict")
        except Exception:
            continue
    # 兜底：容忍坏字符
    return path.open("r", encoding="utf-8", errors="ignore")

# ----------------------
# 解析：单行 dict（单引号）
# ----------------------
def parse_line_to_obj(line: str):
    line = line.strip()
    if not line:
        return None
    # 有些数据可能末尾多一个逗号
    if line.endswith(","):
        line = line[:-1]
    # 用 literal_eval 解析 python dict 风格字符串
    return ast.literal_eval(line)

# ----------------------
# 文件名解析：中文名_{时间戳}_{检查号}.txt
# ----------------------
FNAME_RE = re.compile(r"^(?P<name>.+?)_(?P<ts>\d+?)_(?P<exam>.+?)\.txt$", re.IGNORECASE)

def parse_filename(fname: str):
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return m.group("name"), m.group("ts"), m.group("exam")

# ----------------------
# 主流程
# ----------------------
def main():
    out_root = Path(OUT_DIR)
    (out_root / "npy").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    # 1) 读 xlsx
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    # 统一列名（防止隐藏空格）
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"Name", "LVEF", "ID"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"xlsx 缺列: {missing}. 现有列: {list(df.columns)}")

    # 清洗
    df["Name"] = df["Name"].astype(str).str.strip()
    df["ID"]   = df["ID"].astype(str).str.strip()
    # 确保是 7 位数字（不满足也保留，但标记）
    df["ID_is_7digits"] = df["ID"].str.fullmatch(r"\d{7}").fillna(False)

    # Name -> rows（可能重复）
    name_groups = df.groupby("Name", dropna=False)

    # 2) 遍历 txt，解析并导出
    records = []
    txt_dir = Path(TXT_DIR)
    txt_files = sorted([p for p in txt_dir.glob("*.txt")])

    bad_fname = 0
    total_parse_err = 0

    for p in tqdm(txt_files, desc="Parsing txt"):
        meta = parse_filename(p.name)
        if meta is None:
            bad_fname += 1
            continue
        name, ts, exam = meta

        # 查患者表
        if name in name_groups.groups:
            sub_df = name_groups.get_group(name)
            candidate_ids = list(sub_df["ID"].astype(str).unique())
            id_ambiguous = 1 if len(candidate_ids) != 1 else 0
            patient_id = candidate_ids[0] if len(candidate_ids) == 1 else ""
            # LVEF：如果同名多行且 LVEF 不一致，先全部保留候选，manifest 里取第一个并额外写候选
            lvef_candidates = list(sub_df["LVEF"].tolist())
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
            # 跳过前 25 行
            for _ in range(25):
                _ = f.readline()

            for line_idx, line in enumerate(f, start=26):
                try:
                    obj = parse_line_to_obj(line)
                    if obj is None:
                        continue
                except Exception:
                    parse_error_lines += 1
                    continue

                # 取 sensorData
                sensor = obj.get("sensorData", None)
                if sensor is None:
                    missing_key_lines += 1
                    continue

                # 四路 PPG
                try:
                    g1  = sensor["ppgG1Values"]
                    g2  = sensor["ppgG2Values"]
                    ir  = sensor["ppgIRValues"]
                    red = sensor["ppgRedValues"]
                except KeyError:
                    missing_key_lines += 1
                    continue

                # 变成 numpy 1d
                g1 = np.asarray(g1, dtype=np.float32).reshape(-1)
                g2 = np.asarray(g2, dtype=np.float32).reshape(-1)
                ir = np.asarray(ir, dtype=np.float32).reshape(-1)
                red = np.asarray(red, dtype=np.float32).reshape(-1)

                # 基本一致性：四路长度应当相同
                Ls = {len(g1), len(g2), len(ir), len(red)}
                if len(Ls) != 1:
                    # 视作坏行，但不致命
                    missing_key_lines += 1
                    continue

                g1_list.append(g1)
                g2_list.append(g2)
                ir_list.append(ir)
                red_list.append(red)

        # 拼接
        if len(g1_list) == 0:
            # 没有效数据，记录一下
            out_path = ""
            n_samples = 0
            duration_sec = 0.0
        else:
            g1_all  = np.concatenate(g1_list, axis=0)
            g2_all  = np.concatenate(g2_list, axis=0)
            ir_all  = np.concatenate(ir_list, axis=0)
            red_all = np.concatenate(red_list, axis=0)

            X = np.stack([g1_all, g2_all, ir_all, red_all], axis=1)  # [T,4]
            n_samples = int(X.shape[0])
            duration_sec = n_samples / FS

            # 输出目录：优先 patient_id，否则用 name
            sub_folder = patient_id if patient_id else name
            save_dir = out_root / "npy" / sub_folder
            save_dir.mkdir(parents=True, exist_ok=True)

            out_path = save_dir / f"{name}_{ts}_{exam}.npy"
            np.save(out_path, X)

        total_parse_err += parse_error_lines

        records.append({
            "patient_id": patient_id,
            "candidate_ids": "|".join(candidate_ids),
            "id_ambiguous": id_ambiguous,
            "name": name,
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
            "out_npy_path": str(out_path) if out_path else "",
        })

    # 3) 写 manifest
    manifest = pd.DataFrame(records)
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    # 4) 写 summary log
    summary = {
        "xlsx_path": XLSX_PATH,
        "txt_dir": TXT_DIR,
        "out_dir": OUT_DIR,
        "n_txt_files": len(txt_files),
        "bad_filename_count": bad_fname,
        "total_parse_error_lines": total_parse_err,
        "n_records_in_manifest": len(records),
        "n_saved_npy": int((manifest["out_npy_path"] != "").sum()),
        "n_ambiguous_id": int((manifest["id_ambiguous"] == 1).sum()),
    }
    with (out_root / "logs" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"manifest: {manifest_path}")
    print(f"summary : {out_root / 'logs' / 'summary.json'}")

if __name__ == "__main__":
    main()
