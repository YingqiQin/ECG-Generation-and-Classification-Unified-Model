import numpy as np
import pandas as pd
import re

manifest = pd.DataFrame(records)
manifest_path = out_root / "manifest.csv"
manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

# 生成干净版：只保留可训练样本
clean = manifest.copy()

# 1) 必须有 npy
clean = clean[clean["npy_path"].fillna("") != ""]

# 2) 必须 patient_id 合法（严格7位）
clean = clean[clean["patient_id"].astype(str).str.fullmatch(r"\d{7}", na=False)]

# 3) 去掉 ambiguous（同名对应多个ID 或缺失导致的空ID）
clean = clean[clean["id_ambiguous"] == 0]

# 4) LVEF 必须有值且是数值
clean["lvef"] = pd.to_numeric(clean["lvef"], errors="coerce")
clean = clean[clean["lvef"].notna()]

# （可选）5) 样本长度必须>0
clean = clean[clean["n_samples"] > 0]

clean_manifest_path = out_root / "manifest_clean.csv"
clean.to_csv(clean_manifest_path, index=False, encoding="utf-8-sig")

print("clean_manifest:", clean_manifest_path, "rows=", len(clean))

