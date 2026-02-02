import re
import pandas as pd

def normalize_patient_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()

    # 处理类似 "1234567.0"
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]

    # 只保留数字
    digits = re.sub(r"\D", "", s)

    # 你明确说 ID 是 7 位数字：不足则左侧补 0（前提是你确认就是固定7位）
    if 0 < len(digits) <= 7:
        digits = digits.zfill(7)

    # 最终只接受严格7位，否则置空
    return digits if re.fullmatch(r"\d{7}", digits) else ""

df["ID"] = df["ID"].apply(normalize_patient_id)
