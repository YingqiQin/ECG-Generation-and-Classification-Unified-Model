import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import pandas as pd


XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
XML_NS = {"h": "urn:hl7-org:v3"}


@dataclass
class XmlMeta:
    xml_path: Path
    xml_file: str
    patient_id_hint: str
    visit_id_hint: Optional[str]
    ecg_time: Optional[datetime]
    sampling_rate_hz: int
    n_samples: int


def _stable_record_uid(source: str, xml_rel_path: str) -> str:
    digest = hashlib.sha1("{}|{}".format(source, xml_rel_path).encode("utf-8")).hexdigest()[:16]
    return "{}_{}".format(source, digest)


def _ascii_alias_rel_path(source: str, patient_id: str, visit_id: Optional[str], xml_file: str, xml_rel_path: str) -> str:
    safe_patient = str(patient_id or "unknown").strip() or "unknown"
    safe_visit = str(visit_id or "unknown").strip() or "unknown"
    uid = _stable_record_uid(source, xml_rel_path)
    ext = Path(xml_file).suffix or ".xml"
    return "ascii_xml/{}/{}/{}/{}{}".format(source, safe_patient, safe_visit, uid, ext)


def _column_letters(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def _read_xlsx_rows(xlsx_path: Path, sheet_index: int = 0) -> List[Dict[str, str]]:
    with ZipFile(xlsx_path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", XLSX_NS):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//a:t", XLSX_NS)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall(".//a:sheets/a:sheet", XLSX_NS)
        if not sheets:
            raise ValueError("No sheets found in '{}'.".format(xlsx_path))
        if sheet_index >= len(sheets):
            raise ValueError("sheet_index={} is out of range for '{}'.".format(sheet_index, xlsx_path))
        target_sheet_name = sheets[sheet_index].attrib["name"]

        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet_rid = sheets[sheet_index].attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        sheet_path = "xl/" + rel_map[sheet_rid]

        worksheet = ET.fromstring(archive.read(sheet_path))
        rows = worksheet.findall(".//a:sheetData/a:row", XLSX_NS)
        if not rows:
            return []

        header_map: Dict[str, str] = {}
        for cell in rows[0].findall("a:c", XLSX_NS):
            ref = cell.attrib["r"]
            value_node = cell.find("a:v", XLSX_NS)
            if value_node is None:
                continue
            if cell.attrib.get("t") == "s":
                value = shared_strings[int(value_node.text)]
            else:
                value = value_node.text or ""
            header_map[_column_letters(ref)] = value

        parsed_rows: List[Dict[str, str]] = []
        for row in rows[1:]:
            record: Dict[str, str] = {"__sheet_name__": target_sheet_name}
            for cell in row.findall("a:c", XLSX_NS):
                col = _column_letters(cell.attrib["r"])
                column_name = header_map.get(col, col)
                value_node = cell.find("a:v", XLSX_NS)
                if value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text or ""
                record[column_name] = value
            parsed_rows.append(record)
        return parsed_rows


def _parse_xml_ecg_metadata(xml_path: Path) -> XmlMeta:
    root = ET.parse(xml_path).getroot()

    rhythm_series = None
    for series in root.findall(".//h:series", XML_NS):
        code = series.find("h:code", XML_NS)
        if code is not None and code.attrib.get("code") == "RHYTHM":
            rhythm_series = series
            break
    if rhythm_series is None:
        raise ValueError("Could not find RHYTHM series in '{}'.".format(xml_path))

    increment_s = None
    n_samples = None
    for seq in rhythm_series.findall(".//h:sequence", XML_NS):
        value_node = seq.find("h:value", XML_NS)
        if value_node is None:
            continue

        increment_node = value_node.find("h:increment", XML_NS)
        if increment_node is not None and increment_s is None:
            try:
                increment_s = float(increment_node.attrib.get("value"))
            except Exception:
                increment_s = None

        digits_node = value_node.find("h:digits", XML_NS)
        if digits_node is not None and digits_node.text:
            n_samples = len(digits_node.text.split())
            break

    if n_samples is None:
        raise ValueError("Could not find waveform digits in '{}'.".format(xml_path))

    sampling_rate_hz = 500
    if increment_s is not None and increment_s > 0:
        sampling_rate_hz = int(round(1.0 / increment_s))

    low = root.find(".//h:effectiveTime/h:low", XML_NS)
    ecg_time = None
    if low is not None and "value" in low.attrib:
        raw = low.attrib["value"].split(".")[0]
        try:
            ecg_time = datetime.strptime(raw, "%Y%m%d%H%M%S")
        except Exception:
            ecg_time = None

    stem_parts = xml_path.stem.split("_")
    patient_id_hint = stem_parts[1] if len(stem_parts) > 1 else xml_path.stem
    visit_id_hint = stem_parts[1] if len(stem_parts) > 1 else None
    return XmlMeta(
        xml_path=xml_path,
        xml_file=xml_path.name,
        patient_id_hint=patient_id_hint,
        visit_id_hint=visit_id_hint,
        ecg_time=ecg_time,
        sampling_rate_hz=sampling_rate_hz,
        n_samples=n_samples,
    )


def _parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime(1899, 12, 30) + timedelta(days=float(value))
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    return None


def _normalize_int_string(value: str) -> Optional[str]:
    value = str(value).strip()
    if value == "":
        return None
    if value.endswith(".0"):
        value = value[:-2]
    return value


def _parse_float(value: str) -> Optional[float]:
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _records_to_sorted_df(records: List[Dict], columns: Sequence[str], sort_columns: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=list(columns))
    missing_cols = [col for col in columns if col not in df.columns]
    for col in missing_cols:
        df[col] = None
    return df.sort_values(list(sort_columns)).reset_index(drop=True)


def _normalize_visit_key(value: str) -> Optional[str]:
    value = _normalize_int_string(value)
    if value is None:
        return None
    value = value.lstrip("0")
    return value or "0"


def _parse_binary_flag(value: str) -> Optional[int]:
    value = _normalize_int_string(value)
    if value is None:
        return None
    if value in {"0", "1"}:
        return int(value)
    return None


def _build_shengrenyi_actual_xml_indices(xml_dir: Path) -> Tuple[Dict[str, Path], Dict[Tuple[str, str], List[Path]]]:
    xml_by_name: Dict[str, Path] = {}
    xml_by_uid_visit: Dict[Tuple[str, str], List[Path]] = {}
    for xml_path in xml_dir.glob("*.xml"):
        xml_by_name[xml_path.name] = xml_path
        parts = xml_path.stem.split("_")
        if len(parts) < 2:
            continue
        uid = parts[0].strip()
        visit_key = _normalize_visit_key(parts[1])
        if uid and visit_key:
            xml_by_uid_visit.setdefault((uid, visit_key), []).append(xml_path)
    return xml_by_name, xml_by_uid_visit


def _resolve_shengrenyi_updated_xml_path(
    alias_xml_file: str,
    xml_by_name: Dict[str, Path],
    xml_by_uid_visit: Dict[Tuple[str, str], List[Path]],
) -> Optional[Path]:
    alias_xml_file = str(alias_xml_file).strip()
    if alias_xml_file == "":
        return None
    if alias_xml_file in xml_by_name:
        return xml_by_name[alias_xml_file]

    alias_stem = Path(alias_xml_file).stem
    alias_parts = alias_stem.split("_")
    if len(alias_parts) < 6:
        return None

    visit_key = _normalize_visit_key(alias_parts[0])
    uid = alias_parts[-3].strip()
    if visit_key is None or uid == "":
        return None

    candidates = list(xml_by_uid_visit.get((uid, visit_key), []))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    alias_name = alias_parts[1].strip()
    alias_suffix = alias_parts[-2:]
    for candidate in candidates:
        parts = candidate.stem.split("_")
        if len(parts) >= 6 and parts[3].strip() == alias_name and parts[-2:] == alias_suffix:
            return candidate
    return candidates[0]


def _build_shengrenyi_old_class_index(old_class_xlsx_path: Path) -> Dict[str, List[Dict]]:
    rows = _read_xlsx_rows(old_class_xlsx_path)
    index: Dict[str, List[Dict]] = {}
    for row in rows:
        class_value = _normalize_int_string(row.get("分类", ""))
        visit_key = _normalize_visit_key(row.get("门诊/住院号", ""))
        ecg_time = _parse_datetime(row.get("心电图检查时间", ""))
        if class_value not in {"0", "1", "2", "3"} or visit_key is None or ecg_time is None:
            continue
        index.setdefault(visit_key, []).append(
            {
                "class": int(class_value),
                "patient_id": _normalize_int_string(row.get("患者编号", "")) or "",
                "visit_id": visit_key,
                "ecg_time": ecg_time,
                "LVEF_old": _parse_float(row.get("射血分数（%）", "")),
            }
        )
    for visit_key in index:
        index[visit_key].sort(key=lambda item: item["ecg_time"])
    return index


def _build_updated_clinical_label_index(labels_xlsx_path: Path) -> Tuple[Dict[str, Dict[str, Optional[int]]], Sequence[str]]:
    rows = _read_xlsx_rows(labels_xlsx_path)
    label_columns = [
        column
        for column in rows[0].keys()
        if not column.startswith("__") and column not in {"患者编号", "就诊编号", "XML文件名"}
    ]
    lookup: Dict[str, Dict[str, Optional[int]]] = {}
    for row in rows:
        xml_file = str(row.get("XML文件名", "")).strip()
        if xml_file == "":
            continue
        lookup[xml_file] = {column: _parse_binary_flag(row.get(column, "")) for column in label_columns}
    return lookup, label_columns


def _recover_shengrenyi_class(
    row: Dict[str, str],
    old_class_index: Dict[str, List[Dict]],
) -> Tuple[Optional[int], Optional[float], Optional[Dict]]:
    visit_key = _normalize_visit_key(row.get("门诊/住院号", ""))
    ecg_time = _parse_datetime(row.get("心电_检查时间", ""))
    if visit_key is None or ecg_time is None:
        return None, None, None
    candidates = old_class_index.get(visit_key, [])
    if not candidates:
        return None, None, None

    best = min(candidates, key=lambda item: abs((item["ecg_time"] - ecg_time).total_seconds()))
    gap_hours = abs((best["ecg_time"] - ecg_time).total_seconds()) / 3600.0
    return int(best["class"]), gap_hours, best


def _build_taizhou_manifest(
    xlsx_path: Path,
    xml_dir: Path,
    xml_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    output_columns = [
        "source",
        "xml_path",
        "ascii_xml_path",
        "record_uid",
        "path_format",
        "class",
        "LVEF",
        "sampling_rate_hz",
        "n_samples",
        "xml_file",
        "ascii_xml_file",
        "patient_id",
        "visit_id",
        "ecg_time",
        "match_gap_hours",
        "房扑",
    ]
    rows = _read_xlsx_rows(xlsx_path)
    xml_paths = list(xml_dir.glob("*.xml"))
    xml_by_name = {p.name: p for p in xml_paths}
    xml_by_suffix = {"_".join(p.name.split("_")[1:]): p for p in xml_paths if "_" in p.name}
    xml_meta_cache: Dict[Path, XmlMeta] = {}

    records: List[Dict] = []
    missing_files = 0
    skipped_invalid = 0

    for row in rows:
        class_value = _normalize_int_string(row.get("分类", ""))
        if class_value is None:
            continue
        if class_value not in {"0", "1", "2", "3"}:
            skipped_invalid += 1
            continue

        raw_name = next((str(v).strip() for k, v in row.items() if not k.startswith("__") and str(v).strip().endswith(".xml")), "")
        if raw_name == "":
            skipped_invalid += 1
            continue

        xml_path = xml_by_name.get(raw_name)
        if xml_path is None and "_" in raw_name:
            xml_path = xml_by_suffix.get("_".join(raw_name.split("_")[1:]))
        if xml_path is None:
            missing_files += 1
            continue

        if xml_path not in xml_meta_cache:
            xml_meta_cache[xml_path] = _parse_xml_ecg_metadata(xml_path)
        meta = xml_meta_cache[xml_path]

        lvef = _parse_float(row.get("射血分数", ""))
        patient_id = xml_path.stem.split("_")[1] if len(xml_path.stem.split("_")) > 1 else xml_path.stem
        xml_rel_path = xml_path.relative_to(xml_root).as_posix()
        ascii_xml_path = _ascii_alias_rel_path(
            source="taizhou",
            patient_id=patient_id,
            visit_id=patient_id,
            xml_file=xml_path.name,
            xml_rel_path=xml_rel_path,
        )
        records.append(
            {
                "source": "taizhou",
                "xml_path": xml_rel_path,
                "ascii_xml_path": ascii_xml_path,
                "record_uid": _stable_record_uid("taizhou", xml_rel_path),
                "path_format": "xml",
                "class": int(class_value),
                "LVEF": lvef,
                "sampling_rate_hz": meta.sampling_rate_hz,
                "n_samples": meta.n_samples,
                "xml_file": xml_path.name,
                "ascii_xml_file": Path(ascii_xml_path).name,
                "patient_id": patient_id,
                "visit_id": patient_id,
                "ecg_time": meta.ecg_time.strftime("%Y-%m-%d %H:%M:%S") if meta.ecg_time else "",
                "match_gap_hours": 0.0,
                "房扑": _normalize_int_string(row.get("房扑", "")),
            }
        )

    df = _records_to_sorted_df(records, output_columns, ["patient_id", "ecg_time", "xml_file"])
    summary = {
        "rows_written": int(len(df)),
        "rows_with_label": int(sum(1 for row in rows if _normalize_int_string(row.get("分类", "")) is not None)),
        "missing_files": int(missing_files),
        "skipped_invalid": int(skipped_invalid),
    }
    return df, summary


def _build_shengrenyi_manifest(
    xlsx_path: Path,
    xml_dir: Path,
    xml_root: Path,
    max_match_days: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    output_columns = [
        "source",
        "xml_path",
        "ascii_xml_path",
        "record_uid",
        "path_format",
        "class",
        "LVEF",
        "sampling_rate_hz",
        "n_samples",
        "xml_file",
        "ascii_xml_file",
        "patient_id",
        "visit_id",
        "ecg_time",
        "match_gap_hours",
        "xlsx_ecg_time",
        "就诊编号",
        "门诊住院号原始",
    ]
    rows = _read_xlsx_rows(xlsx_path)

    xml_index: Dict[str, List[XmlMeta]] = {}
    for xml_path in xml_dir.glob("*.xml"):
        meta = _parse_xml_ecg_metadata(xml_path)
        if meta.visit_id_hint is None:
            continue
        xml_index.setdefault(meta.visit_id_hint, []).append(meta)

    excel_groups: Dict[str, List[Dict]] = {}
    skipped_invalid = 0
    for row in rows:
        class_value = _normalize_int_string(row.get("分类", ""))
        if class_value is None or class_value not in {"0", "1", "2", "3"}:
            continue

        visit_id = _normalize_int_string(row.get("门诊/住院号", ""))
        ecg_time = _parse_datetime(row.get("心电图检查时间", ""))
        if visit_id is None or ecg_time is None:
            skipped_invalid += 1
            continue

        row_copy = dict(row)
        row_copy["__visit_id__"] = visit_id
        row_copy["__ecg_time__"] = ecg_time
        excel_groups.setdefault(visit_id, []).append(row_copy)

    matched_records: List[Dict] = []
    unmatched_no_id = 0
    unmatched_time_gap = 0

    for visit_id, excel_rows in excel_groups.items():
        xml_candidates = sorted(xml_index.get(visit_id, []), key=lambda x: (x.ecg_time or datetime.min))
        if not xml_candidates:
            unmatched_no_id += len(excel_rows)
            continue

        remaining = list(xml_candidates)
        for row in sorted(excel_rows, key=lambda r: r["__ecg_time__"]):
            if not remaining:
                unmatched_no_id += 1
                continue

            best_idx = None
            best_gap_seconds = None
            for idx, meta in enumerate(remaining):
                if meta.ecg_time is None:
                    continue
                gap_seconds = abs((meta.ecg_time - row["__ecg_time__"]).total_seconds())
                if best_gap_seconds is None or gap_seconds < best_gap_seconds:
                    best_gap_seconds = gap_seconds
                    best_idx = idx

            if best_idx is None or best_gap_seconds is None:
                unmatched_no_id += 1
                continue

            gap_days = best_gap_seconds / 86400.0
            if gap_days > max_match_days:
                unmatched_time_gap += 1
                continue

            meta = remaining.pop(best_idx)
            xml_rel_path = meta.xml_path.relative_to(xml_root).as_posix()
            patient_id = _normalize_int_string(row.get("患者编号", "")) or visit_id
            ascii_xml_path = _ascii_alias_rel_path(
                source="shengrenyi",
                patient_id=patient_id,
                visit_id=visit_id,
                xml_file=meta.xml_file,
                xml_rel_path=xml_rel_path,
            )
            matched_records.append(
                {
                    "source": "shengrenyi",
                    "xml_path": xml_rel_path,
                    "ascii_xml_path": ascii_xml_path,
                    "record_uid": _stable_record_uid("shengrenyi", xml_rel_path),
                    "path_format": "xml",
                    "class": int(row["分类"]),
                    "LVEF": _parse_float(row.get("射血分数（%）", "")),
                    "sampling_rate_hz": meta.sampling_rate_hz,
                    "n_samples": meta.n_samples,
                    "xml_file": meta.xml_file,
                    "ascii_xml_file": Path(ascii_xml_path).name,
                    "patient_id": patient_id,
                    "visit_id": visit_id,
                    "ecg_time": meta.ecg_time.strftime("%Y-%m-%d %H:%M:%S") if meta.ecg_time else "",
                    "match_gap_hours": round(best_gap_seconds / 3600.0, 6),
                    "xlsx_ecg_time": row["__ecg_time__"].strftime("%Y-%m-%d %H:%M:%S"),
                    "就诊编号": _normalize_int_string(row.get("就诊编号", "")),
                    "门诊住院号原始": row.get("门诊/住院号", ""),
                }
            )

    df = _records_to_sorted_df(matched_records, output_columns, ["patient_id", "ecg_time", "xml_file"])
    summary = {
        "rows_written": int(len(df)),
        "groups_with_visit_id": int(len(excel_groups)),
        "unmatched_no_id": int(unmatched_no_id),
        "unmatched_time_gap": int(unmatched_time_gap),
        "max_match_days": float(max_match_days),
    }
    return df, summary


def _build_shengrenyi_updated_manifest(
    xml_info_xlsx_path: Path,
    clinical_labels_xlsx_path: Path,
    old_class_xlsx_path: Path,
    xml_dir: Path,
    xml_root: Path,
    max_class_gap_days: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    xml_rows = _read_xlsx_rows(xml_info_xlsx_path)
    xml_by_name, xml_by_uid_visit = _build_shengrenyi_actual_xml_indices(xml_dir)
    old_class_index = _build_shengrenyi_old_class_index(old_class_xlsx_path)
    clinical_lookup, clinical_label_columns = _build_updated_clinical_label_index(clinical_labels_xlsx_path)
    xml_meta_cache: Dict[Path, XmlMeta] = {}

    audit_records: List[Dict] = []
    strict_records: List[Dict] = []

    unresolved_xml = 0
    missing_old_class = 0
    missing_clinical_labels = 0
    excluded_large_class_gap = 0

    max_class_gap_hours = max_class_gap_days * 24.0

    for row in xml_rows:
        alias_xml_file = str(row.get("XML文件名", "")).strip()
        if alias_xml_file == "":
            continue

        actual_xml_path = _resolve_shengrenyi_updated_xml_path(
            alias_xml_file=alias_xml_file,
            xml_by_name=xml_by_name,
            xml_by_uid_visit=xml_by_uid_visit,
        )
        if actual_xml_path is None:
            unresolved_xml += 1
            continue

        if actual_xml_path not in xml_meta_cache:
            xml_meta_cache[actual_xml_path] = _parse_xml_ecg_metadata(actual_xml_path)
        meta = xml_meta_cache[actual_xml_path]

        patient_id = _normalize_int_string(row.get("患者编号", "")) or meta.patient_id_hint
        visit_id = _normalize_visit_key(row.get("门诊/住院号", "")) or meta.visit_id_hint or patient_id
        xml_rel_path = actual_xml_path.relative_to(xml_root).as_posix()
        ascii_xml_path = _ascii_alias_rel_path(
            source="shengrenyi",
            patient_id=patient_id,
            visit_id=visit_id,
            xml_file=actual_xml_path.name,
            xml_rel_path=xml_rel_path,
        )

        class_value, class_gap_hours, class_match = _recover_shengrenyi_class(row, old_class_index)
        if class_value is None:
            missing_old_class += 1
        elif class_gap_hours is not None and class_gap_hours > max_class_gap_hours:
            excluded_large_class_gap += 1

        clinical_labels = clinical_lookup.get(alias_xml_file)
        if clinical_labels is None:
            missing_clinical_labels += 1
            clinical_labels = {column: None for column in clinical_label_columns}

        ecg_time = _parse_datetime(row.get("心电_检查时间", ""))
        audit_record = {
            "source": "shengrenyi",
            "xml_path": xml_rel_path,
            "ascii_xml_path": ascii_xml_path,
            "record_uid": _stable_record_uid("shengrenyi", xml_rel_path),
            "path_format": "xml",
            "class": class_value,
            "class_recovered": int(class_value is not None),
            "class_match_gap_hours": round(class_gap_hours, 6) if class_gap_hours is not None else "",
            "class_match_within_7d": int(class_gap_hours is not None and class_gap_hours <= max_class_gap_hours),
            "LVEF": _parse_float(row.get("射血分数（%）", "")),
            "sampling_rate_hz": meta.sampling_rate_hz,
            "n_samples": meta.n_samples,
            "xml_file": actual_xml_path.name,
            "xml_file_alias": alias_xml_file,
            "ascii_xml_file": Path(ascii_xml_path).name,
            "patient_id": patient_id,
            "visit_id": visit_id,
            "ecg_time": meta.ecg_time.strftime("%Y-%m-%d %H:%M:%S") if meta.ecg_time else "",
            "xlsx_ecg_time": ecg_time.strftime("%Y-%m-%d %H:%M:%S") if ecg_time else "",
            "xml_time_error_hours": _parse_float(row.get("XML时间误差(小时)", "")),
            "is_problem": _parse_binary_flag(row.get("is_problem", "")),
            "sex": row.get("性别", ""),
            "ecg_result": row.get("检查结果", ""),
            "ecg_conclusion": row.get("心电检查结论", ""),
            "echo_conclusion": row.get("超声结论", ""),
            "echo_description": row.get("超声描述", ""),
            "bnp_value": _parse_float(row.get("检验定量结果", "")),
            "bnp_time_gap_days": _parse_float(row.get("BNP_时间差", "")),
            "echo_time_gap_days": _parse_float(row.get("心超_时间差", "")),
            "old_class_patient_id": class_match["patient_id"] if class_match else "",
            "old_class_ecg_time": class_match["ecg_time"].strftime("%Y-%m-%d %H:%M:%S") if class_match else "",
        }
        audit_record.update(clinical_labels)
        audit_records.append(audit_record)

        if class_value is None or class_gap_hours is None or class_gap_hours > max_class_gap_hours:
            continue

        strict_records.append(dict(audit_record))

    audit_columns = [
        "source",
        "xml_path",
        "ascii_xml_path",
        "record_uid",
        "path_format",
        "class",
        "class_recovered",
        "class_match_gap_hours",
        "class_match_within_7d",
        "LVEF",
        "sampling_rate_hz",
        "n_samples",
        "xml_file",
        "xml_file_alias",
        "ascii_xml_file",
        "patient_id",
        "visit_id",
        "ecg_time",
        "xlsx_ecg_time",
        "xml_time_error_hours",
        "is_problem",
        "sex",
        "ecg_result",
        "ecg_conclusion",
        "echo_conclusion",
        "echo_description",
        "bnp_value",
        "bnp_time_gap_days",
        "echo_time_gap_days",
        "old_class_patient_id",
        "old_class_ecg_time",
        *clinical_label_columns,
    ]
    audit_df = _records_to_sorted_df(audit_records, audit_columns, ["patient_id", "ecg_time", "xml_file"])
    strict_df = _records_to_sorted_df(strict_records, audit_columns, ["patient_id", "ecg_time", "xml_file"])
    summary = {
        "xml_rows_input": int(len(xml_rows)),
        "rows_resolved_xml": int(len(audit_df)),
        "rows_written_strict": int(len(strict_df)),
        "unresolved_xml": int(unresolved_xml),
        "missing_old_class": int(missing_old_class),
        "excluded_large_class_gap": int(excluded_large_class_gap),
        "missing_clinical_labels": int(missing_clinical_labels),
        "max_class_gap_days": float(max_class_gap_days),
        "is_problem_count": int(sum(int(v) for v in strict_df["is_problem"].fillna(0).astype(int))) if not strict_df.empty else 0,
    }
    return strict_df, audit_df, summary


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Build local XML HF manifests from the Taizhou and Shengrenyi Excel label files plus the XML ECG folders."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory that contains the local data folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"), help="Directory to write the output CSV manifests.")
    parser.add_argument("--max-shengrenyi-match-days", type=float, default=7.0, help="Maximum allowed time gap between Excel ECG time and XML effective time for Shengrenyi matching.")
    parser.add_argument("--max-updated-class-gap-days", type=float, default=7.0, help="Maximum allowed time gap when recovering Shengrenyi class labels from the old class workbook for the updated 20260322 XML list.")
    return parser


def main():
    args = build_argparser().parse_args()
    data_root = args.data_root

    taizhou_xlsx = data_root / "泰州ECG及标签" / "泰州心电标签942.xlsx"
    taizhou_xml_dir = data_root / "泰州ECG及标签" / "泰州ECG"
    shengrenyi_xlsx = data_root / "省人医" / "省人医房颤心超数据（含分类）.xlsx"
    shengrenyi_xml_dir = data_root / "省人医" / "省人医ECG" / "房颤数据"
    shengrenyi_updated_xml_info_xlsx = data_root / "省人医5317份xml检查信息_20260322更新.xlsx"
    shengrenyi_updated_clinical_xlsx = data_root / "省人医5317例补充临床标签_20260322更新.xlsx"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    taizhou_df, taizhou_summary = _build_taizhou_manifest(
        xlsx_path=taizhou_xlsx,
        xml_dir=taizhou_xml_dir,
        xml_root=data_root,
    )
    shengrenyi_df, shengrenyi_summary = _build_shengrenyi_manifest(
        xlsx_path=shengrenyi_xlsx,
        xml_dir=shengrenyi_xml_dir,
        xml_root=data_root,
        max_match_days=args.max_shengrenyi_match_days,
    )

    combined_df = pd.concat([taizhou_df, shengrenyi_df], ignore_index=True)
    combined_df = combined_df.sort_values(["source", "patient_id", "ecg_time", "xml_file"]).reset_index(drop=True)

    def _make_ascii_manifest(df: pd.DataFrame) -> pd.DataFrame:
        ascii_df = df.copy()
        ascii_df["original_xml_path"] = ascii_df["xml_path"]
        ascii_df["xml_path"] = ascii_df["ascii_xml_path"]
        return ascii_df

    taizhou_ascii_df = _make_ascii_manifest(taizhou_df)
    shengrenyi_ascii_df = _make_ascii_manifest(shengrenyi_df)
    combined_ascii_df = _make_ascii_manifest(combined_df)

    mapping_df = combined_df[
        [
            "source",
            "record_uid",
            "patient_id",
            "visit_id",
            "xml_file",
            "ascii_xml_file",
            "xml_path",
            "ascii_xml_path",
        ]
    ].drop_duplicates().reset_index(drop=True)

    taizhou_csv = output_dir / "hf_manifest_taizhou.csv"
    taizhou_ascii_csv = output_dir / "hf_manifest_taizhou_ascii.csv"
    shengrenyi_csv = output_dir / "hf_manifest_shengrenyi.csv"
    shengrenyi_ascii_csv = output_dir / "hf_manifest_shengrenyi_ascii.csv"
    combined_csv = output_dir / "hf_manifest_combined.csv"
    combined_ascii_csv = output_dir / "hf_manifest_combined_ascii.csv"
    mapping_csv = output_dir / "hf_manifest_ascii_mapping.csv"
    taizhou_df.to_csv(taizhou_csv, index=False)
    taizhou_ascii_df.to_csv(taizhou_ascii_csv, index=False)
    shengrenyi_df.to_csv(shengrenyi_csv, index=False)
    shengrenyi_ascii_df.to_csv(shengrenyi_ascii_csv, index=False)
    combined_df.to_csv(combined_csv, index=False)
    combined_ascii_df.to_csv(combined_ascii_csv, index=False)
    mapping_df.to_csv(mapping_csv, index=False)

    summary = {
        "taizhou": taizhou_summary,
        "shengrenyi": shengrenyi_summary,
        "combined_rows": int(len(combined_df)),
        "ascii_mapping_rows": int(len(mapping_df)),
        "output_files": {
            "taizhou_csv": str(taizhou_csv),
            "taizhou_ascii_csv": str(taizhou_ascii_csv),
            "shengrenyi_csv": str(shengrenyi_csv),
            "shengrenyi_ascii_csv": str(shengrenyi_ascii_csv),
            "combined_csv": str(combined_csv),
            "combined_ascii_csv": str(combined_ascii_csv),
            "mapping_csv": str(mapping_csv),
        },
    }
    (output_dir / "hf_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if shengrenyi_updated_xml_info_xlsx.exists() and shengrenyi_updated_clinical_xlsx.exists():
        shengrenyi_updated_df, shengrenyi_updated_audit_df, shengrenyi_updated_summary = _build_shengrenyi_updated_manifest(
            xml_info_xlsx_path=shengrenyi_updated_xml_info_xlsx,
            clinical_labels_xlsx_path=shengrenyi_updated_clinical_xlsx,
            old_class_xlsx_path=shengrenyi_xlsx,
            xml_dir=shengrenyi_xml_dir,
            xml_root=data_root,
            max_class_gap_days=args.max_updated_class_gap_days,
        )
        combined_updated_df = pd.concat([taizhou_df, shengrenyi_updated_df], ignore_index=True)
        combined_updated_df = combined_updated_df.sort_values(["source", "patient_id", "ecg_time", "xml_file"]).reset_index(drop=True)

        def _make_ascii_manifest(df: pd.DataFrame) -> pd.DataFrame:
            ascii_df = df.copy()
            ascii_df["original_xml_path"] = ascii_df["xml_path"]
            ascii_df["xml_path"] = ascii_df["ascii_xml_path"]
            return ascii_df

        shengrenyi_updated_ascii_df = _make_ascii_manifest(shengrenyi_updated_df)
        combined_updated_ascii_df = _make_ascii_manifest(combined_updated_df)
        shengrenyi_updated_audit_ascii_df = _make_ascii_manifest(shengrenyi_updated_audit_df)

        mapping_updated_df = combined_updated_df[
            [
                "source",
                "record_uid",
                "patient_id",
                "visit_id",
                "xml_file",
                "ascii_xml_file",
                "xml_path",
                "ascii_xml_path",
            ]
        ].drop_duplicates().reset_index(drop=True)

        tag = "20260322"
        shengrenyi_updated_csv = output_dir / "hf_manifest_shengrenyi_{}.csv".format(tag)
        shengrenyi_updated_ascii_csv = output_dir / "hf_manifest_shengrenyi_{}_ascii.csv".format(tag)
        shengrenyi_updated_audit_csv = output_dir / "hf_manifest_shengrenyi_{}_audit.csv".format(tag)
        shengrenyi_updated_audit_ascii_csv = output_dir / "hf_manifest_shengrenyi_{}_audit_ascii.csv".format(tag)
        combined_updated_csv = output_dir / "hf_manifest_combined_{}.csv".format(tag)
        combined_updated_ascii_csv = output_dir / "hf_manifest_combined_{}_ascii.csv".format(tag)
        mapping_updated_csv = output_dir / "hf_manifest_ascii_mapping_{}.csv".format(tag)

        shengrenyi_updated_df.to_csv(shengrenyi_updated_csv, index=False)
        shengrenyi_updated_ascii_df.to_csv(shengrenyi_updated_ascii_csv, index=False)
        shengrenyi_updated_audit_df.to_csv(shengrenyi_updated_audit_csv, index=False)
        shengrenyi_updated_audit_ascii_df.to_csv(shengrenyi_updated_audit_ascii_csv, index=False)
        combined_updated_df.to_csv(combined_updated_csv, index=False)
        combined_updated_ascii_df.to_csv(combined_updated_ascii_csv, index=False)
        mapping_updated_df.to_csv(mapping_updated_csv, index=False)

        updated_summary = {
            "taizhou_rows": int(len(taizhou_df)),
            "shengrenyi_updated": shengrenyi_updated_summary,
            "combined_updated_rows": int(len(combined_updated_df)),
            "ascii_mapping_rows": int(len(mapping_updated_df)),
            "output_files": {
                "shengrenyi_csv": str(shengrenyi_updated_csv),
                "shengrenyi_ascii_csv": str(shengrenyi_updated_ascii_csv),
                "shengrenyi_audit_csv": str(shengrenyi_updated_audit_csv),
                "shengrenyi_audit_ascii_csv": str(shengrenyi_updated_audit_ascii_csv),
                "combined_csv": str(combined_updated_csv),
                "combined_ascii_csv": str(combined_updated_ascii_csv),
                "mapping_csv": str(mapping_updated_csv),
            },
        }
        (output_dir / "hf_manifest_summary_{}.json".format(tag)).write_text(
            json.dumps(updated_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["updated_20260322"] = updated_summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
