import argparse
import json
from pathlib import Path
from typing import Any

import torch


SOURCE_PT = Path("best.pt")
JSON_PATH = Path("best.json")
RECOVERED_PT = Path("recovered.pt")


DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.int16": torch.int16,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def encode_obj(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return {
            "__type__": "tensor",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tolist(),
        }
    if isinstance(obj, dict):
        return {
            "__type__": "dict",
            "items": {str(key): encode_obj(value) for key, value in obj.items()},
        }
    if isinstance(obj, list):
        return {
            "__type__": "list",
            "items": [encode_obj(item) for item in obj],
        }
    if isinstance(obj, tuple):
        return {
            "__type__": "tuple",
            "items": [encode_obj(item) for item in obj],
        }
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"Unsupported object type for JSON conversion: {type(obj)!r}")


def decode_obj(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected JSON object type: {type(obj)!r}")

    obj_type = obj.get("__type__")
    if obj_type == "tensor":
        dtype_name = obj["dtype"]
        if dtype_name not in DTYPE_MAP:
            raise KeyError(f"Unsupported tensor dtype during recovery: {dtype_name}")
        return torch.tensor(obj["data"], dtype=DTYPE_MAP[dtype_name])
    if obj_type == "dict":
        return {key: decode_obj(value) for key, value in obj["items"].items()}
    if obj_type == "list":
        return [decode_obj(item) for item in obj["items"]]
    if obj_type == "tuple":
        return tuple(decode_obj(item) for item in obj["items"])

    if "dtype" in obj and "data" in obj:
        dtype_name = obj["dtype"]
        if dtype_name not in DTYPE_MAP:
            raise KeyError(f"Unsupported tensor dtype during recovery: {dtype_name}")
        return torch.tensor(obj["data"], dtype=DTYPE_MAP[dtype_name])
    return {key: decode_obj(value) for key, value in obj.items()}


def destruct_pt_to_json(source_pt: Path = SOURCE_PT, json_path: Path = JSON_PATH) -> None:
    payload = torch.load(source_pt, map_location="cpu", weights_only=False)
    encoded = encode_obj(payload)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(encoded, handle, ensure_ascii=False)


def reconstruct_json_to_pt(json_path: Path = JSON_PATH, recovered_pt: Path = RECOVERED_PT) -> None:
    with json_path.open("r", encoding="utf-8") as handle:
        encoded = json.load(handle)
    recovered = decode_obj(encoded)
    torch.save(recovered, recovered_pt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PyTorch bundle/checkpoint files to JSON and back.")
    parser.add_argument(
        "--mode",
        choices=["destruct", "reconstruct", "both"],
        default="both",
        help="`destruct`: pt -> json, `reconstruct`: json -> pt, `both`: run both in sequence.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SOURCE_PT,
        help="Input `.pt` file for `destruct` mode.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=JSON_PATH,
        help="Intermediate JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RECOVERED_PT,
        help="Output `.pt` file for `reconstruct` mode.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode in {"destruct", "both"}:
        destruct_pt_to_json(args.input, args.json)
        print(f"Saved JSON to: {args.json}")
    if args.mode in {"reconstruct", "both"}:
        reconstruct_json_to_pt(args.json, args.output)
        print(f"Saved PT to: {args.output}")
