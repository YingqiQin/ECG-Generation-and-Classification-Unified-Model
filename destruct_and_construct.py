import json
import torch

#-------------destruct----------------
state_dict = torch.load("best.pt", map_location="cpu", weights_only=True)

json_obj = {}
for k, v in state_dict.items():
    json_obj[k] = {
        "dtype": str(v.dtype),
        "shape": list(v.shape),
        "data": v.tolist()
    }

with open("best.json", "w") as f:
    json.dump(json_obj, f)


#-------------reconstruct----------------
dtype_map = {
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

with open("best.json", "r") as f:
    json_obj = json.load(f)

recovered_state_dict = {}
for k, item in json_obj.items():
    recovered_state_dict[k] = torch.tensor(
        item["data"],
        dtype=dtype_map[item["dtype"]]
    )

torch.save(recovered_state_dict, "recovered.pt")