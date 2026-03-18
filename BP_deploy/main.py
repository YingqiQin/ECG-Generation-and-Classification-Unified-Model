from onnx_wrapper import BPModelWithSleepAffine, load_clean_state_dict
import torch

base_model = YourBPModel(...)   # 你的 HourlyBPModel_Atten 或现在的 event model
base_model.load_state_dict(load_clean_state_dict("best.pt"), strict=True)
base_model.eval()

wrapped = BPModelWithSleepAffine(base_model).eval()

dummy_x = torch.randn(1, 8, 1, 800, dtype=torch.float32)
dummy_a = torch.ones(1, 2, dtype=torch.float32)
dummy_b = torch.zeros(1, 2, dtype=torch.float32)
torch.onnx.export(
    wrapped,
    args=(dummy_x, dummy_a, dummy_b),
    f="bp_with_affine.onnx",
    input_names=["x", "a", "b"],
    output_names=["y_cal", "y_raw"],
    dynamo=True,
    opset_version=18,
    dynamic_shapes={
        "x": {0: "batch"},
        "a": {0: "batch"},
        "b": {0: "batch"},
    },
    verify=True,
)
