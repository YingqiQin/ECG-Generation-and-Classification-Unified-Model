import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "bp_with_affine.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

x = np.random.randn(1, 8, 1, 800).astype(np.float32)
a = np.array([[1.02, 0.98]], dtype=np.float32)
b = np.array([[3.1, -1.7]], dtype=np.float32)

outputs = sess.run(
    ["y_cal", "y_raw"],
    {"x": x, "a": a, "b": b}
)

y_cal, y_raw = outputs
print(y_cal, y_raw)