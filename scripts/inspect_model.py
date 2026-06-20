"""检查 PP-OCRv6-small ONNX 模型的输入输出元信息。

模块向外暴露：直接执行本脚本即可打印 det / rec 两个 ONNX 模型的
输入张量（name/shape/type）与输出张量（name/shape/type）。
"""
import onnxruntime as ort

MODELS = {
    "det": "models/PP-OCRv6_small_det_onnx/inference.onnx",
    "rec": "models/PP-OCRv6_small_rec_onnx/inference.onnx",
}

for name, path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"模型: {name} ({path})")
    print(f"{'='*60}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("\n输入:")
    for i in sess.get_inputs():
        print(f"  name={i.name}  shape={i.shape}  type={i.type}")
    print("\n输出:")
    for o in sess.get_outputs():
        print(f"  name={o.name}  shape={o.shape}  type={o.type}")
