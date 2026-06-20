"""用合成图片实测 det + rec 推理，验证输出形状和数值范围。

模块向外暴露：直接执行本脚本即可运行 det / rec 两个 ONNX 模型的
端到端推理（合成图片 -> det -> rec），并打印各阶段输入输出形状与数值范围。
"""
import numpy as np
import cv2
import onnxruntime as ort

# 生成测试图片：白底黑字
img = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, "Hello OCR 2026", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

# === det 推理 ===
print("=" * 60)
print("DET 推理")
print("=" * 60)
det_sess = ort.InferenceSession("models/PP-OCRv6_small_det_onnx/inference.onnx", providers=["CPUExecutionProvider"])

# det 预处理：BGR -> resize -> normalize -> CHW
# 注意：PP-OCRv6 det 模型要求 H/W 必须是 32 的倍数（DBNet 架构下采样对齐），
# 否则会报 "Attempting to broadcast an axis by a dimension other than 1" 错误。
h, w = img.shape[:2]
max_side = 960
ratio = min(max_side / h, max_side / w, 1.0)
new_h, new_w = int(h * ratio), int(w * ratio)
# 向上对齐到 32 的倍数
new_h = int(np.ceil(new_h / 32.0) * 32)
new_w = int(np.ceil(new_w / 32.0) * 32)
img_resized = cv2.resize(img, (new_w, new_h))
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
img_chw = img_norm.transpose(2, 0, 1)[np.newaxis]  # NCHW

det_input_name = det_sess.get_inputs()[0].name
print(f"输入: name={det_input_name}, shape={img_chw.shape}, dtype={img_chw.dtype}")
det_out = det_sess.run(None, {det_input_name: img_chw})
print(f"输出数量: {len(det_out)}")
for idx, out in enumerate(det_out):
    print(f"  输出[{idx}]: name={det_sess.get_outputs()[idx].name}, shape={out.shape}, dtype={out.dtype}")
    print(f"  数值范围: min={out.min():.6f}, max={out.max():.6f}, mean={out.mean():.6f}")

# === rec 推理 ===
print("\n" + "=" * 60)
print("REC 推理")
print("=" * 60)
rec_sess = ort.InferenceSession("models/PP-OCRv6_small_rec_onnx/inference.onnx", providers=["CPUExecutionProvider"])

# rec 预处理：裁剪一行文字区域，缩放到 [3, 48, 320]
crop = img[70:130, 20:380]  # 裁剪文字行
h2, w2 = crop.shape[:2]
target_h = 48
ratio2 = target_h / h2
new_w2 = min(int(w2 * ratio2), 320)
img_resized2 = cv2.resize(crop, (new_w2, target_h))
# pad 到宽度 320
img_padded = np.ones((target_h, 320, 3), dtype=np.uint8) * 0
img_padded[:, :new_w2] = img_resized2
img_norm2 = (img_padded.astype(np.float32) / 255.0 - mean) / std
img_chw2 = img_norm2.transpose(2, 0, 1)[np.newaxis]

rec_input_name = rec_sess.get_inputs()[0].name
print(f"输入: name={rec_input_name}, shape={img_chw2.shape}, dtype={img_chw2.dtype}")
rec_out = rec_sess.run(None, {rec_input_name: img_chw2})
print(f"输出数量: {len(rec_out)}")
for idx, out in enumerate(rec_out):
    print(f"  输出[{idx}]: name={rec_sess.get_outputs()[idx].name}, shape={out.shape}, dtype={out.dtype}")
    print(f"  数值范围: min={out.min():.6f}, max={out.max():.6f}, mean={out.mean():.6f}")
