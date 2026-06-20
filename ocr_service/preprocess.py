"""OCR 预处理模块。

模块向外暴露：
- det_preprocess: det 模型预处理函数，将 BGR 图像转换为 det 模型输入张量
- rec_preprocess: rec 模型预处理函数，将 BGR 图像转换为 rec 模型输入张量
"""
from __future__ import annotations

import cv2
import numpy as np


def det_preprocess(
    img: np.ndarray, max_side: int = 960
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """det 模型预处理。

    将 BGR 图像等比缩放（不放大）并 32 对齐后归一化，转换为 CHW 格式供 det ONNX 模型推理。
    DBNet 架构要求 H/W 必须是 32 的倍数。

    参数:
        img: BGR 图像 (H, W, 3)，uint8
        max_side: 最大边长限制，默认 960

    返回:
        tensor: 预处理后的张量 [1, 3, H, W]，float32
        (orig_h, orig_w, resized_h, resized_w): 原始和缩放后的尺寸，用于后处理还原坐标

    异常:
        ValueError: 输入图像为空、通道数不为 3、或 max_side 非法
    """
    # 防御性检查
    if img is None or img.size == 0:
        raise ValueError("输入图像不能为空")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"输入图像必须是 (H, W, 3) 格式，当前 shape={img.shape}")
    if max_side <= 0:
        raise ValueError(f"max_side 必须为正数，当前 {max_side}")

    orig_h, orig_w = img.shape[:2]

    # 1. 等比缩放（不放大）
    ratio = min(max_side / orig_h, max_side / orig_w, 1.0)
    new_h = int(orig_h * ratio)
    new_w = int(orig_w * ratio)

    # 2. 32 对齐（DBNet 架构要求 H/W 为 32 的倍数）
    new_h = ((new_h + 31) // 32) * 32
    new_w = ((new_w + 31) // 32) * 32

    # 3. resize
    img_resized = cv2.resize(img, (new_w, new_h))

    # 4. 归一化：(img/255.0 - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std

    # 5. CHW 转置 + batch 维度 → [1, 3, new_h, new_w]
    img_chw = img_norm.transpose(2, 0, 1)[np.newaxis]

    # 6. float32
    tensor = np.ascontiguousarray(img_chw, dtype=np.float32)

    return tensor, (orig_h, orig_w, new_h, new_w)


def rec_preprocess(img: np.ndarray, target_h: int = 48, target_w: int = 320) -> np.ndarray:
    """rec 模型预处理。

    将 BGR 图像等比缩放到目标高度，宽度不足时右侧 padding（黑色），
    然后归一化并转换为 CHW 格式，供 rec ONNX 模型推理使用。

    参数:
        img: BGR 图像 (H, W, 3)，uint8（通常是 det 裁剪出的文本行）
        target_h: 目标高度，默认 48
        target_w: 目标宽度，默认 320

    返回:
        tensor: [1, 3, target_h, target_w]，float32

    异常:
        ValueError: 输入图像为空或通道数不为 3
    """
    # 防御性检查
    if img is None or img.size == 0:
        raise ValueError("输入图像不能为空")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"输入图像必须是 (H, W, 3) 格式，当前 shape={img.shape}")

    h, w = img.shape[:2]

    # 1. 等比缩放：ratio = target_h / h，new_w 不超过 target_w
    ratio = target_h / h
    new_w = min(int(w * ratio), target_w)
    # 保证宽度至少为 1，避免 resize 异常
    new_w = max(1, new_w)

    # 2. resize 到 (new_w, target_h)
    img_resized = cv2.resize(img, (new_w, target_h))

    # 3. 创建黑色画布，将缩放后的图贴到左侧
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:, :new_w] = img_resized

    # 4. 归一化：(img/255.0 - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (canvas.astype(np.float32) / 255.0 - mean) / std

    # 5. CHW 转置 + batch 维度 → [1, 3, target_h, target_w]
    img_chw = img_norm.transpose(2, 0, 1)[np.newaxis]

    # 6. float32
    return np.ascontiguousarray(img_chw, dtype=np.float32)
