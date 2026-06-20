"""OCR 后处理模块。

模块向外暴露：
- DBPostProcess: det 模型后处理，从概率图提取文本框多边形
- load_character_dict: 从 inference.yml 解析字符字典
- CTCLabelDecode: CTC 贪心解码器
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


class DBPostProcess:
    """det 模型后处理（DBNet）。

    从 det 模型输出的概率图提取文本框多边形。

    参数与 inference.yml 中 PostProcess 配置一致：
        thresh=0.2, box_thresh=0.45, max_candidates=3000, unclip_ratio=1.4
    """

    def __init__(
        self,
        thresh: float = 0.2,
        box_thresh: float = 0.45,
        max_candidates: int = 3000,
        unclip_ratio: float = 1.4,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(
        self,
        prob_map: np.ndarray,
        orig_shape: tuple[int, int],
        resized_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        """从概率图提取文本框多边形。

        参数:
            prob_map: [N, 1, H, W] / [1, H, W] / [H, W] 概率图
            orig_shape: (orig_h, orig_w) 原始图像尺寸
            resized_shape: (resized_h, resized_w) 缩放后图像尺寸

        返回:
            list of polygons, 每个 polygon 是 [4, 2] 的 np.ndarray (x, y 坐标，相对于原图)

        异常:
            ValueError: prob_map 维度非法、或 shape 参数非法
        """
        # 防御性检查
        if prob_map is None:
            raise ValueError("prob_map 不能为 None")
        if prob_map.ndim not in (2, 3, 4):
            raise ValueError(f"prob_map 必须是 2D/3D/4D，当前 ndim={prob_map.ndim}")
        orig_h, orig_w = orig_shape
        resized_h, resized_w = resized_shape
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"orig_shape 不能有非正维度: {orig_shape}")
        if resized_h <= 0 or resized_w <= 0:
            raise ValueError(f"resized_shape 不能有非正维度: {resized_shape}")

        # 降维到 [H, W]
        if prob_map.ndim == 4:
            prob_map = prob_map[0, 0]
        elif prob_map.ndim == 3:
            prob_map = prob_map[0]

        h, w = prob_map.shape

        # 1. 二值化
        bin_map = (prob_map > self.thresh).astype(np.uint8)

        # 2. 找轮廓
        contours, _ = cv2.findContours(bin_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 3. 坐标缩放比例
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        boxes: list[np.ndarray] = []
        for contour in contours:
            # 点数过滤
            if len(contour) < 4:
                continue
            if len(contour) > self.max_candidates:
                continue

            # 框得分：轮廓内概率均值
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
            score = float((prob_map * mask).sum() / max(mask.sum(), 1))
            if score < self.box_thresh:
                continue

            # unclip 扩展：基于最小外接矩形，按 sqrt(ratio) 线性扩展
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            # 扩展因子：面积比近似为 unclip_ratio，边长比近似为 sqrt
            expand = self.unclip_ratio ** 0.5
            new_rw = rw * expand
            new_rh = rh * expand
            new_rect = ((cx, cy), (new_rw, new_rh), angle)
            box_points = cv2.boxPoints(new_rect)  # [4, 2]

            # 坐标缩放回原图
            box_points[:, 0] = box_points[:, 0] * scale_x
            box_points[:, 1] = box_points[:, 1] * scale_y

            boxes.append(box_points.astype(np.float32))

        return boxes


def load_character_dict(yml_path: str) -> list[str]:
    """从 inference.yml 解析字符字典。

    参数:
        yml_path: inference.yml 文件路径

    返回:
        字符列表（不含 CTC blank，blank 在解码时用 len(dict) 表示）

    异常:
        FileNotFoundError: 配置文件不存在
        KeyError: 配置文件中缺少必要字段
    """
    path = Path(yml_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yml_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    post_process = config.get("PostProcess")
    if post_process is None:
        raise KeyError("配置文件中缺少 PostProcess 字段")

    character_dict = post_process.get("character_dict")
    if character_dict is None:
        raise KeyError("配置文件中缺少 PostProcess.character_dict 字段")

    # 检查 use_space_char，默认为 True（PP-OCR 惯例）。
    # 注意：实际 inference.yml 中可能未显式设置此字段，
    # 但根据模型输出维度（18710 = 18709 + 1 blank）和字典大小（18708），
    # 需要追加空格使字符表大小为 18709，与模型输出对齐。
    use_space_char = post_process.get("use_space_char", True)
    if use_space_char and " " not in character_dict:
        character_dict = character_dict + [" "]

    return character_dict


class CTCLabelDecode:
    """CTC 贪心解码器。

    将 rec 模型输出的 softmax 概率分布解码为文本。

    属性:
        character: 字符列表（不含 CTC blank）
    """

    def __init__(self, character_dict: list[str]):
        """初始化解码器。

        参数:
            character_dict: 字符列表（不含 CTC blank）
        """
        self.character = character_dict

    def __call__(self, probs: np.ndarray) -> list[tuple[str, float]]:
        """CTC 贪心解码。

        参数:
            probs: [N, T, C] softmax 概率，或 [T, C] 单个样本

        返回:
            list of (text, score)，每个样本一个

        异常:
            ValueError: probs 维度不为 2 或 3
        """
        # 1. 如果 probs 是 2D，扩展为 [1, T, C]
        if probs.ndim == 2:
            probs = probs[np.newaxis, ...]
        elif probs.ndim != 3:
            raise ValueError(f"probs 必须是 2D 或 3D，当前 ndim={probs.ndim}")

        n, t, c = probs.shape
        blank_idx = len(self.character)  # blank 索引 = 字符表长度

        # 2. 对每个样本进行解码
        # argmax 沿 C 维度 → [N, T] 索引序列
        pred_indices = probs.argmax(axis=2)
        # 取每个时间步的最大概率值 → [N, T] 分数
        pred_scores = probs.max(axis=2)

        results: list[tuple[str, float]] = []
        for i in range(n):
            indices = pred_indices[i]
            scores = pred_scores[i]

            # 去重相邻相同索引
            dedup_mask = np.ones(t, dtype=bool)
            dedup_mask[1:] = indices[1:] != indices[:-1]
            dedup_indices = indices[dedup_mask]
            dedup_scores = scores[dedup_mask]

            # 去除 blank 索引
            non_blank_mask = dedup_indices != blank_idx
            final_indices = dedup_indices[non_blank_mask]
            final_scores = dedup_scores[non_blank_mask]

            # 将剩余索引映射为字符
            char_list = [self.character[idx] for idx in final_indices]
            text = "".join(char_list)

            # score = 剩余时间步分数的均值
            if len(final_scores) == 0:
                score = 0.0
            else:
                score = float(np.mean(final_scores))

            results.append((text, score))

        return results
