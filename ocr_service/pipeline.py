"""OCR 流水线模块。

模块向外暴露：
- OCRResult: 单个文本框识别结果的数据类
- OCRPipeline: 完整 OCR 流水线（det 检测 → 裁剪 → rec 识别 → 排序输出）
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from ocr_service.postprocess import CTCLabelDecode, DBPostProcess, load_character_dict
from ocr_service.preprocess import det_preprocess, rec_preprocess


@dataclass
class OCRResult:
    """单个文本框的识别结果。

    属性:
        box: [4, 2] 多边形坐标（相对于原图）
        text: 识别文本
        score: 置信度（0.0 ~ 1.0）
    """

    box: np.ndarray  # [4, 2] 多边形坐标（相对于原图）
    text: str  # 识别文本
    score: float  # 置信度


class OCRPipeline:
    """完整 OCR 流水线：图片 → det 检测文本框 → 裁剪 → rec 识别 → 排序输出。

    流水线步骤：
        1. 解码输入（bytes 或 ndarray）为 BGR 图像
        2. det 预处理 → det 推理 → DBPostProcess 得文本框列表
        3. 对每个文本框：透视变换裁剪 → rec 预处理 → rec 推理 → CTCLabelDecode
        4. 结果排序：按框中心 y 坐标排序（从上到下），同 y 范围内按 x 排序（从左到右）
    """

    def __init__(self, model_dir: str = "models"):
        """初始化 OCR 流水线。

        参数:
            model_dir: 模型目录，下含 PP-OCRv6_small_det_onnx/ 和 PP-OCRv6_small_rec_onnx/

        异常:
            FileNotFoundError: 模型文件或配置文件不存在
        """
        # 防御性检查：模型目录存在
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        # det 模型路径
        det_dir = os.path.join(model_dir, "PP-OCRv6_small_det_onnx")
        det_onnx = os.path.join(det_dir, "inference.onnx")
        if not os.path.isfile(det_onnx):
            raise FileNotFoundError(f"det 模型文件不存在: {det_onnx}")

        # rec 模型路径
        rec_dir = os.path.join(model_dir, "PP-OCRv6_small_rec_onnx")
        rec_onnx = os.path.join(rec_dir, "inference.onnx")
        rec_yml = os.path.join(rec_dir, "inference.yml")
        if not os.path.isfile(rec_onnx):
            raise FileNotFoundError(f"rec 模型文件不存在: {rec_onnx}")
        if not os.path.isfile(rec_yml):
            raise FileNotFoundError(f"rec 配置文件不存在: {rec_yml}")

        # 加载 ONNX 模型（CPU 推理，保证可移植性）
        self.det_sess = ort.InferenceSession(
            det_onnx, providers=["CPUExecutionProvider"]
        )
        self.rec_sess = ort.InferenceSession(
            rec_onnx, providers=["CPUExecutionProvider"]
        )

        # 缓存输入张量名（PP-OCRv6 ONNX 导出时输入名固定为 "x"）
        self.det_input_name = self.det_sess.get_inputs()[0].name
        self.rec_input_name = self.rec_sess.get_inputs()[0].name

        # 初始化后处理器
        # det 后处理参数与 inference.yml 中 PostProcess 配置一致
        self.det_post = DBPostProcess(
            thresh=0.2,
            box_thresh=0.45,
            max_candidates=3000,
            unclip_ratio=1.4,
        )

        # rec 后处理：从 inference.yml 加载字符字典并初始化 CTC 解码器
        character_dict = load_character_dict(rec_yml)
        self.rec_post = CTCLabelDecode(character_dict)

    def _decode_image(self, image: np.ndarray | bytes) -> np.ndarray:
        """将输入解码为 BGR ndarray。

        参数:
            image: BGR 图像 (np.ndarray) 或图片字节 (bytes)

        返回:
            BGR ndarray (H, W, 3) uint8

        异常:
            ValueError: 输入类型非法、解码失败、或图像为空
        """
        if isinstance(image, (bytes, bytearray, memoryview)):
            # bytes 输入：用 cv2.imdecode 解码
            img_array = np.frombuffer(image, dtype=np.uint8)
            if img_array.size == 0:
                raise ValueError("输入 bytes 为空，无法解码图像")
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("bytes 解码失败：不是有效的图像数据")
            return img
        elif isinstance(image, np.ndarray):
            # ndarray 输入：直接返回（防御性检查由 det_preprocess 完成）
            if image.size == 0:
                raise ValueError("输入 ndarray 为空")
            return image
        else:
            raise ValueError(
                f"image 必须是 np.ndarray 或 bytes，当前类型: {type(image).__name__}"
            )

    def _crop_text_region(
        self, img: np.ndarray, box: np.ndarray
    ) -> np.ndarray:
        """用透视变换将多边形文本区域校正为水平矩形并裁剪。

        参数:
            img: 原图 BGR ndarray (H, W, 3)
            box: [4, 2] 多边形坐标（相对于原图）

        返回:
            裁剪后的 BGR ndarray (h, w, 3)，h/w 由多边形外接矩形决定

        异常:
            ValueError: 外接矩形尺寸非法（宽或高为 0）
        """
        # 计算多边形的外接矩形 (x, y, w, h)
        rect = cv2.boundingRect(box.astype(np.float32))
        _, _, w, h = rect
        if w <= 0 or h <= 0:
            raise ValueError(f"文本框外接矩形尺寸非法: w={w}, h={h}")

        # 源点：多边形 4 个点
        src_pts = box.astype(np.float32)
        # 目标点：外接矩形 4 个角（注意 OpenCV 坐标系：x 向右，y 向下）
        dst_pts = np.array(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ],
            dtype=np.float32,
        )

        # 计算透视变换矩阵并应用
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        crop = cv2.warpPerspective(img, M, (w, h))

        # 防御性检查：裁剪结果非空
        if crop is None or crop.size == 0:
            raise ValueError("透视变换裁剪结果为空")
        return crop

    def _sort_results(self, results: list[OCRResult]) -> list[OCRResult]:
        """对识别结果排序：从上到下，同 y 范围内从左到右。

        排序规则：
            - 主排序键：框中心 y 坐标
            - 同 y 范围内（y 差 < 框高度的一半）按 x 坐标排序

        参数:
            results: OCRResult 列表

        返回:
            排序后的 OCRResult 列表
        """
        if len(results) <= 1:
            return list(results)

        # 计算每个框的中心 (cx, cy) 和高度
        infos = []
        for idx, r in enumerate(results):
            cx = float(r.box[:, 0].mean())
            cy = float(r.box[:, 1].mean())
            # 框高度：y 坐标极差
            height = float(r.box[:, 1].max() - r.box[:, 1].min())
            infos.append({"idx": idx, "cx": cx, "cy": cy, "height": height})

        # 先按 y 主排序
        infos.sort(key=lambda d: d["cy"])

        # 在同 y 范围内按 x 二次排序
        # 使用稳定排序：先按 x 排，再按 y 分组稳定排序
        # 简化实现：分组排序——遍历已按 y 排序的列表，相邻 y 差 < 框高度一半视为同一行
        sorted_results: list[OCRResult] = []
        i = 0
        n = len(infos)
        while i < n:
            # 找出与当前行 y 相近的所有框（同一行）
            row_start = i
            row_threshold = max(infos[i]["height"] * 0.5, 1.0)
            j = i + 1
            while j < n and (infos[j]["cy"] - infos[row_start]["cy"]) < row_threshold:
                j += 1
            # 同一行内按 x 排序
            row = infos[i:j]
            row.sort(key=lambda d: d["cx"])
            sorted_results.extend(results[d["idx"]] for d in row)
            i = j

        return sorted_results

    def predict(self, image: np.ndarray | bytes) -> list[OCRResult]:
        """端到端 OCR 识别。

        参数:
            image: BGR 图像 (np.ndarray) 或图片字节 (bytes)

        返回:
            list[OCRResult]，按从上到下、从左到右排序

        异常:
            ValueError: 输入类型非法或解码失败
        """
        # 1. 解码输入图像
        img = self._decode_image(image)

        # 2. det 预处理 → det 推理 → det 后处理
        det_tensor, (orig_h, orig_w, resized_h, resized_w) = det_preprocess(img)
        det_out = self.det_sess.run(None, {self.det_input_name: det_tensor})
        # det 输出: [N, 1, H, W] 概率图
        prob_map = det_out[0]
        boxes = self.det_post(
            prob_map, (orig_h, orig_w), (resized_h, resized_w)
        )

        # 3. 没检测到文本框，直接返回空列表
        if not boxes:
            return []

        # 4. 对每个文本框：裁剪 → rec 预处理 → rec 推理 → rec 后处理
        results: list[OCRResult] = []
        for box in boxes:
            try:
                crop = self._crop_text_region(img, box)
            except ValueError:
                # 跳过尺寸非法的框
                continue

            # rec 预处理
            rec_tensor = rec_preprocess(crop)
            # rec 推理
            rec_out = self.rec_sess.run(None, {self.rec_input_name: rec_tensor})
            # rec 输出: [N, T, C] softmax 概率
            rec_probs = rec_out[0]

            # CTC 解码
            decoded = self.rec_post(rec_probs)
            if not decoded:
                continue

            text, score = decoded[0]
            results.append(
                OCRResult(box=box.astype(np.float32), text=text, score=float(score))
            )

        # 5. 结果排序
        return self._sort_results(results)
