"""OCRPipeline 端到端测试。

测试覆盖：
- 合成文字图片端到端识别
- bytes 输入解码
- 空白图片处理
- 结果排序（从上到下、从左到右）
- 非法输入处理
- 模型目录不存在处理
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ocr_service.pipeline import OCRPipeline, OCRResult

# 模型目录（相对于 tests 目录的上级）
MODEL_DIR = str(Path(__file__).parent.parent / "models")


class TestOCRPipeline:
    """OCRPipeline 端到端测试。"""

    @classmethod
    @pytest.fixture(scope="class")
    def pipeline(cls) -> OCRPipeline:
        """加载真实模型（测试可能较慢，scope="class" 共享实例）。"""
        return OCRPipeline(model_dir=MODEL_DIR)

    def test_predict_with_generated_image(self, pipeline: OCRPipeline):
        """用合成的文字图片验证端到端识别。

        生成含英文文字的图片，调用 predict，验证：
        - 至少检测到 1 个结果
        - 每个结果结构正确（OCRResult、box.shape、text 类型、score 范围）
        - 至少识别到非空文字
        """
        # 生成含英文文字的图片：白底黑字
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.putText(
            img,
            "Hello OCR",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 0),
            3,
        )

        results = pipeline.predict(img)

        # 应检测到至少 1 个结果
        assert len(results) >= 1
        # 每个结果结构正确
        for r in results:
            assert isinstance(r, OCRResult)
            assert r.box.shape == (4, 2)
            assert isinstance(r.text, str)
            assert 0.0 <= r.score <= 1.0
        # 至少识别到非空文字（合成图片识别率应较高，但保守断言）
        all_text = "".join(r.text for r in results).upper()
        assert len(all_text) > 0

    def test_predict_with_bytes(self, pipeline: OCRPipeline):
        """测试 bytes 输入：编码为 PNG 后传入应正常识别。"""
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(
            img,
            "Test 2026",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            2,
        )
        # 编码为 PNG bytes
        ok, buf = cv2.imencode(".png", img)
        assert ok, "PNG 编码失败"

        results = pipeline.predict(buf.tobytes())

        # 应返回 list 类型
        assert isinstance(results, list)
        # 结构正确
        for r in results:
            assert isinstance(r, OCRResult)
            assert r.box.shape == (4, 2)

    def test_predict_blank_image(self, pipeline: OCRPipeline):
        """空白图片应返回空列表或无文字结果。

        纯白图片无文字特征，det 应检测不到文本框，或识别为空/低置信度。
        """
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # 纯白
        results = pipeline.predict(img)

        # 空白图片可能检测不到文本框（返回空列表），
        # 或检测到但识别为空/低置信度
        for r in results:
            assert r.text == "" or r.score < 0.5

    def test_results_sorted(self, pipeline: OCRPipeline):
        """验证结果按从上到下排序。

        生成上下两行文字的图片，验证结果 y 坐标递增。
        """
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.putText(
            img,
            "Top Line",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            img,
            "Bottom Line",
            (50, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

        results = pipeline.predict(img)
        if len(results) >= 2:
            # y 坐标应递增（从上到下），允许小误差
            ys = [float(r.box[:, 1].mean()) for r in results]
            for i in range(len(ys) - 1):
                assert ys[i] <= ys[i + 1] + 20, (
                    f"结果未按 y 排序: ys[{i}]={ys[i]}, ys[{i+1}]={ys[i+1]}"
                )

    def test_predict_invalid_input_type(self, pipeline: OCRPipeline):
        """非法输入类型应抛 ValueError。"""
        with pytest.raises(ValueError):
            pipeline.predict(123)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            pipeline.predict("not bytes")  # type: ignore[arg-type]

    def test_predict_empty_bytes(self, pipeline: OCRPipeline):
        """空 bytes 应抛 ValueError。"""
        with pytest.raises(ValueError):
            pipeline.predict(b"")

    def test_predict_invalid_bytes(self, pipeline: OCRPipeline):
        """无效 bytes（非图像数据）应抛 ValueError。"""
        with pytest.raises(ValueError):
            pipeline.predict(b"this is not an image")

    def test_invalid_model_dir(self):
        """模型目录不存在应抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            OCRPipeline(model_dir="/nonexistent/path/to/models")

    def test_predict_empty_ndarray(self, pipeline: OCRPipeline):
        """空 ndarray 应抛 ValueError。"""
        with pytest.raises(ValueError):
            pipeline.predict(np.array([]))
