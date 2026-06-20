"""rec 预处理和后处理的单元测试。

测试覆盖：
- load_character_dict: 字符字典解析
- rec_preprocess: 图像预处理（形状、宽图截断、自定义尺寸、无效输入）
- CTCLabelDecode: CTC 解码（"AB" 构造、全 blank、2D 输入、批量解码）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ocr_service.postprocess import CTCLabelDecode, load_character_dict
from ocr_service.preprocess import rec_preprocess

# 模型配置文件路径
YML_PATH = (
    Path(__file__).parent.parent
    / "models"
    / "PP-OCRv6_small_rec_onnx"
    / "inference.yml"
)


class TestLoadCharacterDict:
    """测试 load_character_dict 函数。"""

    def test_load_character_dict(self):
        """测试从 inference.yml 解析字符字典。

        验证返回列表长度 > 0，包含常见字符 'a'、'0'，
        且 use_space_char 默认为 True 时包含空格。
        """
        char_dict = load_character_dict(str(YML_PATH))

        # 验证返回列表长度 > 0
        assert isinstance(char_dict, list)
        assert len(char_dict) > 0

        # 验证包含常见字符
        assert "a" in char_dict
        assert "0" in char_dict

        # use_space_char 默认为 True，应包含空格
        assert " " in char_dict


class TestRecPreprocess:
    """测试 rec_preprocess 函数。"""

    def test_rec_preprocess_shape(self):
        """测试输入 (60, 200, 3) uint8 图像的输出形状和类型。"""
        img = np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8)
        tensor = rec_preprocess(img)

        assert tensor.shape == (1, 3, 48, 320)
        assert tensor.dtype == np.float32

    def test_rec_preprocess_wide_image(self):
        """测试宽图输入 (48, 500, 3)，验证宽度被截断到 320。"""
        img = np.random.randint(0, 256, (48, 500, 3), dtype=np.uint8)
        tensor = rec_preprocess(img)

        assert tensor.shape == (1, 3, 48, 320)
        assert tensor.dtype == np.float32

    def test_rec_preprocess_custom_size(self):
        """测试自定义目标尺寸。"""
        img = np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8)
        tensor = rec_preprocess(img, target_h=48, target_w=160)

        assert tensor.shape == (1, 3, 48, 160)
        assert tensor.dtype == np.float32

    def test_rec_preprocess_invalid_input(self):
        """测试无效输入应抛出 ValueError。"""
        # 空图像
        with pytest.raises(ValueError):
            rec_preprocess(np.array([]))

        # 错误的通道数
        with pytest.raises(ValueError):
            rec_preprocess(np.zeros((48, 320, 4), dtype=np.uint8))


class TestCTCLabelDecode:
    """测试 CTCLabelDecode 类。"""

    def test_ctc_decode_ab(self):
        """测试 CTC 解码 "AB"。

        构造 probs [1, 10, 5]（5 = 4 字符 + 1 blank）：
        - 时间步序列：[0, 0, 4, 1, 1, 4, 4, 4, 4, 4]
        - 0=A, 1=B, 4=blank
        - 去重后：[0, 4, 1, 4]
        - 去 blank 后：[0, 1] → "AB"
        """
        character_dict = ["A", "B", "C", "D"]
        decoder = CTCLabelDecode(character_dict)

        # 构造 probs：每个时间步在目标索引处置 1.0
        probs = np.zeros((1, 10, 5), dtype=np.float32)
        sequence = [0, 0, 4, 1, 1, 4, 4, 4, 4, 4]
        for t, idx in enumerate(sequence):
            probs[0, t, idx] = 1.0

        results = decoder(probs)
        assert len(results) == 1

        text, score = results[0]
        assert text == "AB"
        assert score > 0

    def test_ctc_decode_all_blank(self):
        """测试全 blank 输出应返回空字符串。"""
        character_dict = ["A", "B", "C", "D"]
        decoder = CTCLabelDecode(character_dict)

        # 全 blank 输出：所有时间步都是 blank（索引 4）
        probs = np.zeros((1, 10, 5), dtype=np.float32)
        probs[0, :, 4] = 1.0

        results = decoder(probs)
        assert len(results) == 1

        text, score = results[0]
        assert text == ""
        assert score == 0.0

    def test_ctc_decode_2d_input(self):
        """测试 2D 输入 [T, C] 应自动扩展为 [1, T, C]。"""
        character_dict = ["A", "B"]
        decoder = CTCLabelDecode(character_dict)

        # 2D 输入 [5, 3]，全 blank（索引 2）
        probs = np.zeros((5, 3), dtype=np.float32)
        probs[:, 2] = 1.0

        results = decoder(probs)
        assert len(results) == 1
        assert results[0][0] == ""

    def test_ctc_decode_batch(self):
        """测试批量解码 [N, T, C]。"""
        character_dict = ["A", "B"]
        decoder = CTCLabelDecode(character_dict)

        # 批量输入 [2, 5, 3]（3 = 2 字符 + 1 blank）
        # 先将所有时间步设为 blank（索引 2），避免未设置时间步 argmax 返回 0
        probs = np.zeros((2, 5, 3), dtype=np.float32)
        probs[:, :, 2] = 1.0  # 默认全 blank
        # 第一个样本：A
        probs[0, 0, 0] = 1.0  # A
        probs[0, 0, 2] = 0.0  # 覆盖 blank
        # 第二个样本：B
        probs[1, 0, 1] = 1.0  # B
        probs[1, 0, 2] = 0.0  # 覆盖 blank

        results = decoder(probs)
        assert len(results) == 2
        assert results[0][0] == "A"
        assert results[1][0] == "B"
