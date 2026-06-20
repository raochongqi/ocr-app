"""det 预处理与后处理单元测试。

测试覆盖：
- det_preprocess: 形状/dtype/数值范围、32 对齐、小图处理、大图缩放、非法输入
- DBPostProcess: 单框检测、空图、低概率、4D 输入、坐标缩放
"""
import numpy as np
import pytest

from ocr_service.preprocess import det_preprocess
from ocr_service.postprocess import DBPostProcess


class TestDetPreprocess:
    """det_preprocess 单元测试。"""

    def test_shape_and_dtype(self):
        """200x400 图像预处理后应为 [1, 3, 224, 416] float32。"""
        img = np.ones((200, 400, 3), dtype=np.uint8) * 128
        tensor, shape_info = det_preprocess(img)

        # 形状：200->224, 400->416（32 对齐）
        assert tensor.shape == (1, 3, 224, 416)
        assert tensor.dtype == np.float32
        # 尺寸信息
        orig_h, orig_w, resized_h, resized_w = shape_info
        assert orig_h == 200
        assert orig_w == 400
        assert resized_h == 224
        assert resized_w == 416

    def test_value_range(self):
        """预处理后数值范围应合理（归一化后大致在 [-3, 3]）。"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        tensor, _ = det_preprocess(img)
        # 全 255 归一化后为 (1 - mean) / std，约 [1.0, 1.1]
        assert tensor.min() > -3.0
        assert tensor.max() < 3.0

    def test_small_image_alignment(self):
        """30x30 小图预处理后 H/W 至少为 32（向上对齐）。"""
        img = np.ones((30, 30, 3), dtype=np.uint8) * 100
        tensor, shape_info = det_preprocess(img)

        assert tensor.shape[2] >= 32
        assert tensor.shape[3] >= 32
        # 30 对齐到 32
        assert tensor.shape[2] == 32
        assert tensor.shape[3] == 32
        _, _, resized_h, resized_w = shape_info
        assert resized_h == 32
        assert resized_w == 32

    def test_max_side_limit(self):
        """大图应被等比缩放到 max_side 附近（32 对齐后允许小幅超出）。"""
        img = np.ones((2000, 1000, 3), dtype=np.uint8) * 200
        tensor, shape_info = det_preprocess(img, max_side=960)
        _, _, resized_h, resized_w = shape_info
        # ratio = min(960/2000, 960/1000, 1.0) = 0.48
        # new_h = 960, new_w = 480，对齐后 960, 480
        assert resized_h == 960
        assert resized_w == 480
        assert max(resized_h, resized_w) <= 992  # 960 + 32 容差

    def test_no_upscale(self):
        """小图不应被放大（ratio 上限为 1.0）。"""
        img = np.ones((50, 80, 3), dtype=np.uint8) * 100
        tensor, shape_info = det_preprocess(img, max_side=960)
        _, _, resized_h, resized_w = shape_info
        # 50->64(对齐), 80->96(对齐)，不放大
        assert resized_h == 64
        assert resized_w == 96

    def test_invalid_input_shape(self):
        """非法输入形状应抛异常。"""
        with pytest.raises(ValueError):
            det_preprocess(np.zeros((100, 100), dtype=np.uint8))  # 缺通道
        with pytest.raises(ValueError):
            det_preprocess(np.zeros((100, 100, 4), dtype=np.uint8))  # 4 通道

    def test_invalid_max_side(self):
        """非法 max_side 应抛异常。"""
        img = np.ones((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            det_preprocess(img, max_side=0)
        with pytest.raises(ValueError):
            det_preprocess(img, max_side=-1)


class TestDBPostProcess:
    """DBPostProcess 单元测试。"""

    def test_detect_single_box(self):
        """构造单一高概率区域，应检测出 1 个框且坐标在原图范围内。"""
        prob_map = np.full((100, 100), 0.1, dtype=np.float32)
        prob_map[40:60, 20:80] = 0.9

        post = DBPostProcess()
        boxes = post(prob_map, (100, 100), (100, 100))

        assert len(boxes) == 1
        box = boxes[0]
        assert box.shape == (4, 2)
        assert box.dtype == np.float32
        # 坐标在原图范围内（unclip 可能略微外扩，留 1 像素容差）
        assert box[:, 0].min() >= -1.0
        assert box[:, 0].max() <= 101.0
        assert box[:, 1].min() >= -1.0
        assert box[:, 1].max() <= 101.0

    def test_empty_prob_map(self):
        """全 0 概率图应返回空列表。"""
        prob_map = np.zeros((100, 100), dtype=np.float32)
        post = DBPostProcess()
        boxes = post(prob_map, (100, 100), (100, 100))
        assert boxes == []

    def test_low_prob_no_detection(self):
        """概率低于 thresh 的区域不应被检测。"""
        prob_map = np.full((100, 100), 0.15, dtype=np.float32)  # < thresh 0.2
        post = DBPostProcess()
        boxes = post(prob_map, (100, 100), (100, 100))
        assert boxes == []

    def test_below_box_thresh_no_detection(self):
        """高于 thresh 但低于 box_thresh 的区域不应被检测。"""
        # thresh=0.2，box_thresh=0.45，区域均值 0.3 通过二值化但不过框得分
        prob_map = np.full((100, 100), 0.1, dtype=np.float32)
        prob_map[40:60, 20:80] = 0.3
        post = DBPostProcess()
        boxes = post(prob_map, (100, 100), (100, 100))
        assert boxes == []

    def test_4d_input(self):
        """4D 概率图 [N, 1, H, W] 应正确处理。"""
        prob_map = np.full((1, 1, 100, 100), 0.1, dtype=np.float32)
        prob_map[0, 0, 40:60, 20:80] = 0.9
        post = DBPostProcess()
        boxes = post(prob_map, (100, 100), (100, 100))
        assert len(boxes) == 1
        assert boxes[0].shape == (4, 2)

    def test_3d_input(self):
        """3D 概率图 [1, H, W] 应正确处理。"""
        prob_map = np.full((1, 100, 100), 0.1, dtype=np.float32)
        prob_map[0, 40:60, 20:80] = 0.9
        post = DBPostProcess()
        boxes = post(prob_map, (100, 100), (100, 100))
        assert len(boxes) == 1

    def test_coordinate_scaling(self):
        """坐标应正确从 resized_shape 缩放到 orig_shape。"""
        # resized 100x100 -> orig 200x200，比例 2.0
        prob_map = np.full((100, 100), 0.1, dtype=np.float32)
        prob_map[40:60, 20:80] = 0.9
        post = DBPostProcess()
        boxes = post(prob_map, (200, 200), (100, 100))
        assert len(boxes) == 1
        box = boxes[0]
        # 原区域中心约 (50, 50)，缩放后在 (100, 100) 附近
        center_x = box[:, 0].mean()
        center_y = box[:, 1].mean()
        assert 80 < center_x < 120
        assert 80 < center_y < 120

    def test_custom_params(self):
        """自定义参数应生效：提高 thresh 使低概率区域不检测。"""
        prob_map = np.full((100, 100), 0.1, dtype=np.float32)
        prob_map[40:60, 20:80] = 0.5
        # thresh=0.6 使 0.5 区域被过滤
        post = DBPostProcess(thresh=0.6)
        boxes = post(prob_map, (100, 100), (100, 100))
        assert boxes == []

    def test_invalid_prob_map(self):
        """非法 prob_map 应抛异常。"""
        post = DBPostProcess()
        with pytest.raises(ValueError):
            post(np.zeros((100,), dtype=np.float32), (100, 100), (100, 100))  # 1D
        with pytest.raises(ValueError):
            post(None, (100, 100), (100, 100))  # None

    def test_invalid_shape(self):
        """非法 orig_shape/resized_shape 应抛异常。"""
        prob_map = np.full((100, 100), 0.1, dtype=np.float32)
        post = DBPostProcess()
        with pytest.raises(ValueError):
            post(prob_map, (0, 100), (100, 100))  # orig_h=0
        with pytest.raises(ValueError):
            post(prob_map, (100, 100), (-1, 100))  # resized_h<0
