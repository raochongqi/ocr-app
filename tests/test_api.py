"""API 接口测试。

测试覆盖：
- 健康检查 /health
- 上传图片进行 OCR 识别 /ocr
- 空文件、非法文件、未上传文件等边界与错误情况

使用 fastapi.testclient.TestClient，模型加载较慢，scope="class" 共享 client。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocr_service.app import app


class TestAPI:
    """OCR 服务 API 接口测试。"""

    @classmethod
    @pytest.fixture(scope="class")
    def client(cls) -> TestClient:
        """创建测试客户端（加载模型，较慢，scope="class" 共享实例）。"""
        return TestClient(app)

    def test_health(self, client: TestClient):
        """健康检查应返回 ok 且模型已加载。

        路径: GET /health
        预期: 200, { status: "ok", models_loaded: true }
        """
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True

    def test_ocr_with_image(self, client: TestClient):
        """上传图片应返回识别结果。

        路径: POST /ocr
        预期: 200, 响应包含 text 和 items，每个 item 含 box/text/score，box 为 4 个点。
        """
        # 生成测试图片：白底黑字
        img = np.ones((200, 500, 3), dtype=np.uint8) * 255
        cv2.putText(
            img,
            "Hello API",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            2,
        )
        # 编码为 PNG
        ok, buf = cv2.imencode(".png", img)
        assert ok, "PNG 编码失败"

        resp = client.post(
            "/ocr", files={"file": ("test.png", buf.tobytes(), "image/png")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "items" in data
        assert isinstance(data["items"], list)
        for item in data["items"]:
            assert "box" in item
            assert "text" in item
            assert "score" in item
            assert len(item["box"]) == 4  # 4 个点

    def test_ocr_empty_file(self, client: TestClient):
        """空文件应返回 400。

        路径: POST /ocr
        预期: 400（文件为空）
        """
        resp = client.post(
            "/ocr", files={"file": ("empty.png", b"", "image/png")}
        )
        assert resp.status_code == 400

    def test_ocr_invalid_file(self, client: TestClient):
        """非图片文件应返回 400。

        路径: POST /ocr
        预期: 400（无法解码图片）
        """
        resp = client.post(
            "/ocr",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400

    def test_ocr_no_file(self, client: TestClient):
        """未上传文件应返回 422。

        路径: POST /ocr（不带 file 字段）
        预期: 422（FastAPI 参数校验失败）
        """
        resp = client.post("/ocr")
        assert resp.status_code == 422
