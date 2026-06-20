"""OCR 服务 FastAPI 应用。

提供 HTTP 接口接收图片并返回 OCR 识别结果。

模块向外暴露：
- app: FastAPI 应用实例
- get_pipeline: 获取全局 OCRPipeline 实例（懒加载）

接口：
    GET  /health  — 健康检查
    POST /ocr     — 上传图片进行文字识别
"""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ocr_service.pipeline import OCRPipeline, OCRResult

# 全局 pipeline 实例（启动时懒加载，避免重复加载模型）
_pipeline: OCRPipeline | None = None


def get_pipeline() -> OCRPipeline:
    """获取全局 OCRPipeline 实例（懒加载）。

    模型目录通过环境变量 OCR_MODEL_DIR 配置，默认为 "models"。

    返回:
        OCRPipeline 实例

    异常:
        FileNotFoundError: 模型文件或配置文件不存在
    """
    global _pipeline
    if _pipeline is None:
        model_dir = os.environ.get("OCR_MODEL_DIR", "models")
        _pipeline = OCRPipeline(model_dir=model_dir)
    return _pipeline


# ---------- 响应模型（pydantic，遵循工作区规则：复合数据强制使用 pydantic） ----------


class HealthResponse(BaseModel):
    """健康检查响应。

    属性:
        status: 服务状态（"ok" 或 "error"）
        models_loaded: 模型是否已加载
    """

    status: str = Field(..., description="服务状态：ok / error")
    models_loaded: bool = Field(..., description="模型是否已加载")


class OCRItemResponse(BaseModel):
    """单个文本框识别结果。

    属性:
        box: [4, 2] 多边形坐标（相对于原图），4 个 [x, y] 点
        text: 识别文本
        score: 置信度（0.0 ~ 1.0）
    """

    box: list[list[float]] = Field(
        ..., description="4 个 [x, y] 坐标点构成的多边形"
    )
    text: str = Field(..., description="识别文本")
    score: float = Field(..., description="置信度，范围 0.0 ~ 1.0")


class OCRResponse(BaseModel):
    """OCR 识别整体响应。

    属性:
        text: 合并全文（各文本框用换行符分隔）
        items: 各文本框识别结果列表
    """

    text: str = Field(..., description="合并全文，各文本框用换行符分隔")
    items: list[OCRItemResponse] = Field(
        ..., description="各文本框识别结果列表"
    )


# ---------- FastAPI 应用 ----------

app = FastAPI(title="OCR 文字识别服务", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """健康检查接口。

    路径: GET /health
    请求参数: 无
    响应参数: HealthResponse（status, models_loaded）
    异常处理: 模型加载失败时返回 status="error", models_loaded=False（不抛异常）
    """
    try:
        get_pipeline()
        return HealthResponse(status="ok", models_loaded=True)
    except Exception:
        # 模型加载失败时仍返回 200，但标记为 error，便于探活
        return HealthResponse(status="error", models_loaded=False)


@app.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)) -> OCRResponse:
    """上传图片进行 OCR 识别。

    路径: POST /ocr
    请求参数:
        file: 图片文件（支持 png/jpg/jpeg/bmp 等 OpenCV 可解码格式）
    响应参数:
        OCRResponse: { text: 合并全文, items: [{box, text, score}, ...] }

    异常处理:
        400: 文件为空或无法解码（非有效图片）
        500: OCR 识别过程中发生内部错误
    """
    # 1. 读取文件内容（防御性：空文件拒绝）
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")

    # 2. 解码图片为 BGR ndarray
    img_array = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400, detail="无法解码图片，请检查文件格式"
        )

    # 3. OCR 识别
    try:
        pipeline = get_pipeline()
        results: list[OCRResult] = pipeline.predict(img)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"OCR 识别失败: {str(e)}"
        ) from e

    # 4. 构造响应（pydantic 模型，自动类型校验与序列化）
    items: list[OCRItemResponse] = []
    texts: list[str] = []
    for r in results:
        items.append(
            OCRItemResponse(
                box=r.box.tolist(),  # [[x,y], [x,y], [x,y], [x,y]]
                text=r.text,
                score=float(r.score),
            )
        )
        texts.append(r.text)

    return OCRResponse(text="\n".join(texts), items=items)
