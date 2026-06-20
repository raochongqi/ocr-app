# OCR 文字识别服务

基于 PaddlePaddle PP-OCRv6_small_onnx 模型的离线 OCR 服务，提供文字检测（det）和文字识别（rec）的完整流水线。

## 技术栈

- Python 3.12（uv 管理）
- ONNX Runtime（CPU 推理）
- OpenCV（图像处理）
- FastAPI（HTTP 服务）
- 模型：PP-OCRv6_small_det_onnx（2.48M 参数）+ PP-OCRv6_small_rec_onnx（5.2M 参数）

## 项目结构

```
ocr-service/
├── models/                              # ONNX 模型文件（自包含）
│   ├── PP-OCRv6_small_det_onnx/
│   │   ├── inference.onnx               # 文本检测模型
│   │   └── inference.yml                # 预处理/后处理配置
│   └── PP-OCRv6_small_rec_onnx/
│       ├── inference.onnx               # 文本识别模型
│       └── inference.yml                # 预处理/后处理配置 + 内嵌字符字典
├── ocr_service/                         # 核心代码
│   ├── __init__.py
│   ├── preprocess.py                    # det/rec 预处理
│   ├── postprocess.py                   # DBPostProcess / CTCLabelDecode
│   ├── pipeline.py                      # OCRPipeline 完整流水线
│   └── app.py                           # FastAPI 应用
├── scripts/                             # 探索性脚本
│   ├── inspect_model.py                 # 检查模型 I/O 元信息
│   └── test_inference.py                # 实际推理验证
├── tests/                               # 测试
│   ├── test_det_process.py              # det 预处理/后处理测试
│   ├── test_rec_process.py              # rec 预处理/后处理测试
│   ├── test_pipeline.py                 # 端到端流水线测试
│   └── test_api.py                      # API 接口测试
├── pyproject.toml
└── README.md
```

## 快速开始

### 安装依赖

```bash
cd ocr-service
uv sync
```

### 运行测试

```bash
uv run pytest tests/ -v
```

### 启动服务

```bash
uv run uvicorn ocr_service.app:app --host 0.0.0.0 --port 8101
```

服务启动后访问：
- API 文档：http://localhost:8101/docs
- 健康检查：http://localhost:8101/health

## API 接口

### GET /health

健康检查。

**响应**：
```json
{
  "status": "ok",
  "models_loaded": true
}
```

### POST /ocr

上传图片进行 OCR 文字识别。

**请求**：`multipart/form-data`，字段 `file` 为图片文件（支持 png/jpg/jpeg/bmp 等）。

**响应**：
```json
{
  "text": "识别的全文（各文本框用换行分隔）",
  "items": [
    {
      "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "text": "单行文本",
      "score": 0.98
    }
  ]
}
```

**示例**：
```bash
curl -X POST http://localhost:8101/ocr \
  -F "file=@test.png"
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `OCR_MODEL_DIR` | `models` | 模型目录路径 |

## 模型说明

### det 模型（文本检测）

- 输入：`x` `[N, 3, H, W]` float32，BGR，归一化（ImageNet mean/std）
- **约束：H 和 W 必须是 32 的倍数**（DBNet 架构要求）
- 输出：`fetch_name_0` `[N, 1, H, W]` sigmoid 概率图
- 后处理：DBPostProcess（thresh=0.2, box_thresh=0.45, unclip_ratio=1.4）

### rec 模型（文本识别）

- 输入：`x` `[N, 3, 48, W]` float32，BGR，归一化
- 输出：`fetch_name_0` `[N, T, 18710]` softmax（T=W/8，18710=18709字符+1blank）
- 后处理：CTCLabelDecode（greedy argmax + 去重 + 去 blank）
- 字符字典：内嵌在 `inference.yml` 的 `PostProcess.character_dict`（18709 个字符，含空格）
