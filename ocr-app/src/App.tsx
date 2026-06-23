import { useState, useRef, useCallback, useEffect } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/tauri";
import { open } from "@tauri-apps/api/dialog";
import { appWindow } from "@tauri-apps/api/window";
import { z } from "zod";
import "./App.css";

interface OcrPoint {
  x: number;
  y: number;
}

interface OcrTextBlock {
  text: string;
  score: number;
  box_points: OcrPoint[];
}

interface OcrResponse {
  blocks: OcrTextBlock[];
  page_angle: number;
  elapsed_ms: number;
}

interface ModelStatus {
  ready: boolean;
  version: string;
  det_size_mb: number;
  rec_size_mb: number;
  dict_entries: number;
}

/** zod schema：校验 ocr_recognize_base64 响应体，防止后端返回异常数据 */
const ocrPointSchema = z.object({
  x: z.number(),
  y: z.number(),
});

const ocrTextBlockSchema = z.object({
  text: z.string(),
  score: z.number(),
  box_points: z.array(ocrPointSchema),
});

const ocrResponseSchema = z.object({
  blocks: z.array(ocrTextBlockSchema),
  page_angle: z.number(),
  elapsed_ms: z.number(),
});

type AppState = "idle" | "loading-models" | "ready" | "recognizing" | "done" | "error";

function App() {
  const [state, setState] = useState<AppState>("idle");
  const [error, setError] = useState<string>("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [ocrResult, setOcrResult] = useState<OcrResponse | null>(null);
  const [imagePath, setImagePath] = useState<string>("");
  const [imageSrc, setImageSrc] = useState<string>("");
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [displaySize, setDisplaySize] = useState<{ w: number; h: number } | null>(null);
  const [modelVersion, setModelVersion] = useState<string>("Tiny");
  const [copiedAll, setCopiedAll] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [copiedSelected, setCopiedSelected] = useState(false);
  const [selectedBlocks, setSelectedBlocks] = useState<Set<number>>(new Set());
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);
  const imageAreaRef = useRef<HTMLDivElement>(null);
  // 用 ref 持有最新 state，避免 paste 监听器频繁重注册
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    invoke<ModelStatus>("get_model_status").then((status) => {
      if (status.ready) {
        setModelStatus(status);
        setState("ready");
      } else {
        loadModels("Tiny");
      }
    }).catch(() => {
      loadModels("Tiny");
    });
  }, []);

  // 监听 Tauri 拖放事件（桌面端）
  useEffect(() => {
    const unlisten = appWindow.onFileDropEvent((event) => {
      if (event.payload.type === "drop") {
        const paths = event.payload.paths;
        if (paths.length > 0) {
          setImageFromFile(paths[0]);
        }
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  // 监听 Ctrl+V 粘贴剪贴板图片（截图 / 复制的图片）
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      // OCR 或模型加载中时，不拦截粘贴
      if (stateRef.current === "recognizing" || stateRef.current === "loading-models") return;
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          e.preventDefault();
          const file = item.getAsFile();
          if (!file) return;
          const reader = new FileReader();
          reader.onload = () => {
            setImagePath("");
            setImageSrc(reader.result as string);
            setOcrResult(null);
            setSelectedBlocks(new Set());
            setState("ready");
          };
          reader.readAsDataURL(file);
          return;
        }
      }
    };
    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, []);

  useEffect(() => {
    if (!ocrResult || !imageSize) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = imageSize.w;
    canvas.height = imageSize.h;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ocrResult.blocks.forEach((block, i) => {
      if (block.box_points.length < 4) return;
      const selected = selectedBlocks.has(i);
      ctx.strokeStyle = selected ? "rgba(220, 38, 38, 0.85)" : "rgba(0, 180, 255, 0.85)";
      ctx.lineWidth = 2;
      ctx.fillStyle = selected ? "rgba(220, 38, 38, 0.15)" : "rgba(0, 180, 255, 0.1)";
      ctx.beginPath();
      ctx.moveTo(block.box_points[0].x, block.box_points[0].y);
      for (let j = 1; j < block.box_points.length; j++) {
        ctx.lineTo(block.box_points[j].x, block.box_points[j].y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });
  }, [ocrResult, imageSize, selectedBlocks]);

  const loadModels = useCallback(async (ver?: string) => {
    setState("loading-models");
    setError("");
    try {
      const v = ver || modelVersion;
      const status = await invoke<ModelStatus>("ensure_models", {
        version: { [v]: null },
      });
      setModelStatus(status);
      setModelVersion(v);
      setState("ready");
    } catch (e) {
      setError(String(e));
      setState("error");
    }
  }, [modelVersion]);

  const setImageFromFile = (path: string) => {
    setImagePath(path);
    setImageSrc(convertFileSrc(path));
    setOcrResult(null);
    setSelectedBlocks(new Set());
    setState("ready");
  };

  const selectImage = async () => {
    const selected = await open({
      multiple: false,
      filters: [{
        name: "图片",
        extensions: ["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "gif"],
      }],
    });
    if (selected) {
      const path = Array.isArray(selected) ? selected[0] : selected;
      if (path) setImageFromFile(path);
    }
  };

  const runOcr = async () => {
    if (!imageSrc) return;
    setState("recognizing");
    setError("");
    try {
      let result: OcrResponse;
      if (imagePath) {
        // 文件路径模式（对话框选择 / Tauri 拖放）
        result = await invoke<OcrResponse>("ocr_recognize", { imagePath });
      } else {
        // 剪贴板粘贴 / HTML 拖放回退：data URL 模式，用 zod 严格校验响应
        const raw = await invoke<unknown>("ocr_recognize_base64", { imageData: imageSrc });
        result = ocrResponseSchema.parse(raw);
      }
      setOcrResult(result);
      setSelectedBlocks(new Set());
      setState("done");
    } catch (e) {
      setError(String(e));
      setState("error");
    }
  };

  const onImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
  };

  /** 根据容器可用空间和图片原始尺寸，计算等比缩放后的显示尺寸 */
  const computeDisplaySize = useCallback(() => {
    if (!imageSize || !imageAreaRef.current) return;
    const container = imageAreaRef.current;
    const style = getComputedStyle(container);
    const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    const maxW = container.clientWidth - padX;
    const maxH = container.clientHeight - padY;
    if (maxW <= 0 || maxH <= 0) return;

    const scale = Math.min(maxW / imageSize.w, maxH / imageSize.h, 1);
    setDisplaySize({
      w: Math.round(imageSize.w * scale),
      h: Math.round(imageSize.h * scale),
    });
  }, [imageSize]);

  /** 监听容器尺寸变化，自动重新计算缩放 */
  useEffect(() => {
    if (!imageSize) return;
    computeDisplaySize();
    const observer = new ResizeObserver(() => computeDisplaySize());
    if (imageAreaRef.current) observer.observe(imageAreaRef.current);
    return () => observer.disconnect();
  }, [imageSize, computeDisplaySize]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dropRef.current?.classList.add("drag-over");
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dropRef.current?.classList.remove("drag-over");
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dropRef.current?.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      const path = (file as unknown as { path?: string }).path;
      if (path) {
        setImageFromFile(path);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          setImagePath("");
          setImageSrc(reader.result as string);
          setOcrResult(null);
          setSelectedBlocks(new Set());
          setState("ready");
        };
        reader.readAsDataURL(file);
      }
    }
  };

  const copyText = (text: string, type: "single" | "all" | "selected", index?: number) => {
    navigator.clipboard.writeText(text);
    if (type === "single" && index !== undefined) {
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 1500);
    } else if (type === "all") {
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 1500);
    } else if (type === "selected") {
      setCopiedSelected(true);
      setTimeout(() => setCopiedSelected(false), 1500);
    }
  };

  const toggleBlock = (i: number) => {
    setSelectedBlocks((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  /** 射线法判断点是否在多边形内 */
  const pointInPolygon = (px: number, py: number, polygon: OcrPoint[]): boolean => {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x, yi = polygon[i].y;
      const xj = polygon[j].x, yj = polygon[j].y;
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  };

  /** Canvas 点击：将显示坐标转为原图坐标，检测命中的框并切换选中 */
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!ocrResult || !imageSize || !displaySize) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    // 显示坐标 → 原图坐标
    const scaleX = imageSize.w / displaySize.w;
    const scaleY = imageSize.h / displaySize.h;
    const imgX = clickX * scaleX;
    const imgY = clickY * scaleY;

    // 遍历框，找到索引最小的命中框
    for (let i = 0; i < ocrResult.blocks.length; i++) {
      const block = ocrResult.blocks[i];
      if (block.box_points.length < 4) continue;
      if (pointInPolygon(imgX, imgY, block.box_points)) {
        toggleBlock(i);
        return;
      }
    }
  };

  const toggleAll = () => {
    if (!ocrResult) return;
    if (selectedBlocks.size === ocrResult.blocks.length) {
      setSelectedBlocks(new Set());
    } else {
      setSelectedBlocks(new Set(ocrResult.blocks.map((_, i) => i)));
    }
  };

  const switchModel = async (ver: string) => {
    if (ver === modelVersion && modelStatus?.ready) return;
    await loadModels(ver);
  };

  const allSelected = ocrResult ? selectedBlocks.size === ocrResult.blocks.length : false;

  return (
    <div className="app">
      <header className="header">
        <h1>PP-OCRv6</h1>
        <span className="subtitle">离线文字识别</span>
        <div className="model-switcher">
          <button
            className={`model-btn ${modelVersion === "Tiny" ? "active" : ""}`}
            onClick={() => switchModel("Tiny")}
            disabled={state === "loading-models"}
          >
            轻量
          </button>
          <button
            className={`model-btn ${modelVersion === "Small" ? "active" : ""}`}
            onClick={() => switchModel("Small")}
            disabled={state === "loading-models"}
          >
            标准
          </button>
          {state === "loading-models" && <span className="loading-hint">切换中...</span>}
        </div>
      </header>

      {!modelStatus?.ready && (
        <div className="model-section">
          <p className="hint">
            点击上方按钮选择模型并加载。轻量版适合低性能设备，标准版精度更高。
          </p>
        </div>
      )}

      {modelStatus?.ready && (
        <div className="main-content">
          <div className="panel-image">
            <div
              className="image-area"
              ref={(el) => { dropRef.current = el; imageAreaRef.current = el; }}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={!imageSrc ? selectImage : undefined}
            >
              {imageSrc ? (
                <div className="image-container" style={displaySize ? { width: displaySize.w, height: displaySize.h } : undefined}>
                  <img
                    src={imageSrc}
                    alt="输入图片"
                    onLoad={onImageLoad}
                    className="ocr-image"
                    style={displaySize ? { width: displaySize.w, height: displaySize.h } : undefined}
                  />
                  {ocrResult && displaySize && (
                    <canvas
                      ref={canvasRef}
                      className="overlay-canvas"
                      style={{ width: displaySize.w, height: displaySize.h }}
                      onClick={handleCanvasClick}
                    />
                  )}
                </div>
              ) : (
                <div className="drop-zone">
                  <div className="drop-icon">+</div>
                  <p>拖放图片到此处，或点击选择（支持 Ctrl+V 粘贴）</p>
                </div>
              )}
            </div>
            <div className="controls">
              <button onClick={selectImage} className="btn">选择图片</button>
              <button onClick={runOcr} disabled={!imageSrc || state === "recognizing"} className="btn-primary">
                {state === "recognizing" ? "识别中..." : "开始识别"}
              </button>
            </div>
          </div>

          <div className="panel-result">
            {ocrResult ? (
              <div className="results">
                <div className="results-header">
                  <div className="results-header-left">
                    <input
                      type="checkbox"
                      checked={allSelected}
                      onChange={toggleAll}
                      className="checkbox-all"
                      title="全选/取消全选"
                    />
                    <h2>识别结果 ({ocrResult.blocks.length} 行, {ocrResult.elapsed_ms} ms)</h2>
                  </div>
                  <div className="results-header-right">
                    {selectedBlocks.size > 0 && (
                      <button
                        onClick={() => copyText(
                          ocrResult.blocks
                            .filter((_, i) => selectedBlocks.has(i))
                            .map((b) => b.text)
                            .join("\n"),
                          "selected"
                        )}
                        className="btn-small btn-accent"
                      >
                        {copiedSelected ? "已复制" : `复制选中 (${selectedBlocks.size})`}
                      </button>
                    )}
                    <button
                      onClick={() => copyText(ocrResult.blocks.map((b) => b.text).join("\n"), "all")}
                      className="btn-small"
                    >
                      {copiedAll ? "已复制" : "复制全部"}
                    </button>
                  </div>
                </div>
                <div className="results-list">
                  {ocrResult.blocks.map((block, i) => (
                    <div key={i} className={`result-block ${selectedBlocks.has(i) ? "selected" : ""}`}>
                      <input
                        type="checkbox"
                        checked={selectedBlocks.has(i)}
                        onChange={() => toggleBlock(i)}
                        className="checkbox-block"
                      />
                      <span className="block-index">{i + 1}</span>
                      <span className="block-text">{block.text}</span>
                      <span className="block-score">{(block.score * 100).toFixed(1)}%</span>
                      <button
                        className="btn-copy"
                        onClick={() => copyText(block.text, "single", i)}
                        title="复制"
                      >
                        {copiedIndex === i ? "✓" : "⧉"}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="result-placeholder">
                <p>选择图片并点击"开始识别"</p>
              </div>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => { setError(""); setState(modelStatus?.ready ? "ready" : "idle"); }}>
            关闭
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
