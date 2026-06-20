use std::sync::Mutex;
use ppocr_rs::{ModelHub, OcrLite, OcrOptions, PpOcrVersion, OcrResult};
use serde::{Deserialize, Serialize};

/// OCR engine state
pub struct OcrState {
    pub ocr: Option<OcrLite>,
    pub model_version: PpOcrVersion,
    pub models_ready: bool,
}

/// Model status returned to frontend
#[derive(Serialize, Deserialize)]
pub struct ModelStatus {
    pub ready: bool,
    pub version: String,
    pub det_size_mb: f64,
    pub rec_size_mb: f64,
    pub dict_entries: usize,
}

/// OCR result for a single text block
#[derive(Serialize, Deserialize)]
pub struct OcrTextBlock {
    pub text: String,
    pub score: f32,
    pub box_points: Vec<OcrPoint>,
}

#[derive(Serialize, Deserialize)]
pub struct OcrPoint {
    pub x: u32,
    pub y: u32,
}

/// Full OCR response
#[derive(Serialize, Deserialize)]
pub struct OcrResponse {
    pub blocks: Vec<OcrTextBlock>,
    pub page_angle: u32,
    pub elapsed_ms: u64,
}

/// Model version choice (Tiny for low-end devices, Small for standard)
#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum ModelVersion {
    Tiny,
    Small,
}

impl From<ModelVersion> for PpOcrVersion {
    fn from(v: ModelVersion) -> Self {
        match v {
            ModelVersion::Tiny => PpOcrVersion::V6Tiny,
            ModelVersion::Small => PpOcrVersion::V6Small,
        }
    }
}

fn version_str(v: PpOcrVersion) -> &'static str {
    match v {
        PpOcrVersion::V6Tiny => "tiny",
        PpOcrVersion::V6Small => "small",
        _ => "unknown",
    }
}

/// Try to load models from bundled resources first, then fall back to ModelHub download.
/// Search order:
///   1. Tauri resource_dir (works for deb/AppImage installs)
///   2. Exe-relative path (works for portable tar.gz distribution)
///   3. ModelHub cache (download if needed)
fn resolve_model_paths(app: &tauri::AppHandle, version: PpOcrVersion) -> Result<ppocr_rs::ModelPaths, String> {
    let ver_dir = match version {
        PpOcrVersion::V6Tiny => "pp_ocrv6_tiny",
        PpOcrVersion::V6Small => "pp_ocrv6_small",
        _ => "pp_ocrv6_small",
    };

    // Candidate directories to search for bundled models
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // 1. Tauri resource_dir (for installed packages: deb, AppImage)
    if let Some(res_dir) = app.path_resolver().resource_dir() {
        candidates.push(res_dir.join("models").join(ver_dir));
    }

    // 2. Exe-relative path (for portable tar.gz distribution: ./models/pp_ocrv6_*/)
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|p| p.to_path_buf())) {
        candidates.push(exe_dir.join("models").join(ver_dir));
    }

    for bundled in &candidates {
        let det = bundled.join("det.onnx");
        let rec = bundled.join("rec.onnx");
        let dict = bundled.join("dict.txt");
        let rec_yml = bundled.join("rec_inference.yml");

        if det.exists() && rec.exists() {
            // Auto-extract dict from yml if dict.txt is missing
            if !dict.exists() && rec_yml.exists() {
                eprintln!("[ocr-app] Extracting dict from rec_inference.yml...");
                extract_dict_from_yml(&rec_yml, &dict)?;
            }

            if dict.exists() {
                eprintln!("[ocr-app] Using bundled models from: {}", bundled.display());
                return Ok(ppocr_rs::ModelPaths {
                    det_onnx: det,
                    rec_onnx: rec,
                    dict_txt: dict,
                    rec_yml,
                });
            }
        }
    }

    // 3. Try ModelHub cache (already downloaded models)
    eprintln!("[ocr-app] No bundled models found, trying ModelHub...");
    let hub = ModelHub::with_default_cache().map_err(|e| e.to_string())?;
    let paths = hub.ensure(version).map_err(|e| format!("Model download/cache failed: {}", e))?;
    Ok(paths)
}

/// Extract character_dict from rec_inference.yml and write to dict.txt.
/// Uses line-by-line parsing matching ppocr-rs's private implementation.
fn extract_dict_from_yml(yml_path: &std::path::Path, dict_path: &std::path::Path) -> Result<(), String> {
    let content = std::fs::read_to_string(yml_path)
        .map_err(|e| format!("Failed to read yml: {}", e))?;

    let mut chars: Vec<String> = Vec::new();
    let mut in_dict = false;

    for line in content.lines() {
        let trimmed = line.trim_start();

        if !in_dict {
            if trimmed.starts_with("character_dict:") {
                in_dict = true;
            }
            continue;
        }

        // Each entry: `  - 'x'`, `  - x`, or `  - ` (empty/space char)
        if let Some(rest) = trimmed.strip_prefix("- ") {
            let rest_trimmed = rest.trim_end_matches('\r');
            let ch = if rest_trimmed.starts_with('\'') && rest_trimmed.ends_with('\'') && rest_trimmed.len() >= 2 {
                rest_trimmed[1..rest_trimmed.len() - 1].replace("''", "'")
            } else {
                rest_trimmed.to_string()
            };
            chars.push(ch);
        } else if trimmed.starts_with('-') {
            // Bare `-` (no space after) — treat as empty entry
            chars.push(String::new());
        } else if trimmed.is_empty() {
            // Blank line inside the list — skip, not a terminator
            continue;
        } else {
            // Non-list, non-blank line → end of character_dict list
            break;
        }
    }

    if chars.is_empty() {
        return Err("character_dict not found in rec_inference.yml".to_string());
    }

    let mut f = std::fs::File::create(dict_path)
        .map_err(|e| format!("Failed to create dict.txt: {}", e))?;
    use std::io::Write;
    for ch in &chars {
        writeln!(f, "{}", ch).map_err(|e| format!("Write error: {}", e))?;
    }

    eprintln!("[ocr-app] Dict extracted: {} entries", chars.len());
    Ok(())
}

/// Ensure models are loaded. If already loaded with the same version, return status directly.
/// If a different version is requested, reinitialize the engine.
#[tauri::command]
fn ensure_models(
    app: tauri::AppHandle,
    state: tauri::State<'_, Mutex<OcrState>>,
    version: Option<ModelVersion>,
) -> Result<ModelStatus, String> {
    let mut state = state.lock().map_err(|e| e.to_string())?;

    let ver = version.unwrap_or(ModelVersion::Small);
    let pp_ver: PpOcrVersion = ver.into();

    // If already loaded with the same version, return directly
    if state.models_ready && state.model_version == pp_ver {
        return Ok(ModelStatus {
            ready: true,
            version: version_str(pp_ver).to_string(),
            det_size_mb: 0.0,
            rec_size_mb: 0.0,
            dict_entries: 0,
        });
    }

    state.model_version = pp_ver;

    let paths = resolve_model_paths(&app, pp_ver)?;

    // Initialize OcrLite
    let mut ocr = OcrLite::new();
    ocr.init_models_no_angle(
        paths.det_onnx.to_str().ok_or("Invalid det path")?,
        paths.rec_onnx.to_str().ok_or("Invalid rec path")?,
        paths.dict_txt.to_str().ok_or("Invalid dict path")?,
        4,
    ).map_err(|e| format!("Model init failed: {}", e))?;

    let det_size = std::fs::metadata(&paths.det_onnx)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    let rec_size = std::fs::metadata(&paths.rec_onnx)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    let dict_entries = std::fs::read_to_string(&paths.dict_txt)
        .map(|s| s.lines().count())
        .unwrap_or(0);

    state.ocr = Some(ocr);
    state.models_ready = true;

    Ok(ModelStatus {
        ready: true,
        version: version_str(pp_ver).to_string(),
        det_size_mb: det_size,
        rec_size_mb: rec_size,
        dict_entries,
    })
}

/// Check if models are ready
#[tauri::command]
fn get_model_status(
    state: tauri::State<'_, Mutex<OcrState>>,
) -> Result<ModelStatus, String> {
    let state = state.lock().map_err(|e| e.to_string())?;

    Ok(ModelStatus {
        ready: state.models_ready,
        version: if state.models_ready { version_str(state.model_version).to_string() } else { String::new() },
        det_size_mb: 0.0,
        rec_size_mb: 0.0,
        dict_entries: 0,
    })
}

/// Run OCR on an image file
#[tauri::command]
fn ocr_recognize(
    state: tauri::State<'_, Mutex<OcrState>>,
    image_path: String,
) -> Result<OcrResponse, String> {
    let mut state = state.lock().map_err(|e| e.to_string())?;
    let ocr = state.ocr.as_mut().ok_or("OCR engine not initialized. Please load models first.")?;

    let img = image::open(&image_path)
        .map_err(|e| format!("Failed to open image: {}", e))?
        .to_rgb8();

    let start = std::time::Instant::now();
    let result: OcrResult = ocr.detect_with_options(
        &img,
        50, 960, 0.5, 0.3, 1.6,
        false, false,
        OcrOptions::default(),
    ).map_err(|e| format!("OCR failed: {}", e))?;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    let blocks: Vec<OcrTextBlock> = result.text_blocks.iter().map(|b| {
        OcrTextBlock {
            text: b.text.clone(),
            score: b.text_score,
            box_points: b.box_points.iter().map(|p| OcrPoint { x: p.x, y: p.y }).collect(),
        }
    }).collect();

    Ok(OcrResponse { blocks, page_angle: result.page_angle, elapsed_ms })
}

/// Run OCR on base64-encoded image data
#[tauri::command]
fn ocr_recognize_base64(
    state: tauri::State<'_, Mutex<OcrState>>,
    image_data: String,
) -> Result<OcrResponse, String> {
    let mut state = state.lock().map_err(|e| e.to_string())?;
    let ocr = state.ocr.as_mut().ok_or("OCR engine not initialized. Please load models first.")?;

    let b64_data = if image_data.contains(',') {
        image_data.split(',').nth(1).unwrap_or(&image_data)
    } else {
        &image_data
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64_data)
        .map_err(|e| format!("Base64 decode failed: {}", e))?;

    let img = image::load_from_memory(&bytes)
        .map_err(|e| format!("Image decode failed: {}", e))?
        .to_rgb8();

    let start = std::time::Instant::now();
    let result: OcrResult = ocr.detect_with_options(
        &img,
        50, 960, 0.5, 0.3, 1.6,
        false, false,
        OcrOptions::default(),
    ).map_err(|e| format!("OCR failed: {}", e))?;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    let blocks: Vec<OcrTextBlock> = result.text_blocks.iter().map(|b| {
        OcrTextBlock {
            text: b.text.clone(),
            score: b.text_score,
            box_points: b.box_points.iter().map(|p| OcrPoint { x: p.x, y: p.y }).collect(),
        }
    }).collect();

    Ok(OcrResponse { blocks, page_angle: result.page_angle, elapsed_ms })
}

pub fn run() {
    // Initialize ONNX Runtime dynamic library (load-dynamic mode, Linux only)
    // Windows uses download-binaries which links statically at compile time
    // Search order: exe dir / ort-lib/ > ORT_DYLIB_PATH env > system dlopen
    #[cfg(target_os = "linux")]
    {
        let lib_loaded = if let Some(exe_dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|p| p.to_path_buf())) {
            // Try exe directory first (for AppImage installs)
            let lib_path = exe_dir.join("libonnxruntime.so");
            if lib_path.exists() {
                eprintln!("[ocr-app] Loading ONNX Runtime from: {}", lib_path.display());
                ort::init_from(lib_path.to_str().unwrap()).commit().is_ok()
            } else {
                // Try ort-lib/ subdirectory (for portable tar.gz distribution)
                let lib_path = exe_dir.join("ort-lib").join("libonnxruntime.so");
                if lib_path.exists() {
                    eprintln!("[ocr-app] Loading ONNX Runtime from: {}", lib_path.display());
                    ort::init_from(lib_path.to_str().unwrap()).commit().is_ok()
                } else {
                    false
                }
            }
        } else {
            false
        };

        if !lib_loaded {
            eprintln!("[ocr-app] No bundled libonnxruntime.so found, using ORT_DYLIB_PATH or system default");
        }
    }

    tauri::Builder::default()
        .manage(Mutex::new(OcrState {
            ocr: None,
            model_version: PpOcrVersion::V6Small,
            models_ready: false,
        }))
        .invoke_handler(tauri::generate_handler![
            ensure_models,
            get_model_status,
            ocr_recognize,
            ocr_recognize_base64,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
