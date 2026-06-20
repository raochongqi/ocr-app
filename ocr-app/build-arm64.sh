#!/usr/bin/env bash
# ARM64 Linux 构建脚本 (Tauri v1 + libwebkit2gtk-4.0)
# 兼容 Ubuntu 20.04 / Debian Buster 同级别发行版
# 全部使用国内镜像源
#
# 用法:
#   方式1 - 在 ARM Linux VM/设备上直接运行:
#     bash build-arm64.sh
#
#   方式2 - 从 Windows 通过 Docker QEMU 构建:
#     docker run --rm --platform linux/arm64 \
#       -v /c/programing/ocr1:/work -w /work/ocr-app \
#       arm64v8/ubuntu:20.04 bash build-arm64.sh

set -euo pipefail

# 防止交互式安装提示 (tzdata 等)
export DEBIAN_FRONTEND=noninteractive

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== PP-OCRv6 ARM64 Linux Build (Tauri v1) ==="
echo "Script dir: $SCRIPT_DIR"

# ---- 1. 安装系统依赖 (仅 Ubuntu/Debian) ----
if command -v apt-get &>/dev/null; then
    echo "[1/6] Configuring apt mirror and installing system dependencies..."
    APT_CMD="apt-get"
    if command -v sudo &>/dev/null && [ "$(id -u)" != "0" ]; then
        APT_CMD="sudo apt-get"
    fi

    # 配置阿里云镜像源 (ARM ports)
    if [ -f /etc/apt/sources.list ] && ! grep -q "mirrors.aliyun.com" /etc/apt/sources.list 2>/dev/null; then
        echo "  Switching apt to Aliyun mirror..."
        sed -i 's|http://ports.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
        sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
    fi

    $APT_CMD update
    $APT_CMD install -y --no-install-recommends \
        build-essential pkg-config curl wget file \
        ca-certificates \
        libssl-dev \
        libwebkit2gtk-4.0-dev \
        libgtk-3-dev \
        libayatana-appindicator3-dev \
        librsvg2-dev \
        libsoup2.4-dev \
        libjavascriptcoregtk-4.0-dev \
        g++-11
else
    echo "[1/6] Non-apt system, skipping package install."
    echo "  Make sure these are available: libwebkit2gtk-4.0, libgtk-3, libsoup2.4, librsvg2"
fi

# ---- 2. 安装 Rust ----
if ! command -v cargo &>/dev/null; then
    echo "[2/6] Installing Rust..."
    # 优先尝试 rsproxy，失败则回退官方源
    export RUSTUP_DIST_SERVER=https://rsproxy.cn
    export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
    if ! curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y 2>/dev/null; then
        echo "  rsproxy failed, falling back to official source..."
        unset RUSTUP_DIST_SERVER RUSTUP_UPDATE_ROOT
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi
    source "$HOME/.cargo/env"
else
    echo "[2/6] Rust already installed: $(cargo --version)"
fi

# 配置 crates.io 镜像 (优先 rsproxy，回退 tuna)
CARGO_CONFIG="$HOME/.cargo/config.toml"
if [ ! -f "$CARGO_CONFIG" ] || ! grep -q "rsproxy\|ustc\|tuna" "$CARGO_CONFIG" 2>/dev/null; then
    echo "  Configuring crates.io mirror..."
    mkdir -p "$(dirname "$CARGO_CONFIG")"
    cat > "$CARGO_CONFIG" << 'CARGO_EOF'
[source.crates-io]
replace-with = "rsproxy-sparse"

[source.rsproxy]
registry = "https://rsproxy.cn/crates.io-index"

[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"

[net]
git-fetch-with-cli = true
CARGO_EOF
fi

# ---- 3. 安装 Node.js (npmmirror 下载二进制包) ----
if ! command -v node &>/dev/null; then
    echo "[3/6] Installing Node.js 22 (binary from npmmirror)..."
    NODE_VER="22.16.0"
    NODE_ARCH="arm64"
    NODE_TAR="node-v${NODE_VER}-linux-${NODE_ARCH}.tar.xz"
    NODE_URL="https://registry.npmmirror.com/-/binary/node/v${NODE_VER}/${NODE_TAR}"
    echo "  Downloading ${NODE_URL}..."
    curl -fsSL "$NODE_URL" -o "/tmp/${NODE_TAR}"
    tar -xJf "/tmp/${NODE_TAR}" -C /usr/local --strip-components=1
    rm -f "/tmp/${NODE_TAR}"
    echo "  Node.js $(node --version) installed."
else
    echo "[3/6] Node.js already installed: $(node --version)"
fi

# 配置 npm 镜像
npm config set registry https://registry.npmmirror.com

# ---- 4. 安装前端依赖 ----
echo "[4/6] Installing frontend dependencies..."
cd "$SCRIPT_DIR"
npm install --frozen-lockfile 2>/dev/null || npm install

# ---- 5. 构建前端 ----
echo "[5/6] Building frontend..."
cd "$SCRIPT_DIR"
npm run build

# ---- 6. 构建 Tauri 应用 ----
echo "[6/6] Building Tauri v1 app..."
cd "$SCRIPT_DIR"

# 使用 load-dynamic 模式，不需要链接 ort-sys 静态库
# 但需要下载 libonnxruntime.so 供运行时使用
ORT_VERSION="1.20.1"
ORT_DIR="onnxruntime-linux-aarch64-${ORT_VERSION}"
ORT_TAR="${ORT_DIR}.tgz"

# 优先从镜像下载 ONNX Runtime 共享库
if [ ! -f "/tmp/${ORT_TAR}" ]; then
    echo "  Downloading ONNX Runtime ${ORT_VERSION} for aarch64..."
    ORT_BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"
    ORT_MIRROR_URL="https://raw.gitmirror.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"
    if ! curl -fsSL "${ORT_MIRROR_URL}/${ORT_TAR}" -o "/tmp/${ORT_TAR}" 2>/dev/null; then
        echo "  Mirror failed, trying GitHub directly..."
        if ! curl -fsSL "${ORT_BASE_URL}/${ORT_TAR}" -o "/tmp/${ORT_TAR}" 2>/dev/null; then
            echo "  WARNING: Failed to download ONNX Runtime. Build will continue without bundled lib."
            echo "  You can manually download it and place libonnxruntime.so in src-tauri/ort-lib/"
        fi
    fi
fi

# 解压到 src-tauri 目录（将打包进应用）
if [ -f "/tmp/${ORT_TAR}" ]; then
    echo "  Extracting ONNX Runtime..."
    mkdir -p "$SCRIPT_DIR/src-tauri/ort-lib"
    tar -xzf "/tmp/${ORT_TAR}" -C /tmp
    cp "/tmp/${ORT_DIR}/lib/libonnxruntime.so.${ORT_VERSION%%.*}" "$SCRIPT_DIR/src-tauri/ort-lib/libonnxruntime.so" 2>/dev/null || \
    cp "/tmp/${ORT_DIR}/lib/libonnxruntime.so" "$SCRIPT_DIR/src-tauri/ort-lib/libonnxruntime.so"
    echo "  libonnxruntime.so ready at src-tauri/ort-lib/"
fi

npx tauri build --bundles deb,appimage

echo ""
echo "=== Build Complete ==="
echo "Output: $SCRIPT_DIR/src-tauri/target/release/bundle/"
