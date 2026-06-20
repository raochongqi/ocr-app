# ARM Linux (aarch64) 构建脚本
# 用法：
#   1. 在 ARM Linux 设备上直接运行：bash build-linux-arm64.sh
#   2. 通过 Docker QEMU 模拟：docker run --rm -v $(pwd):/work -w /work arm64v8/ubuntu:20.04 bash build-linux-arm64.sh

set -e

# 安装依赖（Ubuntu/Debian）
if command -v apt-get &>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        pkg-config \
        libwebkit2gtk-4.1-dev \
        libappindicator3-dev \
        librsvg2-dev \
        libssl-dev \
        curl \
        file
fi

# 安装 Rust（如果未安装）
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# 安装 Node.js（如果未安装）
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# 构建
cd "$(dirname "$0")"
npm install
npx tauri build --target aarch64-unknown-linux-gnu

echo "Build complete!"
echo "Output: src-tauri/target/aarch64-unknown-linux-gnu/release/bundle/"
