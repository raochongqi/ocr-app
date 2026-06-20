# ARM64 Linux 构建环境 (Ubuntu 20.04 + libwebkit2gtk-4.0)
# 兼容 Tauri v1，全部使用国内镜像源
#
# 用法:
#   docker buildx build --platform linux/arm64 -t ocr-app-builder .
#   docker run --rm --platform linux/arm64 \
#     -v /c/programing/ocr1:/work -w /work/ocr-app \
#     ocr-app-builder bash build-arm64.sh

FROM arm64v8/ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# ---- 1. apt 镜像源 (阿里云) ----
RUN sed -i 's|http://ports.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    curl \
    wget \
    file \
    ca-certificates \
    libssl-dev \
    libwebkit2gtk-4.0-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libsoup2.4-dev \
    libjavascriptcoregtk-4.0-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. Rust (rsproxy 镜像) ----
ENV RUSTUP_DIST_SERVER=https://rsproxy.cn
ENV RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# ---- 3. crates.io 镜像 (rsproxy) ----
RUN mkdir -p /root/.cargo && \
    echo '[source.crates-io]\n\
replace-with = "rsproxy"\n\
\n\
[source.rsproxy]\n\
registry = "https://rsproxy.cn/crates.io-index"\n\
\n\
[registries.rsproxy]\n\
index = "https://rsproxy.cn/crates.io-index"\n\
\n\
[net]\n\
git-fetch-with-cli = true' > /root/.cargo/config.toml

# ---- 4. Node.js 22 (npmmirror 二进制包) ----
ENV NODE_VER=22.16.0
RUN curl -fsSL "https://registry.npmmirror.com/-/binary/node/v${NODE_VER}/node-v${NODE_VER}-linux-arm64.tar.xz" \
    -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm -f /tmp/node.tar.xz
RUN npm config set registry https://registry.npmmirror.com

WORKDIR /work/ocr-app
