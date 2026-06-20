#!/bin/bash
set -e
BINARY=/work/ocr1/ocr-app/src-tauri/target/release/ocr-app
DIST=/tmp/pp-ocrv6-arm64

rm -rf $DIST && mkdir -p $DIST/models/pp_ocrv6_tiny $DIST/models/pp_ocrv6_small $DIST/ort-lib

# Copy binary
cp $BINARY $DIST/pp-ocrv6

# Copy models (tiny)
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_tiny/det.onnx $DIST/models/pp_ocrv6_tiny/
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_tiny/rec.onnx $DIST/models/pp_ocrv6_tiny/
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_tiny/dict.txt $DIST/models/pp_ocrv6_tiny/
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_tiny/rec_inference.yml $DIST/models/pp_ocrv6_tiny/

# Copy models (small)
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_small/det.onnx $DIST/models/pp_ocrv6_small/
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_small/rec.onnx $DIST/models/pp_ocrv6_small/
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_small/dict.txt $DIST/models/pp_ocrv6_small/
cp /work/ocr1/ocr-app/src-tauri/models/pp_ocrv6_small/rec_inference.yml $DIST/models/pp_ocrv6_small/

# Download libonnxruntime.so for ARM64 Linux
if [ ! -f /tmp/onnxruntime-linux-aarch64-1.20.1/lib/libonnxruntime.so ]; then
    cd /tmp && curl -sL -o onnxruntime-aarch64.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz
    tar xzf onnxruntime-aarch64.tgz
fi
cp /tmp/onnxruntime-linux-aarch64-1.20.1/lib/libonnxruntime.so.1.20.1 $DIST/ort-lib/libonnxruntime.so

# Create run script
cat > $DIST/run.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/ort-lib:$LD_LIBRARY_PATH"
exec "$SCRIPT_DIR/pp-ocrv6" "$@"
EOF
chmod +x $DIST/run.sh $DIST/pp-ocrv6

# Create tar.gz
cd /tmp && tar czf /work/ocr1/pp-ocrv6-0.1.0-arm64.tar.gz pp-ocrv6-arm64

# Show sizes
echo "=== Distribution contents ==="
du -sh $DIST/*
echo "=== Total ==="
du -sh $DIST
echo "=== tar.gz size ==="
ls -lh /work/ocr1/pp-ocrv6-0.1.0-arm64.tar.gz
