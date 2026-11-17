#!/usr/bin/env bash
# Download ONNX Runtime C++ (aarch64, PC2) into .deps/onnxruntime.
set -euo pipefail
cd "$(dirname "$0")/.."
VER=1.16.3
ARCH=$(uname -m)
case "$ARCH" in
  aarch64) PKG=onnxruntime-linux-aarch64-$VER ;;
  x86_64)  PKG=onnxruntime-linux-x64-$VER ;;
  *) echo "unsupported arch $ARCH" >&2; exit 1 ;;
esac
mkdir -p .deps
cd .deps
if [ ! -d onnxruntime ]; then
  echo "downloading $PKG.tgz"
  curl -fL -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v$VER/$PKG.tgz"
  tar xzf ort.tgz
  mv "$PKG" onnxruntime
  rm ort.tgz
fi
ls onnxruntime/lib onnxruntime/include | head -6
