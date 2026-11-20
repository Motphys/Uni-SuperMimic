#!/usr/bin/env bash
# Build the Uni-SuperMimic C++ deployment (numeric checker + robot controller).
# Does NOT start anything on the robot.
set -euo pipefail
cd "$(dirname "$0")/.."

# ONNXRUNTIME_ROOT: unpacked onnxruntime release with include/ and lib/.
# Default: .deps/onnxruntime (set up by scripts/setup_ort.sh or manually).
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)"
echo "built: build/check_numeric $( [ -x build/g1_uni_ctrl ] && echo 'build/g1_uni_ctrl' )"
