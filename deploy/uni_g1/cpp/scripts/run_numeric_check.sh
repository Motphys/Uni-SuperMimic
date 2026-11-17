#!/usr/bin/env bash
# Offline numeric gate: golden vectors (training MDP) vs this C++ deployment.
# Must print NUMERIC PASS before any real-robot run.
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT:-$PWD/.deps/onnxruntime}/lib:${LD_LIBRARY_PATH:-}"
./build/check_numeric ../assets
