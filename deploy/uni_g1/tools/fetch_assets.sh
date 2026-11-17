#!/usr/bin/env bash
# Download the deployment assets (policy.onnx + demo clips) from Hugging Face.
# Requires only bash + curl. Usage: bash tools/fetch_assets.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="LeeLeno/uni-supermimic-g1-deploy"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main"

CLIPS=(
  cellphone_record_stand_idle_R_001__A423
  drinking_standing_mug_R_001__A282
  bow_R_001__A428
  clap_enthusiastic_001__A029
  pels_air_punch_001__A492
  fancy_spin_R_001__A428
  dance_basic_chaines_180_R_001__A306
  dance_basic_chaines_180_R_001__A307
  Loop_Forward_Walk_001__A019
  Relaxed_walk_forward_001__A057
  Neutral_kick_trash_001__A057
  skipping_R_001__A359
)

mkdir -p assets/clips
echo ">> policy.onnx"
curl -fL --retry 3 -o assets/policy.onnx "${BASE}/policy.onnx"
for n in "${CLIPS[@]}"; do
  echo ">> clips/${n}.npz"
  curl -fL --retry 3 -o "assets/clips/${n}.npz" "${BASE}/clips/${n}.npz"
done
echo "assets ready: $(du -sh assets/policy.onnx assets/clips | tr '\n' ' ')"
