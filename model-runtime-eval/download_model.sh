#!/usr/bin/env bash
# Downloads the submission's GGUF weight file to model/.
# Must be idempotent, require no credentials, and land at the exact
# path referenced by metadata.json -> _runtime.model_path.
set -euo pipefail

MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH="model/qwen2.5-1.5b-instruct-q4_k_m.gguf"   # must match metadata.json _runtime.model_path exactly

mkdir -p model

if [ -f "$MODEL_PATH" ]; then
    echo "Model already present at $MODEL_PATH — skipping download (idempotent)."
    exit 0
fi

echo "Downloading model to $MODEL_PATH ..."
curl -L --fail -o "$MODEL_PATH" "$MODEL_URL"
echo "Done."
