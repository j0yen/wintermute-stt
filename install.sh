#!/usr/bin/env bash
# install.sh — install wintermute-stt (the `wm-stt` binary).
#
# Modes:
#   1. Repo-local: invoked as `./install.sh` from a checkout.
#   2. Curl-piped: invoked as `curl ... | bash`. Self-clones into
#      ~/.local/share/wintermute-stt/ then continues.
#
# Default build uses the StubEngine (no whisper.cpp). Pass
# WM_STT_FEATURES=whisper to enable the real engine — requires cmake
# + a C/C++ toolchain to compile whisper.cpp via whisper-rs-sys.
#
# Model download flags (only meaningful with WM_STT_FEATURES=whisper):
#   --download-model <name>   download a specific model (e.g. distil-small.en)
#   --download-models all     download all allowed models
#
# Models are cached at /usr/share/wintermute/models/ (requires sudo or
# pre-created directory with user write permission).
# Source: https://huggingface.co/ggerganov/whisper.cpp
# License: Apache 2.0 — see README for attribution.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=""
if [ -f "$SCRIPT_PATH" ]; then
  SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
fi

if [ -z "$SCRIPT_DIR" ] || [ ! -f "$SCRIPT_DIR/Cargo.toml" ] \
   || ! grep -q '^name = "wintermute-stt"' "$SCRIPT_DIR/Cargo.toml" 2>/dev/null; then
  echo "→ self-cloning j0yen/wintermute-stt..."
  command -v git >/dev/null 2>&1 || { echo "fatal: git not found"; exit 1; }

  CLONE_ROOT="${WINTERMUTE_STT_CLONE_ROOT:-$HOME/.local/share/wintermute-stt}"
  mkdir -p "$(dirname "$CLONE_ROOT")"

  if [ -d "$CLONE_ROOT/.git" ]; then
    echo "→ existing clone at $CLONE_ROOT — refreshing"
    git -C "$CLONE_ROOT" fetch --depth 1 origin main
    git -C "$CLONE_ROOT" reset --hard origin/main
  else
    git clone --depth 1 https://github.com/j0yen/wintermute-stt.git "$CLONE_ROOT"
  fi

  SCRIPT_DIR="$CLONE_ROOT"
fi

cd "$SCRIPT_DIR"

command -v cargo >/dev/null 2>&1 || {
  echo "fatal: cargo not found. Install Rust: https://rustup.rs/"
  exit 1
}

FEATURES="${WM_STT_FEATURES:-}"
INSTALL_ARGS=(install --path . --locked)
if [ -n "$FEATURES" ]; then
  INSTALL_ARGS+=(--features "$FEATURES")
  echo "→ building wm-stt with features: $FEATURES"
else
  echo "→ building wm-stt (default features; StubEngine only)"
  echo "  set WM_STT_FEATURES=whisper for real inference (requires cmake + clang)"
fi

cargo "${INSTALL_ARGS[@]}"

if ! command -v wm-stt >/dev/null 2>&1; then
  echo
  echo "! wm-stt installed but not on PATH. Add ~/.cargo/bin to PATH:"
  echo "    export PATH=\"\$HOME/.cargo/bin:\$PATH\""
fi

# ---------------------------------------------------------------------------
# Model download (whisper feature only)
# ---------------------------------------------------------------------------
DOWNLOAD_MODEL=""
for arg in "$@"; do
  case "$arg" in
    --download-model) DOWNLOAD_MODEL="${2:-}" ;;
    --download-model=*) DOWNLOAD_MODEL="${arg#--download-model=}" ;;
    --download-models=all|--download-models\ all) DOWNLOAD_MODEL="all" ;;
  esac
done

# Parse --download-model <name> pattern (positional shift)
for i in "$@"; do
  if [ "$i" = "--download-model" ] || [ "$i" = "--download-models" ]; then
    # handled above via positional $2 which is unreliable in this loop; skip
    :
  fi
done

# Re-parse with proper positional handling
DOWNLOAD_MODEL_NAME=""
i=0
for arg in "$@"; do
  i=$((i + 1))
  if [ "$arg" = "--download-model" ]; then
    # next arg
    DOWNLOAD_MODEL_NAME="${!i:-}"
  fi
  if [[ "$arg" == "--download-model="* ]]; then
    DOWNLOAD_MODEL_NAME="${arg#--download-model=}"
  fi
  if [ "$arg" = "--download-models" ]; then
    NEXT="${!i:-}"
    if [ "$NEXT" = "all" ]; then DOWNLOAD_MODEL_NAME="all"; fi
  fi
done

if [ -n "$DOWNLOAD_MODEL_NAME" ]; then
  MODELS_ROOT="${WM_STT_MODELS_ROOT:-/usr/share/wintermute/models}"
  MODELS_ROOT="${MODELS_ROOT%/}/whisper"
  mkdir -p "$MODELS_ROOT" 2>/dev/null || sudo mkdir -p "$MODELS_ROOT"

  HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml"

  download_model() {
    local name="$1"
    local sanitised="${name//./-}"
    local dest="$MODELS_ROOT/whisper-${sanitised}.bin"
    if [ -f "$dest" ]; then
      echo "✓ model already present: $dest"
      return
    fi
    # Map PRD name to ggml filename convention
    local ggml_name
    case "$name" in
      distil-small.en) ggml_name="distil-small.en" ;;
      small.en)        ggml_name="small.en-q5_1"  ;;
      medium.en)       ggml_name="medium.en-q5_0" ;;
      large-v3-turbo)  ggml_name="large-v3-turbo-q5_0" ;;
      *) echo "unknown model: $name"; return 1 ;;
    esac
    local url="${HF_BASE}-${ggml_name}.bin"
    echo "→ downloading $name from $url"
    command -v curl >/dev/null 2>&1 || { echo "fatal: curl not found"; exit 1; }
    curl -fL --progress-bar -o "$dest.tmp" "$url" && mv "$dest.tmp" "$dest"
    echo "✓ model saved to $dest"
  }

  if [ "$DOWNLOAD_MODEL_NAME" = "all" ]; then
    for m in distil-small.en small.en medium.en large-v3-turbo; do
      download_model "$m"
    done
  else
    download_model "$DOWNLOAD_MODEL_NAME"
  fi
fi

echo "✓ wm-stt installed."
echo
echo "Next:"
echo "  # with WM_STT_FEATURES=whisper, download the default model:"
echo "  ./install.sh --download-model distil-small.en"
echo "  wm-stt start                       # subscribe to wm.audio.speech.* and publish wm.stt.*"
echo "  wm-stt reload-model small.en       # hot-swap the active model"
