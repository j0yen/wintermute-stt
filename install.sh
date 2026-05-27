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

echo "✓ wm-stt installed."
echo
echo "Next:"
echo "  # drop a whisper model under /usr/share/wintermute/models/ (whisper feature only)"
echo "  wm-stt start                       # subscribe to wm.audio.speech.* and publish wm.stt.*"
echo "  wm-stt reload-model small.en       # hot-swap the active model"
