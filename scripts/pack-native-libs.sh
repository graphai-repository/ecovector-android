#!/usr/bin/env bash
# pack-native-libs.sh — Package pre-built native binaries into a zip for GitHub Releases.
#
# Usage:
#   ./scripts/pack-native-libs.sh [output.zip]
#
# Run from the repository root. The zip is structured so that extracting into
# ecovector/src/main/ places every file in the correct location.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_MAIN="$REPO_ROOT/ecovector/src/main"
OUTPUT="${1:-native-libs-arm64-v8a.zip}"

if [[ ! -d "$SRC_MAIN" ]]; then
  echo "ERROR: $SRC_MAIN not found. Run from the repository root." >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Collecting native libraries ..."

# --- third_party static/shared libs ---
THIRD_PARTY="cpp/third_party"

collect() {
  local src="$SRC_MAIN/$1"
  local dest="$TMPDIR/$1"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp -r "$src" "$dest"
    echo "  + $1"
  else
    echo "  SKIP $1 (not found)"
  fi
}

# FAISS
collect "$THIRD_PARTY/faiss/lib/arm64-v8a"

# Kiwi
collect "$THIRD_PARTY/kiwi/lib/arm64-v8a"

# ObjectBox (arm64-v8a + x86_64)
collect "$THIRD_PARTY/objectbox/lib/arm64-v8a"
collect "$THIRD_PARTY/objectbox/lib/x86_64"

# ONNX Runtime — .so only (skip ~1.1GB .a files)
for abi in arm64-v8a x86_64; do
  src_dir="$SRC_MAIN/$THIRD_PARTY/onnxruntime/lib/$abi"
  dest_dir="$TMPDIR/$THIRD_PARTY/onnxruntime/lib/$abi"
  if [[ -d "$src_dir" ]]; then
    mkdir -p "$dest_dir"
    find "$src_dir" -name '*.so' -exec cp {} "$dest_dir/" \;
    echo "  + $THIRD_PARTY/onnxruntime/lib/$abi (*.so only)"
  else
    echo "  SKIP $THIRD_PARTY/onnxruntime/lib/$abi (not found)"
  fi
done

# tokenizers-cpp
collect "$THIRD_PARTY/tokenizers-cpp/lib/arm64-v8a"

# --- JNI shared libs ---
collect "jniLibs/arm64-v8a"

echo ""
echo "Creating archive: $OUTPUT"
(cd "$TMPDIR" && zip -r - .) > "$OUTPUT"
echo "Done. $(du -h "$OUTPUT" | cut -f1) written to $OUTPUT"
