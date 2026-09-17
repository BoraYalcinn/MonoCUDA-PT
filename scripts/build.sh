#!/usr/bin/env bash
# Convenience wrapper around the CMake presets.
#
# Usage:
#   ./scripts/build.sh              # release build
#   ./scripts/build.sh debug        # debug build
#   ./scripts/build.sh release run  # build then run with default settings

set -euo pipefail

PRESET="${1:-release}"

cmake --preset "$PRESET"
cmake --build --preset "$PRESET" -j"$(nproc)"

if [[ "${2:-}" == "run" ]]; then
    ./build/"$PRESET"/monocuda --out render.ppm
fi