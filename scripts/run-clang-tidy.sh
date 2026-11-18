#!/usr/bin/env bash
# Runs clang-tidy over the sources with the flags a CUDA file needs.
#
# clang does not support CUDA 13.x, so three things have to be patched up from
# the outside before it can even parse path_tracer.cu:
#
#   1. clang picks the highest installed GCC, which may be one whose libstdc++
#      headers are not installed (it then fails to find <cmath>). We pick the
#      highest GCC that actually has its C++ headers.
#   2. clang's __clang_cuda_runtime_wrapper.h includes crt/math_functions.hpp
#      directly, and that file does not define _NV_RSQRT_SPECIFIER itself (its
#      .h sibling does), so the macro reaches the parser unexpanded. On glibc
#      2.42 and newer NVIDIA expands it to noexcept(true), so we do the same.
#   3. CUDA 13 removed texture_fetch_functions.h (the old texture reference
#      API), but clang's wrapper still includes it unconditionally. An empty
#      shim satisfies it; the texture object API this renderer actually uses
#      lives in texture_indirect_functions.h, which is still shipped.
#
# Usage:
#   ./scripts/run-clang-tidy.sh                      # all sources under src/
#   ./scripts/run-clang-tidy.sh src/image_io.cpp     # one file
#   ./scripts/run-clang-tidy.sh --fix src/foo.cu     # flags go to clang-tidy
#
# Environment overrides: BUILD_DIR, CUDA_PATH, SM_ARCH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
SM_ARCH="${SM_ARCH:-sm_120}"

# The dependency headers are only there once a build has been configured.
BUILD_DIR="${BUILD_DIR:-}"
if [[ -z "$BUILD_DIR" ]]; then
    for candidate in build/release build/debug; do
        [[ -d "$candidate/_deps" ]] && BUILD_DIR="$candidate" && break
    done
fi
if [[ -z "$BUILD_DIR" || ! -d "$BUILD_DIR/_deps" ]]; then
    echo "error: no configured build found (looked for build/release, build/debug)." >&2
    echo "       run 'cmake --preset release' first, or set BUILD_DIR." >&2
    exit 1
fi
DEPS="$BUILD_DIR/_deps"

# (1) Highest GCC whose libstdc++ headers are present.
GCC_DIR=""
for version_dir in $(ls -d /usr/lib/gcc/x86_64-linux-gnu/*/ 2>/dev/null | sort -Vr); do
    version="$(basename "${version_dir%/}")"
    if [[ -d "/usr/include/c++/$version" ]]; then
        GCC_DIR="${version_dir%/}"
        break
    fi
done
if [[ -z "$GCC_DIR" ]]; then
    echo "error: found no GCC installation with matching libstdc++ headers." >&2
    exit 1
fi

# (3) Empty stand-in for the header CUDA 13 dropped. Lives in the build
# directory so it stays out of the repository.
SHIM_DIR="$BUILD_DIR/clang-tidy-shim"
mkdir -p "$SHIM_DIR"
: > "$SHIM_DIR/texture_fetch_functions.h"

# The dependencies go in as -isystem, the way CMakeLists.txt marks them, so
# clang-tidy does not report stb/tinyexr/minipbrt code as ours. (A plain -I
# would: HeaderFilterRegex 'src/.*' matches paths like _deps/stb-src/stb_image.h
# too, which is 730 warnings out of someone else's code.)
INCLUDES=(
    -I"$SHIM_DIR"
    -Isrc
    -isystem "$DEPS/minipbrt-src"
    -isystem "$DEPS/stb-src"
    -isystem "$DEPS/tinyexr-src"
    -isystem "$DEPS/tinyexr-src/deps/miniz"
)

# clang-tidy flags from the command line, source files collected separately so
# each one can be given the flags its language needs.
TIDY_FLAGS=()
SOURCES=()
for argument in "$@"; do
    if [[ -f "$argument" ]]; then
        SOURCES+=("$argument")
    else
        TIDY_FLAGS+=("$argument")
    fi
done
if [[ ${#SOURCES[@]} -eq 0 ]]; then
    SOURCES=(src/path_tracer.cu src/image_io.cpp)
fi

status=0
for source in "${SOURCES[@]}"; do
    compiler_flags=(-std=c++17 "${INCLUDES[@]}" --gcc-install-dir="$GCC_DIR")
    if [[ "$source" == *.cu ]]; then
        compiler_flags+=(
            -x cuda
            --cuda-gpu-arch="$SM_ARCH"
            --cuda-path="$CUDA_PATH"
            -D_NV_RSQRT_SPECIFIER="noexcept(true)"   # (2)
        )
    fi

    echo "=== $source ==="
    clang-tidy "${TIDY_FLAGS[@]+"${TIDY_FLAGS[@]}"}" "$source" -- "${compiler_flags[@]}" || status=$?
done

exit "$status"
