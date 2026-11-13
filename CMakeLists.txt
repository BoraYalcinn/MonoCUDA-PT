cmake_minimum_required(VERSION 3.24)

# Defaults to the architecture of the GPU present on the build machine.
# Override with -DCMAKE_CUDA_ARCHITECTURES=89 (Ada) / 120 (Blackwell) etc.
# for cross-compiling or CI (no GPU present -> falls back to a safe list).
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120-real)
endif()

project(MonoCUDA LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



option(MONOCUDA_ENABLE_WARNINGS "Enable strict host+device warnings" ON)
option(MONOCUDA_BENCH "Build the benchmark target" OFF)

# Third-party code, fetched at configure time and pinned to a commit:
#   minipbrt  - pbrt-v3 scene parser (MIT)
#   stb       - stb_image / stb_image_write, PNG/JPG/HDR decoding (public domain)
#   tinyexr   - OpenEXR decoding, uses its bundled miniz (BSD)
# SOURCE_SUBDIR points at a directory without a CMakeLists.txt on purpose, so
# FetchContent only downloads the sources and we compile what we need below.
include(FetchContent)
FetchContent_Declare(minipbrt
    GIT_REPOSITORY https://github.com/vilya/minipbrt.git
    GIT_TAG        a6f9dadb55dae3a8a051c66863fe9b2622f8c7b3
    SOURCE_SUBDIR  cmake_not_used)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        9600522de53973336fa033a289858a4354cbb04c
    SOURCE_SUBDIR  cmake_not_used)
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        e48a52fb3711da47494e518a229f04aa5ae8e0ac
    SOURCE_SUBDIR  cmake_not_used)
FetchContent_MakeAvailable(minipbrt stb tinyexr)

add_library(monocuda_deps STATIC
    src/image_io.cpp
    ${minipbrt_SOURCE_DIR}/minipbrt.cpp
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c)
target_include_directories(monocuda_deps PUBLIC src)
target_include_directories(monocuda_deps SYSTEM PUBLIC
    ${minipbrt_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz)
target_compile_features(monocuda_deps PRIVATE cxx_std_17)

add_executable(monocuda src/path_tracer.cu)
target_link_libraries(monocuda PRIVATE monocuda_deps)

target_compile_features(monocuda PRIVATE cxx_std_17)

set_target_properties(monocuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF   # single translation unit by design
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(MONOCUDA_ENABLE_WARNINGS)
    target_compile_options(monocuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endif()

# Debug builds: device-side debug symbols so cuda-gdb / Nsight can step
# through kernels; no optimization so line numbers stay accurate.
target_compile_options(monocuda PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G;-g>
)
target_compile_options(monocuda PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>
)

# Release builds: aggressive optimization + fast math (path tracer doesn't
# need strict IEEE compliance; revisit if correctness diffs demand it).
target_compile_options(monocuda PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;--use_fast_math>
)

if(MONOCUDA_BENCH)
    add_subdirectory(bench)
endif()