cmake_minimum_required(VERSION 3.24)

project(MonoCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Defaults to the architecture of the GPU present on the build machine.
# Override with -DCMAKE_CUDA_ARCHITECTURES=89 (Ada) / 120 (Blackwell) etc.
# for cross-compiling or CI (no GPU present -> falls back to a safe list).
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120-real)
endif()

option(MONOCUDA_ENABLE_WARNINGS "Enable strict host+device warnings" ON)
option(MONOCUDA_BENCH "Build the benchmark target" OFF)

add_executable(monocuda src/path_tracer.cu)

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

# Release builds: aggressive optimization + fast math (path tracer doesn't
# need strict IEEE compliance; revisit if correctness diffs demand it).
target_compile_options(monocuda PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;--use_fast_math>
)

if(MONOCUDA_BENCH)
    add_subdirectory(bench)
endif()