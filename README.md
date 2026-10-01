# MonoCUDA-PT
MonoCUDA is an original CUDA path tracer written as a single source file. It was started after completing Peter Shirley's Ray Tracing in One Weekend series in C++, with the goal of understanding how physically based rendering, specifically Monte Carlo path integration, maps onto GPU hardware rather than a single CPU thread. It is not based on or derived from NVIDIA's official CUDA port of the same book.

The renderer is deliberately kept in one file. The intent is for the source to read as a continuous explanation of a GPU path tracer's design, from ray generation through BVH traversal, material evaluation, and light transport, rather than as a general-purpose rendering engine split across many files. Kernels are written iteratively rather than recursively, since recursion on GPU forces register spilling and reduces occupancy; 

The project follows a standard Monte Carlo path tracing formulation: primary rays are generated per pixel, intersected against scene geometry accelerated by a bounding volume hierarchy, and shaded using physically based materials (diffuse, dielectric, conductor, and emissive) with multiple importance sampling between light and BSDF sampling. Correctness is checked against a CPU reference implementation on shared test scenes, and performance is tracked with benchmark numbers will be recorded in bench/results.md.

Building requires the CUDA Toolkit (12.x or later), CMake 3.24 or later, and a CUDA-capable NVIDIA GPU. From the repository root:

cmake --preset release
cmake --build --preset release
./build/release/monocuda --out render.ppm

The project's implementation status is tracked in CHECKLIST.md. It is licensed under the MIT License.

# Implementation Checklist

## 1. Core Rendering Pipeline
- [ x ] Device query + context setup (print GPU name, compute capability, SM count)
- [ x ] `vec3`/`float3` math library (add/sub/scale/dot/cross/normalize/reflect/refract)
- [ x ] `ray` type + camera-to-ray generation
- [ x ] Pinhole camera (position, look-at, FOV, aspect ratio)
- [ x ] Thin-lens camera model — aperture + focus distance (depth of field)
- [ x ] HDR framebuffer (float3/float4 per pixel), accumulated across samples
- [ x ] Progressive rendering loop (N samples/pixel, accumulate + save incrementally)

## 2. Scene Representation & Acceleration Structures
- [ x ] Primitive types: sphere, and at least one more
- [ ] Structure-of-arrays scene layout (not array-of-structs) for coalesced access
- [ x ] BVH construction (GPU-side LBVH build via Morton codes + Karras parallel radix-tree algorithm)
- [ x ] BVH traversal (device-side, iterative, stack-based — no recursion)
- [ x ] Bounding-box (AABB) intersection + slab test
- [ ] **(stretch)** SAH (surface area heuristic) BVH build for better traversal quality
- [ ] **(stretch)** Basic OBJ mesh loading so you can render more than primitives

## 3. Materials / BSDFs
- [ x ] Lambertian (diffuse) BSDF — cosine-weighted hemisphere sampling
- [ x ] Specular/mirror reflection
- [ x ] Dielectric (glass) BSDF — Fresnel + refraction (Schlick approximation)
- [ x ] Rough conductor (metal) BSDF — at least a fuzz/roughness parameter
- [ x ] Emissive materials (area light sources built from geometry)
- [ ] **(stretch)** Real microfacet BSDF (GGX distribution) instead of ad-hoc fuzz
- [ ] **(stretch)** Energy-conserving material blending (e.g. dielectric-coated diffuse)

## 4. Textures & Surface Detail
- [ ] UV coordinates on primitives (spherical mapping for sphere, barycentric for triangle)
- [ ] Image loading (e.g. `stb_image`) — PNG/JPG/HDR into host memory
- [ ] Hardware-accelerated sampling via `cudaTextureObject_t` (bilinear filtering on the GPU, not a manual lerp)
- [ ] Albedo/diffuse texture map, wired into existing materials
- [ ] **(stretch)** Normal mapping (tangent-space perturbation of the shading normal)
- [ ] **(stretch)** Roughness/metallic texture maps for the conductor BSDF

## 5. Light Transport & Sampling
- [ x ] Monte Carlo path integration — iterative bounce loop, not recursive
- [ x ] Russian roulette path termination (unbiased early exit)
- [ ] Light list construction — collect emissive primitives from the scene into a separate array
- [ ] Light sampling — pick a point on a light (solid-angle sampling for sphere, area sampling for triangle) and compute its PDF
- [ ] Light selection strategy for multiple lights (uniform, or power-weighted)
- [ ] Next event estimation (NEE) — explicitly sample lights each bounce instead of relying on BSDF sampling alone to find them
- [ ] Multiple importance sampling (MIS) between light sampling and BSDF sampling
- [ ] Cosine-weighted / importance-sampled BSDF sampling (not uniform hemisphere)
- [ ] **(stretch)** Environment map (HDRI) lighting with importance sampling
- [ ] **(stretch)** Stratified or low-discrepancy sampling (e.g. Sobol/blue-noise) instead of pure `curand` uniform — measurably reduces noise at equal sample count

## 6. Volumetric Rendering
- [ ] Homogeneous participating media (constant-density fog/smoke)
- [ ] Ray marching through volume bounds — absorption + scattering along the path
- [ ] Henyey-Greenstein phase function for scattering direction
- [ ] **(stretch)** Heterogeneous volumes (3D voxel grid for density — real smoke/cloud data)

## 7. Motion Blur
- [ ] Per-sample time value on each ray (random within the shutter interval)
- [ ] Time-interpolated primitive transforms (e.g. `center0`/`center1` lerp'ed by `time` for a moving sphere)
- [ ] BVH handling for moving primitives — expanded bounding box covering the full motion range
- [ ] **(stretch)** Proper motion-aware BVH instead of a single expanded box per moving primitive

## 8. GPU Architecture-Specific Design
- [ ] Iterative (not recursive) kernels throughout — verify with register usage report (`nvcc --ptxas-options=-v`)
- [ ] Persistent per-pixel RNG state, not reinitialized per sample
- [ ] Memory access audit: confirm scene/material reads go through `__ldg`/`const __restrict__`
- [ ] Occupancy measurement (Nsight Compute or `nvidia-smi`) at a baseline milestone
- [ ] **(stretch)** Wavefront path tracing — split trace/shade into separate kernels with stream compaction between bounces (this is how real production renderers like PBRT-on-GPU or Blender Cycles handle divergence; a genuine architectural upgrade over "one big kernel does everything")
- [ ] **(stretch)** Material-sorted shading — sort active paths by material ID before shading to reduce warp divergence

## 9. Image Output & Post-Processing
- [ ] Tonemapping operator (start with Reinhard or ACES-approx, not just clamp)
- [ ] Gamma correction (linear → sRGB)
- [ ] PNG output (not just PPM) so renders are shareable without conversion
- [ ] **(stretch)** Simple denoiser pass (even a basic bilateral/edge-aware filter counts) to demonstrate you understand the variance-reduction vs. denoising trade-off

## 10. Tooling & Validation
- [ ] Benchmark harness reporting rays/sec and ms/frame at fixed sample counts
- [ ] Numbers for at least one "before vs. after" optimization (e.g. BVH vs. brute force, or single-kernel vs. wavefront) recorded in the repo
- [ ] Command-line scene selection (a couple of hardcoded test scenes, switchable via flag)

## References

- Peter Shirley, Trevor David Black, Steve Hollasch — *Ray Tracing in One Weekend* / *Ray Tracing: The Next Week* / *Ray Tracing: The Rest of Your Life* (https://raytracing.github.io/)
- NVIDIA Developer Blog — "Accelerated Ray Tracing in One Weekend in CUDA" (https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- Tero Karras, NVIDIA Developer Blog — "Thinking Parallel, Part II: Tree Traversal on the GPU" (https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)
- Tero Karras, NVIDIA Developer Blog — "Thinking Parallel, Part III: Tree Construction on the GPU" (https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)
- ToruNiina/lbvh (https://github.com/ToruNiina/lbvh) — used as a reference while designing and debugging MonoCUDA-PT's own LBVH construction and traversal; not used as source code