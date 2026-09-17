// ==============================================================================
// === INCLUDE ===
// ==============================================================================
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include <vector>
// ==============================================================================


// ==============================================================================
// === ERROR DEFINITIONS ===
// ==============================================================================
// CUDA has layers for error handling one being for API calls and other for Kernel
// I will wrap API calls like memory allocations with CUDA_CHECK and I will use
// CUDA_CHECK_KERNEL after calling a kernel. 
// ==============================================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                          __LINE__, cudaGetErrorString(err__));              \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                  \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)
// ============================================================================



// ============================================================================
// === PRINT OUT DEVICE INFO
// ============================================================================
void fetchDeviceInfo(){
    int nDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    for( int i = 0;i < nDevices; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        std::cout << "==================== DEVICE INFO ===================\n";
        printf("Device number: %d\n",i);
        printf("Device name: %s\n",prop.name);
        int clockRateKHz;
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrMemoryClockRate, i);
        printf("Memory Clock Rate (KHz): %d\n", clockRateKHz);
        printf("Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("Peak Memory Bandwith (GB/s): %f\n",2.*clockRateKHz*(prop.memoryBusWidth/8)/1.0e6);
        printf("Total Global Memory: %lu\n",prop.totalGlobalMem);
        printf("Compute Capability: %d.%d\n",prop.major,prop.minor);
        printf("Number of SMs: %d\n",prop.multiProcessorCount);
        printf("Max Threads per Block: %d\n",prop.maxThreadsPerBlock);
        printf("Max Grid Dimension: x = %d, y = %d, z = %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
        std::cout << "====================================================\n";
    }
}
// ============================================================================




// ============================================================================
// === MATH CLASSES ===
// ============================================================================
class vec3{
    public:
        float x,y,z;

        __host__ __device__ vec3() : x(0),y(0),z(0){}
        __host__ __device__ vec3(float x_, float y_,float z_ ) : x(x_),y(y_),z(z_){}

        __host__ __device__ vec3 operator+(const vec3& other) const { return vec3{x+other.x, y+other.y , z+other.z};}
        __host__ __device__ vec3 operator-(const vec3& other) const { return vec3{x-other.x, y-other.y , z-other.z};}
        __host__ __device__ vec3 operator*(float n) const { return vec3{x*n,y*n,z*n};}

        __host__ __device__ float length_squared() const {
            return x*x + y*y + z*z;
        }

        __device__ vec3 cross(const vec3& other) const {
            return { y*other.z - z*other.y ,
                     z*other.z - x*other.z ,
                     x*other.y - y*other.x}; 
            }
        __device__ float dot(const vec3& other) const {
            return x*other.x + y*other.y + z*other.z;
        }

        __device__ vec3 normalize() const {
            float length = __sqrtf(this->length_squared());
            return vec3{x/length,y/length,z/length};
        }
};

// ============================================================================


// ============================================================================
// === Mesh & Primitive Types ===
// ============================================================================
__device__ float edge_function(const vec3& a, const vec3& b, const vec3& c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

struct triangle {
    vec3 v0, v1, v2;

    __host__ __device__ triangle() {}
    __host__ __device__ triangle(vec3 v0_, vec3 v1_, vec3 v2_) : v0(v0_), v1(v1_), v2(v2_) {}
};

struct sphere {

};
// ============================================================================


// ============================================================================
// === Render ===
// ============================================================================

__global__ void render_kernel(vec3* h_fb, int maximumX, int maximumY ){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maximumX) || (j >= maximumY)) return;
    int pixel_index = maximumX * j + i;
    h_fb[pixel_index] = vec3{float(i)/maximumX ,float(j)/ maximumY,0};
}
__global__ void render_mesh_kernel(vec3* h_fb,triangle tri,int maximumX,int maximumY){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    vec3 point = {float(i), float(j), 0.0f};
    if((i >= maximumX) || (j >= maximumY)) return;

    float w0 = edge_function(tri.v1, tri.v2, point);
    float w1 = edge_function(tri.v2, tri.v0, point);
    float w2 = edge_function(tri.v0, tri.v1, point);

    bool inside = (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0);
    if (inside) {
        int pixel_index = maximumX * j + i;
        h_fb[pixel_index] = vec3(1.0f, 1.0f, 1.0f); 
    }
}
// ============================================================================

// ============================================================================
// === MAIN ===
// ============================================================================
int main() {

    // display device info
    fetchDeviceInfo();
    
    // display size
    int numberOfPixels_X = 800;
    int numberOfPixels_Y = 400;
    int totalNumberOfPixels = numberOfPixels_X * numberOfPixels_Y;

    // determine the number of block and grid numbers to be used
    dim3 block(16, 16);
    dim3 grid((numberOfPixels_X + block.x - 1) / block.x, (numberOfPixels_Y + block.y - 1) / block.y);

    // initialize the host and device frame buffer
    vec3* host_frameBuffer = new vec3[totalNumberOfPixels];    
    vec3* device_frameBuffer; 

    size_t frameBufferSize = totalNumberOfPixels * sizeof(vec3); 
    CUDA_CHECK(cudaMalloc((vec3**) &device_frameBuffer,frameBufferSize));

    // call the render_kernel
    render_kernel<<<grid,block>>>(device_frameBuffer,numberOfPixels_X,numberOfPixels_Y);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(host_frameBuffer,device_frameBuffer,frameBufferSize,cudaMemcpyDeviceToHost));

    // initialize triangle
    triangle firstTriangle = triangle(vec3{400,50,0},vec3{200,350,0},vec3{600,350,0});

    render_mesh_kernel<<<grid,block>>>(device_frameBuffer,firstTriangle,numberOfPixels_X,numberOfPixels_Y);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(host_frameBuffer,device_frameBuffer,frameBufferSize,cudaMemcpyDeviceToHost));


    // draw the ppm
    FILE* out = fopen("render.ppm", "w");
    if (!out) {
        std::fprintf(stderr, "failed to open render.ppm for writing\n");
        return EXIT_FAILURE;
    }
    fprintf(out, "P3\n%d %d\n255\n", numberOfPixels_X, numberOfPixels_Y);
    for (int j = 0; j < numberOfPixels_Y; j++) {
        for (int i = 0; i < numberOfPixels_X; i++) {
            size_t pixel_index = numberOfPixels_X * j + i;
            int r = int(255.99 * host_frameBuffer[pixel_index].x);
            int g = int(255.99 * host_frameBuffer[pixel_index].y);
            int b = int(255.99 * host_frameBuffer[pixel_index].z);
            fprintf(out, "%d %d %d\n", r, g, b);
        }
    }
    fclose(out);

    // free the memory
    cudaFree(device_frameBuffer);
    delete[] host_frameBuffer;

    return 0;
}
// ============================================================================