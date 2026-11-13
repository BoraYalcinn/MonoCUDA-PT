// ==============================================================================
// This project is fully written by Bora Yalçın (Undergraduate CSE Student at Yeditepe University)
// Checkout out my website for related blogs : https://borayalcinn.github.io/
// Checkout the related repository           : https://github.com/BoraYalcinn/MonoCUDA-PT
// I have shared the sources such as papers, websites and learning courses etc. under the README.md
// ==============================================================================
// === INCLUDE AND MACROS ===
// ==============================================================================
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <map>
#include <set>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include "minipbrt.h"
#include "image_io.h"

#if defined(__unix__) || defined(__APPLE__)
    #include <unistd.h>
    #define MONOCUDA_IS_TERMINAL(fileDescriptor) isatty(fileDescriptor)
#else
    #include <io.h>
    #define MONOCUDA_IS_TERMINAL(fileDescriptor) _isatty(fileDescriptor)
#endif

#include "cuda_runtime.h"
#include <curand_kernel.h>


#define MONOCUDA_PI 3.14159265358979323846f

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
        printf("Max Threads Dimension: x = %d, y = %d, z = %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
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
        __host__ __device__ vec3 operator*(const vec3& other) const { return vec3{x*other.x, y*other.y, z*other.z};}
        __host__ __device__ vec3 operator-() const {return vec3{-x,-y,-z};}
        __host__ __device__ vec3 operator*(float n) const { return vec3{x*n,y*n,z*n};}
        __host__ __device__ vec3 operator/(float n) const { return vec3{x/n,y/n,z/n};}


        __host__ __device__ float length_squared() const {
            return x*x + y*y + z*z;
        }

        __host__ __device__ float length() const {
            return sqrtf(this->length_squared());
        }

        __host__ __device__ vec3 cross(const vec3& other) const {
            return { y*other.z - z*other.y ,
                    z*other.x - x*other.z ,
                    x*other.y - y*other.x};
        }
        __host__ __device__ float dot(const vec3& other) const {
            return x*other.x + y*other.y + z*other.z;
        }

        __host__ __device__ vec3 normalize() const {
            float length = sqrtf(this->length_squared());
            return vec3{x/length,y/length,z/length};
        }

        __host__ __device__ vec3 reflect(const vec3& surfaceNormal) {
            return *this - surfaceNormal * (2.0f * this->dot(surfaceNormal));
        }

        __host__ __device__ vec3 refract(const vec3& surfaceNormal,float refraction_cof){
            vec3 direction = this->normalize();
            float cosTheta1 = fminf((-direction).dot(surfaceNormal), 1.0f);
            float theta1 = acosf(cosTheta1);

            float sinTheta2 = sinf(theta1) * refraction_cof;
            if (sinTheta2 > 1.f) {
                return direction.reflect(surfaceNormal);
            }
            float theta2 = asinf(sinTheta2);
            vec3 tangent = direction + surfaceNormal * cosTheta1;
            float tangentLength = tangent.length();
            if (tangentLength > 1e-8f) {
                tangent = tangent / tangentLength;
            }

            return tangent * sinf(theta2) - surfaceNormal * cosf(theta2);
        }

        __host__ float calculateDistanceTo(const vec3& other){
            return sqrtf((x - other.x)*(x - other.x) + (y - other.y)*(y - other.y) + (z - other.z)*(z - other.z));
        }
};

class vec2{
    public:
        float x,y;

        __host__ __device__ vec2() : x(0),y(0){}
        __host__ __device__ vec2(float x_, float y_ ) : x(x_),y(y_){}

        __host__ __device__ vec2 operator+(const vec2 &other)const{return vec2{x+other.x,y+other.y};}
        __host__ __device__ vec2 operator-(const vec2 &other)const{return vec2{x-other.x,y-other.y};}
        __host__ __device__ vec2 operator*(const vec2 &other)const{return vec2{x*other.x,y*other.y};}
        __host__ __device__ vec2 operator-()const{return vec2{-x,-y};}
        __host__ __device__ vec2 operator/(float n)const{return vec2{x/n,y/n};}
        __host__ __device__ vec2 operator*(float n)const{return vec2{x*n,y*n};}
};

// ============================================================================

// ============================================================================
// === TEXTURES ===
// ============================================================================

enum class TextureType { Solid, Checker, Image };

struct Texture {
    TextureType type;
    vec3 colorA;        // Solid color only 1 color
    vec3 colorB;        // second color for the checker texture
    float scale;        // square density in the checker texture
    cudaTextureObject_t imageHandle;   // for Image textures
};

__host__ __device__ inline float srgbToLinear(float encodedValue){
    return encodedValue <= 0.04045f ? encodedValue / 12.92f : powf((encodedValue + 0.055f) / 1.055f, 2.4f);
}
__host__ __device__ inline vec3 srgbToLinear(const vec3& encodedColor){
    return vec3(srgbToLinear(encodedColor.x), srgbToLinear(encodedColor.y), srgbToLinear(encodedColor.z));
}

__host__ __device__ Texture make_solid_texture(vec3 color) {
    return Texture{TextureType::Solid, srgbToLinear(color), vec3(0,0,0), 1.f, 0};
}

__host__ __device__ Texture make_checker_texture(vec3 colorA, vec3 colorB, float checkerScale) {
    return Texture{TextureType::Checker, srgbToLinear(colorA), srgbToLinear(colorB), checkerScale, 0};
}

__device__ vec3 sampleTexture(const Texture& texture, float u, float v) {
    switch (texture.type) {
        case TextureType::Solid:
            return texture.colorA;

        case TextureType::Checker: {
            int cellIndexU = (int)floorf(u * texture.scale);
            int cellIndexV = (int)floorf(v * texture.scale);
            bool isEvenCell = ((cellIndexU + cellIndexV) & 1) == 0;
            return isEvenCell ? texture.colorA : texture.colorB;
        }
        case TextureType::Image: {
            float4 texel = tex2D<float4>(texture.imageHandle, u, 1.0f - v);
            return vec3(texel.x, texel.y, texel.z);
        }
        default:
            return vec3(1.f, 0.f, 1.f);
    }
}

static bool readNextPpmInteger(FILE* file, int& value) {
    int character = fgetc(file);
    while (character != EOF) {
        if (character == '#') {                       // comment line
            while (character != '\n' && character != EOF){
                 character = fgetc(file);
            }
        } else if (isspace(character)) {
            character = fgetc(file);
        } else break;
    }
    if (character == EOF || !isdigit(character)) {
        return false;
    }
    value = 0;
    while (character != EOF && isdigit(character)) {
        value = value * 10 + (character - '0');
        character = fgetc(file);                      // the separator after the number is consumed here
    }
    return true;
}

// reads a binary PPM (P6) and returns RGBA8 ( !! CUDA textures cannot be 3-channel)
static bool loadPpmAsRgba8(const char* filePath, int& imageWidth, int& imageHeight, std::vector<unsigned char>& rgbaPixels) {
    FILE* file = fopen(filePath, "rb");
    if (!file) return false;

    if (fgetc(file) != 'P' || fgetc(file) != '6'){
        fclose(file);
        return false;
    }
    int maximumValue = 0;
    if (!readNextPpmInteger(file, imageWidth) || !readNextPpmInteger(file, imageHeight) ||!readNextPpmInteger(file, maximumValue) || maximumValue != 255){
        fclose(file);
        return false;
    }

    std::vector<unsigned char> rgbPixels((size_t)imageWidth * imageHeight * 3);
    size_t readCount = fread(rgbPixels.data(), 1, rgbPixels.size(), file);
    fclose(file);
    if (readCount != rgbPixels.size()){
        return false;
    }

    rgbaPixels.resize((size_t)imageWidth * imageHeight * 4);
    for (size_t pixelIndex = 0; pixelIndex < (size_t)imageWidth * imageHeight; pixelIndex++) {
        rgbaPixels[pixelIndex * 4 + 0] = rgbPixels[pixelIndex * 3 + 0];
        rgbaPixels[pixelIndex * 4 + 1] = rgbPixels[pixelIndex * 3 + 1];
        rgbaPixels[pixelIndex * 4 + 2] = rgbPixels[pixelIndex * 3 + 2];
        rgbaPixels[pixelIndex * 4 + 3] = 255;
    }
    return true;
}

static cudaTextureObject_t createImageTextureObject(const std::vector<unsigned char>& rgbaPixels, int imageWidth, int imageHeight, cudaArray_t& imageArray) {
    cudaChannelFormatDesc channelDescription = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&imageArray, &channelDescription, imageWidth, imageHeight));
    CUDA_CHECK(cudaMemcpy2DToArray(imageArray, 0, 0, rgbaPixels.data(), imageWidth * 4, imageWidth * 4, imageHeight, cudaMemcpyHostToDevice));

    cudaResourceDesc resourceDescription = {};
    resourceDescription.resType = cudaResourceTypeArray;
    resourceDescription.res.array.array = imageArray;

    cudaTextureDesc textureDescription = {};
    textureDescription.addressMode[0] = cudaAddressModeWrap;
    textureDescription.addressMode[1] = cudaAddressModeWrap;
    textureDescription.filterMode = cudaFilterModeLinear;
    textureDescription.readMode = cudaReadModeNormalizedFloat;
    textureDescription.normalizedCoords = 1;
    textureDescription.sRGB = 1; // image textures are sRGB encoded now

    cudaTextureObject_t textureHandle = 0;
    CUDA_CHECK(cudaCreateTextureObject(&textureHandle, &resourceDescription, &textureDescription, nullptr));
    return textureHandle;
}

__host__ __device__ Texture make_image_texture(cudaTextureObject_t imageHandle) {
    return Texture{TextureType::Image, vec3(1,1,1), vec3(0,0,0), 1.f, imageHandle};
}

struct EnvironmentMap {
    cudaTextureObject_t texture;
    int enabled;
    float intensity;
    float rotationRadians;
    const float* rowCdf;
    const float* columnCdf;
    int width;
    int height;
    float totalWeight;
};
__constant__ EnvironmentMap g_environment;

static bool loadPfmAsRgba32f(const char* filePath, int& imageWidth, int& imageHeight, std::vector<float>& rgbaPixels){
    FILE* file = fopen(filePath, "rb");
    if(!file) return false;
    char magic[3] = {0, 0, 0};
    float scale = 0.0f;
    if(fscanf(file, "%2s %d %d %f", magic, &imageWidth, &imageHeight, &scale) != 4 || std::strcmp(magic, "PF") != 0 || imageWidth <= 0 || imageHeight <= 0){
        fclose(file);
        return false;
    }
    fgetc(file); // exactly one whitespace character separates the header from the pixels
    if(scale > 0.0f){
        std::fprintf(stderr, "big-endian PFM files are not supported\n");
        fclose(file);
        return false;
    }
    std::vector<float> rgbPixels((size_t)imageWidth * imageHeight * 3);
    size_t floatsRead = fread(rgbPixels.data(), sizeof(float), rgbPixels.size(), file);
    fclose(file);
    if(floatsRead != rgbPixels.size()) return false;

    rgbaPixels.resize((size_t)imageWidth * imageHeight * 4);
    for(int row = 0; row < imageHeight; row++){
        int sourceRow = imageHeight - 1 - row; // PFM stores the bottom row first
        for(int column = 0; column < imageWidth; column++){
            const float* source = &rgbPixels[((size_t)sourceRow * imageWidth + column) * 3];
            float* target = &rgbaPixels[((size_t)row * imageWidth + column) * 4];
            target[0] = source[0]; target[1] = source[1]; target[2] = source[2]; target[3] = 1.0f;
        }
    }
    return true;
}

static void buildEnvironmentSamplingTables(const std::vector<float>& rgbaPixels, int imageWidth, int imageHeight, std::vector<float>& rowCdf, std::vector<float>& columnCdf, float& totalWeight){
    rowCdf.assign(imageHeight + 1, 0.0f);
    columnCdf.assign((size_t)imageHeight * (imageWidth + 1), 0.0f);
    for(int row = 0; row < imageHeight; row++){
        float sinTheta = sinf(MONOCUDA_PI * (row + 0.5f) / imageHeight);
        float* cdf = &columnCdf[(size_t)row * (imageWidth + 1)];
        cdf[0] = 0.0f;
        for(int column = 0; column < imageWidth; column++){
            const float* texel = &rgbaPixels[((size_t)row * imageWidth + column) * 4];
            float luminance = 0.2126f * texel[0] + 0.7152f * texel[1] + 0.0722f * texel[2];
            cdf[column + 1] = cdf[column] + fmaxf(luminance, 0.0f) * sinTheta;
        }
        rowCdf[row + 1] = rowCdf[row] + cdf[imageWidth];
    }
    totalWeight = rowCdf[imageHeight];
}

static cudaTextureObject_t createEnvironmentTexture(const std::vector<float>& rgbaPixels, int imageWidth, int imageHeight, cudaArray_t& imageArray){
    cudaChannelFormatDesc channelDescription = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&imageArray, &channelDescription, imageWidth, imageHeight));
    CUDA_CHECK(cudaMemcpy2DToArray(imageArray, 0, 0, rgbaPixels.data(), imageWidth * sizeof(float4), imageWidth * sizeof(float4), imageHeight, cudaMemcpyHostToDevice));

    cudaResourceDesc resourceDescription = {};
    resourceDescription.resType = cudaResourceTypeArray;
    resourceDescription.res.array.array = imageArray;

    cudaTextureDesc textureDescription = {};
    textureDescription.addressMode[0] = cudaAddressModeWrap;  // longitude wraps around
    textureDescription.addressMode[1] = cudaAddressModeClamp; // latitude stops at the poles
    textureDescription.filterMode = cudaFilterModeLinear;
    textureDescription.readMode = cudaReadModeElementType;    // HDR values stay as they are, no normalization
    textureDescription.normalizedCoords = 1;

    cudaTextureObject_t textureHandle = 0;
    CUDA_CHECK(cudaCreateTextureObject(&textureHandle, &resourceDescription, &textureDescription, nullptr));
    return textureHandle;
}

__device__ vec2 environmentUv(const vec3& unitDirection){
    float u = atan2f(unitDirection.z, unitDirection.x) / (2.0f * MONOCUDA_PI) + 0.5f + g_environment.rotationRadians / (2.0f * MONOCUDA_PI);
    float v = acosf(fminf(fmaxf(unitDirection.y, -1.0f), 1.0f)) / MONOCUDA_PI;
    return vec2(u - floorf(u), v);
}

__device__ vec3 environmentDirection(float u, float v){
    float phi = (u - 0.5f) * 2.0f * MONOCUDA_PI - g_environment.rotationRadians;
    float theta = v * MONOCUDA_PI;
    float sinTheta = sinf(theta);
    return vec3(cosf(phi) * sinTheta, cosf(theta), sinf(phi) * sinTheta);
}

__device__ vec3 environmentRadiance(const vec3& direction, const vec3& fallbackColor){
    if(!g_environment.enabled) return fallbackColor;
    vec2 uv = environmentUv(direction.normalize());
    float4 texel = tex2D<float4>(g_environment.texture, uv.x, uv.y);
    return vec3(texel.x, texel.y, texel.z) * g_environment.intensity;
}

__device__ float environmentSelectionProbability(int lightCount){
    if(!g_environment.enabled || g_environment.totalWeight <= 0.0f) return 0.0f;
    return (lightCount > 0) ? 0.5f : 1.0f;
}

__device__ int findCdfInterval(const float* cdf, int intervalCount, float value){
    int low = 0;
    int high = intervalCount - 1;
    while(low < high){
        int middle = (low + high) >> 1;
        if(cdf[middle + 1] <= value) low = middle + 1;
        else high = middle;
    }
    return low;
}

__device__ float environmentTexelPdf(int row, int column, float sinTheta){
    if(sinTheta < 1e-6f) return 0.0f;
    const float* cdf = g_environment.columnCdf + (size_t)row * (g_environment.width + 1);
    float texelWeight = cdf[column + 1] - cdf[column];
    return texelWeight * g_environment.width * g_environment.height / (g_environment.totalWeight * 2.0f * MONOCUDA_PI * MONOCUDA_PI * sinTheta);
}

__device__ float environmentDirectionPdf(const vec3& direction){
    vec3 unitDirection = direction.normalize();
    vec2 uv = environmentUv(unitDirection);
    int column = min((int)(uv.x * g_environment.width), g_environment.width - 1);
    int row = min((int)(uv.y * g_environment.height), g_environment.height - 1);
    return environmentTexelPdf(row, column, sqrtf(fmaxf(1.0f - unitDirection.y * unitDirection.y, 0.0f)));
}

__device__ vec3 sampleEnvironmentDirection(curandState* rngState, float& directionPdf){
    int row = findCdfInterval(g_environment.rowCdf, g_environment.height, curand_uniform(rngState) * g_environment.totalWeight);
    const float* cdf = g_environment.columnCdf + (size_t)row * (g_environment.width + 1);
    int column = findCdfInterval(cdf, g_environment.width, curand_uniform(rngState) * cdf[g_environment.width]);
    float u = (column + curand_uniform(rngState)) / g_environment.width;
    float v = (row + curand_uniform(rngState)) / g_environment.height;
    directionPdf = environmentTexelPdf(row, column, sinf(v * MONOCUDA_PI));
    return environmentDirection(u, v);
}
// ============================================================================




// ============================================================================
// === MATERIALS ===
// ============================================================================
__device__ float schlick_approx(float cosine,float refraction_cof){
    float r0 = (1.f - refraction_cof) / (1.f + refraction_cof);
    r0 = r0 * r0;
    return r0 + (1.f - r0) * powf((1-cosine),5.f);
}

enum class MaterialType {Lambertian, Dielectric, Conductor,Emissive,Isotropic};

struct Material{
    MaterialType type;
    Texture albedoTexture;
    vec3 emittedColor;
    float fuzz;
    float refractionIndex;
    float density = 0.f;
};

__device__ vec3 random_unit_vector(curandState* rngState){

    while (true){
        float x = curand_uniform(rngState) * 2.f - 1.f;
        float y = curand_uniform(rngState) * 2.f - 1.f;
        float z = curand_uniform(rngState) * 2.f - 1.f;
        vec3 randomVec(x,y,z);
        if (randomVec.length_squared() < 1.0f) {
            return randomVec.normalize();
        }
    }
}

__device__ vec3 lambertian_scatter_direction(const vec3& surfaceNormal, curandState* rngState) {
    vec3 randomDirection = surfaceNormal + random_unit_vector(rngState);

    if (randomDirection.length_squared() < 1e-8f) {
        return surfaceNormal;
    }
    return randomDirection.normalize();
}

__device__ vec3 specular_scatter_direction(const vec3& incomingLightDirection,const vec3& surfaceNormal){
    return incomingLightDirection.normalize().reflect(surfaceNormal);
}

__device__ vec3 dielectric_scatter_direction(const vec3& incomingLightDirection,const vec3& surfaceNormal,float refraction_cof,curandState* rngState){
    vec3 unitDirection = incomingLightDirection.normalize();
    float cosTheta = fminf((-unitDirection).dot(surfaceNormal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    bool cannotRefract = (refraction_cof * sinTheta) > 1.0f;
    float reflectProbability;
    if(cannotRefract){
        reflectProbability = 1.0f;
    }else{
        reflectProbability = schlick_approx(cosTheta, refraction_cof);
    }
    if (curand_uniform(rngState) < reflectProbability) {
        return unitDirection.reflect(surfaceNormal);
    }
    return unitDirection.refract(surfaceNormal, refraction_cof);
}

__device__ vec3 isotropic_scatter_direction(curandState* rngState){
    float u1 = curand_uniform(rngState);
    float u2 = curand_uniform(rngState);

    float z = 1.0f - 2.0f * u1;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * MONOCUDA_PI * u2;

    return vec3(r * cosf(phi), r * sinf(phi), z);
}

// Constructors for material types
__host__ __device__ Material make_lambertian(vec3 albedo) {
    return Material{MaterialType::Lambertian, make_solid_texture(albedo), vec3(0,0,0), 0.f, 0.f};
}

__host__ __device__ Material make_lambertian(Texture albedoTexture) {
    return Material{MaterialType::Lambertian, albedoTexture, vec3(0,0,0), 0.f, 0.f};
}

__host__ __device__ Material make_conductor(vec3 albedo, float fuzz) {
    return Material{MaterialType::Conductor, make_solid_texture(albedo), vec3(0,0,0), fuzz < 1 ? fuzz : 1.f, 0.f};
}

__host__ __device__ Material make_dielectric(float refractionIndex) {
    return Material{MaterialType::Dielectric, make_solid_texture(vec3(1,1,1)), vec3(0,0,0), 0.f, refractionIndex};
}

__host__ __device__ Material make_emissive(vec3 emittedColor) {
    return Material{MaterialType::Emissive, make_solid_texture(vec3(0,0,0)), emittedColor, 0.f, 0.f};
}

__host__ __device__ Material make_isotropic(vec3 albedo, float density){
    return Material{MaterialType::Isotropic,make_solid_texture(albedo),vec3(0,0,0),0.f,0.f,density};
}

// ============================================================================


// ============================================================================
// === AABB ===
// ============================================================================

class alignas(16) AABB{
public:
    vec3 minInterval;
    float paddingToMakeMinIntervalSixteenBytes;

    vec3 maxInterval;
    float paddingToMakeMaxIntervalSixteenBytes;


    __host__ __device__ AABB(){}
    __host__ __device__ AABB(const vec3& minInterval_,const vec3& maxInterval_) : minInterval(minInterval_) , maxInterval(maxInterval_){}
};
static_assert(sizeof(AABB) == 32,  "AABB must be exactly 32 bytes so one box is two 16-byte vector loads");
static_assert(alignof(AABB) == 16, "AABB must be 16-byte aligned for LDG.128");
// ============================================================================

// ============================================================================
// === MESH & PRIMITIVE TYPES ===
// ============================================================================
__host__ __device__ inline AABB merge(const AABB& leftHandSide, const AABB& rightHandSide);
__device__ float edge_function(const vec3& a, const vec3& b, const vec3& c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

struct triangle {
    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    vec3 n0, n1, n2;
    Material mat;
    AABB aabb;

    __host__ __device__ triangle() {}
    __host__ __device__ triangle(vec3 v0_, vec3 v1_, vec3 v2_) : v0(v0_), v1(v1_), v2(v2_) {}
    __host__ __device__ triangle(vec3 v0_, vec3 v1_, vec3 v2_,Material mat_) : v0(v0_), v1(v1_), v2(v2_), mat(mat_) {}

    __host__ __device__ vec3 calculateCentroid(){
        return (v0 + v1 + v2)/3.;
    }

    __host__ __device__ void calculate_bounding_box(){
        const float eps = 1e-3f;
        aabb.minInterval = { fminf(v0.x, fminf(v1.x, v2.x)) - eps, fminf(v0.y, fminf(v1.y, v2.y)) - eps, fminf(v0.z, fminf(v1.z, v2.z)) - eps };
        aabb.maxInterval = { fmaxf(v0.x, fmaxf(v1.x, v2.x)) + eps, fmaxf(v0.y, fmaxf(v1.y, v2.y)) + eps, fmaxf(v0.z, fmaxf(v1.z, v2.z)) + eps };
    }

    __host__ __device__ AABB get_bounding_box(){return aabb;}
};

__device__ vec2 computeTriangleUvFromHitPoint(const triangle& hitTriangle, const vec3& hitPoint){
    float wholeTriangleAreaTimesTwo = (hitTriangle.v1 - hitTriangle.v0).cross(hitTriangle.v2 - hitTriangle.v0).length();

    float weightOfVertex0 = (hitTriangle.v1 - hitPoint).cross(hitTriangle.v2 - hitPoint).length() / wholeTriangleAreaTimesTwo;
    float weightOfVertex1 = (hitTriangle.v0 - hitPoint).cross(hitTriangle.v2 - hitPoint).length() / wholeTriangleAreaTimesTwo;
    float weightOfVertex2 = (hitTriangle.v1 - hitPoint).cross(hitTriangle.v0 - hitPoint).length() / wholeTriangleAreaTimesTwo;

    return hitTriangle.uv0 * weightOfVertex0 + hitTriangle.uv1 * weightOfVertex1+ hitTriangle.uv2 * weightOfVertex2;
}

__device__ vec3 computeTriangleShadingNormal(const triangle& hitTriangle, const vec3& hitPoint, const vec3& faceNormal){
    if (hitTriangle.n0.length_squared() == 0.0f) return faceNormal;

    float wholeTriangleAreaTimesTwo = (hitTriangle.v1 - hitTriangle.v0).cross(hitTriangle.v2 - hitTriangle.v0).length();

    float weightOfVertex0 = (hitTriangle.v1 - hitPoint).cross(hitTriangle.v2 - hitPoint).length() / wholeTriangleAreaTimesTwo;
    float weightOfVertex1 = (hitTriangle.v0 - hitPoint).cross(hitTriangle.v2 - hitPoint).length() / wholeTriangleAreaTimesTwo;
    float weightOfVertex2 = (hitTriangle.v1 - hitPoint).cross(hitTriangle.v0 - hitPoint).length() / wholeTriangleAreaTimesTwo;

    float weightSum = weightOfVertex0 + weightOfVertex1 + weightOfVertex2;
    if (!(weightSum > 0.0f) || weightSum > 1.01f) return faceNormal;
    vec3 shadingNormal = hitTriangle.n0 * weightOfVertex0 + hitTriangle.n1 * weightOfVertex1 + hitTriangle.n2 * weightOfVertex2;
    if (shadingNormal.length_squared() < 1e-12f) return faceNormal;
    shadingNormal = shadingNormal.normalize();
    if (shadingNormal.dot(faceNormal) < 0.0f) shadingNormal = -shadingNormal;
    return shadingNormal;
}

__device__ bool hit_triangle(const triangle &tri,const vec3& rayOrig,const vec3& rayDir, float& intersectionDistance){
    vec3 edge1 = tri.v1 - tri.v0;
    vec3 edge2 = tri.v2 - tri.v0;
    vec3 triangleNormal = edge1.cross(edge2).normalize();

    float denominator = rayDir.dot(triangleNormal);
    if (fabsf(denominator) < 1e-6f) return false; // ray is parallel

    float t =  (tri.v0 - rayOrig).dot(triangleNormal) / denominator;
    if(t < 0.0001f) return false;
    vec3 hitPoint = rayOrig + rayDir *t;
    // which side of the plane test with the edge function 3D
    vec3 edge0 = tri.v1 - tri.v0;
    vec3 edgeTestVector0 = hitPoint - tri.v0;
    if (triangleNormal.dot(edge0.cross(edgeTestVector0)) < 0) return false;

    vec3 edge1b = tri.v2 - tri.v1;
    vec3 edgeTestVector1 = hitPoint - tri.v1;
    if (triangleNormal.dot(edge1b.cross(edgeTestVector1)) < 0) return false;

    vec3 edge2b = tri.v0 - tri.v2;
    vec3 edgeTestVector2 = hitPoint - tri.v2;
    if (triangleNormal.dot(edge2b.cross(edgeTestVector2)) < 0) return false;

    intersectionDistance = t;
    return true;
}

// to pick a random point on triangles surface
__device__ vec3 sample_triangle(const triangle& tri,curandState* rngState){
    float randomValue_1 = curand_uniform(rngState);
    float randomValue_2 = curand_uniform(rngState);
    // we are taking its squareRoot so that the random values spread out evenly inside the triangle
    float sqrtRandom_1 = sqrtf(randomValue_1);

    float weightOfVertex0 = 1.0f - sqrtRandom_1;
    float weightOfVertex1 = randomValue_2 * sqrtRandom_1;
    float weightOfVertex2 = 1.0f - weightOfVertex0 - weightOfVertex1;

    return tri.v0 * weightOfVertex0 + tri.v1 * weightOfVertex1 + tri.v2 * weightOfVertex2;
}

__device__ float powerHeuristic(float primaryPdf, float otherPdf) {
    float primarySquared = primaryPdf * primaryPdf;
    float otherSquared = otherPdf * otherPdf;
    return primarySquared / (primarySquared + otherSquared);
}

__device__ float lambertianCosinePdf(float cosineAtSurface) {
    return cosineAtSurface / MONOCUDA_PI;
}

// isotropic scattering spreads uniformly over the sphere, value and pdf are the same constant
__device__ float isotropicPhasePdf() {
    return 0.25f / MONOCUDA_PI;
}

__device__ float triangleSurfaceArea(const triangle& tri) {
    return 0.5f * (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).length();
}

__device__ float lightSolidAnglePdf(float lightSelectionProbability, float distanceToLight, float cosineAtLight, float lightArea) {
    return lightSelectionProbability * (distanceToLight * distanceToLight) / (cosineAtLight * lightArea);
}

struct sphere {
    vec3 center;
    vec3 center1;
    bool isMoving = false;
    float radius;
    Material mat;
    AABB aabb;

    __host__ __device__ sphere(){}
    __host__ __device__ sphere(vec3 center_,double radius_) : center(center_), radius(radius_){}
    __host__ __device__ sphere(vec3 center_,double radius_,Material mat_) : center(center_), radius(radius_), mat(mat_){}
    __host__ __device__ sphere(vec3 c0, vec3 c1, double radius_, Material mat_) : center(c0), center1(c1), isMoving(true), radius(radius_), mat(mat_){}

    __host__ __device__ vec3 calculateCentroid(){
        return isMoving ? (center + center1) * 0.5f : center;
    }
    __host__ __device__ void calculate_bounding_box(){
        vec3 rv(radius, radius, radius);
        AABB b0(center - rv, center + rv);
        if (!isMoving) {
            aabb = b0;
            return;
        }
        AABB b1(center1 - rv, center1 + rv);
        aabb = merge(b0, b1);
    }
    __host__ __device__ vec3 center_at(float t) const {
        return isMoving ? center + (center1 - center) * t : center;
    }
    __host__ __device__ AABB get_bounding_box(){return aabb;}
};

__device__ vec2 computeSphereUvFromOutwardNormal(const vec3& outwardUnitNormal){
    float clampedNegativeY = fminf(1.0f, fmaxf(-1.0f, -outwardUnitNormal.y));
    float polarAngleTheta = acosf(clampedNegativeY);
    float azimuthAnglePhi = atan2f(-outwardUnitNormal.z, outwardUnitNormal.x) + MONOCUDA_PI;

    float u = azimuthAnglePhi / (2.0f * MONOCUDA_PI);
    float v = polarAngleTheta / MONOCUDA_PI;
    return vec2(u, v);
}
__device__ bool hit_sphere(const sphere& targetSphere,const vec3& rayOrigin, const vec3& rayDirection,float time,float& intersectionDistance,bool& frontFace, vec3& outwardNormal) {
    vec3 cur_center = targetSphere.center_at(time);
    vec3 vectorFromRayOriginToSphereCenter = cur_center - rayOrigin;
    float distanceToClosestPointOnRay =  vectorFromRayOriginToSphereCenter.dot(rayDirection) / rayDirection.dot(rayDirection);
    vec3 closestPointOnRay = rayOrigin + rayDirection * distanceToClosestPointOnRay;

    float distanceSquaredFromCenterToClosest = (cur_center - closestPointOnRay).length_squared();
    float discriminant = targetSphere.radius * targetSphere.radius - distanceSquaredFromCenterToClosest;

    if (discriminant < 0.0f) {
        return false;
    }
    float halfChordLength = sqrtf(discriminant);
    float nearRoot = distanceToClosestPointOnRay - halfChordLength;
    float farRoot = distanceToClosestPointOnRay + halfChordLength;
    float t = nearRoot;
    if (t < 0.0001f) t = farRoot;      // if ray starts from inside
    if (t < 0.0001f) return false;

    vec3 hitPoint = rayOrigin + rayDirection * t;
    outwardNormal = (hitPoint - cur_center).normalize();
    frontFace = rayDirection.dot(outwardNormal) < 0.0f;   // is ray coming from outside

    intersectionDistance = t;
    return true;
}

__host__ __device__ inline unsigned int expandBits(const unsigned int point){
    unsigned int expandedPoint = 0;
    for(int i = 0; i < 10; i++){
        unsigned int  bit = (point >> i ) & 1;
        expandedPoint |= (bit << (3 * i));
    }
    return expandedPoint;
}

__host__ __device__ inline unsigned int quantize(float value, float minVal, float maxVal){
    float normalized = (value - minVal) / (maxVal - minVal);
    return (unsigned int)(normalized * 1023.0f);
}

__host__ __device__ inline unsigned int morton3D(const vec3& centroid, const AABB& sceneBounds){
    unsigned int clampedX = quantize(centroid.x, sceneBounds.minInterval.x, sceneBounds.maxInterval.x);
    unsigned int clampedY = quantize(centroid.y, sceneBounds.minInterval.y, sceneBounds.maxInterval.y);
    unsigned int clampedZ = quantize(centroid.z, sceneBounds.minInterval.z, sceneBounds.maxInterval.z);
    clampedX = expandBits(clampedX);
    clampedY = expandBits(clampedY);
    clampedZ = expandBits(clampedZ);
    return clampedX | clampedY << 1| clampedZ << 2;
}

// ============================================================================

// ============================================================================
// === Lights ===
// ============================================================================
struct Light{
    triangle sourceTriangle;
    float surfaceArea;
    vec3 surfaceNormal;
};

// ============================================================================

// ============================================================================
// === Ray & HitRecord ===
// ============================================================================
struct Ray{
public:
    vec3 origin;
    vec3 direction;
    float time;

    __device__ Ray(){}
    __device__ Ray(vec3 origin_,vec3 direction_) : origin(origin_),direction(direction_),time(0.f) {}
    __device__ Ray(vec3 origin_,vec3 direction_,float time_) : origin(origin_),direction(direction_),time(time_){}

    __device__ vec3 isAt(float t) const {
        return origin + direction * t;
    }

};

struct hitRecord{
    bool didHit =  false;
    float closestDistance = 1e30f;
    vec3 hitPoint;
    vec3 hitNormal;
    Material mat;
    bool frontFace = true;
    int primitiveType = -1;     // 0 = sphere, 1 = triangle
    int primitiveIndex = -1;

    float u = 0.f;
    float v = 0.f;
};

// for deferred shading
struct ClosestHitCandidate{
    float closestDistance;
    int primitiveType;
    int primitiveIndex;
};

struct alignas(16) PackedHit{
    float hitDistance;
    int primitiveType;
    int primitiveIndex;
    int didHit;
};

// ============================================================================



// ============================================================================
// === BVH ===
// ============================================================================

struct BVH_Node{
    AABB bounds;
    int  leftChild;
    int  rightChild;
    bool leftIsLeaf;
    bool rightIsLeaf;
    int  parent;
    int  primitiveIndex;
    bool isLeaf;
};
struct PrimitiveRef {
    AABB bounds;
    vec3 centroid;
    int  primitiveType;
    int  primitiveIndex;
    unsigned int mortonCode;
};

// == FORWARD DECLERATIONS =======================================================
__global__ void buildLeavesKernel(BVH_Node* d_leafNode, const int* sortedPrimitiveIDs, const AABB* sortedBounds, int numObjects);
__global__ void buildInternalNodesKernel(BVH_Node* d_internalNodes, BVH_Node* d_leafNode, const unsigned int* sortedMortonCodes, int numObjects);
__global__ void refitBoundsKernel(BVH_Node* d_internalNodes, BVH_Node* d_leafNode, int* d_atomicCounters, int numObjects);
__host__ __device__ inline bool intersects(const AABB& leftHandSide, const AABB& rightHandSide);
__host__ __device__ inline AABB merge(const AABB& leftHandSide, const AABB& rightHandSide);
__host__ __device__ inline bool hit_aabb(const AABB& box, const vec3& rayOrigin, const vec3& rayDirection, float& nearestEntryDistance, float& farthestExitDistance);
__device__ inline void testPrimitiveHit(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection, float time, const sphere* spheres, const triangle* triangles, hitRecord& nearestHit);
__device__ inline bool doesPrimitiveBlockRay(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection, float time, const sphere* spheres, const triangle* triangles, float maximumDistance);
__device__ inline void testPrimitiveClosest(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection, float time, const sphere* spheres, const triangle* triangles, ClosestHitCandidate& nearestHit);
// ===============================================================================
// One ray's traversal progress. Lives in a lane for as long as the ray is being traversed,
// so a lane can be handed a new ray the moment its old one is finished.
// applying persistent threads to increase the number of threads per instruction to prevent thread divergence
static constexpr int TRAVERSAL_STACK_CAPACITY = 64;

struct TraversalState{
    vec3 rayOrigin;
    vec3 rayDirection;
    float rayTime;

    ClosestHitCandidate nearestHit;

    int stackIdx[TRAVERSAL_STACK_CAPACITY];
    bool stackIsLeaf[TRAVERSAL_STACK_CAPACITY];
    int stackPointer;

    int nodeIdx;
    bool nodeIsLeaf;
};


class BVH{

public:
    BVH_Node* d_internalNodes = nullptr;
    BVH_Node* d_leafNode = nullptr;
    int* d_primitiveTypes = nullptr;
    int primitiveCount;


    __host__ void generateHierarchy(const unsigned int* d_sortedMortonCodes, const int* d_sortedPrimitiveIDs, const AABB* d_sortedBounds, const int* d_sortedPrimitiveTypes, int numObjects){
        primitiveCount = numObjects;

        CUDA_CHECK(cudaMalloc(&d_leafNode, numObjects * sizeof(BVH_Node)));
        CUDA_CHECK(cudaMalloc(&d_internalNodes, (numObjects - 1) * sizeof(BVH_Node)));

        CUDA_CHECK(cudaMalloc(&d_primitiveTypes, numObjects * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_primitiveTypes, d_sortedPrimitiveTypes, numObjects * sizeof(int), cudaMemcpyDeviceToDevice));

        int* d_atomicCounters;
        CUDA_CHECK(cudaMalloc(&d_atomicCounters, (numObjects - 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_atomicCounters, 0, (numObjects - 1) * sizeof(int)));

        int blockSize = 256;
        buildLeavesKernel<<<(numObjects + blockSize - 1) / blockSize, blockSize>>>(d_leafNode, d_sortedPrimitiveIDs, d_sortedBounds, numObjects);
        CUDA_CHECK_KERNEL();

        buildInternalNodesKernel<<<(numObjects - 1 + blockSize - 1) / blockSize, blockSize>>>(d_internalNodes, d_leafNode, d_sortedMortonCodes, numObjects);
        CUDA_CHECK_KERNEL();

        int rootParent = -1;
        CUDA_CHECK(cudaMemcpy(&d_internalNodes[0].parent, &rootParent, sizeof(int), cudaMemcpyHostToDevice));

        refitBoundsKernel<<<(numObjects + blockSize - 1) / blockSize, blockSize>>>(d_internalNodes, d_leafNode, d_atomicCounters, numObjects);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaFree(d_atomicCounters));
    }

    __device__ hitRecord traverseRay(const vec3& rayOrigin, const vec3& rayDirection, float time, const sphere* spheres, const triangle* triangles,float maxDistance = 1e30f) const {
        hitRecord nearestHit;
        nearestHit.closestDistance = maxDistance;

        const int MAX_STACK = 64;
        int stackIdx[MAX_STACK];
        bool stackIsLeaf[MAX_STACK];
        int stackPtr = 0;

        int nodeIdx = 0;
        bool nodeIsLeaf = false;

        while (true) {
            const BVH_Node& node = nodeIsLeaf ? d_leafNode[nodeIdx] : d_internalNodes[nodeIdx];

            int childLIdx = node.leftChild;
            bool childLIsLeaf = node.leftIsLeaf;
            int childRIdx = node.rightChild;
            bool childRIsLeaf = node.rightIsLeaf;

            AABB childLBounds = childLIsLeaf ? d_leafNode[childLIdx].bounds : d_internalNodes[childLIdx].bounds;
            AABB childRBounds = childRIsLeaf ? d_leafNode[childRIdx].bounds : d_internalNodes[childRIdx].bounds;

            float nearL = 0.0001f;
            float farL = nearestHit.closestDistance;
            bool overlapL = hit_aabb(childLBounds, rayOrigin, rayDirection, nearL, farL);

            float nearR = 0.0001f;
            float farR = nearestHit.closestDistance;
            bool overlapR = hit_aabb(childRBounds, rayOrigin, rayDirection, nearR, farR);

            if (overlapL && childLIsLeaf) {
                int primitiveIdx = d_leafNode[childLIdx].primitiveIndex;
                int primitiveType = d_primitiveTypes[childLIdx];
                testPrimitiveHit(primitiveType, primitiveIdx, rayOrigin, rayDirection,time, spheres, triangles, nearestHit);
            }
            if (overlapR && childRIsLeaf) {
                int primitiveIdx = d_leafNode[childRIdx].primitiveIndex;
                int primitiveType = d_primitiveTypes[childRIdx];
                testPrimitiveHit(primitiveType, primitiveIdx, rayOrigin, rayDirection, time, spheres, triangles, nearestHit);
            }


            bool traverseL = overlapL && !childLIsLeaf;
            bool traverseR = overlapR && !childRIsLeaf;

            if (!traverseL && !traverseR) {
                if (stackPtr == 0) break;
                stackPtr--;
                nodeIdx = stackIdx[stackPtr];
                nodeIsLeaf = stackIsLeaf[stackPtr];
            } else {
                if (traverseL && traverseR) {
                    stackIdx[stackPtr] = childRIdx;
                    stackIsLeaf[stackPtr] = childRIsLeaf;
                    stackPtr++;
                }
                nodeIdx = traverseL ? childLIdx : childRIdx;
                nodeIsLeaf = traverseL ? childLIsLeaf : childRIsLeaf;
            }
        }

        return nearestHit;
    }

    __device__ bool traverseAnyHit(const vec3& rayOrigin, const vec3& rayDirection, float time, const sphere* spheres, const triangle* triangles,float maxDistance = 1e30f) const {

        const int MAX_STACK = 64;
        int stackIdx[MAX_STACK];
        bool stackIsLeaf[MAX_STACK];
        int stackPtr = 0;

        int nodeIdx = 0;
        bool nodeIsLeaf = false;

        while (true) {
            const BVH_Node& node = nodeIsLeaf ? d_leafNode[nodeIdx] : d_internalNodes[nodeIdx];

            int childLIdx = node.leftChild;
            bool childLIsLeaf = node.leftIsLeaf;
            int childRIdx = node.rightChild;
            bool childRIsLeaf = node.rightIsLeaf;

            AABB childLBounds = childLIsLeaf ? d_leafNode[childLIdx].bounds : d_internalNodes[childLIdx].bounds;
            AABB childRBounds = childRIsLeaf ? d_leafNode[childRIdx].bounds : d_internalNodes[childRIdx].bounds;

            float nearL = 0.0001f;
            float farL = maxDistance;
            bool overlapL = hit_aabb(childLBounds, rayOrigin, rayDirection, nearL, farL);

            float nearR = 0.0001f;
            float farR = maxDistance;
            bool overlapR = hit_aabb(childRBounds, rayOrigin, rayDirection, nearR, farR);

            if (overlapL && childLIsLeaf) {
                int primitiveIdx = d_leafNode[childLIdx].primitiveIndex;
                int primitiveType = d_primitiveTypes[childLIdx];
                if(doesPrimitiveBlockRay(primitiveType, primitiveIdx, rayOrigin, rayDirection, time, spheres, triangles, maxDistance))return true;
            }
            if (overlapR && childRIsLeaf) {
                int primitiveIdx = d_leafNode[childRIdx].primitiveIndex;
                int primitiveType = d_primitiveTypes[childRIdx];
                if(doesPrimitiveBlockRay(primitiveType,primitiveIdx,rayOrigin,rayDirection,time,spheres,triangles,maxDistance))return true;
            }


            bool traverseL = overlapL && !childLIsLeaf;
            bool traverseR = overlapR && !childRIsLeaf;

            if (!traverseL && !traverseR) {
                if (stackPtr == 0) break;
                stackPtr--;
                nodeIdx = stackIdx[stackPtr];
                nodeIsLeaf = stackIsLeaf[stackPtr];
            } else {
                if (traverseL && traverseR) {
                    stackIdx[stackPtr] = childRIdx;
                    stackIsLeaf[stackPtr] = childRIsLeaf;
                    stackPtr++;
                }
                nodeIdx = traverseL ? childLIdx : childRIdx;
                nodeIsLeaf = traverseL ? childLIsLeaf : childRIsLeaf;
            }
        }
        return false;
    }

    __device__ void traverseHierarchy(const PrimitiveRef* d_primitiveRefs, int queryObjectIdx, int* d_collisionCounts) const {
        AABB queryBounds = d_primitiveRefs[queryObjectIdx].bounds;
        int  querySelfIdx = d_primitiveRefs[queryObjectIdx].primitiveIndex;
        const int MAX_STACK = 64;
        int  stackIdx[MAX_STACK];
        bool stackIsLeaf[MAX_STACK];
        int  stackPtr = 0;

        int  nodeIdx = 0;      // root = d_internalNodes[0]
        bool nodeIsLeaf = false;

        while (true) {
            const BVH_Node& node = nodeIsLeaf ? d_leafNode[nodeIdx] : d_internalNodes[nodeIdx];

            int  childLIdx = node.leftChild;
            bool childLIsLeaf = node.leftIsLeaf;
            int  childRIdx = node.rightChild;
            bool childRIsLeaf = node.rightIsLeaf;

            AABB childLBounds = childLIsLeaf ? d_leafNode[childLIdx].bounds : d_internalNodes[childLIdx].bounds;
            AABB childRBounds = childRIsLeaf ? d_leafNode[childRIdx].bounds : d_internalNodes[childRIdx].bounds;

            bool overlapL = intersects(queryBounds, childLBounds);
            bool overlapR = intersects(queryBounds, childRBounds);

            if (overlapL && childLIsLeaf) {
                int hitPrimitiveIdx = d_leafNode[childLIdx].primitiveIndex;
                if (hitPrimitiveIdx != querySelfIdx) {
                    atomicAdd(&d_collisionCounts[queryObjectIdx], 1);
                }
            }
            if (overlapR && childRIsLeaf) {
                int hitPrimitiveIdx = d_leafNode[childRIdx].primitiveIndex;
                if (hitPrimitiveIdx != querySelfIdx) {
                    atomicAdd(&d_collisionCounts[queryObjectIdx], 1);
                }
            }

            bool traverseL = overlapL && !childLIsLeaf;
            bool traverseR = overlapR && !childRIsLeaf;

            if (!traverseL && !traverseR) {
                if (stackPtr == 0) break;
                stackPtr--;
                nodeIdx = stackIdx[stackPtr];
                nodeIsLeaf = stackIsLeaf[stackPtr];
            } else {
                if (traverseL && traverseR) {
                    stackIdx[stackPtr] = childRIdx;
                    stackIsLeaf[stackPtr] = childRIsLeaf;
                    stackPtr++;
                }
                nodeIdx = traverseL ? childLIdx : childRIdx;
                nodeIsLeaf = traverseL ? childLIsLeaf : childRIsLeaf;
            }
        }
    }

    __device__ void beginTraversal(TraversalState& traversal, const vec3& rayOrigin, const vec3& rayDirection, float rayTime, float maxDistance = 1e30f) const {
        traversal.rayOrigin = rayOrigin;
        traversal.rayDirection = rayDirection;
        traversal.rayTime = rayTime;

        traversal.nearestHit.closestDistance = maxDistance;
        traversal.nearestHit.primitiveIndex = -1;
        traversal.nearestHit.primitiveType = -1;

        traversal.stackPointer = 0;
        traversal.nodeIdx = 0;
        traversal.nodeIsLeaf = false;
    }

    // Executes exactly one iteration of the traverseRay loop. Returns true when the traversal is finished.
    __device__ bool stepTraversal(TraversalState& traversal, const sphere* spheres, const triangle* triangles) const {
        const BVH_Node& node = traversal.nodeIsLeaf ? d_leafNode[traversal.nodeIdx] : d_internalNodes[traversal.nodeIdx];

        int childLIdx = node.leftChild;
        bool childLIsLeaf = node.leftIsLeaf;
        int childRIdx = node.rightChild;
        bool childRIsLeaf = node.rightIsLeaf;

        AABB childLBounds = childLIsLeaf ? d_leafNode[childLIdx].bounds : d_internalNodes[childLIdx].bounds;
        AABB childRBounds = childRIsLeaf ? d_leafNode[childRIdx].bounds : d_internalNodes[childRIdx].bounds;

        float nearL = 0.0001f;
        float farL = traversal.nearestHit.closestDistance;
        bool overlapL = hit_aabb(childLBounds, traversal.rayOrigin, traversal.rayDirection, nearL, farL);

        float nearR = 0.0001f;
        float farR = traversal.nearestHit.closestDistance;
        bool overlapR = hit_aabb(childRBounds, traversal.rayOrigin, traversal.rayDirection, nearR, farR);

        if (overlapL && childLIsLeaf) {
            int primitiveIdx = d_leafNode[childLIdx].primitiveIndex;
            int primitiveType = d_primitiveTypes[childLIdx];
            testPrimitiveClosest(primitiveType, primitiveIdx, traversal.rayOrigin, traversal.rayDirection, traversal.rayTime, spheres, triangles, traversal.nearestHit);
        }
        if (overlapR && childRIsLeaf) {
            int primitiveIdx = d_leafNode[childRIdx].primitiveIndex;
            int primitiveType = d_primitiveTypes[childRIdx];
            testPrimitiveClosest(primitiveType, primitiveIdx, traversal.rayOrigin, traversal.rayDirection, traversal.rayTime, spheres, triangles, traversal.nearestHit);
        }

        bool traverseL = overlapL && !childLIsLeaf;
        bool traverseR = overlapR && !childRIsLeaf;

        if (!traverseL && !traverseR) {
            if (traversal.stackPointer == 0) return true;
            traversal.stackPointer--;
            traversal.nodeIdx = traversal.stackIdx[traversal.stackPointer];
            traversal.nodeIsLeaf = traversal.stackIsLeaf[traversal.stackPointer];
        } else {
            if (traverseL && traverseR) {
                traversal.stackIdx[traversal.stackPointer] = childRIdx;
                traversal.stackIsLeaf[traversal.stackPointer] = childRIsLeaf;
                traversal.stackPointer++;
            }
            traversal.nodeIdx = traverseL ? childLIdx : childRIdx;
            traversal.nodeIsLeaf = traverseL ? childLIsLeaf : childRIsLeaf;
        }
        return false;
    }

    __host__ void free(){
        if (d_internalNodes) CUDA_CHECK(cudaFree(d_internalNodes));
        if (d_leafNode) CUDA_CHECK(cudaFree(d_leafNode));
        if (d_primitiveTypes) CUDA_CHECK(cudaFree(d_primitiveTypes));
        d_internalNodes = nullptr;
        d_leafNode = nullptr;
        d_primitiveTypes = nullptr;
    }



};


__device__ inline int delta(const unsigned int* sortedMortonCodes, int numObjects, int i, int j){
    if (j < 0 || j >= numObjects) return -1;

    unsigned int codeI = sortedMortonCodes[i];
    unsigned int codeJ = sortedMortonCodes[j];

    if (codeI == codeJ){
        return 32 + __clz((unsigned int)(i ^ j));
    }
    return __clz(codeI ^ codeJ);
}

__device__ inline int2 determineRange(const unsigned int* sortedMortonCodes, int numObjects, int idx){
    int d = (delta(sortedMortonCodes, numObjects, idx, idx+1) - delta(sortedMortonCodes, numObjects, idx, idx-1)) >= 0 ? 1 : -1;
    int deltaMin = delta(sortedMortonCodes, numObjects, idx, idx - d);

    int lmax = 2;
    while (delta(sortedMortonCodes, numObjects, idx, idx + lmax * d) > deltaMin){
        lmax *= 2;
    }

    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2){
        if (delta(sortedMortonCodes, numObjects, idx, idx + (l + t) * d) > deltaMin){
            l += t;
        }
    }

    int jdx = idx + l * d;
    int first = min(idx, jdx);
    int last  = max(idx, jdx);
    return make_int2(first, last);
}

__device__ inline int findSplit(const unsigned int* sortedMortonCodes, int numObjects, int first, int last){
    int commonPrefix = delta(sortedMortonCodes, numObjects, first, last);

    int split = first;
    int step  = last - first;
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if (newSplit < last){
            int splitPrefix = delta(sortedMortonCodes, numObjects, first, newSplit);
            if (splitPrefix > commonPrefix) split = newSplit;
        }
    } while (step > 1);

    return split;
}
// KERNEL FUNCTIONS FOR GENERATING THE HIERARCHY
__global__ void buildLeavesKernel(BVH_Node* d_leafNode, const int* sortedPrimitiveIDs, const AABB* sortedBounds, int numObjects){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numObjects) return;
    d_leafNode[idx].primitiveIndex = sortedPrimitiveIDs[idx];
    d_leafNode[idx].bounds = sortedBounds[idx];
    d_leafNode[idx].isLeaf = true;
    d_leafNode[idx].leftChild  = -1;
    d_leafNode[idx].rightChild = -1;
}

__global__ void refitBoundsKernel(BVH_Node* d_internalNodes, BVH_Node* d_leafNode, int* d_atomicCounters, int numObjects){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numObjects) return;

    int currentParent = d_leafNode[idx].parent;

    while (currentParent != -1) {
        int old = atomicAdd(&d_atomicCounters[currentParent], 1);
        if (old == 0) return;

        BVH_Node& node = d_internalNodes[currentParent];
        AABB leftBounds = node.leftIsLeaf  ? d_leafNode[node.leftChild].bounds  : d_internalNodes[node.leftChild].bounds;
        AABB rightBounds = node.rightIsLeaf ? d_leafNode[node.rightChild].bounds : d_internalNodes[node.rightChild].bounds;
        node.bounds = merge(leftBounds, rightBounds);
        __threadfence();
        currentParent = node.parent;
    }
}

__global__ void buildInternalNodesKernel(BVH_Node* d_internalNodes, BVH_Node* d_leafNode,const unsigned int* sortedMortonCodes, int numObjects){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numObjects - 1) return;

    int2 range = determineRange(sortedMortonCodes, numObjects, idx);
    int first = range.x;
    int last  = range.y;
    int split = findSplit(sortedMortonCodes, numObjects, first, last);

    d_internalNodes[idx].isLeaf = false;

    if (split == first){
        d_internalNodes[idx].leftChild = split;
        d_internalNodes[idx].leftIsLeaf = true;
        d_leafNode[split].parent = idx;
    } else {
        d_internalNodes[idx].leftChild = split;
        d_internalNodes[idx].leftIsLeaf = false;
        d_internalNodes[split].parent = idx;
    }

    if (split + 1 == last){
        d_internalNodes[idx].rightChild = split + 1;
        d_internalNodes[idx].rightIsLeaf = true;
        d_leafNode[split + 1].parent = idx;
    } else {
        d_internalNodes[idx].rightChild = split + 1;
        d_internalNodes[idx].rightIsLeaf = false;
        d_internalNodes[split + 1].parent = idx;
    }
}

__host__ __device__ inline bool intersects(const AABB& leftHandSide, const AABB& rightHandSide){
    if (leftHandSide.maxInterval.x < rightHandSide.minInterval.x || rightHandSide.maxInterval.x < leftHandSide.minInterval.x) { return false; }
    if (leftHandSide.maxInterval.y < rightHandSide.minInterval.y || rightHandSide.maxInterval.y < leftHandSide.minInterval.y) { return false; }
    if (leftHandSide.maxInterval.z < rightHandSide.minInterval.z || rightHandSide.maxInterval.z < leftHandSide.minInterval.z) { return false; }
    return true;
}

__host__ __device__ inline AABB merge(const AABB& leftHandSide, const AABB& rightHandSide){
    AABB mergedAABB;
    mergedAABB.maxInterval.x = fmaxf(leftHandSide.maxInterval.x ,rightHandSide.maxInterval.x);
    mergedAABB.maxInterval.y = fmaxf(leftHandSide.maxInterval.y ,rightHandSide.maxInterval.y);
    mergedAABB.maxInterval.z = fmaxf(leftHandSide.maxInterval.z ,rightHandSide.maxInterval.z);
    mergedAABB.minInterval.x = fminf(leftHandSide.minInterval.x ,rightHandSide.minInterval.x);
    mergedAABB.minInterval.y = fminf(leftHandSide.minInterval.y ,rightHandSide.minInterval.y);
    mergedAABB.minInterval.z = fminf(leftHandSide.minInterval.z ,rightHandSide.minInterval.z);
    return mergedAABB;
}

__host__ __device__ inline bool hit_aabb(const AABB& box, const vec3& rayOrigin, const vec3& rayDirection, float& nearestEntryDistance, float& farthestExitDistance){
    for (int axisIndex = 0; axisIndex < 3; axisIndex++) {
        float originOnAxis = (axisIndex == 0) ? rayOrigin.x : (axisIndex == 1) ? rayOrigin.y : rayOrigin.z;
        float directionOnAxis = (axisIndex == 0) ? rayDirection.x : (axisIndex == 1) ? rayDirection.y : rayDirection.z;
        float boxMinOnAxis = (axisIndex == 0) ? box.minInterval.x : (axisIndex == 1) ? box.minInterval.y : box.minInterval.z;
        float boxMaxOnAxis = (axisIndex == 0) ? box.maxInterval.x : (axisIndex == 1) ? box.maxInterval.y : box.maxInterval.z;

        float inverseDirection = 1.0f / directionOnAxis;
        float entryDistance = (boxMinOnAxis - originOnAxis) * inverseDirection;
        float exitDistance = (boxMaxOnAxis - originOnAxis) * inverseDirection;
        if (inverseDirection < 0.0f) {
            float swapTemp = entryDistance;
            entryDistance = exitDistance;
            exitDistance = swapTemp;
        }

        nearestEntryDistance = fmaxf(nearestEntryDistance, entryDistance);
        farthestExitDistance = fminf(farthestExitDistance, exitDistance);
        if (farthestExitDistance <= nearestEntryDistance) return false;
    }
    return true;
}

__device__ inline void testPrimitiveHit(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection,float time, const sphere* spheres, const triangle* triangles, hitRecord& nearestHit){
    if (primitiveType == 0) {
        float intersectionDistance;
        bool frontFace;
        vec3 outwardNormal;
        if (hit_sphere(spheres[primitiveIdx], rayOrigin, rayDirection, time, intersectionDistance, frontFace, outwardNormal)) {
            if (intersectionDistance < nearestHit.closestDistance && intersectionDistance > 0.0001f) {
                nearestHit.didHit = true;
                nearestHit.primitiveType = primitiveType;
                nearestHit.primitiveIndex = primitiveIdx;
                nearestHit.closestDistance = intersectionDistance;
                nearestHit.hitPoint = rayOrigin + rayDirection * intersectionDistance;
                nearestHit.mat = spheres[primitiveIdx].mat;
                nearestHit.frontFace = frontFace;
                nearestHit.hitNormal = frontFace ? outwardNormal : -outwardNormal;
                vec2 sphereUv = computeSphereUvFromOutwardNormal(outwardNormal);
                nearestHit.u = sphereUv.x;
                nearestHit.v = sphereUv.y;
            }
        }
    } else {
        float intersectionDistance;
        if (hit_triangle(triangles[primitiveIdx], rayOrigin, rayDirection, intersectionDistance)) {
            if (intersectionDistance < nearestHit.closestDistance && intersectionDistance > 0.0001f) {
                nearestHit.didHit = true;
                nearestHit.primitiveType = primitiveType;
                nearestHit.primitiveIndex = primitiveIdx;
                nearestHit.closestDistance = intersectionDistance;
                nearestHit.hitPoint = rayOrigin + rayDirection * intersectionDistance;
                nearestHit.mat = triangles[primitiveIdx].mat;

                vec3 edge1 = triangles[primitiveIdx].v1 - triangles[primitiveIdx].v0;
                vec3 edge2 = triangles[primitiveIdx].v2 - triangles[primitiveIdx].v0;
                vec3 n = edge1.cross(edge2).normalize();

                bool front = rayDirection.dot(n) < 0.0f;
                nearestHit.frontFace = front;
                vec3 shadingNormal = computeTriangleShadingNormal(triangles[primitiveIdx], nearestHit.hitPoint, n);
                nearestHit.hitNormal = front ? shadingNormal : -shadingNormal;
                vec2 triangleUv = computeTriangleUvFromHitPoint(triangles[primitiveIdx], nearestHit.hitPoint);
                nearestHit.u = triangleUv.x;
                nearestHit.v = triangleUv.y;
            }
        }
    }
}

__device__ inline void testPrimitiveClosest(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection, float time, const sphere* spheres, const triangle* triangles, ClosestHitCandidate& nearestHit){
    float intersectionDistance;
    if (primitiveType == 0) {
        bool frontFace;
        vec3 outwardNormal;
        if (!hit_sphere(spheres[primitiveIdx], rayOrigin, rayDirection, time, intersectionDistance, frontFace, outwardNormal)) return;
    } else {
        if (!hit_triangle(triangles[primitiveIdx], rayOrigin, rayDirection, intersectionDistance)) return;
    }
    if (intersectionDistance < nearestHit.closestDistance && intersectionDistance > 0.0001f) {
        nearestHit.closestDistance = intersectionDistance;
        nearestHit.primitiveType = primitiveType;
        nearestHit.primitiveIndex = primitiveIdx;
    }
}

__device__ inline bool doesPrimitiveBlockRay(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection, float time, const sphere* spheres, const triangle* triangles, float maximumDistance){
    float intersectionDistance;
    if (primitiveType == 0) {
        bool frontFace;
        vec3 outwardNormal;
        if (!hit_sphere(spheres[primitiveIdx], rayOrigin, rayDirection, time, intersectionDistance, frontFace, outwardNormal)) return false;
    } else {
        if (!hit_triangle(triangles[primitiveIdx], rayOrigin, rayDirection, intersectionDistance)) return false;
    }
    return (intersectionDistance > 0.0001f) && (intersectionDistance < maximumDistance);
}

__global__ void findCollisions(const PrimitiveRef* d_primitiveRefs, BVH* d_bvh, int primitiveCount, int* d_collisionCounts){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < primitiveCount){
        d_bvh->traverseHierarchy(d_primitiveRefs, idx, d_collisionCounts);
    }
}
// ============================================================================


// ============================================================================
// === CAMERA ===
// ============================================================================
__host__ __device__ float degrees_to_radians(float degree) {
    return degree * (3.14159265358979323846f / 180.0f);
}

class Camera {
public:
    float aspect_ratio = 1.f;
    int image_width = 100;
    int image_height;
    int samples_per_pixel = 20;
    int max_depth = 10;

    float vfov = 90.f;
    vec3 look_from = vec3(0,0,0);
    vec3 look_at = vec3(0,0,-1);
    vec3 up = vec3(0,1,0);

    vec3 center;
    vec3 pixel00_location;
    vec3 pixel_delta_u, pixel_delta_v;
    vec3 u, v, w;
    vec3 backround_color;

    float aperture;
    float lens_radius;
    float focus_distance = 10;

    __host__ __device__ Camera(){}

    __host__ __device__ void initialize() {
        w = (look_from - look_at).normalize();
        u = up.cross(w).normalize();
        v = w.cross(u);
        lens_radius = aperture / 2.;
        image_height = int(image_width / aspect_ratio);
        center = look_from;

        float theta = degrees_to_radians(vfov);
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h * focus_distance;
        float viewport_width = viewport_height * (float(image_width) / image_height);

        vec3 viewport_u = u * viewport_width;
        vec3 viewport_v = v * -viewport_height;

        pixel_delta_u = viewport_u / float(image_width);
        pixel_delta_v = viewport_v / float(image_height);

        vec3 viewport_upper_left = center - (w * focus_distance) - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel00_location = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5f;
    }

    __device__ vec3 random_in_unit_disk(curandState* rngState) const{
        while(true){
                float randomX = curand_uniform(rngState) * 2.f - 1.f;
                float randomY = curand_uniform(rngState) * 2.f - 1.f;
                vec3 candidatePoint = vec3(randomX,randomY,0.0f);
                if(candidatePoint.length_squared() < 1.0f){
                    return candidatePoint;
                }
        }
    }

    __device__ double random_double(curandState* rngState,double low,double high) const{
        return low + (high - low) * curand_uniform_double(rngState);
    }

    __device__ Ray getRay(int pixelX,int pixelY,curandState* rngState) const{
        vec3 pixelCenter = pixel00_location + (pixel_delta_u * float(pixelX)) + (pixel_delta_v * float(pixelY));

        vec3 randomPointOnLens = random_in_unit_disk(rngState) * lens_radius;
        vec3 lensOffSetInWorldSpace = u * randomPointOnLens.x + v * randomPointOnLens.y;
        vec3 rayOriginOnLens = center + lensOffSetInWorldSpace;
        float ray_time = curand_uniform(rngState);

        vec3 rayDirection = (pixelCenter - rayOriginOnLens).normalize();
        return Ray(rayOriginOnLens, rayDirection, ray_time);
    }
};
// ============================================================================


// ============================================================================
// === SCENE SETUP ===
// ============================================================================

struct hostScene {
    std::vector<sphere> spheres;
    std::vector<triangle> triangles;
    std::vector<Light> lights;
    std::vector<PrimitiveRef> primitiveRefs;
    std::vector<cudaArray_t> imageArrays;
    std::vector<cudaTextureObject_t> imageTextureHandles;
    AABB bounds;
};

static void add_quad(hostScene& s, vec3 a, vec3 b, vec3 c, vec3 d, Material m){
    s.triangles.push_back(triangle(a, b, c, m));
    s.triangles.push_back(triangle(a, c, d, m));

    triangle& firstTriangle = s.triangles[s.triangles.size() - 2];
    triangle& secondTriangle = s.triangles[s.triangles.size() - 1];

    firstTriangle.uv0 = vec2(0.f, 0.f);
    firstTriangle.uv1 = vec2(1.f, 0.f);
    firstTriangle.uv2 = vec2(1.f, 1.f);

    secondTriangle.uv0 = vec2(0.f, 0.f);
    secondTriangle.uv1 = vec2(1.f, 1.f);
    secondTriangle.uv2 = vec2(0.f, 1.f);
}

static void add_box(hostScene& s, vec3 center, vec3 half, float rotY, Material m){
    float cs = cosf(rotY), sn = sinf(rotY);
    vec3 p[8];
    for (int i = 0; i < 8; i++) {
        float x = (i & 1) ? half.x : -half.x;
        float y = (i & 2) ? half.y : -half.y;
        float z = (i & 4) ? half.z : -half.z;
        p[i] = center + vec3(x*cs + z*sn, y, -x*sn + z*cs);
    }
    add_quad(s, p[0], p[1], p[3], p[2], m);  // -z
    add_quad(s, p[4], p[5], p[7], p[6], m);  // +z
    add_quad(s, p[0], p[2], p[6], p[4], m);  // -x
    add_quad(s, p[1], p[3], p[7], p[5], m);  // +x
    add_quad(s, p[0], p[1], p[5], p[4], m);  // -y
    add_quad(s, p[2], p[3], p[7], p[6], m);  // +y
}

enum class SceneChoice {Cornell, Sponza, Bunny, Dragon, LivingRoom, Sibenik, Sphere, Barcelona, Bathroom, Book, SanMiguel, Pbrt};
static const char* sceneNames[] = {"cornell", "sponza", "bunny", "dragon", "living_room", "sibenik", "sphere", "barcelona", "bathroom", "book", "sanmiguel", "pbrt"};
static const char* scenePbrtPaths[] = {"", "", "", "", "", "", "",
    "models/pbrt-v3-scenes/barcelona-pavilion/pavilion-day.pbrt",
    "models/pbrt-v3-scenes/contemporary-bathroom/contemporary-bathroom.pbrt",
    "models/pbrt-v3-scenes/pbrt-book/book.pbrt",
    "models/pbrt-v3-scenes/sanmiguel/sanmiguel.pbrt", ""};
static const char* sceneDescriptions[] = {
    "Cornell box (small scene with a smoke sphere)",
    "Sponza atrium (262k triangles, needs models/sponza/sponza.obj)",
    "Stanford bunny in the Cornell box (144k triangles, needs models/bunny/bunny.obj)",
    "Stanford dragon in the Cornell box (871k triangles, needs models/dragon/dragon.obj)",
    "Living room (580k triangles, needs models/living_room/living_room.obj)",
    "Sibenik cathedral (75k triangles, needs models/sibenik/sibenik.obj)",
    "Textured UV sphere in the Cornell box (8k triangles, needs models/sphere/sphere-cylcoords-16k.obj)",
    "Barcelona pavilion, pbrt-v3 (130k triangles, sky from an EXR environment map)",
    "Contemporary bathroom, pbrt-v3 (536k triangles, lit through the window blinds)",
    "The pbrt book cover, pbrt-v3 (23k triangles, two spherical area lights)",
    "San Miguel courtyard, pbrt-v3 (10.5M triangles, needs about 3 GB)",
    "pbrt-v3 scene file (camera, lights and materials come from the file, pass it with --pbrt)"
};
static const int sceneCount = 12;

// == FORWARD DECLERATIONS =======================================================
static bool loadObject(hostScene& scene,const char* filePath,Material mat,vec3 translation,float uniformScale,float rotationAroundYRadians);
static std::string getDirectoryOfFile(const std::string& filePath);
static cudaTextureObject_t getOrLoadObjTexture(hostScene& scene,const std::string& ppmPath,const std::string& originalPath,std::map<std::string,cudaTextureObject_t>& textureCache);
// ===============================================================================

__host__ void buildCornellRoom(hostScene& scene) {

    Material floor = make_lambertian(make_checker_texture(vec3(0.9f,0.9f,0.9f), vec3(0.1f,0.1f,0.1f), 8.f));
    Material white  = make_lambertian(vec3(0.73f, 0.73f, 0.73f));
    Material red = make_lambertian(vec3(0.65f, 0.05f, 0.05f));
    Material green  = make_lambertian(vec3(0.12f, 0.45f, 0.15f));
    Material light  = make_emissive(vec3(15.0f, 15.0f, 15.0f));

    const float L = -2.f, R = 2.f, B = 0.f, T = 4.f, F = -4.f, N = 0.f;
    add_quad(scene, vec3(L,B,N), vec3(R,B,N), vec3(R,B,F), vec3(L,B,F), floor);  // floor
    add_quad(scene, vec3(L,T,N), vec3(R,T,N), vec3(R,T,F), vec3(L,T,F), white);  // ceiling
    add_quad(scene, vec3(L,B,F), vec3(R,B,F), vec3(R,T,F), vec3(L,T,F), white);  // back
    add_quad(scene, vec3(L,B,N), vec3(L,B,F), vec3(L,T,F), vec3(L,T,N), red);    // left
    add_quad(scene, vec3(R,B,N), vec3(R,B,F), vec3(R,T,F), vec3(R,T,N), green);  // right

    add_quad(scene, vec3(-1.f,T-0.01f,-2.6f), vec3(1.f,T-0.01f,-2.6f),vec3(1.f,T-0.01f,-1.4f), vec3(-1.f,T-0.01f,-1.4f), light);
}

__host__ void buildCornellScene(hostScene& scene) {
    buildCornellRoom(scene);

    Material white  = make_lambertian(vec3(0.73f, 0.73f, 0.73f));
    Material mirror = make_conductor(vec3(0.9f, 0.9f, 0.9f), 0.02f);
    Material smoke  = make_isotropic(vec3(0.8f, 0.8f, 0.85f), 1.f);

    vec3 boxCenter(-0.8f, 1.1f, -2.8f);
    add_box(scene, boxCenter, vec3(0.55f, 1.1f, 0.55f), 0.3f, white);
    vec3 bigCenter(0.85f, 0.7f, -1.7f);
    scene.spheres.push_back(sphere(bigCenter, 0.7f, mirror));
    vec3 smokeCenter(0.f, 2.9f, -3.0f);
    scene.spheres.push_back(sphere(smokeCenter, 0.6f, smoke));
    // earth
    vec3 earthCenter(1.3f, 2.8f, -3.0f);
    float earthRadius = 0.5f;

    int earthImageWidth = 0, earthImageHeight = 0;
    std::vector<unsigned char> earthImagePixels;
    if (!loadPpmAsRgba8("earth.ppm", earthImageWidth, earthImageHeight, earthImagePixels)) {
        std::fprintf(stderr, "could not load earth.ppm\n");
        std::exit(EXIT_FAILURE);
    }
    cudaArray_t earthImageArray = nullptr;
    cudaTextureObject_t earthTextureHandle = createImageTextureObject(earthImagePixels, earthImageWidth, earthImageHeight, earthImageArray);
    scene.imageArrays.push_back(earthImageArray);
    scene.imageTextureHandles.push_back(earthTextureHandle);

    Material earthMaterial = make_lambertian(make_image_texture(earthTextureHandle));
    scene.spheres.push_back(sphere(earthCenter, earthRadius, earthMaterial));

    srand(42);
    const float r = 0.1f;
    for (int a = 0; a < 13; a++) {
        for (int b = 0; b < 13; b++) {
            float jx = (rand() / (float)RAND_MAX - 0.5f) * 0.08f;
            float jz = (rand() / (float)RAND_MAX - 0.5f) * 0.08f;
            vec3 c(-1.8f + 0.3f * a + jx, r, -3.8f + 0.3f * b + jz);

            if ((c - bigCenter).length() < 0.7f + r + 0.02f) continue;
            if ((c - vec3(boxCenter.x, r, boxCenter.z)).length() < 0.9f) continue;

            float chooseMat = rand() / (float)RAND_MAX;

            if (chooseMat < 0.8f) {
                Material m = make_lambertian(vec3(rand()/(float)RAND_MAX, rand()/(float)RAND_MAX, rand()/(float)RAND_MAX));
                float h = 0.5f * (rand() / (float)RAND_MAX);
                scene.spheres.push_back(sphere(c, c + vec3(0.f, h, 0.f), r, m));
            } else {
                Material m;
                if (chooseMat < 0.95f) m = make_conductor(vec3(0.5f + 0.5f*(rand()/(float)RAND_MAX), 0.5f + 0.5f*(rand()/(float)RAND_MAX), 0.5f + 0.5f*(rand()/(float)RAND_MAX)), 0.3f*(rand()/(float)RAND_MAX));
                else m = make_dielectric(1.5f);
                scene.spheres.push_back(sphere(c, r, m));
            }
        }
    }

}

static bool g_environmentEnabled = false;

__host__ void buildSponzaScene(hostScene& scene){
    Material wallMaterial = make_lambertian(vec3(0.75f, 0.72f, 0.68f));
    if (!loadObject(scene, "models/sponza/sponza.obj", wallMaterial, vec3(0.f, 0.f, 0.f), 37.f, 0.f)){
        std::fprintf(stderr, "could not load models/sponza/sponza.obj\n");
        std::exit(EXIT_FAILURE);
    }

    if(!g_environmentEnabled){
        Material skylight = make_emissive(vec3(1.5f, 2.1f, 3.0f));
        const float skylightHeight = 14.6f;
        add_quad(scene, vec3(-8.8f, skylightHeight, -1.6f), vec3(8.3f, skylightHeight, -1.6f), vec3(8.3f, skylightHeight, 1.7f), vec3(-8.8f, skylightHeight, 1.7f), skylight);

        Material sun = make_emissive(vec3(320.f, 300.f, 260.f));
        const float sunHeight = 22.f;
        add_quad(scene, vec3(11.5f, sunHeight, -1.5f), vec3(14.5f, sunHeight, -1.5f), vec3(14.5f, sunHeight, 1.5f), vec3(11.5f, sunHeight, 1.5f), sun);
    }
}

// minipbrt brings a ply reader along, so a mesh can be dropped into one of the
// built-in scenes without going through a wavefront obj first. Placement matches
// loadObject: centred on x/z, lowest point at y = 0, largest side scaled to
// uniformScale, then rotated around y and translated.
static bool loadPlyObject(hostScene& scene,const char* filePath,Material mat,vec3 translation,float uniformScale,float rotationAroundYRadians){
    minipbrt::PLYMesh plyShape;
    size_t pathLength = std::strlen(filePath);
    plyShape.filename = new char[pathLength + 1];
    std::memcpy(plyShape.filename, filePath, pathLength + 1);

    minipbrt::TriangleMesh* mesh = plyShape.triangle_mesh();
    if (!mesh) return false;

    vec3 boxMin(mesh->P[0], mesh->P[1], mesh->P[2]);
    vec3 boxMax = boxMin;
    for (unsigned int vertexIndex = 1; vertexIndex < mesh->num_vertices; vertexIndex++){
        const float* position = &mesh->P[vertexIndex * 3];
        boxMin = vec3(fminf(boxMin.x, position[0]), fminf(boxMin.y, position[1]), fminf(boxMin.z, position[2]));
        boxMax = vec3(fmaxf(boxMax.x, position[0]), fmaxf(boxMax.y, position[1]), fmaxf(boxMax.z, position[2]));
    }
    vec3 modelSize = boxMax - boxMin;
    float largestSide = fmaxf(modelSize.x, fmaxf(modelSize.y, modelSize.z));
    if (largestSide <= 0.f){
        delete mesh;
        return false;
    }
    vec3 bottomCenter((boxMin.x + boxMax.x) * 0.5f, boxMin.y, (boxMin.z + boxMax.z) * 0.5f);
    float cosine = cosf(rotationAroundYRadians);
    float sine = sinf(rotationAroundYRadians);

    std::vector<vec3> transformedPositions(mesh->num_vertices);
    std::vector<vec3> transformedNormals(mesh->N ? mesh->num_vertices : 0);
    for (unsigned int vertexIndex = 0; vertexIndex < mesh->num_vertices; vertexIndex++){
        const float* position = &mesh->P[vertexIndex * 3];
        vec3 normalizedPosition = (vec3(position[0], position[1], position[2]) - bottomCenter) * (uniformScale / largestSide);
        transformedPositions[vertexIndex] = vec3(normalizedPosition.x * cosine + normalizedPosition.z * sine, normalizedPosition.y, -normalizedPosition.x * sine + normalizedPosition.z * cosine) + translation;
        if (mesh->N){
            const float* normal = &mesh->N[vertexIndex * 3];
            vec3 rotatedNormal(normal[0] * cosine + normal[2] * sine, normal[1], -normal[0] * sine + normal[2] * cosine);
            if (rotatedNormal.length_squared() > 0.f) transformedNormals[vertexIndex] = rotatedNormal.normalize();
        }
    }

    int addedTriangleCount = 0;
    int skippedFaceCount = 0;
    for (unsigned int cornerIndex = 0; cornerIndex + 2 < mesh->num_indices; cornerIndex += 3){
        int cornerIndices[3] = {mesh->indices[cornerIndex], mesh->indices[cornerIndex + 1], mesh->indices[cornerIndex + 2]};
        bool isValid = true;
        for (int corner = 0; corner < 3; corner++){
            if (cornerIndices[corner] < 0 || cornerIndices[corner] >= (int)mesh->num_vertices) isValid = false;
        }
        if (!isValid){
            skippedFaceCount++;
            continue;
        }
        vec3 vertex0 = transformedPositions[cornerIndices[0]];
        vec3 vertex1 = transformedPositions[cornerIndices[1]];
        vec3 vertex2 = transformedPositions[cornerIndices[2]];
        if ((vertex1 - vertex0).cross(vertex2 - vertex0).length() < 1e-9f){
            skippedFaceCount++;
            continue;
        }
        triangle objectTriangle(vertex0, vertex1, vertex2, mat);
        if (mesh->uv){
            vec2* uvTargets[3] = {&objectTriangle.uv0, &objectTriangle.uv1, &objectTriangle.uv2};
            for (int corner = 0; corner < 3; corner++){
                const float* uv = &mesh->uv[cornerIndices[corner] * 2];
                *uvTargets[corner] = vec2(uv[0], uv[1]);
            }
        }
        if (mesh->N){
            vec3* normalTargets[3] = {&objectTriangle.n0, &objectTriangle.n1, &objectTriangle.n2};
            for (int corner = 0; corner < 3; corner++) *normalTargets[corner] = transformedNormals[cornerIndices[corner]];
        }
        scene.triangles.push_back(objectTriangle);
        addedTriangleCount++;
    }
    delete mesh;

    printf("PLY '%s': %d triangles added, %d faces skipped\n", filePath, addedTriangleCount, skippedFaceCount);
    return addedTriangleCount > 0;
}

__host__ void buildBunnyScene(hostScene& scene){
    buildCornellRoom(scene);
    Material bunnyMaterial = make_lambertian(vec3(0.75f, 0.75f, 0.75f));
    if (!loadObject(scene, "models/bunny/bunny.obj", bunnyMaterial, vec3(0.f, 0.f, -2.f), 2.2f, 0.f)){
        std::fprintf(stderr, "could not load models/bunny/bunny.obj\n");
        std::exit(EXIT_FAILURE);
    }
}

__host__ void buildDragonScene(hostScene& scene){
    buildCornellRoom(scene);
    Material dragonMaterial = make_conductor(vec3(0.95f, 0.78f, 0.45f), 0.04f);
    if (!loadPlyObject(scene, "models/pbrt-v3-scenes/dragon/geometry/dragon_remeshed.ply", dragonMaterial, vec3(-0.1f, 0.f, -2.4f), 2.4f, MONOCUDA_PI * 0.62f)){
        std::fprintf(stderr, "could not load models/pbrt-v3-scenes/dragon/geometry/dragon_remeshed.ply\n");
        std::exit(EXIT_FAILURE);
    }
}

__host__ void buildSphereScene(hostScene& scene){
    buildCornellRoom(scene);
    Material sphereMaterial = make_lambertian(vec3(0.75f, 0.75f, 0.75f));
    if (!loadObject(scene, "models/sphere/sphere-cylcoords-16k.obj", sphereMaterial, vec3(0.f, 0.f, -2.f), 2.f, 0.f)){
        std::fprintf(stderr, "could not load models/sphere/sphere-cylcoords-16k.obj\n");
        std::exit(EXIT_FAILURE);
    }
}

__host__ void buildLivingRoomScene(hostScene& scene){
    Material fallbackMaterial = make_lambertian(vec3(0.75f, 0.72f, 0.68f));
    if (!loadObject(scene, "models/living_room/living_room.obj", fallbackMaterial, vec3(0.f, 0.f, 0.f), 8.6f, 0.f)){
        std::fprintf(stderr, "could not load models/living_room/living_room.obj\n");
        std::exit(EXIT_FAILURE);
    }
}

__host__ void buildSibenikScene(hostScene& scene){
    Material fallbackMaterial = make_lambertian(vec3(0.75f, 0.72f, 0.68f));
    if (!loadObject(scene, "models/sibenik/sibenik.obj", fallbackMaterial, vec3(0.f, 0.f, 0.f), 40.f, 0.f)){
        std::fprintf(stderr, "could not load models/sibenik/sibenik.obj\n");
        std::exit(EXIT_FAILURE);
    }
}

// bounding boxes, primitive references for the BVH and the light list (shared by every scene)
__host__ void finalizeScene(hostScene& scene){
    for (size_t i = 0; i < scene.spheres.size(); i++){
        scene.spheres[i].calculate_bounding_box();
    }
    for (size_t i = 0; i < scene.triangles.size(); i++){
        scene.triangles[i].calculate_bounding_box();
    }

    AABB sceneBounds;
    bool hasBounds = false;
    for (size_t i = 0; i < scene.spheres.size(); i++){
        sceneBounds = hasBounds ? merge(sceneBounds, scene.spheres[i].aabb) : scene.spheres[i].aabb;
        hasBounds = true;
    }
    for (size_t i = 0; i < scene.triangles.size(); i++){
        sceneBounds = hasBounds ? merge(sceneBounds, scene.triangles[i].aabb) : scene.triangles[i].aabb;
        hasBounds = true;
    }
    scene.bounds = sceneBounds;

        printf("scene bounds: min(%.2f %.2f %.2f) max(%.2f %.2f %.2f), %d triangles\n",
           sceneBounds.minInterval.x, sceneBounds.minInterval.y, sceneBounds.minInterval.z,
           sceneBounds.maxInterval.x, sceneBounds.maxInterval.y, sceneBounds.maxInterval.z,
           (int)scene.triangles.size());

    for (size_t i = 0; i < scene.spheres.size(); i++) {
        PrimitiveRef ref;
        ref.bounds = scene.spheres[i].aabb;
        ref.centroid = scene.spheres[i].calculateCentroid();
        ref.primitiveType = 0;
        ref.primitiveIndex = i;
        ref.mortonCode = morton3D(ref.centroid, scene.bounds);
        scene.primitiveRefs.push_back(ref);
    }
    for (size_t i = 0; i < scene.triangles.size(); i++) {
        PrimitiveRef ref;
        ref.bounds = scene.triangles[i].aabb;
        ref.centroid = scene.triangles[i].calculateCentroid();
        ref.primitiveType = 1;
        ref.primitiveIndex = i;
        ref.mortonCode = morton3D(ref.centroid, scene.bounds);
        scene.primitiveRefs.push_back(ref);

        if (scene.triangles[i].mat.type == MaterialType::Emissive){
            triangle& emissiveTriangle = scene.triangles[i];
            vec3 edgeOne = emissiveTriangle.v1 - emissiveTriangle.v0;
            vec3 edgeTwo = emissiveTriangle.v2 - emissiveTriangle.v0;
            vec3 areaVector = edgeOne.cross(edgeTwo);

            Light light;
            light.sourceTriangle = emissiveTriangle;
            light.surfaceArea = 0.5f * areaVector.length();
            light.surfaceNormal = areaVector.normalize();
            scene.lights.push_back(light);
        }
    }
}

// ============================================================================
// === PBRT SCENE LOADING ===
// ============================================================================
// pbrt-v3 files are parsed by minipbrt; what comes out is flattened into the
// same hostScene the built-in scenes use, geometry and camera taken as they are.
// Colors in pbrt files are already linear, so the solid textures are built
// directly instead of through make_solid_texture (which expects sRGB).
struct PbrtSceneSettings {
    bool hasCamera = false;
    vec3 lookFrom;
    vec3 lookAt;
    vec3 up;
    float verticalFovDegrees = 40.f;
    int imageWidth = 0;
    int imageHeight = 0;
    vec3 backgroundColor;
    std::string environmentPath;
    float environmentIntensity = 1.f;
    float environmentRotationDegrees = 0.f;
};
static std::string g_pbrtPath;
static PbrtSceneSettings g_pbrtSettings;
static long long g_pbrtTriangleBudget = 0;

// Instances are expanded into real triangles, so a scene that pbrt renders from a
// handful of instanced meshes can turn into hundreds of millions of them here. At
// 272 bytes per triangle on the host that runs the machine out of memory, and the
// process is killed before anything is drawn. Counting first turns that into a
// message. The default budget is what the free device memory can hold.
static long long pbrtDefaultTriangleBudget(){
    size_t freeDeviceMemory = 0, totalDeviceMemory = 0;
    if (cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory) != cudaSuccess) return 20000000ll;
    const long long bytesPerTriangleOnDevice = (long long)sizeof(triangle) + 96ll;   // triangle, bvh node and sorted reference
    return (long long)((double)freeDeviceMemory * 0.75 / (double)bytesPerTriangleOnDevice);
}

static long long pbrtCountTriangles(const minipbrt::Scene* pbrtScene){
    std::vector<long long> trianglesOfShape(pbrtScene->shapes.size(), 0);
    for (size_t shapeIndex = 0; shapeIndex < pbrtScene->shapes.size(); shapeIndex++){
        const minipbrt::Shape* shape = pbrtScene->shapes[shapeIndex];
        if (shape->type() == minipbrt::ShapeType::TriangleMesh){
            trianglesOfShape[shapeIndex] = static_cast<const minipbrt::TriangleMesh*>(shape)->num_indices / 3;
        } else if (shape->type() == minipbrt::ShapeType::Sphere){
            trianglesOfShape[shapeIndex] = 1;
        }
    }
    long long triangleCount = 0;
    for (size_t shapeIndex = 0; shapeIndex < pbrtScene->shapes.size(); shapeIndex++){
        if (pbrtScene->shapes[shapeIndex]->object == minipbrt::kInvalidIndex) triangleCount += trianglesOfShape[shapeIndex];
    }
    for (size_t instanceIndex = 0; instanceIndex < pbrtScene->instances.size(); instanceIndex++){
        const minipbrt::Instance* instance = pbrtScene->instances[instanceIndex];
        if (instance->object == minipbrt::kInvalidIndex || instance->object >= pbrtScene->objects.size()) continue;
        const minipbrt::Object* object = pbrtScene->objects[instance->object];
        for (unsigned int shapeOffset = 0; shapeOffset < object->numShapes; shapeOffset++){
            uint32_t shapeIndex = object->firstShape + shapeOffset;
            if (shapeIndex < trianglesOfShape.size()) triangleCount += trianglesOfShape[shapeIndex];
        }
    }
    return triangleCount;
}

static void pbrtMultiply(const float a[4][4], const float b[4][4], float out[4][4]){
    for (int row = 0; row < 4; row++){
        for (int column = 0; column < 4; column++){
            out[row][column] = a[row][0] * b[0][column] + a[row][1] * b[1][column] + a[row][2] * b[2][column] + a[row][3] * b[3][column];
        }
    }
}

static vec3 pbrtTransformPoint(const float m[4][4], const vec3& p){
    return vec3(m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
                m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
                m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]);
}

static vec3 pbrtTransformVector(const float m[4][4], const vec3& v){
    return vec3(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
}

static Texture pbrtLinearSolidTexture(const vec3& color){
    return Texture{TextureType::Solid, color, vec3(0,0,0), 1.f, 0};
}

static std::string pbrtResolvePath(const std::string& baseDirectory, const char* fileName){
    if (!fileName) return "";
    std::string path = fileName;
    if (!path.empty() && path[0] == '/') return path;
    FILE* probe = fopen(path.c_str(), "rb");
    if (probe){
        fclose(probe);
        return path;
    }
    return baseDirectory + "/" + path;
}

static Texture pbrtColorTexture(const minipbrt::Scene* pbrtScene, const minipbrt::ColorTex& colorTex, const std::string& baseDirectory, hostScene& scene, std::map<std::string,cudaTextureObject_t>& textureCache, int& unsupportedTextureCount){
    vec3 color(colorTex.value[0], colorTex.value[1], colorTex.value[2]);
    if (colorTex.texture != minipbrt::kInvalidIndex && colorTex.texture < pbrtScene->textures.size()){
        const minipbrt::Texture* texture = pbrtScene->textures[colorTex.texture];
        if (texture->type() == minipbrt::TextureType::ImageMap){
            const minipbrt::ImageMapTexture* imageMap = static_cast<const minipbrt::ImageMapTexture*>(texture);
            std::string imagePath = pbrtResolvePath(baseDirectory, imageMap->filename);
            cudaTextureObject_t textureHandle = getOrLoadObjTexture(scene, imagePath, imagePath, textureCache);
            if (textureHandle != 0) return make_image_texture(textureHandle);
        } else if (texture->type() == minipbrt::TextureType::Constant){
            const minipbrt::ConstantTexture* constant = static_cast<const minipbrt::ConstantTexture*>(texture);
            color = vec3(constant->value[0], constant->value[1], constant->value[2]);
        } else if (texture->type() == minipbrt::TextureType::Scale){
            const minipbrt::ScaleTexture* scaleTexture = static_cast<const minipbrt::ScaleTexture*>(texture);
            return pbrtColorTexture(pbrtScene, scaleTexture->tex1, baseDirectory, scene, textureCache, unsupportedTextureCount);
        } else {
            unsupportedTextureCount++;
        }
    }
    return pbrtLinearSolidTexture(color);
}

// reflectance of a conductor at normal incidence from its complex index of refraction
static vec3 pbrtConductorColor(const float eta[3], const float k[3]){
    vec3 color;
    float* channels[3] = {&color.x, &color.y, &color.z};
    for (int channel = 0; channel < 3; channel++){
        float etaMinus = eta[channel] - 1.0f;
        float etaPlus = eta[channel] + 1.0f;
        float kSquared = k[channel] * k[channel];
        *channels[channel] = (etaMinus * etaMinus + kSquared) / (etaPlus * etaPlus + kSquared);
    }
    return color;
}

static Material pbrtMaterial(const minipbrt::Scene* pbrtScene, uint32_t materialIndex, const std::string& baseDirectory, hostScene& scene, std::map<std::string,cudaTextureObject_t>& textureCache, int& unsupportedTextureCount, std::set<std::string>& approximatedMaterials){
    Material fallback = Material{MaterialType::Lambertian, pbrtLinearSolidTexture(vec3(0.5f, 0.5f, 0.5f)), vec3(0,0,0), 0.f, 0.f, 0.f};
    if (materialIndex == minipbrt::kInvalidIndex || materialIndex >= pbrtScene->materials.size()) return fallback;

    const minipbrt::Material* material = pbrtScene->materials[materialIndex];
    switch (material->type()){
        case minipbrt::MaterialType::Matte: {
            const minipbrt::MatteMaterial* matte = static_cast<const minipbrt::MatteMaterial*>(material);
            return make_lambertian(pbrtColorTexture(pbrtScene, matte->Kd, baseDirectory, scene, textureCache, unsupportedTextureCount));
        }
        case minipbrt::MaterialType::Plastic: {
            const minipbrt::PlasticMaterial* plastic = static_cast<const minipbrt::PlasticMaterial*>(material);
            approximatedMaterials.insert("plastic");
            return make_lambertian(pbrtColorTexture(pbrtScene, plastic->Kd, baseDirectory, scene, textureCache, unsupportedTextureCount));
        }
        case minipbrt::MaterialType::Substrate: {
            const minipbrt::SubstrateMaterial* substrate = static_cast<const minipbrt::SubstrateMaterial*>(material);
            approximatedMaterials.insert("substrate");
            return make_lambertian(pbrtColorTexture(pbrtScene, substrate->Kd, baseDirectory, scene, textureCache, unsupportedTextureCount));
        }
        case minipbrt::MaterialType::Uber: {
            const minipbrt::UberMaterial* uber = static_cast<const minipbrt::UberMaterial*>(material);
            approximatedMaterials.insert("uber");
            return make_lambertian(pbrtColorTexture(pbrtScene, uber->Kd, baseDirectory, scene, textureCache, unsupportedTextureCount));
        }
        case minipbrt::MaterialType::Disney: {
            const minipbrt::DisneyMaterial* disney = static_cast<const minipbrt::DisneyMaterial*>(material);
            approximatedMaterials.insert("disney");
            return make_lambertian(pbrtColorTexture(pbrtScene, disney->color, baseDirectory, scene, textureCache, unsupportedTextureCount));
        }
        case minipbrt::MaterialType::Glass: {
            const minipbrt::GlassMaterial* glass = static_cast<const minipbrt::GlassMaterial*>(material);
            return make_dielectric(glass->eta.value);
        }
        case minipbrt::MaterialType::Mirror: {
            const minipbrt::MirrorMaterial* mirror = static_cast<const minipbrt::MirrorMaterial*>(material);
            return Material{MaterialType::Conductor, pbrtLinearSolidTexture(vec3(mirror->Kr.value[0], mirror->Kr.value[1], mirror->Kr.value[2])), vec3(0,0,0), 0.f, 0.f, 0.f};
        }
        case minipbrt::MaterialType::Metal: {
            const minipbrt::MetalMaterial* metal = static_cast<const minipbrt::MetalMaterial*>(material);
            float fuzz = fminf(fmaxf(metal->uroughness.value, 0.f), 1.f);
            return Material{MaterialType::Conductor, pbrtLinearSolidTexture(pbrtConductorColor(metal->eta.value, metal->k.value)), vec3(0,0,0), fuzz, 0.f, 0.f};
        }
        case minipbrt::MaterialType::Mix: {
            const minipbrt::MixMaterial* mix = static_cast<const minipbrt::MixMaterial*>(material);
            approximatedMaterials.insert("mix");
            return pbrtMaterial(pbrtScene, mix->namedmaterial1, baseDirectory, scene, textureCache, unsupportedTextureCount, approximatedMaterials);
        }
        default:
            approximatedMaterials.insert(material->name ? material->name : "unnamed");
            return fallback;
    }
}

static Material pbrtMediumMaterial(const minipbrt::Medium* medium){
    vec3 albedo;
    float density = 0.f;
    float* albedoChannels[3] = {&albedo.x, &albedo.y, &albedo.z};
    for (int channel = 0; channel < 3; channel++){
        float extinction = (medium->sigma_a[channel] + medium->sigma_s[channel]) * medium->scale;
        *albedoChannels[channel] = extinction > 0.f ? medium->sigma_s[channel] * medium->scale / extinction : 0.f;
        density += extinction / 3.f;
    }
    return Material{MaterialType::Isotropic, pbrtLinearSolidTexture(albedo), vec3(0,0,0), 0.f, 0.f, density};
}

static int pbrtAddMesh(hostScene& scene, const minipbrt::TriangleMesh* mesh, const float toWorld[4][4], const Material& material){
    int addedTriangleCount = 0;
    for (unsigned int triangleIndex = 0; triangleIndex + 2 < mesh->num_indices; triangleIndex += 3){
        int cornerIndices[3] = {mesh->indices[triangleIndex], mesh->indices[triangleIndex + 1], mesh->indices[triangleIndex + 2]};
        bool isValid = true;
        for (int corner = 0; corner < 3; corner++){
            if (cornerIndices[corner] < 0 || cornerIndices[corner] >= (int)mesh->num_vertices) isValid = false;
        }
        if (!isValid) continue;

        vec3 corners[3];
        for (int corner = 0; corner < 3; corner++){
            const float* position = &mesh->P[cornerIndices[corner] * 3];
            corners[corner] = (pbrtTransformPoint(toWorld, vec3(position[0], position[1], position[2])));
        }
        if ((corners[1] - corners[0]).cross(corners[2] - corners[0]).length_squared() < 1e-24f) continue;

        triangle objectTriangle(corners[0], corners[1], corners[2], material);
        if (mesh->uv){
            vec2* uvTargets[3] = {&objectTriangle.uv0, &objectTriangle.uv1, &objectTriangle.uv2};
            for (int corner = 0; corner < 3; corner++){
                const float* uv = &mesh->uv[cornerIndices[corner] * 2];
                *uvTargets[corner] = vec2(uv[0], uv[1]);
            }
        }
        if (mesh->N){
            vec3* normalTargets[3] = {&objectTriangle.n0, &objectTriangle.n1, &objectTriangle.n2};
            for (int corner = 0; corner < 3; corner++){
                const float* normal = &mesh->N[cornerIndices[corner] * 3];
                vec3 worldNormal = (pbrtTransformVector(toWorld, vec3(normal[0], normal[1], normal[2])));
                if (worldNormal.length_squared() > 0.f) *normalTargets[corner] = worldNormal.normalize();
            }
        }
        scene.triangles.push_back(objectTriangle);
        addedTriangleCount++;
    }
    return addedTriangleCount;
}

// pbrt keeps spheres as quadrics and minipbrt does not triangulate them. A sphere
// that emits has to become triangles so finalizeScene() picks it up as an area
// light, every other one goes in as the renderer's own sphere primitive.
static int pbrtAddTessellatedSphere(hostScene& scene, const vec3& center, float radius, const Material& material){
    const int meridianCount = 32;
    const int parallelCount = 16;
    int addedTriangleCount = 0;
    for (int parallel = 0; parallel < parallelCount; parallel++){
        float theta0 = MONOCUDA_PI * parallel / parallelCount;
        float theta1 = MONOCUDA_PI * (parallel + 1) / parallelCount;
        for (int meridian = 0; meridian < meridianCount; meridian++){
            float phi0 = 2.f * MONOCUDA_PI * meridian / meridianCount;
            float phi1 = 2.f * MONOCUDA_PI * (meridian + 1) / meridianCount;
            vec3 corners[4];
            float thetas[4] = {theta0, theta1, theta1, theta0};
            float phis[4] = {phi0, phi0, phi1, phi1};
            for (int corner = 0; corner < 4; corner++){
                corners[corner] = center + vec3(sinf(thetas[corner]) * cosf(phis[corner]), cosf(thetas[corner]), sinf(thetas[corner]) * sinf(phis[corner])) * radius;
            }
            int cornerIndices[2][3] = {{0, 1, 2}, {0, 2, 3}};
            for (int triangleIndex = 0; triangleIndex < 2; triangleIndex++){
                vec3 v0 = corners[cornerIndices[triangleIndex][0]];
                vec3 v1 = corners[cornerIndices[triangleIndex][1]];
                vec3 v2 = corners[cornerIndices[triangleIndex][2]];
                if ((v1 - v0).cross(v2 - v0).length_squared() < 1e-24f) continue;
                scene.triangles.push_back(triangle(v0, v1, v2, material));
                addedTriangleCount++;
            }
        }
    }
    return addedTriangleCount;
}

// shape, area light and medium of one shape placed with a given transform
static int pbrtAddShape(hostScene& scene, const minipbrt::Scene* pbrtScene, const minipbrt::Shape* shape, const float toWorld[4][4], uint32_t areaLightOverride,
                        const std::string& baseDirectory, std::map<std::string,cudaTextureObject_t>& textureCache, std::map<uint32_t,Material>& materialCache,
                        int& unsupportedTextureCount, std::set<std::string>& approximatedMaterials, int& skippedShapeCount){
    if (shape->type() != minipbrt::ShapeType::TriangleMesh && shape->type() != minipbrt::ShapeType::Sphere){
        skippedShapeCount++;
        return 0;
    }

    Material material;
    uint32_t areaLightIndex = (areaLightOverride != minipbrt::kInvalidIndex) ? areaLightOverride : shape->areaLight;
    if (areaLightIndex != minipbrt::kInvalidIndex && areaLightIndex < pbrtScene->areaLights.size()){
        const minipbrt::AreaLight* areaLight = pbrtScene->areaLights[areaLightIndex];
        const minipbrt::DiffuseAreaLight* diffuse = static_cast<const minipbrt::DiffuseAreaLight*>(areaLight);
        material = make_emissive(vec3(diffuse->L[0] * diffuse->scale[0], diffuse->L[1] * diffuse->scale[1], diffuse->L[2] * diffuse->scale[2]));
    } else if (shape->insideMedium != minipbrt::kInvalidIndex && shape->insideMedium < pbrtScene->mediums.size()){
        material = pbrtMediumMaterial(pbrtScene->mediums[shape->insideMedium]);
    } else {
        bool isInvisibleInterface = shape->material != minipbrt::kInvalidIndex && shape->material < pbrtScene->materials.size()
                                 && pbrtScene->materials[shape->material]->type() == minipbrt::MaterialType::None;
        if (isInvisibleInterface){
            skippedShapeCount++;
            return 0;
        }
        std::map<uint32_t,Material>::iterator cached = materialCache.find(shape->material);
        if (cached == materialCache.end()){
            material = pbrtMaterial(pbrtScene, shape->material, baseDirectory, scene, textureCache, unsupportedTextureCount, approximatedMaterials);
            materialCache[shape->material] = material;
        } else {
            material = cached->second;
        }
    }

    if (shape->type() == minipbrt::ShapeType::Sphere){
        const minipbrt::Sphere* pbrtSphere = static_cast<const minipbrt::Sphere*>(shape);
        vec3 center = pbrtTransformPoint(toWorld, vec3(0.f, 0.f, 0.f));
        float axisLengths[3];
        for (int axis = 0; axis < 3; axis++){
            vec3 unitAxis(axis == 0 ? 1.f : 0.f, axis == 1 ? 1.f : 0.f, axis == 2 ? 1.f : 0.f);
            axisLengths[axis] = pbrtTransformVector(toWorld, unitAxis).length();
        }
        float radius = pbrtSphere->radius * (axisLengths[0] + axisLengths[1] + axisLengths[2]) / 3.f;
        if (radius <= 0.f){
            skippedShapeCount++;
            return 0;
        }
        if (material.type == MaterialType::Emissive) return pbrtAddTessellatedSphere(scene, center, radius, material);
        scene.spheres.push_back(sphere(center, radius, material));
        return 1;
    }

    const minipbrt::TriangleMesh* mesh = static_cast<const minipbrt::TriangleMesh*>(shape);
    return pbrtAddMesh(scene, mesh, toWorld, material);
}

static bool loadPbrtScene(hostScene& scene, const char* filePath){
    minipbrt::Loader loader;
    if (!loader.load(filePath)){
        const minipbrt::Error* error = loader.error();
        std::fprintf(stderr, "pbrt '%s': %s", filePath, error ? error->message() : "parse error");
        if (error && error->has_line_and_column()) std::fprintf(stderr, " (line %lld, column %lld)", (long long)error->line(), (long long)error->column());
        std::fprintf(stderr, "\n");
        return false;
    }
    minipbrt::Scene* pbrtScene = loader.borrow_scene();
    pbrtScene->load_all_ply_meshes(false);
    pbrtScene->all_to_triangle_mesh(false);

    long long triangleBudget = g_pbrtTriangleBudget > 0 ? g_pbrtTriangleBudget : pbrtDefaultTriangleBudget();
    long long triangleCount = pbrtCountTriangles(pbrtScene);
    if (triangleCount > triangleBudget){
        std::fprintf(stderr, "pbrt '%s': the scene expands to %lld triangles, more than the %lld that fit (instances are flattened, %zu bytes each).\n",
                     filePath, triangleCount, triangleBudget, sizeof(triangle));
        std::fprintf(stderr, "  raise it with --max-triangles N if there is memory for it, or render a variant of the scene with less instanced geometry.\n");
        return false;
    }

    std::string baseDirectory = getDirectoryOfFile(filePath);
    std::map<std::string,cudaTextureObject_t> textureCache;
    std::map<uint32_t,Material> materialCache;
    std::set<std::string> approximatedMaterials;
    int unsupportedTextureCount = 0;
    int skippedShapeCount = 0;
    int addedTriangleCount = 0;
    int instanceCount = 0;

    for (size_t shapeIndex = 0; shapeIndex < pbrtScene->shapes.size(); shapeIndex++){
        const minipbrt::Shape* shape = pbrtScene->shapes[shapeIndex];
        if (shape->object != minipbrt::kInvalidIndex) continue;
        addedTriangleCount += pbrtAddShape(scene, pbrtScene, shape, shape->shapeToWorld.start, minipbrt::kInvalidIndex, baseDirectory, textureCache, materialCache, unsupportedTextureCount, approximatedMaterials, skippedShapeCount);
    }
    for (size_t instanceIndex = 0; instanceIndex < pbrtScene->instances.size(); instanceIndex++){
        const minipbrt::Instance* instance = pbrtScene->instances[instanceIndex];
        if (instance->object == minipbrt::kInvalidIndex || instance->object >= pbrtScene->objects.size()) continue;
        const minipbrt::Object* object = pbrtScene->objects[instance->object];
        float objectToWorld[4][4];
        pbrtMultiply(instance->instanceToWorld.start, object->objectToInstance.start, objectToWorld);
        instanceCount++;
        for (unsigned int shapeOffset = 0; shapeOffset < object->numShapes; shapeOffset++){
            uint32_t shapeIndex = object->firstShape + shapeOffset;
            if (shapeIndex >= pbrtScene->shapes.size()) break;
            const minipbrt::Shape* shape = pbrtScene->shapes[shapeIndex];
            float toWorld[4][4];
            pbrtMultiply(objectToWorld, shape->shapeToWorld.start, toWorld);
            addedTriangleCount += pbrtAddShape(scene, pbrtScene, shape, toWorld, instance->areaLight, baseDirectory, textureCache, materialCache, unsupportedTextureCount, approximatedMaterials, skippedShapeCount);
        }
    }

    // the lights that are not geometry need the extent of what was loaded
    vec3 sceneMinimum(1e30f, 1e30f, 1e30f);
    vec3 sceneMaximum(-1e30f, -1e30f, -1e30f);
    for (size_t triangleIndex = 0; triangleIndex < scene.triangles.size(); triangleIndex++){
        const vec3* corners[3] = {&scene.triangles[triangleIndex].v0, &scene.triangles[triangleIndex].v1, &scene.triangles[triangleIndex].v2};
        for (int corner = 0; corner < 3; corner++){
            sceneMinimum = vec3(fminf(sceneMinimum.x, corners[corner]->x), fminf(sceneMinimum.y, corners[corner]->y), fminf(sceneMinimum.z, corners[corner]->z));
            sceneMaximum = vec3(fmaxf(sceneMaximum.x, corners[corner]->x), fmaxf(sceneMaximum.y, corners[corner]->y), fmaxf(sceneMaximum.z, corners[corner]->z));
        }
    }
    vec3 sceneCenter = (sceneMinimum + sceneMaximum) * 0.5f;
    float sceneRadius = fmaxf((sceneMaximum - sceneMinimum).length() * 0.5f, 1e-3f);

    int approximatedLightCount = 0;
    for (size_t lightIndex = 0; lightIndex < pbrtScene->lights.size(); lightIndex++){
        const minipbrt::Light* light = pbrtScene->lights[lightIndex];
        switch (light->type()){
            case minipbrt::LightType::Infinite: {
                const minipbrt::InfiniteLight* infinite = static_cast<const minipbrt::InfiniteLight*>(light);
                vec3 radiance(infinite->L[0] * light->scale[0], infinite->L[1] * light->scale[1], infinite->L[2] * light->scale[2]);
                if (infinite->mapname){
                    g_pbrtSettings.environmentPath = pbrtResolvePath(baseDirectory, infinite->mapname);
                    g_pbrtSettings.environmentIntensity = light->scale[0];
                    // pbrt keeps +z up in light space and puts phi = 90 degrees (the light's +y axis) at u = 0.25;
                    // find where that axis ends up in the world and turn the map so the same direction reads u = 0.25
                    vec3 lightAxisY = (pbrtTransformVector(light->lightToWorld.start, vec3(0.f, 1.f, 0.f)));
                    float rotationRadians = -0.5f * MONOCUDA_PI - atan2f(lightAxisY.z, lightAxisY.x);
                    g_pbrtSettings.environmentRotationDegrees = rotationRadians * 180.f / MONOCUDA_PI;
                } else {
                    g_pbrtSettings.backgroundColor = radiance;
                }
                break;
            }
            case minipbrt::LightType::Distant: {
                const minipbrt::DistantLight* distant = static_cast<const minipbrt::DistantLight*>(light);
                vec3 towardsLight = vec3(distant->from[0] - distant->to[0], distant->from[1] - distant->to[1], distant->from[2] - distant->to[2]);
                towardsLight = (pbrtTransformVector(light->lightToWorld.start, towardsLight)).normalize();
                const float halfAngleTangent = tanf(1.0f * MONOCUDA_PI / 180.f);
                float distance = sceneRadius * 10.f;
                float halfSize = distance * halfAngleTangent;
                vec3 radiance(distant->L[0] * light->scale[0], distant->L[1] * light->scale[1], distant->L[2] * light->scale[2]);
                radiance = radiance / (4.f * halfAngleTangent * halfAngleTangent);
                vec3 helper = fabsf(towardsLight.y) < 0.9f ? vec3(0.f, 1.f, 0.f) : vec3(1.f, 0.f, 0.f);
                vec3 sideU = helper.cross(towardsLight).normalize() * halfSize;
                vec3 sideV = towardsLight.cross(sideU).normalize() * halfSize;
                vec3 center = sceneCenter + towardsLight * distance;
                add_quad(scene, center - sideU - sideV, center + sideU - sideV, center + sideU + sideV, center - sideU + sideV, make_emissive(radiance));
                approximatedLightCount++;
                break;
            }
            case minipbrt::LightType::Point:
            case minipbrt::LightType::Spot: {
                const float* from = (light->type() == minipbrt::LightType::Point) ? static_cast<const minipbrt::PointLight*>(light)->from : static_cast<const minipbrt::SpotLight*>(light)->from;
                const float* intensity = (light->type() == minipbrt::LightType::Point) ? static_cast<const minipbrt::PointLight*>(light)->I : static_cast<const minipbrt::SpotLight*>(light)->I;
                vec3 position = (pbrtTransformPoint(light->lightToWorld.start, vec3(from[0], from[1], from[2])));
                float halfSize = sceneRadius * 0.005f;
                vec3 radiance(intensity[0] * light->scale[0], intensity[1] * light->scale[1], intensity[2] * light->scale[2]);
                radiance = radiance / (6.f * halfSize * halfSize);
                add_box(scene, position, vec3(halfSize, halfSize, halfSize), 0.f, make_emissive(radiance));
                approximatedLightCount++;
                break;
            }
            default:
                approximatedLightCount++;
                break;
        }
    }

    if (pbrtScene->film && pbrtScene->film->type() == minipbrt::FilmType::Image){
        pbrtScene->film->get_resolution(g_pbrtSettings.imageWidth, g_pbrtSettings.imageHeight);
    }
    if (pbrtScene->camera){
        const float (*toWorld)[4] = pbrtScene->camera->cameraToWorld.start;
        g_pbrtSettings.lookFrom = (pbrtTransformPoint(toWorld, vec3(0.f, 0.f, 0.f)));
        vec3 forward = (pbrtTransformVector(toWorld, vec3(0.f, 0.f, 1.f))).normalize();
        g_pbrtSettings.lookAt = g_pbrtSettings.lookFrom + forward;
        g_pbrtSettings.up = (pbrtTransformVector(toWorld, vec3(0.f, 1.f, 0.f))).normalize();
        g_pbrtSettings.hasCamera = true;
        float fov = 90.f;
        if (pbrtScene->camera->type() == minipbrt::CameraType::Perspective){
            fov = static_cast<const minipbrt::PerspectiveCamera*>(pbrtScene->camera)->fov;
        }
        // pbrt's fov belongs to the shorter image side, this camera wants the vertical one
        float width = (float)(g_pbrtSettings.imageWidth > 0 ? g_pbrtSettings.imageWidth : 1);
        float height = (float)(g_pbrtSettings.imageHeight > 0 ? g_pbrtSettings.imageHeight : 1);
        if (width >= height){
            g_pbrtSettings.verticalFovDegrees = fov;
        } else {
            g_pbrtSettings.verticalFovDegrees = 2.f * atanf(tanf(fov * 0.5f * MONOCUDA_PI / 180.f) * height / width) * 180.f / MONOCUDA_PI;
        }
    }

    printf("pbrt '%s': %d triangles added, %zu shapes, %d instances, %zu materials, %d skipped shapes, %d approximated lights, %d unsupported textures\n",
           filePath, addedTriangleCount, pbrtScene->shapes.size(), instanceCount, pbrtScene->materials.size(), skippedShapeCount, approximatedLightCount, unsupportedTextureCount);
    if (!approximatedMaterials.empty()){
        printf("pbrt '%s': rendered as lambertian or first component:", filePath);
        for (std::set<std::string>::iterator name = approximatedMaterials.begin(); name != approximatedMaterials.end(); ++name) printf(" %s", name->c_str());
        printf("\n");
    }
    return addedTriangleCount > 0;
}

__host__ void buildPbrtScene(hostScene& scene){
    if (g_pbrtPath.empty()){
        std::fprintf(stderr, "the pbrt scene needs a file: --pbrt path/to/scene.pbrt\n");
        std::exit(EXIT_FAILURE);
    }
    if (!loadPbrtScene(scene, g_pbrtPath.c_str())){
        std::fprintf(stderr, "could not load %s\n", g_pbrtPath.c_str());
        std::exit(EXIT_FAILURE);
    }
}
// ============================================================================

__host__ hostScene setup_scene(SceneChoice sceneChoice){
    hostScene scene;
    switch (sceneChoice){
        case SceneChoice::Sponza:     buildSponzaScene(scene);     break;
        case SceneChoice::Bunny:      buildBunnyScene(scene);      break;
        case SceneChoice::Dragon:     buildDragonScene(scene);     break;
        case SceneChoice::LivingRoom: buildLivingRoomScene(scene); break;
        case SceneChoice::Sibenik:    buildSibenikScene(scene);    break;
        case SceneChoice::Sphere:     buildSphereScene(scene);     break;
        case SceneChoice::Barcelona:
        case SceneChoice::Bathroom:
        case SceneChoice::Book:
        case SceneChoice::SanMiguel:
        case SceneChoice::Pbrt:       buildPbrtScene(scene);       break;
        case SceneChoice::Cornell:
        default:                      buildCornellScene(scene);    break;
    }
    finalizeScene(scene);
    return scene;
}
// ============================================================================

// ============================================================================
// === OBJECT LOADING ===
// ============================================================================
struct ObjVertexReference {
    int positionIndex;   // -1 = missing
    int uvIndex;         // -1 = missing
    int normalIndex;
};

struct ObjTriangleCorners {
    ObjVertexReference corner[3];
    int materialIndex = -1;    // index into the material library, -1 = use the default material
};

struct ObjMaterialLibrary {
    std::vector<Material> materials;
    std::map<std::string,int> indexByName;
};

// obj indices start from 1 and negative ones count from the end of the list
static int resolveObjIndex(int rawIndex,int listSize){
    if (rawIndex > 0) return rawIndex - 1;
    if (rawIndex < 0) return listSize + rawIndex;
    return -1;
}

// token examples: "7", "7/3", "7/3/5", "7//5"
static ObjVertexReference parseObjFaceToken(const std::string& token,int positionCount,int uvCount,int normalCount){
    int rawIndices[3] = {0, 0, 0};
    size_t startOfPiece = 0;
    for (int partIndex = 0; partIndex < 3; partIndex++){
        size_t slashPosition = token.find('/', startOfPiece);
        size_t pieceLength = (slashPosition == std::string::npos) ? std::string::npos : slashPosition - startOfPiece;
        std::string piece = token.substr(startOfPiece, pieceLength);
        if (!piece.empty()){
            rawIndices[partIndex] = std::atoi(piece.c_str());
        }
        if (slashPosition == std::string::npos) break;
        startOfPiece = slashPosition + 1;
    }

    ObjVertexReference reference;
    reference.positionIndex = resolveObjIndex(rawIndices[0], positionCount);
    reference.uvIndex = resolveObjIndex(rawIndices[1], uvCount);
    reference.normalIndex = resolveObjIndex(rawIndices[2], normalCount);
    return reference;
}

static std::string getDirectoryOfFile(const std::string& filePath){
    size_t lastSlashPosition = filePath.find_last_of("/\\");
    if (lastSlashPosition == std::string::npos) return ".";
    return filePath.substr(0, lastSlashPosition);
}

// textures are converted to ppm offline, "textures\wall.png" becomes "textures/wall.ppm"
static std::string convertTexturePathToPpm(std::string texturePath){
    for (size_t characterIndex = 0; characterIndex < texturePath.size(); characterIndex++){
        if (texturePath[characterIndex] == '\\') texturePath[characterIndex] = '/';
    }
    size_t lastDotPosition = texturePath.find_last_of('.');
    size_t lastSlashPosition = texturePath.find_last_of('/');
    bool dotBelongsToFileName = (lastDotPosition != std::string::npos) && (lastSlashPosition == std::string::npos || lastDotPosition > lastSlashPosition);
    if (dotBelongsToFileName){
        texturePath = texturePath.substr(0, lastDotPosition);
    }
    return texturePath + ".ppm";
}

// every ppm is uploaded to the gpu only once even if many materials use it, returns 0 if it could not be loaded
static cudaTextureObject_t getOrLoadObjTexture(hostScene& scene,const std::string& ppmPath,const std::string& originalPath,std::map<std::string,cudaTextureObject_t>& textureCache){
    std::map<std::string,cudaTextureObject_t>::iterator cachedEntry = textureCache.find(ppmPath);
    if (cachedEntry != textureCache.end()) return cachedEntry->second;

    cudaTextureObject_t textureHandle = 0;
    int imageWidth = 0, imageHeight = 0;
    std::vector<unsigned char> imagePixels;
    if (loadPpmAsRgba8(ppmPath.c_str(), imageWidth, imageHeight, imagePixels) || loadImageRgba8(originalPath.c_str(), imageWidth, imageHeight, imagePixels)){
        cudaArray_t imageArray = nullptr;
        textureHandle = createImageTextureObject(imagePixels, imageWidth, imageHeight, imageArray);
        scene.imageArrays.push_back(imageArray);
        scene.imageTextureHandles.push_back(textureHandle);
    } else {
        printf("could not load texture '%s'\n", originalPath.c_str());
    }
    textureCache[ppmPath] = textureHandle;
    return textureHandle;
}

// reads newmtl / Kd / map_Kd / Ke, every material becomes a lambertian (flat color or image texture)
// unless it has a non-zero Ke, then it becomes an emissive. Ka, Ks, bump maps and alpha masks
// are ignored, so are map_Kd options like "-s 2 2 2"
static bool loadMtlFile(hostScene& scene,const std::string& mtlPath,const std::string& objectDirectory,ObjMaterialLibrary& library,std::map<std::string,cudaTextureObject_t>& textureCache,int& missingTextureCount){
    std::ifstream file(mtlPath);
    if (!file.is_open()) return false;

    std::string currentName;
    vec3 currentColor(0.8f, 0.8f, 0.8f);
    vec3 currentEmission(0.f, 0.f, 0.f);
    std::string currentTexturePath;
    bool hasCurrentMaterial = false;

    // pushes the material that was being read, called when the next one starts and at the end of the file
    auto finishCurrentMaterial = [&](){
        if (!hasCurrentMaterial) return;
        Material finishedMaterial = make_lambertian(currentColor);
        bool emitsLight = fmaxf(currentEmission.x, fmaxf(currentEmission.y, currentEmission.z)) > 0.f;
        if (emitsLight){
            finishedMaterial = make_emissive(currentEmission);
        } else if (!currentTexturePath.empty()){
            std::string ppmPath = objectDirectory + "/" + convertTexturePathToPpm(currentTexturePath);
            std::string originalPath = currentTexturePath;
            for (size_t characterIndex = 0; characterIndex < originalPath.size(); characterIndex++){
                if (originalPath[characterIndex] == '\\') originalPath[characterIndex] = '/';
            }
            originalPath = objectDirectory + "/" + originalPath;
            cudaTextureObject_t textureHandle = getOrLoadObjTexture(scene, ppmPath, originalPath, textureCache);
            if (textureHandle != 0){
                finishedMaterial = make_lambertian(make_image_texture(textureHandle));
            } else {
                missingTextureCount++;
            }
        }
        library.indexByName[currentName] = (int)library.materials.size();
        library.materials.push_back(finishedMaterial);
        hasCurrentMaterial = false;
    };

    std::string line;
    while (std::getline(file, line)){
        std::istringstream lineStream(line);
        std::string keyword;
        lineStream >> keyword;

        if (keyword == "newmtl"){
            finishCurrentMaterial();
            lineStream >> currentName;
            currentColor = vec3(0.8f, 0.8f, 0.8f);
            currentEmission = vec3(0.f, 0.f, 0.f);
            currentTexturePath.clear();
            hasCurrentMaterial = true;
        } else if (keyword == "Kd" && hasCurrentMaterial){
            float red, green, blue;
            lineStream >> red >> green >> blue;
            currentColor = vec3(red, green, blue);
        } else if (keyword == "Ke" && hasCurrentMaterial){
            float red, green, blue;
            lineStream >> red >> green >> blue;
            currentEmission = vec3(red, green, blue);
        } else if (keyword == "map_Kd" && hasCurrentMaterial){
            lineStream >> currentTexturePath;
        }
    }
    finishCurrentMaterial();
    return true;
}

// model is centered on x/z, its lowest point sits at y = 0 and its largest side becomes 1
// then uniformScale, rotation around Y and translation (where the bottom center ends up) are applied
// mat is only used for faces that have no material from the .mtl file
static bool loadObject(hostScene& scene,const char* filePath,Material mat,vec3 translation,float uniformScale,float rotationAroundYRadians){
    std::ifstream file(filePath);
    if (!file.is_open()) return false;

    std::string objectDirectory = getDirectoryOfFile(filePath);

    std::vector<vec3> positions;
    std::vector<vec2> uvs;
    std::vector<vec3> normals;
    std::vector<ObjTriangleCorners> faces;

    ObjMaterialLibrary materialLibrary;
    std::map<std::string,cudaTextureObject_t> textureCache;
    int missingTextureCount = 0;
    int currentMaterialIndex = -1;
    std::set<std::string> unknownMaterialNames;

    std::string line;
    while (std::getline(file, line)){
        std::istringstream lineStream(line);
        std::string keyword;
        lineStream >> keyword;

        if (keyword == "v"){
            float x, y, z;
            lineStream >> x >> y >> z;
            positions.push_back(vec3(x, y, z));
        } else if (keyword == "vt"){
            float u, v;
            lineStream >> u >> v;
            uvs.push_back(vec2(u, v));
        } else if (keyword == "vn"){
            float x, y, z;
            lineStream >> x >> y >> z;
            normals.push_back(vec3(x, y, z));
        } else if (keyword == "mtllib"){
            std::string mtlFileName;
            while (lineStream >> mtlFileName){
                if (!loadMtlFile(scene, objectDirectory + "/" + mtlFileName, objectDirectory, materialLibrary, textureCache, missingTextureCount)){
                    printf("OBJ '%s': could not open material library '%s'\n", filePath, mtlFileName.c_str());
                }
            }
        } else if (keyword == "usemtl"){
            std::string materialName;
            lineStream >> materialName;
            std::map<std::string,int>::iterator foundMaterial = materialLibrary.indexByName.find(materialName);
            if (foundMaterial != materialLibrary.indexByName.end()){
                currentMaterialIndex = foundMaterial->second;
            } else {
                currentMaterialIndex = (materialLibrary.materials.size() == 1) ? 0 : -1;
                if (unknownMaterialNames.insert(materialName).second){
                    printf("OBJ '%s': material '%s' is not defined in the .mtl file, %s\n", filePath, materialName.c_str(),
                           currentMaterialIndex >= 0 ? "using the only material it defines" : "falling back to the scene material");
                }
            }
        } else if (keyword == "f"){
            std::vector<ObjVertexReference> corners;
            std::string token;
            while (lineStream >> token){
                corners.push_back(parseObjFaceToken(token, (int)positions.size(), (int)uvs.size(), (int)normals.size()));
            }
            // fan triangulation (0, fanIndex, fanIndex + 1)
            for (size_t fanIndex = 1; fanIndex + 1 < corners.size(); fanIndex++){
                ObjTriangleCorners face;
                face.corner[0] = corners[0];
                face.corner[1] = corners[fanIndex];
                face.corner[2] = corners[fanIndex + 1];
                face.materialIndex = currentMaterialIndex;
                faces.push_back(face);
            }
        }
    }
    if (positions.empty() || faces.empty()) return false;

    // bounding box of the raw model
    vec3 boxMin = positions[0];
    vec3 boxMax = positions[0];
    for (size_t positionIndex = 1; positionIndex < positions.size(); positionIndex++){
        boxMin = vec3(fminf(boxMin.x, positions[positionIndex].x), fminf(boxMin.y, positions[positionIndex].y), fminf(boxMin.z, positions[positionIndex].z));
        boxMax = vec3(fmaxf(boxMax.x, positions[positionIndex].x), fmaxf(boxMax.y, positions[positionIndex].y), fmaxf(boxMax.z, positions[positionIndex].z));
    }
    vec3 modelSize = boxMax - boxMin;
    float largestSide = fmaxf(modelSize.x, fmaxf(modelSize.y, modelSize.z));
    vec3 bottomCenter((boxMin.x + boxMax.x) * 0.5f, boxMin.y, (boxMin.z + boxMax.z) * 0.5f);

    float cosine = cosf(rotationAroundYRadians);
    float sine = sinf(rotationAroundYRadians);

    // transform every position only once
    std::vector<vec3> transformedPositions(positions.size());
    for (size_t positionIndex = 0; positionIndex < positions.size(); positionIndex++){
        vec3 normalizedPosition = (positions[positionIndex] - bottomCenter) * (uniformScale / largestSide);
        vec3 rotatedPosition(normalizedPosition.x * cosine + normalizedPosition.z * sine, normalizedPosition.y, -normalizedPosition.x * sine + normalizedPosition.z * cosine);
        transformedPositions[positionIndex] = rotatedPosition + translation;
    }
    // models without vn get area-weighted averaged vertex normals, so organic meshes (bunny, dragon, sphere) look smooth
    std::vector<vec3> averagedNormals;
    if (normals.empty()){
        averagedNormals.assign(positions.size(), vec3(0.f, 0.f, 0.f));
        for (size_t faceIndex = 0; faceIndex < faces.size(); faceIndex++){
            const ObjTriangleCorners& face = faces[faceIndex];
            int i0 = face.corner[0].positionIndex, i1 = face.corner[1].positionIndex, i2 = face.corner[2].positionIndex;
            if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= (int)positions.size() || i1 >= (int)positions.size() || i2 >= (int)positions.size()) continue;
            // the cross product length is twice the triangle area, so big triangles weigh more
            vec3 areaWeightedNormal = (transformedPositions[i1] - transformedPositions[i0]).cross(transformedPositions[i2] - transformedPositions[i0]);
            averagedNormals[i0] = averagedNormals[i0] + areaWeightedNormal;
            averagedNormals[i1] = averagedNormals[i1] + areaWeightedNormal;
            averagedNormals[i2] = averagedNormals[i2] + areaWeightedNormal;
        }
    }

    int addedTriangleCount = 0;
    int skippedFaceCount = 0;
    for (size_t faceIndex = 0; faceIndex < faces.size(); faceIndex++){
        const ObjTriangleCorners& face = faces[faceIndex];

        bool isFaceValid = true;
        for (int cornerIndex = 0; cornerIndex < 3; cornerIndex++){
            int positionIndex = face.corner[cornerIndex].positionIndex;
            if (positionIndex < 0 || positionIndex >= (int)positions.size()) isFaceValid = false;
        }
        if (!isFaceValid){
            skippedFaceCount++;
            continue;
        }

        vec3 vertex0 = transformedPositions[face.corner[0].positionIndex];
        vec3 vertex1 = transformedPositions[face.corner[1].positionIndex];
        vec3 vertex2 = transformedPositions[face.corner[2].positionIndex];

        // degenerate (zero area) triangles would give NaN normals
        if ((vertex1 - vertex0).cross(vertex2 - vertex0).length() < 1e-9f){
            skippedFaceCount++;
            continue;
        }

        Material triangleMaterial = mat;
        if (face.materialIndex >= 0 && face.materialIndex < (int)materialLibrary.materials.size()){
            triangleMaterial = materialLibrary.materials[face.materialIndex];
        }

        triangle objectTriangle(vertex0, vertex1, vertex2, triangleMaterial);
        vec2* uvTargets[3] = {&objectTriangle.uv0, &objectTriangle.uv1, &objectTriangle.uv2};
        for (int cornerIndex = 0; cornerIndex < 3; cornerIndex++){
            int uvIndex = face.corner[cornerIndex].uvIndex;
            if (uvIndex >= 0 && uvIndex < (int)uvs.size()){
                *uvTargets[cornerIndex] = uvs[uvIndex];
            }
        }
        vec3* normalTargets[3] = {&objectTriangle.n0, &objectTriangle.n1, &objectTriangle.n2};
        for (int cornerIndex = 0; cornerIndex < 3; cornerIndex++){
            int normalIndex = face.corner[cornerIndex].normalIndex;
            if (normalIndex >= 0 && normalIndex < (int)normals.size() && normals[normalIndex].length_squared() > 0.0f){
                vec3 normal = normals[normalIndex];
                *normalTargets[cornerIndex] = vec3(normal.x * cosine + normal.z * sine, normal.y, -normal.x * sine + normal.z * cosine).normalize();
            }else if (!averagedNormals.empty()){
                const vec3& averaged = averagedNormals[face.corner[cornerIndex].positionIndex];
                if (averaged.length_squared() > 0.0f) *normalTargets[cornerIndex] = averaged.normalize();
            }
        }
        scene.triangles.push_back(objectTriangle);
        addedTriangleCount++;
    }

    int loadedTextureCount = 0;
    for (std::map<std::string,cudaTextureObject_t>::iterator cachedEntry = textureCache.begin(); cachedEntry != textureCache.end(); ++cachedEntry){
        if (cachedEntry->second != 0) loadedTextureCount++;
    }
    printf("OBJ '%s': %d triangles added, %d faces skipped, %d materials, %d textures loaded, %d materials with a missing texture\n",
           filePath, addedTriangleCount, skippedFaceCount, (int)materialLibrary.materials.size(), loadedTextureCount, missingTextureCount);
    return addedTriangleCount > 0;
}
// ============================================================================

// ============================================================================
// === RENDER ===
// ============================================================================
__global__ void init_rng_kernel(curandState* rngStates, int maximumX, int maximumY, unsigned long long seed){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= maximumX || j >= maximumY) return;
    int pixel_index = maximumX * j + i;
    curand_init(seed, pixel_index, 0, &rngStates[pixel_index]);
}


__device__ float mediumTransmittance(const BVH& bvh, vec3 rayOrigin, const vec3& rayDirection, float rayTime,
                                     const sphere* spheres, const triangle* triangles, float maximumDistance){
    float transmittance = 1.0f;
    float distanceTravelled = 0.0f;

    for (int boundaryCrossing = 0; boundaryCrossing < 4; boundaryCrossing++){
        float remainingDistance = maximumDistance - distanceTravelled;
        if (remainingDistance <= 0.0f) break;

        hitRecord blocker = bvh.traverseRay(rayOrigin, rayDirection, rayTime, spheres, triangles, remainingDistance);
        if (!blocker.didHit) break;
        if (blocker.mat.type != MaterialType::Isotropic) return 0.0f;

        if (!blocker.frontFace) transmittance = transmittance * expf(-blocker.mat.density * blocker.closestDistance);
        distanceTravelled = distanceTravelled + blocker.closestDistance;
        rayOrigin = blocker.hitPoint;
    }
    return transmittance;
}

__global__ void trace_sample_kernel(vec3* accumBuffer, Camera camera, BVH bvh, sphere* spheres, int sphereCount, triangle* triangles, int triangleCount,
                                    int maximumX, int maximumY, curandState* rngStates, int sampleIndex,Light* lights,int lightCount){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= maximumX || j >= maximumY) return;
    int pixel_index = maximumX * j + i;

    curandState rngState = rngStates[pixel_index];

    Ray r = camera.getRay(i, j, &rngState);
    vec3 attenuation(1.0f, 1.0f, 1.0f);
    vec3 sampleColor(0.0f, 0.0f, 0.0f);
    bool previousBounceWasSpecular = true;
    float previousBsdfPdf = 1.0f;

    for(int depth = 0 ; depth < camera.max_depth;depth++){
        hitRecord hit = bvh.traverseRay(r.origin, r.direction,r.time, spheres, triangles);

        if(!hit.didHit){
            float environmentWeight = 1.0f;
            float environmentProbability = environmentSelectionProbability(lightCount);
            if(!(depth == 0 || previousBounceWasSpecular) && environmentProbability > 0.0f){
                environmentWeight = powerHeuristic(previousBsdfPdf, environmentProbability * environmentDirectionPdf(r.direction));
            }
            sampleColor = sampleColor + attenuation * environmentRadiance(r.direction, camera.backround_color) * environmentWeight;
            break;
        }
        float emissionWeight = 1.0f;
        bool emissionComesFromBsdfSampling = !(depth == 0 || previousBounceWasSpecular);
        if (emissionComesFromBsdfSampling && hit.mat.type == MaterialType::Emissive && hit.primitiveType == 1) {
            float cosineAtLight = fmaxf(fabsf(r.direction.dot(hit.hitNormal)), 1e-4f);
            float lightArea = triangleSurfaceArea(triangles[hit.primitiveIndex]);
            float lightPdf = lightSolidAnglePdf((1.0f - environmentSelectionProbability(lightCount)) / float(lightCount), hit.closestDistance, cosineAtLight, lightArea);
            emissionWeight = powerHeuristic(previousBsdfPdf, lightPdf);
        }
        sampleColor = sampleColor + attenuation * hit.mat.emittedColor * emissionWeight;

        if (hit.mat.type == MaterialType::Emissive) {
            break;
        }

        vec3 scatterOrigin = hit.hitPoint;
        if (hit.mat.type == MaterialType::Isotropic) {
            float scatterDistance = hit.frontFace ? hit.closestDistance
                                                  : -logf(1.0f - curand_uniform(&rngState)) / hit.mat.density;
            if (scatterDistance >= hit.closestDistance) {
                r = Ray(hit.hitPoint, r.direction, r.time);
                continue;
            }
            scatterOrigin = r.origin + r.direction * scatterDistance;
        }

        attenuation = attenuation * sampleTexture(hit.mat.albedoTexture, hit.u, hit.v);

        // ---- NEXT EVENT ESTIMATION (direct light) ----
        float environmentProbability = environmentSelectionProbability(lightCount);
        if ((hit.mat.type == MaterialType::Lambertian || hit.mat.type == MaterialType::Isotropic) && (lightCount > 0 || environmentProbability > 0.0f)) {
            vec3 directionToLight;
            vec3 lightColor;
            float distanceToLight;
            float solidAnglePdf;

            if (environmentProbability > 0.0f && (lightCount == 0 || curand_uniform(&rngState) < environmentProbability)) {
                float directionPdf;
                directionToLight = sampleEnvironmentDirection(&rngState, directionPdf);
                lightColor = environmentRadiance(directionToLight, vec3(0.f, 0.f, 0.f));
                distanceToLight = 1e30f;
                solidAnglePdf = environmentProbability * directionPdf;
            } else {
                int selectedLightIndex = min((int)(curand_uniform(&rngState) * lightCount), lightCount - 1);
                const Light& selectedLight = lights[selectedLightIndex];
                float lightSelectionProbability = (1.0f - environmentProbability) / float(lightCount);

                vec3 sampledPointOnLight = sample_triangle(selectedLight.sourceTriangle, &rngState);
                vec3 vectorToLight = sampledPointOnLight - scatterOrigin;
                distanceToLight = vectorToLight.length();
                directionToLight = vectorToLight / distanceToLight;
                lightColor = selectedLight.sourceTriangle.mat.emittedColor;

                float cosineAtLight = fabsf(selectedLight.surfaceNormal.dot(directionToLight));
                solidAnglePdf = (cosineAtLight > 1e-4f) ? lightSelectionProbability * (distanceToLight * distanceToLight) / (cosineAtLight * selectedLight.surfaceArea) : 0.0f;
            }

            // isotropic scattering has no surface cosine and accepts light from any side
            bool scatteredIsotropically = (hit.mat.type == MaterialType::Isotropic);
            float cosineAtSurface = scatteredIsotropically ? 1.0f : hit.hitNormal.dot(directionToLight);

            if (cosineAtSurface > 0.0f && solidAnglePdf > 0.0f) {
                float transmittance = mediumTransmittance(bvh, scatterOrigin, directionToLight, r.time, spheres, triangles, distanceToLight * 0.999f);

                if (transmittance > 0.0f) {
                    float scatterPdf = scatteredIsotropically ? isotropicPhasePdf() : lambertianCosinePdf(cosineAtSurface);
                    float scatterValue = scatteredIsotropically ? isotropicPhasePdf() : cosineAtSurface / MONOCUDA_PI;
                    float neeWeight = powerHeuristic(solidAnglePdf, scatterPdf);
                    sampleColor = sampleColor + attenuation * lightColor * (scatterValue / solidAnglePdf) * neeWeight * transmittance;
                }
            }
        }
        // ----------------------------------------------


        if(depth > 3){
            float maxComponent = fmaxf(attenuation.x , fmaxf(attenuation.y,attenuation.z));
            float continueProbability =  fminf(maxComponent,0.95f);
            if(curand_uniform(&rngState) > continueProbability)break;

            attenuation = attenuation / continueProbability;
        }

        vec3 newDirection;
        switch (hit.mat.type) {
            case MaterialType::Conductor:{
                vec3 reflected = specular_scatter_direction(r.direction, hit.hitNormal);
                newDirection = (reflected + random_unit_vector(&rngState) * hit.mat.fuzz).normalize();
                break;
            }
            case MaterialType::Dielectric:{
                float ratio;
                if(hit.frontFace){
                    ratio = 1.0f / hit.mat.refractionIndex;
                }else{
                    ratio = hit.mat.refractionIndex;
                }
                newDirection = dielectric_scatter_direction(r.direction, hit.hitNormal, ratio, &rngState);
                break;
            }
            case MaterialType::Isotropic:{
                newDirection = isotropic_scatter_direction(&rngState);
                break;
            }
            case MaterialType::Lambertian:
            default:
                newDirection = lambertian_scatter_direction(hit.hitNormal, &rngState);
                break;
        }
        previousBounceWasSpecular = (hit.mat.type != MaterialType::Lambertian && hit.mat.type != MaterialType::Isotropic);
        previousBsdfPdf = (hit.mat.type == MaterialType::Isotropic) ? isotropicPhasePdf()
                                                                   : lambertianCosinePdf(fmaxf(newDirection.dot(hit.hitNormal), 0.0f));
        r = Ray(scatterOrigin, newDirection, r.time);
    }
    // to reduce firefly points
    float brightestChannel = fmaxf(sampleColor.x,fmaxf(sampleColor.y,sampleColor.z));
    const float maximumSampleBrightness = 10.0f;
    if(brightestChannel > maximumSampleBrightness){
        sampleColor = sampleColor * (maximumSampleBrightness / brightestChannel);
    }

    accumBuffer[pixel_index] = accumBuffer[pixel_index] + sampleColor;
    rngStates[pixel_index] = rngState;
}


__device__ float linearToSrgb(float linearValue){
    linearValue = fmaxf(linearValue,0.0f);
    float srgbValue;
    if (linearValue <= 0.0031308f){
        srgbValue = 12.92f * linearValue;
    }
    else{
        srgbValue = 1.055f * powf(linearValue, 1.0f / 2.4f) - 0.055f;
    }
    return srgbValue;
}

enum class ToneMapOption{None,Reinhard,ACES };

__device__ float toneMapChannel(float linearValue, ToneMapOption toneMap){
    linearValue = fmaxf(linearValue,0.f);
    switch(toneMap){
        case ToneMapOption::Reinhard:{
            return linearValue / (1.f + linearValue);
        }
        case ToneMapOption::ACES:{
            float mapped = (linearValue * (2.51f * linearValue + 0.03f)) / (linearValue * (2.43f * linearValue + 0.59f) + 0.14f);
            return fminf(fmaxf(mapped, 0.0f), 1.0f);
        }
        case ToneMapOption::None:
        default:
            return linearValue;
    }
}

__global__ void resolve_kernel(const vec3* accumBuffer, vec3* outputBuffer,int maximumX, int maximumY, int samplesSoFar, ToneMapOption toneMap,float exposure) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= maximumX || j >= maximumY) return;
    int pixel_index = maximumX * j + i;

    vec3 averagedRadiance = accumBuffer[pixel_index] / float(samplesSoFar) * exposure;
    outputBuffer[pixel_index] = vec3(linearToSrgb(toneMapChannel(averagedRadiance.x, toneMap)),linearToSrgb(toneMapChannel(averagedRadiance.y, toneMap)),linearToSrgb(toneMapChannel(averagedRadiance.z, toneMap)));
}
// Sums ln(delta + luminance) over the image so the host can form the geometric mean (log-average luminance).
// Every thread of a warp takes part in the shuffle (out-of-range threads add 0), then one lane per warp does the atomicAdd.
__global__ void logLuminanceSumKernel(const vec3* accumBuffer, int maximumX, int maximumY, int samplesSoFar, float* logLuminanceSum){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    const float delta = 1e-4f;
    float logLuminance = 0.0f;
    if (i < maximumX && j < maximumY){
        vec3 radiance = accumBuffer[maximumX * j + i] / float(samplesSoFar);
        float luminance = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
        logLuminance = logf(delta + (isfinite(luminance) ? fmaxf(luminance, 0.0f) : 0.0f));
    }
    for (int offset = 16; offset > 0; offset >>= 1){
        logLuminance += __shfl_down_sync(0xffffffffu, logLuminance, offset);
    }
    int laneIndex = (threadIdx.y * blockDim.x + threadIdx.x) & 31;
    if (laneIndex == 0) atomicAdd(logLuminanceSum, logLuminance);
}

// middle gray of 0.18 in linear radiance is the usual target for the log-average luminance
static float computeAutoExposure(const vec3* accumBuffer, int imageWidth, int imageHeight, int samplesSoFar, dim3 grid, dim3 block, float targetGray = 0.18f){
    float* deviceSum = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceSum, sizeof(float)));
    CUDA_CHECK(cudaMemset(deviceSum, 0, sizeof(float)));
    logLuminanceSumKernel<<<grid, block>>>(accumBuffer, imageWidth, imageHeight, samplesSoFar, deviceSum);
    CUDA_CHECK_KERNEL();
    float hostSum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&hostSum, deviceSum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceSum));
    float logAverageLuminance = expf(hostSum / float(imageWidth * imageHeight));
    return targetGray / logAverageLuminance;
}
// ============================================================================

// ============================================================================
// === WAVEFRONT & PERSISTENT THREADS ===
// ============================================================================
struct PathStateBuffers {
    float* rayOriginX = nullptr;
    float* rayOriginY = nullptr;
    float* rayOriginZ = nullptr;

    float* rayDirectionX = nullptr;
    float* rayDirectionY = nullptr;
    float* rayDirectionZ = nullptr;

    float* rayTime = nullptr;

    float* throughputRed = nullptr;
    float* throughputGreen = nullptr;
    float* throughputBlue = nullptr;

    float* currentSampleRed = nullptr;
    float* currentSampleGreen = nullptr;
    float* currentSampleBlue = nullptr;

    unsigned char* previousBounceWasSpecular = nullptr;
    float* previousBsdfPdf = nullptr;
};

struct HitResultBuffers {
    PackedHit* packedHits = nullptr;
};

struct PathQueue {
    int* pathIndices = nullptr;   // capacity = pixelCount
    int* itemCount = nullptr;
};

struct ShadowRayQueue {
    float* originX = nullptr;
    float* originY = nullptr;
    float* originZ = nullptr;

    float* directionX = nullptr;
    float* directionY = nullptr;
    float* directionZ = nullptr;

    float* maximumDistance = nullptr;
    float* contributionRed = nullptr;
    float* contributionGreen = nullptr;
    float* contributionBlue = nullptr;

    int* pathIndices = nullptr;
    int* itemCount = nullptr;
};

struct WavefrontRenderContext {
    Camera camera;
    BVH bvh;
    sphere* spheres;
    int sphereCount;

    triangle* triangles;
    int triangleCount;

    Light* lights;
    int lightCount;

    vec3* accumBuffer;
    vec3* frameBuffer;
    curandState* randomNumberStates;

    int imageWidth;
    int imageHeight;
    dim3 gridDimensions;
    dim3 blockDimensions;

    bool usePersistentThreads = false;
    bool sceneHasMedium = false;
    ToneMapOption toneMap = ToneMapOption::None;
    float exposure = 1.f;
};

template <typename ElementType>
static void allocateDeviceArray(ElementType*& devicePointer, size_t elementCount){
    CUDA_CHECK(cudaMalloc(&devicePointer, elementCount * sizeof(ElementType)));
    CUDA_CHECK(cudaMemset(devicePointer, 0, elementCount * sizeof(ElementType)));
}

template <typename ElementType>
static void freeDeviceArray(ElementType*& devicePointer) {
    if (devicePointer) {
        CUDA_CHECK(cudaFree(devicePointer));
        devicePointer = nullptr;
    }
}
// ----------------------------------------------------------------------------
// Path queue
// ----------------------------------------------------------------------------
static void allocatePathQueue(PathQueue& queue, int pathCapacity) {
    allocateDeviceArray(queue.pathIndices, pathCapacity);
    allocateDeviceArray(queue.itemCount, 1);
}

static void freePathQueue(PathQueue& queue) {
    freeDeviceArray(queue.pathIndices);
    freeDeviceArray(queue.itemCount);
}

// ----------------------------------------------------------------------------
// Path state (structure of arrays)
// ----------------------------------------------------------------------------
static void allocatePathStateBuffers(PathStateBuffers& buffers, int pathCapacity) {
    allocateDeviceArray(buffers.rayOriginX, pathCapacity);
    allocateDeviceArray(buffers.rayOriginY, pathCapacity);
    allocateDeviceArray(buffers.rayOriginZ, pathCapacity);

    allocateDeviceArray(buffers.rayDirectionX, pathCapacity);
    allocateDeviceArray(buffers.rayDirectionY, pathCapacity);
    allocateDeviceArray(buffers.rayDirectionZ, pathCapacity);

    allocateDeviceArray(buffers.rayTime, pathCapacity);

    allocateDeviceArray(buffers.throughputRed, pathCapacity);
    allocateDeviceArray(buffers.throughputGreen, pathCapacity);
    allocateDeviceArray(buffers.throughputBlue, pathCapacity);

    allocateDeviceArray(buffers.currentSampleRed, pathCapacity);
    allocateDeviceArray(buffers.currentSampleGreen, pathCapacity);
    allocateDeviceArray(buffers.currentSampleBlue, pathCapacity);

    allocateDeviceArray(buffers.previousBounceWasSpecular, pathCapacity);
    allocateDeviceArray(buffers.previousBsdfPdf, pathCapacity);
}

static void freePathStateBuffers(PathStateBuffers& buffers) {
    freeDeviceArray(buffers.rayOriginX);
    freeDeviceArray(buffers.rayOriginY);
    freeDeviceArray(buffers.rayOriginZ);

    freeDeviceArray(buffers.rayDirectionX);
    freeDeviceArray(buffers.rayDirectionY);
    freeDeviceArray(buffers.rayDirectionZ);

    freeDeviceArray(buffers.rayTime);

    freeDeviceArray(buffers.throughputRed);
    freeDeviceArray(buffers.throughputGreen);
    freeDeviceArray(buffers.throughputBlue);

    freeDeviceArray(buffers.currentSampleRed);
    freeDeviceArray(buffers.currentSampleGreen);
    freeDeviceArray(buffers.currentSampleBlue);

    freeDeviceArray(buffers.previousBounceWasSpecular);
    freeDeviceArray(buffers.previousBsdfPdf);
}

// ----------------------------------------------------------------------------
// Hit results
// ----------------------------------------------------------------------------
static void allocateHitResultBuffers(HitResultBuffers& buffers, int pathCapacity) {
    allocateDeviceArray(buffers.packedHits,pathCapacity);
}

static void freeHitResultBuffers(HitResultBuffers& buffers) {
    freeDeviceArray(buffers.packedHits);
}

// ----------------------------------------------------------------------------
// Shadow ray queue
// ----------------------------------------------------------------------------
static void allocateShadowRayQueue(ShadowRayQueue& queue, int pathCapacity) {
    allocateDeviceArray(queue.originX, pathCapacity);
    allocateDeviceArray(queue.originY, pathCapacity);
    allocateDeviceArray(queue.originZ, pathCapacity);

    allocateDeviceArray(queue.directionX, pathCapacity);
    allocateDeviceArray(queue.directionY, pathCapacity);
    allocateDeviceArray(queue.directionZ, pathCapacity);

    allocateDeviceArray(queue.maximumDistance, pathCapacity);

    allocateDeviceArray(queue.contributionRed, pathCapacity);
    allocateDeviceArray(queue.contributionGreen, pathCapacity);
    allocateDeviceArray(queue.contributionBlue, pathCapacity);

    allocateDeviceArray(queue.pathIndices, pathCapacity);
    allocateDeviceArray(queue.itemCount, 1);
}

static void freeShadowRayQueue(ShadowRayQueue& queue) {
    freeDeviceArray(queue.originX);
    freeDeviceArray(queue.originY);
    freeDeviceArray(queue.originZ);

    freeDeviceArray(queue.directionX);
    freeDeviceArray(queue.directionY);
    freeDeviceArray(queue.directionZ);

    freeDeviceArray(queue.maximumDistance);

    freeDeviceArray(queue.contributionRed);
    freeDeviceArray(queue.contributionGreen);
    freeDeviceArray(queue.contributionBlue);

    freeDeviceArray(queue.pathIndices);
    freeDeviceArray(queue.itemCount);
}

__global__ void generateCameraRaysKernel(Camera camera, int imageWidth, int imageHeight,curandState* randomNumberStates,PathStateBuffers pathState, PathQueue queue){
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;
    int pathIndex = imageWidth * pixelY + pixelX;

    curandState randomNumberState = randomNumberStates[pathIndex];
    Ray cameraRay = camera.getRay(pixelX, pixelY, &randomNumberState);

    pathState.rayOriginX[pathIndex] = cameraRay.origin.x;
    pathState.rayOriginY[pathIndex] = cameraRay.origin.y;
    pathState.rayOriginZ[pathIndex] = cameraRay.origin.z;

    pathState.rayDirectionX[pathIndex] = cameraRay.direction.x;
    pathState.rayDirectionY[pathIndex] = cameraRay.direction.y;
    pathState.rayDirectionZ[pathIndex] = cameraRay.direction.z;
    pathState.rayTime[pathIndex] = cameraRay.time;

    pathState.throughputRed[pathIndex] = 1.0f;
    pathState.throughputGreen[pathIndex] = 1.0f;
    pathState.throughputBlue[pathIndex] = 1.0f;

    pathState.currentSampleRed[pathIndex] = 0.0f;
    pathState.currentSampleGreen[pathIndex] = 0.0f;
    pathState.currentSampleBlue[pathIndex] = 0.0f;

    pathState.previousBounceWasSpecular[pathIndex] = 1;
    pathState.previousBsdfPdf[pathIndex] = 1.0f;

    queue.pathIndices[pathIndex] = pathIndex;

    randomNumberStates[pathIndex] = randomNumberState;
}

__global__ void extendKernel(BVH bvh, const sphere* spheres, const triangle* triangles, PathStateBuffers pathState, HitResultBuffers hitResults, PathQueue queue){
    int queueSlot = threadIdx.x + blockIdx.x * blockDim.x;
    if (queueSlot >= *queue.itemCount) return;
    int pathIndex = queue.pathIndices[queueSlot];

    vec3 rayOrigin(pathState.rayOriginX[pathIndex], pathState.rayOriginY[pathIndex], pathState.rayOriginZ[pathIndex]);
    vec3 rayDirection(pathState.rayDirectionX[pathIndex], pathState.rayDirectionY[pathIndex], pathState.rayDirectionZ[pathIndex]);
    float rayTime = pathState.rayTime[pathIndex];

    hitRecord hit = bvh.traverseRay(rayOrigin, rayDirection, rayTime, spheres, triangles);

    PackedHit packedHit;
    packedHit.hitDistance = hit.closestDistance;
    packedHit.primitiveType = hit.primitiveType;
    packedHit.primitiveIndex = hit.primitiveIndex;
    if(hit.didHit){
        packedHit.didHit = 1;
    }else{
        packedHit.didHit = 0;
    }
    hitResults.packedHits[pathIndex] = packedHit;
}

// A lane that finishes its ray asks for a new one while the other lanes keep traversing.
// A warp refills only when at least this many of its lanes are idle (tunable: 1, 8, 16).
static constexpr int minimumIdleLanesBeforeRefill = 8;

__device__ inline void writeHitResult(HitResultBuffers& hitResults, int pathIndex, const ClosestHitCandidate& hit){
    PackedHit packedHit;
    packedHit.hitDistance = hit.closestDistance;
    packedHit.primitiveType = hit.primitiveType;
    packedHit.primitiveIndex = hit.primitiveIndex;

    if(hit.primitiveIndex >= 0){
        packedHit.didHit = 1;
    }else{
        packedHit.didHit = 0;
    }
    hitResults.packedHits[pathIndex] = packedHit;
}

__global__ void extendKernelPersistent(BVH bvh, const sphere* spheres, const triangle* triangles, PathStateBuffers pathState, HitResultBuffers hitResults, PathQueue queue, int* workCounter){
    const unsigned int fullWarpMask = 0xffffffffu;
    const int laneIndex = threadIdx.x & 31;
    const int totalPathCount = *queue.itemCount;

    TraversalState traversal;
    bool laneHasRay = false;
    int currentPathIndex = -1;
    bool workQueueExhausted = false;     // identical for all 32 lanes of the warp

    while (true) {
        // --- refill idle lanes: one atomicAdd per warp, ranks distributed with a prefix popcount ---
        if (!workQueueExhausted) {
            unsigned int idleLaneMask = __ballot_sync(fullWarpMask, !laneHasRay);
            int idleLaneCount = __popc(idleLaneMask);

            if (idleLaneCount >= minimumIdleLanesBeforeRefill) {
                int firstIdleLane = __ffs(idleLaneMask) - 1;
                int firstQueueSlot = 0;
                if (laneIndex == firstIdleLane) firstQueueSlot = atomicAdd(workCounter, idleLaneCount);
                firstQueueSlot = __shfl_sync(fullWarpMask, firstQueueSlot, firstIdleLane);

                if (!laneHasRay) {
                    int idleRank = __popc(idleLaneMask & ((1u << laneIndex) - 1u));
                    int queueSlot = firstQueueSlot + idleRank;
                    if (queueSlot < totalPathCount) {
                        int pathIndex = queue.pathIndices[queueSlot];
                        vec3 rayOrigin(pathState.rayOriginX[pathIndex], pathState.rayOriginY[pathIndex], pathState.rayOriginZ[pathIndex]);
                        vec3 rayDirection(pathState.rayDirectionX[pathIndex], pathState.rayDirectionY[pathIndex], pathState.rayDirectionZ[pathIndex]);
                        float rayTime = pathState.rayTime[pathIndex];

                        bvh.beginTraversal(traversal, rayOrigin, rayDirection, rayTime);
                        currentPathIndex = pathIndex;
                        laneHasRay = true;
                    }
                }
                if (firstQueueSlot + idleLaneCount >= totalPathCount) workQueueExhausted = true;
            }
        }

        // --- the warp is done when the queue is empty and no lane holds a ray ---
        if (workQueueExhausted && __ballot_sync(fullWarpMask, laneHasRay) == 0u) break;

        // --- one traversal step for every lane that has a ray ---
        if (laneHasRay) {
            bool traversalFinished = bvh.stepTraversal(traversal, spheres, triangles);
            if (traversalFinished) {
                writeHitResult(hitResults, currentPathIndex, traversal.nearestHit);
                laneHasRay = false;
            }
        }
    }
}

__global__ void shadeKernel(int depth, vec3 backgroundColor,
                            const sphere* spheres, const triangle* triangles,
                            const Light* lights, int lightCount,
                            curandState* randomNumberStates,
                            PathStateBuffers pathState, HitResultBuffers hitResults,
                            PathQueue currentQueue, PathQueue nextQueue,
                            ShadowRayQueue shadowQueue)
{
    int queueSlot = threadIdx.x + blockIdx.x * blockDim.x;
    if (queueSlot >= *currentQueue.itemCount) return;
    int pathIndex = currentQueue.pathIndices[queueSlot];

    vec3 throughput(pathState.throughputRed[pathIndex], pathState.throughputGreen[pathIndex], pathState.throughputBlue[pathIndex]);
    vec3 currentSample(pathState.currentSampleRed[pathIndex], pathState.currentSampleGreen[pathIndex], pathState.currentSampleBlue[pathIndex]);

    PackedHit packedHit = hitResults.packedHits[pathIndex];
    // missed
    if (!packedHit.didHit) {
        vec3 missDirection(pathState.rayDirectionX[pathIndex], pathState.rayDirectionY[pathIndex], pathState.rayDirectionZ[pathIndex]);
        float environmentWeight = 1.0f;
        float environmentProbability = environmentSelectionProbability(lightCount);
        if(!(depth == 0 || pathState.previousBounceWasSpecular[pathIndex] != 0) && environmentProbability > 0.0f){
            environmentWeight = powerHeuristic(pathState.previousBsdfPdf[pathIndex], environmentProbability * environmentDirectionPdf(missDirection));
        }
        currentSample = currentSample + throughput * environmentRadiance(missDirection, backgroundColor) * environmentWeight;
        pathState.currentSampleRed[pathIndex] = currentSample.x;
        pathState.currentSampleGreen[pathIndex] = currentSample.y;
        pathState.currentSampleBlue[pathIndex] = currentSample.z;
        return;
    }

    // material
    int primitiveIndex = packedHit.primitiveIndex;
    int primitiveType = packedHit.primitiveType;
    Material material = (primitiveType == 0) ? spheres[primitiveIndex].mat : triangles[primitiveIndex].mat;

    vec3 incomingRayDirection(pathState.rayDirectionX[pathIndex], pathState.rayDirectionY[pathIndex], pathState.rayDirectionZ[pathIndex]);

    // The extend kernel no longer stores these: recompute them with the same formulas as testPrimitiveHit.
    vec3 rayOriginAtHit(pathState.rayOriginX[pathIndex], pathState.rayOriginY[pathIndex], pathState.rayOriginZ[pathIndex]);
    float rayTimeAtHit = pathState.rayTime[pathIndex];
    vec3 hitPoint = rayOriginAtHit + incomingRayDirection * packedHit.hitDistance;
    vec3 hitNormal;
    bool frontFace;
    float hitU, hitV;
    if (primitiveType == 0) {
        vec3 sphereCenter = spheres[primitiveIndex].center_at(rayTimeAtHit);
        vec3 outwardNormal = (hitPoint - sphereCenter).normalize();
        frontFace = incomingRayDirection.dot(outwardNormal) < 0.0f;
        hitNormal = frontFace ? outwardNormal : -outwardNormal;
        vec2 sphereUv = computeSphereUvFromOutwardNormal(outwardNormal);
        hitU = sphereUv.x;
        hitV = sphereUv.y;
    } else {
        const triangle& hitTriangle = triangles[primitiveIndex];
        vec3 edge1 = hitTriangle.v1 - hitTriangle.v0;
        vec3 edge2 = hitTriangle.v2 - hitTriangle.v0;
        vec3 triangleNormal = edge1.cross(edge2).normalize();
        frontFace = incomingRayDirection.dot(triangleNormal) < 0.0f;
        vec3 shadingNormal = computeTriangleShadingNormal(hitTriangle, hitPoint, triangleNormal);
        hitNormal = frontFace ? shadingNormal : -shadingNormal;
        vec2 triangleUv = computeTriangleUvFromHitPoint(hitTriangle, hitPoint);
        hitU = triangleUv.x;
        hitV = triangleUv.y;
    }

    // emission (MIS weighted when it comes from BSDF sampling)
    if (material.type == MaterialType::Emissive) {
        bool previousBounceWasSpecular = pathState.previousBounceWasSpecular[pathIndex] != 0;
        float emissionWeight = 1.0f;
        bool emissionComesFromBsdfSampling = !(depth == 0 || previousBounceWasSpecular);
        if (emissionComesFromBsdfSampling && primitiveType == 1) {
            float cosineAtLight = fmaxf(fabsf(incomingRayDirection.dot(hitNormal)), 1e-4f);
            float lightArea = triangleSurfaceArea(triangles[primitiveIndex]);
            float lightPdf = lightSolidAnglePdf((1.0f - environmentSelectionProbability(lightCount)) / float(lightCount), packedHit.hitDistance, cosineAtLight, lightArea);
            emissionWeight = powerHeuristic(pathState.previousBsdfPdf[pathIndex], lightPdf);
        }
        currentSample = currentSample + throughput * material.emittedColor * emissionWeight;
        pathState.currentSampleRed[pathIndex] = currentSample.x;
        pathState.currentSampleGreen[pathIndex] = currentSample.y;
        pathState.currentSampleBlue[pathIndex] = currentSample.z;
        return;
    }

    curandState randomNumberState = randomNumberStates[pathIndex];

    vec3 scatterOrigin = hitPoint;
    if (material.type == MaterialType::Isotropic) {
        float scatterDistance = frontFace ? packedHit.hitDistance
                                          : -logf(1.0f - curand_uniform(&randomNumberState)) / material.density;
        if (scatterDistance >= packedHit.hitDistance) {
            pathState.rayOriginX[pathIndex] = hitPoint.x;
            pathState.rayOriginY[pathIndex] = hitPoint.y;
            pathState.rayOriginZ[pathIndex] = hitPoint.z;

            int passThroughSlot = atomicAdd(nextQueue.itemCount, 1);
            nextQueue.pathIndices[passThroughSlot] = pathIndex;
            randomNumberStates[pathIndex] = randomNumberState;
            return;
        }
        scatterOrigin = rayOriginAtHit + incomingRayDirection * scatterDistance;
    }

    // throughput (with albedo)
    throughput = throughput * sampleTexture(material.albedoTexture, hitU, hitV);

    // NEE again same as before !
    float environmentProbability = environmentSelectionProbability(lightCount);
    if ((material.type == MaterialType::Lambertian || material.type == MaterialType::Isotropic) && (lightCount > 0 || environmentProbability > 0.0f)) {
        vec3 directionToLight;
        vec3 lightColor;
        float distanceToLight;
        float solidAnglePdf;

        if (environmentProbability > 0.0f && (lightCount == 0 || curand_uniform(&randomNumberState) < environmentProbability)) {
            float directionPdf;
            directionToLight = sampleEnvironmentDirection(&randomNumberState, directionPdf);
            lightColor = environmentRadiance(directionToLight, vec3(0.f, 0.f, 0.f));
            distanceToLight = 1e30f;
            solidAnglePdf = environmentProbability * directionPdf;
        } else {
            int selectedLightIndex = min((int)(curand_uniform(&randomNumberState) * lightCount), lightCount - 1);
            const Light& selectedLight = lights[selectedLightIndex];
            float lightSelectionProbability = (1.0f - environmentProbability) / float(lightCount);

            vec3 sampledPointOnLight = sample_triangle(selectedLight.sourceTriangle, &randomNumberState);
            vec3 vectorToLight = sampledPointOnLight - scatterOrigin;
            distanceToLight = vectorToLight.length();
            directionToLight = vectorToLight / distanceToLight;
            lightColor = selectedLight.sourceTriangle.mat.emittedColor;

            float cosineAtLight = fabsf(selectedLight.surfaceNormal.dot(directionToLight));
            solidAnglePdf = (cosineAtLight > 1e-4f) ? lightSelectionProbability * (distanceToLight * distanceToLight) / (cosineAtLight * selectedLight.surfaceArea) : 0.0f;
        }

        // isotropic scattering has no surface cosine and accepts light from any side
        bool scatteredIsotropically = (material.type == MaterialType::Isotropic);
        float cosineAtSurface = scatteredIsotropically ? 1.0f : hitNormal.dot(directionToLight);

        if (cosineAtSurface > 0.0f && solidAnglePdf > 0.0f) {
            float scatterPdf = scatteredIsotropically ? isotropicPhasePdf() : lambertianCosinePdf(cosineAtSurface);
            float scatterValue = scatteredIsotropically ? isotropicPhasePdf() : cosineAtSurface / MONOCUDA_PI;
            float neeWeight = powerHeuristic(solidAnglePdf, scatterPdf);
            vec3 contribution = throughput * lightColor * (scatterValue / solidAnglePdf) * neeWeight;

            int shadowSlot = atomicAdd(shadowQueue.itemCount, 1);
            shadowQueue.originX[shadowSlot] = scatterOrigin.x;
            shadowQueue.originY[shadowSlot] = scatterOrigin.y;
            shadowQueue.originZ[shadowSlot] = scatterOrigin.z;
            shadowQueue.directionX[shadowSlot] = directionToLight.x;
            shadowQueue.directionY[shadowSlot] = directionToLight.y;
            shadowQueue.directionZ[shadowSlot] = directionToLight.z;
            shadowQueue.maximumDistance[shadowSlot] = distanceToLight * 0.999f;
            shadowQueue.contributionRed[shadowSlot] = contribution.x;
            shadowQueue.contributionGreen[shadowSlot] = contribution.y;
            shadowQueue.contributionBlue[shadowSlot] = contribution.z;
            shadowQueue.pathIndices[shadowSlot] = pathIndex;
        }
    }

    // Russian roulette
    bool pathSurvives = true;
    if (depth > 3) {
        float maxComponent = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        float continueProbability = fminf(maxComponent, 0.95f);
        if (curand_uniform(&randomNumberState) > continueProbability) {
            pathSurvives = false;
        } else {
            throughput = throughput / continueProbability;
        }
    }

    if (pathSurvives) {
        vec3 rayDirection(pathState.rayDirectionX[pathIndex], pathState.rayDirectionY[pathIndex], pathState.rayDirectionZ[pathIndex]);

        // same old scatter switch case
        vec3 newDirection;
        switch (material.type) {
            case MaterialType::Conductor: {
                vec3 reflected = specular_scatter_direction(rayDirection, hitNormal);
                newDirection = (reflected + random_unit_vector(&randomNumberState) * material.fuzz).normalize();
                break;
            }
            case MaterialType::Dielectric: {
                float ratio = frontFace ? 1.0f / material.refractionIndex : material.refractionIndex;
                newDirection = dielectric_scatter_direction(rayDirection, hitNormal, ratio, &randomNumberState);
                break;
            }
            case MaterialType::Isotropic:{
                newDirection = isotropic_scatter_direction(&randomNumberState);
                break;
            }
            case MaterialType::Lambertian:
            default:
                newDirection = lambertian_scatter_direction(hitNormal, &randomNumberState);
                break;
        }

        pathState.rayOriginX[pathIndex] = scatterOrigin.x;
        pathState.rayOriginY[pathIndex] = scatterOrigin.y;
        pathState.rayOriginZ[pathIndex] = scatterOrigin.z;
        pathState.rayDirectionX[pathIndex] = newDirection.x;
        pathState.rayDirectionY[pathIndex] = newDirection.y;
        pathState.rayDirectionZ[pathIndex] = newDirection.z;
        pathState.throughputRed[pathIndex] = throughput.x;
        pathState.throughputGreen[pathIndex] = throughput.y;
        pathState.throughputBlue[pathIndex] = throughput.z;
        pathState.previousBounceWasSpecular[pathIndex] = (material.type != MaterialType::Lambertian && material.type != MaterialType::Isotropic) ? 1 : 0;
        pathState.previousBsdfPdf[pathIndex] = (material.type == MaterialType::Isotropic) ? isotropicPhasePdf()
                                                                                          : lambertianCosinePdf(fmaxf(newDirection.dot(hitNormal), 0.0f));

        int nextQueueSlot = atomicAdd(nextQueue.itemCount, 1);
        nextQueue.pathIndices[nextQueueSlot] = pathIndex;
    }

    randomNumberStates[pathIndex] = randomNumberState;
}

__global__ void shadowKernel(BVH bvh, const sphere* spheres, const triangle* triangles,
                             PathStateBuffers pathState, ShadowRayQueue shadowQueue, bool sceneHasMedium)
{
    int shadowSlot = threadIdx.x + blockIdx.x * blockDim.x;
    if (shadowSlot >= *shadowQueue.itemCount) return;

    int pathIndex = shadowQueue.pathIndices[shadowSlot];
    vec3 shadowRayOrigin(shadowQueue.originX[shadowSlot], shadowQueue.originY[shadowSlot], shadowQueue.originZ[shadowSlot]);
    vec3 shadowRayDirection(shadowQueue.directionX[shadowSlot], shadowQueue.directionY[shadowSlot], shadowQueue.directionZ[shadowSlot]);
    float rayTime = pathState.rayTime[pathIndex];     // motion blur same as before

    float transmittance;
    if (sceneHasMedium) {
        transmittance = mediumTransmittance(bvh, shadowRayOrigin, shadowRayDirection, rayTime, spheres, triangles, shadowQueue.maximumDistance[shadowSlot]);
    } else {
        bool isOccluded = bvh.traverseAnyHit(shadowRayOrigin, shadowRayDirection, rayTime, spheres, triangles, shadowQueue.maximumDistance[shadowSlot]);
        transmittance = isOccluded ? 0.0f : 1.0f;
    }

    if (transmittance > 0.0f) {

        pathState.currentSampleRed[pathIndex] += shadowQueue.contributionRed[shadowSlot] * transmittance;
        pathState.currentSampleGreen[pathIndex] += shadowQueue.contributionGreen[shadowSlot] * transmittance;
        pathState.currentSampleBlue[pathIndex] += shadowQueue.contributionBlue[shadowSlot] * transmittance;
    }
}

__global__ void debugShowRayDirectionKernel(vec3* frameBuffer, int imageWidth, int imageHeight, PathStateBuffers pathState){
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;
    int pathIndex = imageWidth * pixelY + pixelX;

    frameBuffer[pathIndex] = vec3(pathState.rayDirectionX[pathIndex] * 1.5f + 0.5f,pathState.rayDirectionY[pathIndex] * 1.5f + 0.5f,pathState.rayDirectionZ[pathIndex] * -1.0f);
}


__global__ void finalizeSampleKernel(vec3* accumBuffer, int imageWidth, int imageHeight, PathStateBuffers pathState){
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;
    int pathIndex = imageWidth * pixelY + pixelX;

    vec3 sampleColor(pathState.currentSampleRed[pathIndex], pathState.currentSampleGreen[pathIndex], pathState.currentSampleBlue[pathIndex]);


    float brightestChannel = fmaxf(sampleColor.x, fmaxf(sampleColor.y, sampleColor.z));
    const float maximumSampleBrightness = 10.0f;
    if (brightestChannel > maximumSampleBrightness) {
        sampleColor = sampleColor * (maximumSampleBrightness / brightestChannel);
    }
    accumBuffer[pathIndex] = accumBuffer[pathIndex] + sampleColor;
}

static void renderWithWavefront(const WavefrontRenderContext& context){
    int pathCapacity = context.imageWidth * context.imageHeight;
    PathStateBuffers pathState;
    HitResultBuffers hitResults;
    PathQueue currentQueue, nextQueue;
    ShadowRayQueue shadowQueue;


    allocatePathStateBuffers(pathState, pathCapacity);
    allocateHitResultBuffers(hitResults, pathCapacity);
    allocatePathQueue(currentQueue, pathCapacity);
    allocatePathQueue(nextQueue, pathCapacity);
    allocateShadowRayQueue(shadowQueue, pathCapacity);

    auto renderStart = std::chrono::steady_clock::now();

    int* persistentWorkCounter = nullptr;
    int persistentBlockCount = 0;
    if (context.usePersistentThreads) {
        CUDA_CHECK(cudaMalloc(&persistentWorkCounter, sizeof(int)));
        int deviceIndex = 0, multiprocessorCount = 0, residentBlocksPerMultiprocessor = 0;
        CUDA_CHECK(cudaGetDevice(&deviceIndex));
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, deviceIndex));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentBlocksPerMultiprocessor, extendKernelPersistent, 256, 0));
        persistentBlockCount = multiprocessorCount * residentBlocksPerMultiprocessor;
    }

    for (int sampleIndex = 0; sampleIndex < context.camera.samples_per_pixel; ++sampleIndex) {

        generateCameraRaysKernel<<<context.gridDimensions, context.blockDimensions>>>(context.camera, context.imageWidth, context.imageHeight,context.randomNumberStates, pathState, currentQueue);
        CUDA_CHECK_KERNEL();

        int activePathCount = pathCapacity;
        CUDA_CHECK(cudaMemcpy(currentQueue.itemCount, &activePathCount, sizeof(int), cudaMemcpyHostToDevice));


        for (int depth = 0; depth < context.camera.max_depth; ++depth) {
            if (activePathCount == 0) break;
            CUDA_CHECK(cudaMemset(nextQueue.itemCount, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(shadowQueue.itemCount, 0, sizeof(int)));

            int blockSize = 256;
            int blockCount = (activePathCount + blockSize - 1) / blockSize;

            if (context.usePersistentThreads) {
                CUDA_CHECK(cudaMemset(persistentWorkCounter, 0, sizeof(int)));
                int persistentLaunchBlockCount = std::min(blockCount, persistentBlockCount);
                extendKernelPersistent<<<persistentLaunchBlockCount, blockSize>>>(context.bvh, context.spheres, context.triangles, pathState, hitResults, currentQueue, persistentWorkCounter);
            } else {
                extendKernel<<<blockCount, blockSize>>>(context.bvh, context.spheres, context.triangles, pathState, hitResults, currentQueue);
            }
            CUDA_CHECK_KERNEL();

            shadeKernel<<<blockCount, blockSize>>>(depth, context.camera.backround_color,context.spheres, context.triangles, context.lights, context.lightCount,context.randomNumberStates, pathState, hitResults, currentQueue, nextQueue, shadowQueue);
            CUDA_CHECK_KERNEL();

            shadowKernel<<<blockCount, blockSize>>>(context.bvh, context.spheres, context.triangles, pathState, shadowQueue, context.sceneHasMedium);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpy(&activePathCount, nextQueue.itemCount, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(currentQueue, nextQueue);
        }

        finalizeSampleKernel<<<context.gridDimensions, context.blockDimensions>>>(context.accumBuffer, context.imageWidth, context.imageHeight, pathState);
        CUDA_CHECK_KERNEL();

        resolve_kernel<<<context.gridDimensions, context.blockDimensions>>>(context.accumBuffer, context.frameBuffer, context.imageWidth, context.imageHeight, sampleIndex + 1,context.toneMap,context.exposure);
        CUDA_CHECK_KERNEL();

        float percent = 100.0f * float(sampleIndex + 1) / float(context.camera.samples_per_pixel);
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - renderStart).count();
        printf("\rSample %d / %d (%.1f%%) — %.3fs elapsed", sampleIndex + 1, context.camera.samples_per_pixel, percent, elapsed);
        fflush(stdout);
    }


    freeShadowRayQueue(shadowQueue);
    CUDA_CHECK(cudaFree(persistentWorkCounter));

    freePathQueue(nextQueue);
    freePathQueue(currentQueue);
    freeHitResultBuffers(hitResults);
    freePathStateBuffers(pathState);
}


// ============================================================================
// === LAUNCH CONFIGURATION ===
// ============================================================================
// A hardcoded 16x16 block is only a good guess on the GPU it was written for.
// The thread budget is asked from the device instead: the occupancy API reports
// how many threads of this particular kernel the hardware can keep resident,
// which depends on its register pressure and therefore on the architecture.
//
// That budget is then spent on a square tile rather than on the widest block
// that fits. Rays inside a square tile point in similar directions and walk
// similar parts of the BVH, and measuring both shapes showed the wide variant
// losing ~3-7% on the megakernel even though it had the higher occupancy.
template <typename KernelFunction>
static dim3 chooseImageBlockShape(KernelFunction kernel, const cudaDeviceProp& deviceProperties){
    int minimumGridSize = 0;
    int threadsPerBlock = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minimumGridSize, &threadsPerBlock, kernel, 0, 0));

    threadsPerBlock = std::min(threadsPerBlock, deviceProperties.maxThreadsPerBlock);
    if (threadsPerBlock < 64) threadsPerBlock = 64;   // two warps is the smallest tile worth launching

    int blockSide = 8;
    while (blockSide * 2 <= deviceProperties.maxThreadsDim[0] &&
           blockSide * 2 <= deviceProperties.maxThreadsDim[1] &&
           (blockSide * 2) * (blockSide * 2) <= threadsPerBlock){
        blockSide *= 2;
    }

    return dim3((unsigned int)blockSide, (unsigned int)blockSide);
}

// one block per tile of the image, rounded up, clamped to the grid the device accepts
static dim3 chooseImageGridShape(dim3 blockShape, int imageWidth, int imageHeight, const cudaDeviceProp& deviceProperties){
    unsigned int gridWidth = (imageWidth + blockShape.x - 1) / blockShape.x;
    unsigned int gridHeight = (imageHeight + blockShape.y - 1) / blockShape.y;
    gridWidth = std::min(gridWidth, (unsigned int)deviceProperties.maxGridSize[0]);
    gridHeight = std::min(gridHeight, (unsigned int)deviceProperties.maxGridSize[1]);
    return dim3(gridWidth, gridHeight);
}
// ============================================================================

// ============================================================================
// === OPTIONS AND UI ===
// ============================================================================

// Every setting the user can choose lives here. The command line fills it today and the
// terminal menu will fill the very same struct later, so main() never cares where it came from.
struct RenderOptions {
    bool useWavefront = false;
    bool usePersistent = false;
    SceneChoice scene = SceneChoice::Cornell;
    int samplesPerPixelOverride = 0;
    ToneMapOption toneMap = ToneMapOption::None;
    float exposure = 1.0f;
    bool autoExposure = false;
    std::string environmentPath;
    float environmentIntensity = 1.0f;
    float environmentRotationDegrees = 0.0f;
    std::string pbrtPath;
    long long maximumTriangles = 0;
};

static RenderOptions parseCommandLine(int argc, char** argv){
    RenderOptions options;
    for(int argumentIndex = 1; argumentIndex < argc; ++argumentIndex){
        if(std::strcmp(argv[argumentIndex], "--wavefront") == 0){
            options.useWavefront = true;
        }else if(std::strcmp(argv[argumentIndex], "--persistent") == 0){
            options.usePersistent = true;
        }else if(std::strcmp(argv[argumentIndex], "--scene") == 0 && argumentIndex + 1 < argc){
            ++argumentIndex;
            for(int sceneIndex = 0; sceneIndex < sceneCount; sceneIndex++){
                if(std::strcmp(argv[argumentIndex], sceneNames[sceneIndex]) == 0) options.scene = (SceneChoice)sceneIndex;
            }
        }else if(std::strcmp(argv[argumentIndex], "--spp") == 0 && argumentIndex + 1 < argc){
            ++argumentIndex;
            options.samplesPerPixelOverride = std::atoi(argv[argumentIndex]);
        }else if(std::strcmp(argv[argumentIndex], "--tonemap") == 0 && argumentIndex + 1 < argc){
            ++argumentIndex;
            if(std::strcmp(argv[argumentIndex], "reinhard") == 0) options.toneMap = ToneMapOption::Reinhard;
            else if(std::strcmp(argv[argumentIndex], "aces") == 0) options.toneMap = ToneMapOption::ACES;
        }else if(std::strcmp(argv[argumentIndex], "--exposure") == 0 && argumentIndex + 1 < argc){
            ++argumentIndex;
            float parsedExposure = (float)std::atof(argv[argumentIndex]);
            if(std::strcmp(argv[argumentIndex], "auto") == 0){
                options.autoExposure = true;
            }else{
                float parsedExposure = (float)std::atof(argv[argumentIndex]);
                if(parsedExposure > 0.0f) options.exposure = parsedExposure;
            }
        }else if(std::strcmp(argv[argumentIndex], "--env") == 0 && argumentIndex + 1 < argc){
            options.environmentPath = argv[++argumentIndex];
        }else if(std::strcmp(argv[argumentIndex], "--env-intensity") == 0 && argumentIndex + 1 < argc){
            options.environmentIntensity = (float)std::atof(argv[++argumentIndex]);
        }else if(std::strcmp(argv[argumentIndex], "--env-rotation") == 0 && argumentIndex + 1 < argc){
            options.environmentRotationDegrees = (float)std::atof(argv[++argumentIndex]);
        }else if(std::strcmp(argv[argumentIndex], "--pbrt") == 0 && argumentIndex + 1 < argc){
            options.pbrtPath = argv[++argumentIndex];
            options.scene = SceneChoice::Pbrt;
        }else if(std::strcmp(argv[argumentIndex], "--max-triangles") == 0 && argumentIndex + 1 < argc){
            options.maximumTriangles = std::atoll(argv[++argumentIndex]);
        }
    }
    return options;
}

static std::string readLine(){
    std::string line;
    if(!std::getline(std::cin, line)) return "";
    return line;
}

// prints a numbered list, returns the chosen 0-based index, Enter keeps the default
static int askChoice(const char* title, const std::vector<std::string>& labels, int defaultIndex){
    printf("\n%s\n", title);
    for(size_t i = 0; i < labels.size(); i++){
        printf("  [%zu] %s%s\n", i + 1, labels[i].c_str(), (int)i == defaultIndex ? "   (default)" : "");
    }
    while(true){
        printf("> ");
        fflush(stdout);
        std::string line = readLine();
        if(line.empty()) return defaultIndex;
        int choice = std::atoi(line.c_str());
        if(choice >= 1 && choice <= (int)labels.size()) return choice - 1;
        printf("Please enter a number between 1 and %zu, or press Enter for the default.\n", labels.size());
    }
}

static float askNumber(const char* prompt, float defaultValue, float minimumValue){
    while(true){
        printf("\n%s [%g]: ", prompt, defaultValue);
        fflush(stdout);
        std::string line = readLine();
        if(line.empty()) return defaultValue;
        char* parsedEnd = nullptr;
        float value = std::strtof(line.c_str(), &parsedEnd);
        if(parsedEnd != line.c_str() && value >= minimumValue) return value;
        printf("Please enter a number >= %g, or press Enter for the default.\n", minimumValue);
    }
}

static RenderOptions runInteractiveMenu(RenderOptions options){
    printf("\n==================== MonoCUDA-PT ====================\n");
    printf("Press Enter to keep the default of any question.\n");

    int sceneIndex = askChoice("Scene", std::vector<std::string>(sceneDescriptions, sceneDescriptions + sceneCount), (int)options.scene);
    options.scene = (SceneChoice)sceneIndex;

    int modeIndex = askChoice("Execution model",{"megakernel (one thread per path)", "wavefront", "wavefront + persistent threads"},options.useWavefront ? (options.usePersistent ? 2 : 1) : 0);
    options.useWavefront = (modeIndex >= 1);
    options.usePersistent = (modeIndex == 2);

    int toneMapIndex = askChoice("Tone mapping", {"none", "Reinhard", "ACES approximation"}, (int)options.toneMap);
    options.toneMap = (toneMapIndex == 1) ? ToneMapOption::Reinhard
                    : (toneMapIndex == 2) ? ToneMapOption::ACES
                                          : ToneMapOption::None;

    int exposureMode = askChoice("Exposure", {"manual multiplier", "automatic (log-average luminance)"}, options.autoExposure ? 1 : 0);
    options.autoExposure = (exposureMode == 1);
    if(!options.autoExposure) options.exposure = askNumber("Exposure multiplier", options.exposure, 0.0001f);
    options.samplesPerPixelOverride = (int)askNumber("Samples per pixel (0 = scene default)", (float)options.samplesPerPixelOverride, 0.0f);

    const char* toneMapNames[] = {"none", "reinhard", "aces"};
    printf("\nSame run from the command line:\n  ./build/release/monocuda");
    if(options.useWavefront)  printf(" --wavefront");
    if(options.usePersistent) printf(" --persistent");
    printf(" --scene %s", sceneNames[(int)options.scene]);
    if(options.samplesPerPixelOverride > 0) printf(" --spp %d", options.samplesPerPixelOverride);
    if(options.autoExposure) printf(" --tonemap %s --exposure auto\n\n", toneMapNames[(int)options.toneMap]);
    else printf(" --tonemap %s --exposure %g\n\n", toneMapNames[(int)options.toneMap], options.exposure);
    return options;
}
// ============================================================================

// ============================================================================
// === MAIN ===
// ============================================================================
int main(int argc,char** argv) {

    RenderOptions options = parseCommandLine(argc, argv);
    if(argc == 1 && MONOCUDA_IS_TERMINAL(0) && MONOCUDA_IS_TERMINAL(1)){
        options = runInteractiveMenu(options);
    }
    bool usePersistentThreads = options.usePersistent;
    bool useWavefrontPathTracing = options.useWavefront;
    SceneChoice sceneChoice = options.scene;
    int samplesPerPixelOverride = options.samplesPerPixelOverride;
    ToneMapOption toneMap = options.toneMap;
    float exposure = options.exposure;
    bool autoExposure = options.autoExposure;
    if(usePersistentThreads && !useWavefrontPathTracing){
        printf("--persistent only applies to the wavefront path tracer, ignoring it (add --wavefront)\n");
    }

    // display device info
    fetchDeviceInfo();

    // display size
    g_pbrtPath = options.pbrtPath;
    if (g_pbrtPath.empty()) g_pbrtPath = scenePbrtPaths[(int)sceneChoice];
    g_pbrtTriangleBudget = options.maximumTriangles;
    g_environmentEnabled = !options.environmentPath.empty();
    hostScene scene = setup_scene(sceneChoice);

    int numberOfPixels_X = 600;
    int numberOfPixels_Y = 600;
    if (g_pbrtSettings.imageWidth > 0 && g_pbrtSettings.imageHeight > 0){
        numberOfPixels_X = g_pbrtSettings.imageWidth;
        numberOfPixels_Y = g_pbrtSettings.imageHeight;
    }
    int totalNumberOfPixels = numberOfPixels_X * numberOfPixels_Y;

    // determine the number of block and grid numbers to be used, sized against
    // the device this build is running on rather than a fixed 16x16 tile
    int currentDevice = 0;
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, currentDevice));

    // the two paths drive different image kernels, so each one is measured on its own
    dim3 block = useWavefrontPathTracing ? chooseImageBlockShape(generateCameraRaysKernel, deviceProperties)
                                         : chooseImageBlockShape(trace_sample_kernel, deviceProperties);
    dim3 grid = chooseImageGridShape(block, numberOfPixels_X, numberOfPixels_Y, deviceProperties);
    printf("Image kernel launch: block %ux%u (%u threads), grid %ux%u\n", block.x, block.y, block.x * block.y, grid.x, grid.y);

    // initialize the host and device frame buffer
    vec3* host_frameBuffer = new vec3[totalNumberOfPixels];
    vec3* device_frameBuffer;

    size_t frameBufferSize = totalNumberOfPixels * sizeof(vec3);
    CUDA_CHECK(cudaMalloc((vec3**) &device_frameBuffer,frameBufferSize));

    // initialize camera
    Camera cam;
    cam.aspect_ratio = numberOfPixels_X / float(numberOfPixels_Y);
    cam.image_width = numberOfPixels_X;
    cam.max_depth = 20;
    cam.look_from = vec3(0, 2, 5.5f);
    cam.look_at = vec3(0, 2, -2);
    cam.vfov = 40.0f;
    cam.focus_distance = 7.5f;
    cam.aperture = 0.0f;
    cam.backround_color = vec3(0, 0, 0);
    cam.samples_per_pixel = 500;

    if (samplesPerPixelOverride > 0) cam.samples_per_pixel = samplesPerPixelOverride;

    if (sceneChoice == SceneChoice::Sponza){
        // the lowest geometry of the model is the outer ground plate, so the atrium
        // floor lands at y ~ 1.23 and the camera has to sit above that, inside the atrium
        cam.look_from = vec3(-8.f, 3.f, 0.f);
        cam.look_at = vec3(8.f, 3.5f, 0.f);
        cam.vfov = 60.f;
        cam.focus_distance = 10.f;
    }
    if (sceneChoice == SceneChoice::LivingRoom){
        cam.look_from = vec3(0.f, 1.5f, 3.6f);
        cam.look_at = vec3(0.f, 1.4f, -3.f);
        cam.vfov = 60.f;
        cam.focus_distance = 6.f;
    }
    if (sceneChoice == SceneChoice::Sibenik){
        cam.look_from = vec3(-17.f, 4.f, 0.f);
        cam.look_at = vec3(15.f, 9.f, 0.f);
        cam.vfov = 60.f;
        cam.focus_distance = 20.f;
    }
    if (g_pbrtSettings.hasCamera){
        cam.look_from = g_pbrtSettings.lookFrom;
        cam.look_at = g_pbrtSettings.lookAt;
        cam.up = g_pbrtSettings.up;
        cam.vfov = g_pbrtSettings.verticalFovDegrees;
        cam.focus_distance = 1.f;
        cam.backround_color = g_pbrtSettings.backgroundColor;
    }

    cam.initialize();

    if (options.environmentPath.empty() && !g_pbrtSettings.environmentPath.empty()){
        options.environmentPath = g_pbrtSettings.environmentPath;
        options.environmentIntensity = g_pbrtSettings.environmentIntensity;
        options.environmentRotationDegrees = g_pbrtSettings.environmentRotationDegrees;
    }
    g_environmentEnabled = !options.environmentPath.empty();

    cudaArray_t environmentArray = nullptr;
    cudaTextureObject_t environmentTexture = 0;
    float* d_environmentRowCdf = nullptr;
    float* d_environmentColumnCdf = nullptr;
    {
        EnvironmentMap environmentToUpload = {0, 0, options.environmentIntensity, options.environmentRotationDegrees * MONOCUDA_PI / 180.0f, nullptr, nullptr, 0, 0, 0.0f};
        if(g_environmentEnabled){
            int environmentWidth = 0, environmentHeight = 0;
            std::vector<float> environmentPixels;
            bool isPfm = options.environmentPath.size() >= 4 && options.environmentPath.compare(options.environmentPath.size() - 4, 4, ".pfm") == 0;
            bool environmentLoaded = isPfm ? loadPfmAsRgba32f(options.environmentPath.c_str(), environmentWidth, environmentHeight, environmentPixels)
                                           : loadImageRgbaFloat(options.environmentPath.c_str(), environmentWidth, environmentHeight, environmentPixels);
            if(!environmentLoaded){
                std::fprintf(stderr, "could not load environment map '%s' (PFM, EXR or HDR)\n", options.environmentPath.c_str());
                return EXIT_FAILURE;
            }
            environmentTexture = createEnvironmentTexture(environmentPixels, environmentWidth, environmentHeight, environmentArray);
            environmentToUpload.texture = environmentTexture;
            environmentToUpload.enabled = 1;

            std::vector<float> environmentRowCdf, environmentColumnCdf;
            buildEnvironmentSamplingTables(environmentPixels, environmentWidth, environmentHeight, environmentRowCdf, environmentColumnCdf, environmentToUpload.totalWeight);
            CUDA_CHECK(cudaMalloc(&d_environmentRowCdf, environmentRowCdf.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_environmentColumnCdf, environmentColumnCdf.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_environmentRowCdf, environmentRowCdf.data(), environmentRowCdf.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_environmentColumnCdf, environmentColumnCdf.data(), environmentColumnCdf.size() * sizeof(float), cudaMemcpyHostToDevice));
            environmentToUpload.rowCdf = d_environmentRowCdf;
            environmentToUpload.columnCdf = d_environmentColumnCdf;
            environmentToUpload.width = environmentWidth;
            environmentToUpload.height = environmentHeight;
            printf("Environment map: %s (%dx%d), intensity %g, rotation %g deg\n", options.environmentPath.c_str(), environmentWidth, environmentHeight, options.environmentIntensity, options.environmentRotationDegrees);
        }
        CUDA_CHECK(cudaMemcpyToSymbol(g_environment, &environmentToUpload, sizeof(EnvironmentMap)));
    }

    // === BVH MORTON CODE SORT AND DEVICE COPY ===
    std::sort(scene.primitiveRefs.begin(), scene.primitiveRefs.end(),[](const PrimitiveRef& a, const PrimitiveRef& b) {
        return a.mortonCode < b.mortonCode;
    });

    int numPrimitives = (int)scene.primitiveRefs.size();

    std::vector<unsigned int> sortedMortonCodes(numPrimitives);
    std::vector<int> sortedPrimitiveIDs(numPrimitives);
    std::vector<AABB> sortedBounds(numPrimitives);
    std::vector<int> sortedPrimitiveTypes(numPrimitives);

    for (int i = 0; i < numPrimitives; i++) {
        sortedMortonCodes[i] = scene.primitiveRefs[i].mortonCode;
        sortedPrimitiveIDs[i] = scene.primitiveRefs[i].primitiveIndex;
        sortedBounds[i] = scene.primitiveRefs[i].bounds;
        sortedPrimitiveTypes[i] = scene.primitiveRefs[i].primitiveType;
    }

    unsigned int* d_sortedMortonCodes;
    int* d_sortedPrimitiveIDs;
    AABB* d_sortedBounds;
    int* d_sortedPrimitiveTypes;

    CUDA_CHECK(cudaMalloc(&d_sortedMortonCodes, numPrimitives * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_sortedPrimitiveIDs, numPrimitives * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sortedBounds, numPrimitives * sizeof(AABB)));
    CUDA_CHECK(cudaMalloc(&d_sortedPrimitiveTypes, numPrimitives * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_sortedMortonCodes, sortedMortonCodes.data(), numPrimitives * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sortedPrimitiveIDs, sortedPrimitiveIDs.data(), numPrimitives * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sortedBounds, sortedBounds.data(), numPrimitives * sizeof(AABB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sortedPrimitiveTypes, sortedPrimitiveTypes.data(), numPrimitives * sizeof(int), cudaMemcpyHostToDevice));

    BVH bvh;
    bvh.generateHierarchy(d_sortedMortonCodes, d_sortedPrimitiveIDs, d_sortedBounds, d_sortedPrimitiveTypes, numPrimitives);

    CUDA_CHECK(cudaFree(d_sortedMortonCodes));
    CUDA_CHECK(cudaFree(d_sortedPrimitiveIDs));
    CUDA_CHECK(cudaFree(d_sortedBounds));
    CUDA_CHECK(cudaFree(d_sortedPrimitiveTypes));

    // call the render_kernel
    sphere* d_spheres;
    int sphereCount = (int)scene.spheres.size();
    CUDA_CHECK(cudaMalloc(&d_spheres, sphereCount * sizeof(sphere)));
    CUDA_CHECK(cudaMemcpy(d_spheres, scene.spheres.data(),sphereCount * sizeof(sphere), cudaMemcpyHostToDevice));

    triangle* d_triangles;
    int triangleCount = (int)scene.triangles.size();
    CUDA_CHECK(cudaMalloc(&d_triangles, triangleCount * sizeof(triangle)));
    CUDA_CHECK(cudaMemcpy(d_triangles, scene.triangles.data(),triangleCount * sizeof(triangle), cudaMemcpyHostToDevice));

    vec3* d_accumBuffer;
    CUDA_CHECK(cudaMalloc(&d_accumBuffer, frameBufferSize));
    CUDA_CHECK(cudaMemset(d_accumBuffer, 0, frameBufferSize));

    curandState* d_rngStates;
    CUDA_CHECK(cudaMalloc(&d_rngStates, totalNumberOfPixels * sizeof(curandState)));
    init_rng_kernel<<<grid, block>>>(d_rngStates, numberOfPixels_X, numberOfPixels_Y, 1234ULL);
    CUDA_CHECK_KERNEL();

    Light* d_lights;
    int lightCount = (int)scene.lights.size();
    CUDA_CHECK(cudaMalloc(&d_lights, lightCount * sizeof(Light)));
    CUDA_CHECK(cudaMemcpy(d_lights, scene.lights.data(), lightCount * sizeof(Light), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(device_frameBuffer, 0, frameBufferSize));




    auto renderStart = std::chrono::steady_clock::now();
    if(useWavefrontPathTracing){
        // Wavefront init
        WavefrontRenderContext wavefrontContext;
        wavefrontContext.camera = cam;
        wavefrontContext.bvh = bvh;
        wavefrontContext.spheres = d_spheres;
        wavefrontContext.sphereCount = sphereCount;
        wavefrontContext.triangles = d_triangles;
        wavefrontContext.triangleCount = triangleCount;
        wavefrontContext.lights = d_lights;
        wavefrontContext.lightCount = lightCount;
        wavefrontContext.accumBuffer = d_accumBuffer;
        wavefrontContext.frameBuffer = device_frameBuffer;
        wavefrontContext.randomNumberStates = d_rngStates;
        wavefrontContext.imageWidth = numberOfPixels_X;
        wavefrontContext.imageHeight = numberOfPixels_Y;
        wavefrontContext.gridDimensions = grid;
        wavefrontContext.blockDimensions = block;
        wavefrontContext.toneMap = toneMap;
        wavefrontContext.exposure = exposure;
        wavefrontContext.usePersistentThreads = usePersistentThreads;
        bool sceneHasMedium = false;
        for (const sphere& s : scene.spheres)    if (s.mat.type == MaterialType::Isotropic) sceneHasMedium = true;
        for (const triangle& t : scene.triangles) if (t.mat.type == MaterialType::Isotropic) sceneHasMedium = true;
        wavefrontContext.sceneHasMedium = sceneHasMedium;
        renderWithWavefront(wavefrontContext);
    }else{
        for (int s = 0; s < cam.samples_per_pixel; s++) {
            trace_sample_kernel<<<grid, block>>>(d_accumBuffer, cam, bvh, d_spheres, sphereCount, d_triangles, triangleCount, numberOfPixels_X, numberOfPixels_Y, d_rngStates, s,d_lights,lightCount);
            CUDA_CHECK_KERNEL();

            resolve_kernel<<<grid, block>>>(d_accumBuffer, device_frameBuffer,numberOfPixels_X, numberOfPixels_Y, s + 1,toneMap,exposure);
            CUDA_CHECK_KERNEL();

            // PROGRESS BAR
            float percent = 100.0f * float(s + 1) / float(cam.samples_per_pixel);
            auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - renderStart).count();
            printf("\rSample %d / %d (%.1f%%) — %.3fs elapsed", s + 1, cam.samples_per_pixel, percent, elapsed);
            fflush(stdout);
        }
    }


    printf("\nLight triangles in the scene: %zu\n", scene.lights.size());
    printf("\n");
    if(autoExposure){
        float chosenExposure = computeAutoExposure(d_accumBuffer, numberOfPixels_X, numberOfPixels_Y, cam.samples_per_pixel, grid, block);
        printf("Auto exposure: %.3f\n", chosenExposure);
        resolve_kernel<<<grid, block>>>(d_accumBuffer, device_frameBuffer, numberOfPixels_X, numberOfPixels_Y, cam.samples_per_pixel, toneMap, chosenExposure);
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaMemcpy(host_frameBuffer,device_frameBuffer,frameBufferSize,cudaMemcpyDeviceToHost));

    // draw the ppm
    const char* outputFileName = useWavefrontPathTracing ? "render_wavefront.png" : "render_megakernel.png";
    std::vector<unsigned char> outputPixels((size_t)numberOfPixels_X * numberOfPixels_Y * 3);
    for (int j = 0; j < numberOfPixels_Y; j++) {
        for (int i = 0; i < numberOfPixels_X; i++) {
            size_t pixel_index = numberOfPixels_X * j + i;

            vec3 c = host_frameBuffer[pixel_index];
            // NaN shield
            float rf = isfinite(c.x) ? fminf(fmaxf(c.x, 0.0f), 1.0f) : 0.0f;
            float gf = isfinite(c.y) ? fminf(fmaxf(c.y, 0.0f), 1.0f) : 0.0f;
            float bf = isfinite(c.z) ? fminf(fmaxf(c.z, 0.0f), 1.0f) : 0.0f;

            unsigned char* targetPixel = &outputPixels[pixel_index * 3];
            targetPixel[0] = (unsigned char)(255.99f * rf);
            targetPixel[1] = (unsigned char)(255.99f * gf);
            targetPixel[2] = (unsigned char)(255.99f * bf);
        }
    }
    if (!saveImageRgb8Png(outputFileName, numberOfPixels_X, numberOfPixels_Y, outputPixels)) {
        std::fprintf(stderr, "failed to write %s\n", outputFileName);
        return EXIT_FAILURE;
    }

    // free the memory
    CUDA_CHECK(cudaFree(device_frameBuffer));
    CUDA_CHECK(cudaFree(d_spheres));
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_accumBuffer));
    CUDA_CHECK(cudaFree(d_rngStates));
    CUDA_CHECK(cudaFree(d_lights));
    for(size_t i = 0; i < scene.imageTextureHandles.size(); i++){
        CUDA_CHECK(cudaDestroyTextureObject(scene.imageTextureHandles[i]));
    }
    for(size_t i = 0; i < scene.imageArrays.size(); i++){
        CUDA_CHECK(cudaFreeArray(scene.imageArrays[i]));
    }
    if(environmentTexture) CUDA_CHECK(cudaDestroyTextureObject(environmentTexture));
    if(environmentArray) CUDA_CHECK(cudaFreeArray(environmentArray));
    if(d_environmentRowCdf) CUDA_CHECK(cudaFree(d_environmentRowCdf));
    if(d_environmentColumnCdf) CUDA_CHECK(cudaFree(d_environmentColumnCdf));
    bvh.free();
    delete[] host_frameBuffer;

    return 0;
}
// ============================================================================
