// ==============================================================================
// This project is fully written by Bora Yalçın (Undergraduate CSE Student at Yeditepe University)
// Checkout out my website for related blogs : https://borayalcinn.github.io/
// Checkout the related repository           : https://github.com/BoraYalcinn/MonoCUDA-PT 
// I have shared the sources such as papers, websites and learning courses etc. under the README.mdHitResultBuffers
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
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
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

__host__ __device__ Texture make_solid_texture(vec3 color) {
    return Texture{TextureType::Solid, color, vec3(0,0,0), 1.f, 0};
}

__host__ __device__ Texture make_checker_texture(vec3 colorA, vec3 colorB, float checkerScale) {
    return Texture{TextureType::Checker, colorA, colorB, checkerScale, 0};
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
    textureDescription.sRGB = 0;      // 1 once we output gamma-corrected images

    cudaTextureObject_t textureHandle = 0;
    CUDA_CHECK(cudaCreateTextureObject(&textureHandle, &resourceDescription, &textureDescription, nullptr));
    return textureHandle;
}

__host__ __device__ Texture make_image_texture(cudaTextureObject_t imageHandle) {
    return Texture{TextureType::Image, vec3(1,1,1), vec3(0,0,0), 1.f, imageHandle};
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

enum class MaterialType {Lambertian, Dielectric, Conductor,Emissive};

struct Material{
    MaterialType type;
    Texture albedoTexture;
    vec3 emittedColor;
    float fuzz;
    float refractionIndex;

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

// ============================================================================


// ============================================================================
// === AABB ===
// ============================================================================

class AABB{
public:
    vec3 minInterval;
    vec3 maxInterval;
    vec3 centroid;

    __host__ __device__ AABB(){}
    __host__ __device__ AABB(const vec3& minInterval_,const vec3& maxInterval_) : minInterval(minInterval_) , maxInterval(maxInterval_){}


};

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
// ===============================================================================
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
                nearestHit.hitNormal = front ? n : -n;
                vec2 triangleUv = computeTriangleUvFromHitPoint(triangles[primitiveIdx], nearestHit.hitPoint);
                nearestHit.u = triangleUv.x;
                nearestHit.v = triangleUv.y;
            }
        }
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

enum class SceneChoice {Cornell, Sponza};

// == FORWARD DECLERATIONS =======================================================
static bool loadObject(hostScene& scene,const char* filePath,Material mat,vec3 translation,float uniformScale,float rotationAroundYRadians);
// ===============================================================================

__host__ void buildCornellScene(hostScene& scene) {

    Material floor = make_lambertian(make_checker_texture(vec3(0.9f,0.9f,0.9f), vec3(0.1f,0.1f,0.1f), 8.f));
    Material white  = make_lambertian(vec3(0.73f, 0.73f, 0.73f));
    Material red = make_lambertian(vec3(0.65f, 0.05f, 0.05f));
    Material green  = make_lambertian(vec3(0.12f, 0.45f, 0.15f));
    Material mirror = make_conductor(vec3(0.9f, 0.9f, 0.9f), 0.02f);
    Material light  = make_emissive(vec3(15.0f, 15.0f, 15.0f));

    const float L = -2.f, R = 2.f, B = 0.f, T = 4.f, F = -4.f, N = 0.f;
    add_quad(scene, vec3(L,B,N), vec3(R,B,N), vec3(R,B,F), vec3(L,B,F), floor);  // floor
    add_quad(scene, vec3(L,T,N), vec3(R,T,N), vec3(R,T,F), vec3(L,T,F), white);  // ceiling
    add_quad(scene, vec3(L,B,F), vec3(R,B,F), vec3(R,T,F), vec3(L,T,F), white);  // back
    add_quad(scene, vec3(L,B,N), vec3(L,B,F), vec3(L,T,F), vec3(L,T,N), red);    // left
    add_quad(scene, vec3(R,B,N), vec3(R,B,F), vec3(R,T,F), vec3(R,T,N), green);  // right
    
    add_quad(scene, vec3(-1.f,T-0.01f,-2.6f), vec3(1.f,T-0.01f,-2.6f),vec3(1.f,T-0.01f,-1.4f), vec3(-1.f,T-0.01f,-1.4f), light);

    vec3 boxCenter(-0.8f, 1.1f, -2.8f);
    add_box(scene, boxCenter, vec3(0.55f, 1.1f, 0.55f), 0.3f, white);
    vec3 bigCenter(0.85f, 0.7f, -1.7f);
    scene.spheres.push_back(sphere(bigCenter, 0.7f, mirror));
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

__host__ void buildSponzaScene(hostScene& scene){
    Material wallMaterial = make_lambertian(vec3(0.75f, 0.72f, 0.68f));
    if (!loadObject(scene, "models/sponza.obj", wallMaterial, vec3(0.f, 0.f, 0.f), 37.f, 0.f)){
        std::fprintf(stderr, "could not load models/sponza.obj\n");
        std::exit(EXIT_FAILURE);
    }

    // sponza has no light of its own, a quad that fills the open roof acts as the sky.
    // after the scale above the opening sits at x [-9, 8.5] and z [-1.8, 1.9], anything
    // wider than that ends up buried in the roof and only wastes shadow rays
    Material skylight = make_emissive(vec3(6.f, 6.f, 6.f));
    const float skylightHeight = 14.6f;
    add_quad(scene, vec3(-8.8f, skylightHeight, -1.6f), vec3(8.3f, skylightHeight, -1.6f), vec3(8.3f, skylightHeight, 1.7f), vec3(-8.8f, skylightHeight, 1.7f), skylight);
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

__host__ hostScene setup_scene(SceneChoice sceneChoice){
    hostScene scene;
    if (sceneChoice == SceneChoice::Sponza){
        buildSponzaScene(scene);
    }else{
        buildCornellScene(scene);
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
static ObjVertexReference parseObjFaceToken(const std::string& token,int positionCount,int uvCount){
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
static cudaTextureObject_t getOrLoadObjTexture(hostScene& scene,const std::string& ppmPath,std::map<std::string,cudaTextureObject_t>& textureCache){
    std::map<std::string,cudaTextureObject_t>::iterator cachedEntry = textureCache.find(ppmPath);
    if (cachedEntry != textureCache.end()) return cachedEntry->second;

    cudaTextureObject_t textureHandle = 0;
    int imageWidth = 0, imageHeight = 0;
    std::vector<unsigned char> imagePixels;
    if (loadPpmAsRgba8(ppmPath.c_str(), imageWidth, imageHeight, imagePixels)){
        cudaArray_t imageArray = nullptr;
        textureHandle = createImageTextureObject(imagePixels, imageWidth, imageHeight, imageArray);
        scene.imageArrays.push_back(imageArray);
        scene.imageTextureHandles.push_back(textureHandle);
    } else {
        printf("could not load texture '%s' (convert the png to a binary ppm first)\n", ppmPath.c_str());
    }
    textureCache[ppmPath] = textureHandle;
    return textureHandle;
}

// reads newmtl / Kd / map_Kd, every material becomes a lambertian (flat color or image texture)
// Ka, Ks, bump maps and alpha masks are ignored, so are map_Kd options like "-s 2 2 2"
static bool loadMtlFile(hostScene& scene,const std::string& mtlPath,const std::string& objectDirectory,ObjMaterialLibrary& library,std::map<std::string,cudaTextureObject_t>& textureCache,int& missingTextureCount){
    std::ifstream file(mtlPath);
    if (!file.is_open()) return false;

    std::string currentName;
    vec3 currentColor(0.8f, 0.8f, 0.8f);
    std::string currentTexturePath;
    bool hasCurrentMaterial = false;

    // pushes the material that was being read, called when the next one starts and at the end of the file
    auto finishCurrentMaterial = [&](){
        if (!hasCurrentMaterial) return;
        Material finishedMaterial = make_lambertian(currentColor);
        if (!currentTexturePath.empty()){
            std::string ppmPath = objectDirectory + "/" + convertTexturePathToPpm(currentTexturePath);
            cudaTextureObject_t textureHandle = getOrLoadObjTexture(scene, ppmPath, textureCache);
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
            currentTexturePath.clear();
            hasCurrentMaterial = true;
        } else if (keyword == "Kd" && hasCurrentMaterial){
            float red, green, blue;
            lineStream >> red >> green >> blue;
            currentColor = vec3(red, green, blue);
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
    std::vector<ObjTriangleCorners> faces;

    ObjMaterialLibrary materialLibrary;
    std::map<std::string,cudaTextureObject_t> textureCache;
    int missingTextureCount = 0;
    int currentMaterialIndex = -1;

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
            currentMaterialIndex = (foundMaterial != materialLibrary.indexByName.end()) ? foundMaterial->second : -1;
        } else if (keyword == "f"){
            std::vector<ObjVertexReference> corners;
            std::string token;
            while (lineStream >> token){
                corners.push_back(parseObjFaceToken(token, (int)positions.size(), (int)uvs.size()));
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
            sampleColor = sampleColor + attenuation * camera.backround_color;
            break;
        }
        float emissionWeight = 1.0f;
        bool emissionComesFromBsdfSampling = !(depth == 0 || previousBounceWasSpecular);
        if (emissionComesFromBsdfSampling && hit.mat.type == MaterialType::Emissive && hit.primitiveType == 1) {
            float cosineAtLight = fmaxf(fabsf(r.direction.dot(hit.hitNormal)), 1e-4f);
            float lightArea = triangleSurfaceArea(triangles[hit.primitiveIndex]);
            float lightPdf = lightSolidAnglePdf(1.0f / float(lightCount), hit.closestDistance, cosineAtLight, lightArea);
            emissionWeight = powerHeuristic(previousBsdfPdf, lightPdf);
        }
        sampleColor = sampleColor + attenuation * hit.mat.emittedColor * emissionWeight;

        if (hit.mat.type == MaterialType::Emissive) {
            break;   
        }

        attenuation = attenuation * sampleTexture(hit.mat.albedoTexture, hit.u, hit.v);

        // ---- NEXT EVENT ESTIMATION (direct light) ----
        if (hit.mat.type == MaterialType::Lambertian && lightCount > 0) {
            int selectedLightIndex = min((int)(curand_uniform(&rngState) * lightCount), lightCount - 1);
            const Light& selectedLight = lights[selectedLightIndex];
            float lightSelectionProbability = 1.0f / float(lightCount);

            vec3 sampledPointOnLight = sample_triangle(selectedLight.sourceTriangle, &rngState);
            vec3 vectorToLight = sampledPointOnLight - hit.hitPoint;
            float distanceToLight = vectorToLight.length();
            vec3 directionToLight = vectorToLight / distanceToLight;

            float cosineAtSurface = hit.hitNormal.dot(directionToLight);
            float cosineAtLight = fabsf(selectedLight.surfaceNormal.dot(directionToLight));

            if (cosineAtSurface > 0.0f && cosineAtLight > 1e-4f) {
                hitRecord shadowHit = bvh.traverseRay(hit.hitPoint, directionToLight, r.time, spheres, triangles, distanceToLight * 0.999f);

                if (!shadowHit.didHit) {
                    float solidAnglePdf = lightSelectionProbability * (distanceToLight * distanceToLight) / (cosineAtLight * selectedLight.surfaceArea);
                    vec3 lightColor = selectedLight.sourceTriangle.mat.emittedColor;
                    float neeWeight = powerHeuristic(solidAnglePdf, lambertianCosinePdf(cosineAtSurface));
                    sampleColor = sampleColor + attenuation * lightColor * (cosineAtSurface / (MONOCUDA_PI * solidAnglePdf)) * neeWeight;
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
            case MaterialType::Lambertian:
            default:
                newDirection = lambertian_scatter_direction(hit.hitNormal, &rngState);
                break;
        }
        previousBounceWasSpecular = (hit.mat.type != MaterialType::Lambertian);
        previousBsdfPdf = lambertianCosinePdf(fmaxf(newDirection.dot(hit.hitNormal), 0.0f));
        r = Ray(hit.hitPoint, newDirection, r.time);   
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

__global__ void resolve_kernel(const vec3* accumBuffer, vec3* outputBuffer,int maximumX, int maximumY, int samplesSoFar) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= maximumX || j >= maximumY) return;
    int pixel_index = maximumX * j + i;

    outputBuffer[pixel_index] = accumBuffer[pixel_index] / float(samplesSoFar);
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
    unsigned char* didHit = nullptr;
    float* hitDistance = nullptr;
    
    float* hitPointX = nullptr;
    float* hitPointY = nullptr;   
    float* hitPointZ = nullptr;
    
    float* hitNormalX = nullptr; 
    float* hitNormalY = nullptr;  
    float* hitNormalZ = nullptr;
    
    unsigned char* hitWasFrontFace = nullptr;
    int* hitPrimitiveType = nullptr;     // 0 = sphere, 1 = triangle 
    int* hitPrimitiveIndex = nullptr;
    float* hitU = nullptr;
    float* hitV = nullptr;
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
    allocateDeviceArray(buffers.didHit, pathCapacity);
    allocateDeviceArray(buffers.hitDistance, pathCapacity);

    allocateDeviceArray(buffers.hitPointX, pathCapacity);
    allocateDeviceArray(buffers.hitPointY, pathCapacity);
    allocateDeviceArray(buffers.hitPointZ, pathCapacity);

    allocateDeviceArray(buffers.hitNormalX, pathCapacity);
    allocateDeviceArray(buffers.hitNormalY, pathCapacity);
    allocateDeviceArray(buffers.hitNormalZ, pathCapacity);

    allocateDeviceArray(buffers.hitWasFrontFace, pathCapacity);
    allocateDeviceArray(buffers.hitPrimitiveType, pathCapacity);
    allocateDeviceArray(buffers.hitPrimitiveIndex, pathCapacity);

    // for textures uv
    allocateDeviceArray(buffers.hitU, pathCapacity);
    allocateDeviceArray(buffers.hitV, pathCapacity);

}

static void freeHitResultBuffers(HitResultBuffers& buffers) {
    freeDeviceArray(buffers.didHit);
    freeDeviceArray(buffers.hitDistance);

    freeDeviceArray(buffers.hitPointX);
    freeDeviceArray(buffers.hitPointY);
    freeDeviceArray(buffers.hitPointZ);

    freeDeviceArray(buffers.hitNormalX);
    freeDeviceArray(buffers.hitNormalY);
    freeDeviceArray(buffers.hitNormalZ);

    freeDeviceArray(buffers.hitWasFrontFace);
    freeDeviceArray(buffers.hitPrimitiveType);
    freeDeviceArray(buffers.hitPrimitiveIndex);

    // for textures uv
    freeDeviceArray(buffers.hitU);
    freeDeviceArray(buffers.hitV);
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

    hitResults.didHit[pathIndex] = hit.didHit ? 1 : 0;
    if (!hit.didHit) return;

    hitResults.hitDistance[pathIndex] = hit.closestDistance;
    hitResults.hitPointX[pathIndex] = hit.hitPoint.x;
    hitResults.hitPointY[pathIndex] = hit.hitPoint.y;
    hitResults.hitPointZ[pathIndex] = hit.hitPoint.z;
    hitResults.hitNormalX[pathIndex] = hit.hitNormal.x;
    hitResults.hitNormalY[pathIndex] = hit.hitNormal.y;
    hitResults.hitNormalZ[pathIndex] = hit.hitNormal.z;
    hitResults.hitWasFrontFace[pathIndex] = hit.frontFace ? 1 : 0;
    hitResults.hitPrimitiveType[pathIndex] = hit.primitiveType;
    hitResults.hitPrimitiveIndex[pathIndex] = hit.primitiveIndex;
    hitResults.hitU[pathIndex] = hit.u;
    hitResults.hitV[pathIndex] = hit.v;
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

    // missed
    if (!hitResults.didHit[pathIndex]) {
        currentSample = currentSample + throughput * backgroundColor;
        pathState.currentSampleRed[pathIndex] = currentSample.x;
        pathState.currentSampleGreen[pathIndex] = currentSample.y;
        pathState.currentSampleBlue[pathIndex] = currentSample.z;
        return;
    }

    // material
    int primitiveIndex = hitResults.hitPrimitiveIndex[pathIndex];
    int primitiveType = hitResults.hitPrimitiveType[pathIndex];
    Material material = (primitiveType == 0) ? spheres[primitiveIndex].mat : triangles[primitiveIndex].mat;

    vec3 hitPoint(hitResults.hitPointX[pathIndex], hitResults.hitPointY[pathIndex], hitResults.hitPointZ[pathIndex]);
    vec3 hitNormal(hitResults.hitNormalX[pathIndex], hitResults.hitNormalY[pathIndex], hitResults.hitNormalZ[pathIndex]);
    vec3 incomingRayDirection(pathState.rayDirectionX[pathIndex], pathState.rayDirectionY[pathIndex], pathState.rayDirectionZ[pathIndex]);

    // emission (MIS weighted when it comes from BSDF sampling)
    if (material.type == MaterialType::Emissive) {
        bool previousBounceWasSpecular = pathState.previousBounceWasSpecular[pathIndex] != 0;
        float emissionWeight = 1.0f;
        bool emissionComesFromBsdfSampling = !(depth == 0 || previousBounceWasSpecular);
        if (emissionComesFromBsdfSampling && primitiveType == 1) {
            float cosineAtLight = fmaxf(fabsf(incomingRayDirection.dot(hitNormal)), 1e-4f);
            float lightArea = triangleSurfaceArea(triangles[primitiveIndex]);
            float lightPdf = lightSolidAnglePdf(1.0f / float(lightCount), hitResults.hitDistance[pathIndex], cosineAtLight, lightArea);
            emissionWeight = powerHeuristic(pathState.previousBsdfPdf[pathIndex], lightPdf);
        }
        currentSample = currentSample + throughput * material.emittedColor * emissionWeight;
        pathState.currentSampleRed[pathIndex] = currentSample.x;
        pathState.currentSampleGreen[pathIndex] = currentSample.y;
        pathState.currentSampleBlue[pathIndex] = currentSample.z;
        return;
    }

    // throughput (with albedo)
    float hitU = hitResults.hitU[pathIndex];
    float hitV = hitResults.hitV[pathIndex];
    throughput = throughput * sampleTexture(material.albedoTexture, hitU, hitV);

    curandState randomNumberState = randomNumberStates[pathIndex];

    // NEE again same as before !
    if (material.type == MaterialType::Lambertian && lightCount > 0) {
        int selectedLightIndex = min((int)(curand_uniform(&randomNumberState) * lightCount), lightCount - 1);
        const Light& selectedLight = lights[selectedLightIndex];
        float lightSelectionProbability = 1.0f / float(lightCount);

        vec3 sampledPointOnLight = sample_triangle(selectedLight.sourceTriangle, &randomNumberState);
        vec3 vectorToLight = sampledPointOnLight - hitPoint;
        float distanceToLight = vectorToLight.length();
        vec3 directionToLight = vectorToLight / distanceToLight;

        float cosineAtSurface = hitNormal.dot(directionToLight);
        float cosineAtLight = fabsf(selectedLight.surfaceNormal.dot(directionToLight));

        if (cosineAtSurface > 0.0f && cosineAtLight > 1e-4f) {
            float solidAnglePdf = lightSelectionProbability * (distanceToLight * distanceToLight) / (cosineAtLight * selectedLight.surfaceArea);
            vec3 lightColor = selectedLight.sourceTriangle.mat.emittedColor;
            float neeWeight = powerHeuristic(solidAnglePdf, lambertianCosinePdf(cosineAtSurface));
            vec3 contribution = throughput * lightColor * (cosineAtSurface / (MONOCUDA_PI * solidAnglePdf)) * neeWeight;

            int shadowSlot = atomicAdd(shadowQueue.itemCount, 1);
            shadowQueue.originX[shadowSlot] = hitPoint.x;
            shadowQueue.originY[shadowSlot] = hitPoint.y;
            shadowQueue.originZ[shadowSlot] = hitPoint.z;
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
        bool frontFace = hitResults.hitWasFrontFace[pathIndex] != 0;

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
            case MaterialType::Lambertian:
            default:
                newDirection = lambertian_scatter_direction(hitNormal, &randomNumberState);
                break;
        }
        
        pathState.rayOriginX[pathIndex] = hitPoint.x;
        pathState.rayOriginY[pathIndex] = hitPoint.y;
        pathState.rayOriginZ[pathIndex] = hitPoint.z;
        pathState.rayDirectionX[pathIndex] = newDirection.x;
        pathState.rayDirectionY[pathIndex] = newDirection.y;
        pathState.rayDirectionZ[pathIndex] = newDirection.z;
        pathState.throughputRed[pathIndex] = throughput.x;
        pathState.throughputGreen[pathIndex] = throughput.y;
        pathState.throughputBlue[pathIndex] = throughput.z;
        pathState.previousBounceWasSpecular[pathIndex] = (material.type != MaterialType::Lambertian) ? 1 : 0;
            pathState.previousBsdfPdf[pathIndex] = lambertianCosinePdf(fmaxf(newDirection.dot(hitNormal), 0.0f));

        int nextQueueSlot = atomicAdd(nextQueue.itemCount, 1);
        nextQueue.pathIndices[nextQueueSlot] = pathIndex;
    }

    randomNumberStates[pathIndex] = randomNumberState;
}

__global__ void shadowKernel(BVH bvh, const sphere* spheres, const triangle* triangles,
                             PathStateBuffers pathState, ShadowRayQueue shadowQueue)
{
    int shadowSlot = threadIdx.x + blockIdx.x * blockDim.x;
    if (shadowSlot >= *shadowQueue.itemCount) return;

    int pathIndex = shadowQueue.pathIndices[shadowSlot];
    vec3 shadowRayOrigin(shadowQueue.originX[shadowSlot], shadowQueue.originY[shadowSlot], shadowQueue.originZ[shadowSlot]);
    vec3 shadowRayDirection(shadowQueue.directionX[shadowSlot], shadowQueue.directionY[shadowSlot], shadowQueue.directionZ[shadowSlot]);
    float rayTime = pathState.rayTime[pathIndex];     // motion blur same as before

    bool isOccluded = bvh.traverseAnyHit(shadowRayOrigin, shadowRayDirection, rayTime, spheres, triangles, shadowQueue.maximumDistance[shadowSlot]);

    if (!isOccluded) {
        
        pathState.currentSampleRed[pathIndex] += shadowQueue.contributionRed[shadowSlot];
        pathState.currentSampleGreen[pathIndex] += shadowQueue.contributionGreen[shadowSlot];
        pathState.currentSampleBlue[pathIndex] += shadowQueue.contributionBlue[shadowSlot];
    }
}

__global__ void debugShowRayDirectionKernel(vec3* frameBuffer, int imageWidth, int imageHeight, PathStateBuffers pathState){
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;
    int pathIndex = imageWidth * pixelY + pixelX;

    frameBuffer[pathIndex] = vec3(pathState.rayDirectionX[pathIndex] * 1.5f + 0.5f,pathState.rayDirectionY[pathIndex] * 1.5f + 0.5f,pathState.rayDirectionZ[pathIndex] * -1.0f);
}

__global__ void debugShowHitNormalKernel(vec3* frameBuffer, int imageWidth, int imageHeight, HitResultBuffers hitResults)
{
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;
    int pathIndex = imageWidth * pixelY + pixelX;

    if (!hitResults.didHit[pathIndex]){ 
        frameBuffer[pathIndex] = vec3(0, 0, 0); 
        return; 
    }
    frameBuffer[pathIndex] = vec3(hitResults.hitNormalX[pathIndex] * 0.5f + 0.5f,
                                  hitResults.hitNormalY[pathIndex] * 0.5f + 0.5f,
                                  hitResults.hitNormalZ[pathIndex] * 0.5f + 0.5f);
}

__global__ void finalizeSampleKernel(vec3* accumBuffer, int imageWidth, int imageHeight, PathStateBuffers pathState)
{
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

            extendKernel<<<blockCount, blockSize>>>(context.bvh, context.spheres, context.triangles,pathState, hitResults, currentQueue);
            CUDA_CHECK_KERNEL();

            shadeKernel<<<blockCount, blockSize>>>(depth, context.camera.backround_color,context.spheres, context.triangles, context.lights, context.lightCount,context.randomNumberStates, pathState, hitResults, currentQueue, nextQueue, shadowQueue);
            CUDA_CHECK_KERNEL();

            shadowKernel<<<blockCount, blockSize>>>(context.bvh, context.spheres, context.triangles,pathState, shadowQueue);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpy(&activePathCount, nextQueue.itemCount, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(currentQueue, nextQueue);
        }

        finalizeSampleKernel<<<context.gridDimensions, context.blockDimensions>>>(context.accumBuffer, context.imageWidth, context.imageHeight, pathState);
        CUDA_CHECK_KERNEL();

        resolve_kernel<<<context.gridDimensions, context.blockDimensions>>>(context.accumBuffer, context.frameBuffer, context.imageWidth, context.imageHeight, sampleIndex + 1);
        CUDA_CHECK_KERNEL();

        float percent = 100.0f * float(sampleIndex + 1) / float(context.camera.samples_per_pixel);
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - renderStart).count();
        printf("\rSample %d / %d (%.1f%%) — %.3fs elapsed", sampleIndex + 1, context.camera.samples_per_pixel, percent, elapsed);
        fflush(stdout);
    }
    

    freeShadowRayQueue(shadowQueue);
    freePathQueue(nextQueue);
    freePathQueue(currentQueue);
    freeHitResultBuffers(hitResults);
    freePathStateBuffers(pathState);
}


// ============================================================================
// === MAIN ===
// ============================================================================
int main(int argc,char** argv) {

    bool useWavefrontPathTracing = false;
    SceneChoice sceneChoice = SceneChoice::Cornell;
    int samplesPerPixelOverride = 0;
    for(int argumentIndex = 1; argumentIndex < argc; ++argumentIndex){
        if(std::strcmp(argv[argumentIndex], "--wavefront") == 0){
            useWavefrontPathTracing = true;
        }else if(std::strcmp(argv[argumentIndex], "--scene") == 0 && argumentIndex + 1 < argc){
            ++argumentIndex;
            if(std::strcmp(argv[argumentIndex], "sponza") == 0) sceneChoice = SceneChoice::Sponza;
        }else if(std::strcmp(argv[argumentIndex], "--spp") == 0 && argumentIndex + 1 < argc){
            ++argumentIndex;
            samplesPerPixelOverride = std::atoi(argv[argumentIndex]);
        }
    }


    // display device info
    fetchDeviceInfo();
    
    // display size
    int numberOfPixels_X = 600;
    int numberOfPixels_Y = 600;     
    int totalNumberOfPixels = numberOfPixels_X * numberOfPixels_Y;

    // determine the number of block and grid numbers to be used
    dim3 block(16, 16);
    dim3 grid((numberOfPixels_X + block.x - 1) / block.x, (numberOfPixels_Y + block.y - 1) / block.y);

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
  
    cam.initialize();

    // scene setup
    hostScene scene = setup_scene(sceneChoice);

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
        renderWithWavefront(wavefrontContext);
    }else{
        for (int s = 0; s < cam.samples_per_pixel; s++) {
            trace_sample_kernel<<<grid, block>>>(d_accumBuffer, cam, bvh, d_spheres, sphereCount, d_triangles, triangleCount, numberOfPixels_X, numberOfPixels_Y, d_rngStates, s,d_lights,lightCount);
            CUDA_CHECK_KERNEL();

            resolve_kernel<<<grid, block>>>(d_accumBuffer, device_frameBuffer,numberOfPixels_X, numberOfPixels_Y, s + 1);
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

    CUDA_CHECK(cudaMemcpy(host_frameBuffer,device_frameBuffer,frameBufferSize,cudaMemcpyDeviceToHost));

    // draw the ppm
    const char* outputFileName = useWavefrontPathTracing ? "render_wavefront.ppm" : "render_megakernel.ppm";
    FILE* out = fopen(outputFileName, "w");
    if (!out) {
        std::fprintf(stderr, "failed to open %s for writing\n",outputFileName);
        return EXIT_FAILURE;
    }
    fprintf(out, "P3\n%d %d\n255\n", numberOfPixels_X, numberOfPixels_Y);
    for (int j = 0; j < numberOfPixels_Y; j++) {
        for (int i = 0; i < numberOfPixels_X; i++) {
            size_t pixel_index = numberOfPixels_X * j + i;
            
            vec3 c = host_frameBuffer[pixel_index];
            // NaN shield
            float rf = isfinite(c.x) ? fminf(fmaxf(c.x, 0.0f), 1.0f) : 0.0f;
            float gf = isfinite(c.y) ? fminf(fmaxf(c.y, 0.0f), 1.0f) : 0.0f;
            float bf = isfinite(c.z) ? fminf(fmaxf(c.z, 0.0f), 1.0f) : 0.0f;

            int r = int(255.99 * rf);
            int g = int(255.99 * gf);
            int b = int(255.99 * bf);
            fprintf(out, "%d %d %d\n", r, g, b);
        }
    }
    fclose(out);

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
    bvh.free();
    delete[] host_frameBuffer;
    

    return 0;
}
// ============================================================================
