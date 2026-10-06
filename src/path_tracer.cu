// ==============================================================================
// This project is fully written by Bora Yalçın (Undergraduate CSE Student at Yeditepe University)
// Checkout out my website for related blogs : https://borayalcinn.github.io/
// Checkout the related repository           : https://github.com/BoraYalcinn/MonoCUDA-PT 
// I have shared the sources such as papers, websites and learning courses etc. under the README.md
// ==============================================================================
// === INCLUDE AND MACROS ===
// ==============================================================================
#include <cstdio>
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
    vec3 albedo;
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
    return Material{MaterialType::Lambertian, albedo, vec3(0,0,0), 0.f, 0.f};
}

__host__ __device__ Material make_conductor(vec3 albedo, float fuzz) {
    return Material{MaterialType::Conductor, albedo, vec3(0,0,0), fuzz < 1 ? fuzz : 1.f, 0.f};
}

__host__ __device__ Material make_dielectric(float refractionIndex) {
    return Material{MaterialType::Dielectric, vec3(1,1,1), vec3(0,0,0), 0.f, refractionIndex};
}

__host__ __device__ Material make_emissive(vec3 emittedColor) {
    return Material{MaterialType::Emissive, vec3(0,0,0), emittedColor, 0.f, 0.f};
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
__device__ float edge_function(const vec3& a, const vec3& b, const vec3& c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

struct triangle {
    vec3 v0, v1, v2;
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

struct sphere {
    vec3 center;
    float radius;
    Material mat;
    AABB aabb;

    __host__ __device__ sphere(){}
    __host__ __device__ sphere(vec3 center_,double radius_) : center(center_), radius(radius_){}
    __host__ __device__ sphere(vec3 center_,double radius_,Material mat_) : center(center_), radius(radius_), mat(mat_){} 

    __host__ __device__ vec3 calculateCentroid(){
        return center;
    }
    __host__ __device__ void calculate_bounding_box(){
        aabb.minInterval = {(center.x - radius),(center.y - radius),(center.z - radius)};
        aabb.maxInterval = {(center.x + radius),(center.y + radius),(center.z + radius)};
    }
    __host__ __device__ AABB get_bounding_box(){return aabb;}
};

__device__ bool hit_sphere(const sphere& targetSphere,const vec3& rayOrigin, const vec3& rayDirection,float& intersectionDistance,bool& frontFace, vec3& outwardNormal) {
    vec3 vectorFromRayOriginToSphereCenter = targetSphere.center - rayOrigin;
    float distanceToClosestPointOnRay =  vectorFromRayOriginToSphereCenter.dot(rayDirection) / rayDirection.dot(rayDirection);
    vec3 closestPointOnRay = rayOrigin + rayDirection * distanceToClosestPointOnRay;

    float distanceSquaredFromCenterToClosest = (targetSphere.center - closestPointOnRay).length_squared();
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
    outwardNormal = (hitPoint - targetSphere.center).normalize();
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
// === Ray & HitRecord ===
// ============================================================================
struct Ray{
    vec3 origin;
    vec3 direction;

    __device__ Ray(){}
    __device__ Ray(vec3 origin_,vec3 direction_) : origin(origin_),direction(direction_) {}

    __device__ vec3 isAt(double t) const {
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
};

__device__ hitRecord find_nearest_hit(const vec3& rayOrig,const vec3& rayDir,
                                        const sphere* sphereArray,const int sphereCount,
                                        const triangle* triangleArray,const int triangleCount){
    hitRecord nearestHit;

    bool frontFace;
    vec3 outwardNormal;
    for(int i = 0; i < sphereCount;i++ ){
        float t;
        if(hit_sphere(sphereArray[i],rayOrig,rayDir,t,frontFace,outwardNormal)){
            if(t < nearestHit.closestDistance && t > 0.0001f){
                nearestHit.didHit = true;
                nearestHit.closestDistance = t;
                nearestHit.hitPoint = rayOrig + rayDir * t;
                nearestHit.hitNormal = (nearestHit.hitPoint - sphereArray[i].center).normalize();
                nearestHit.mat = sphereArray[i].mat;
                nearestHit.frontFace = frontFace;
                if(frontFace){
                    nearestHit.hitNormal = outwardNormal;
                }else{
                    nearestHit.hitNormal = -outwardNormal;
                }
            }
        }
    }

    for (int i = 0; i < triangleCount; i++) {
    float t;
    if (hit_triangle(triangleArray[i], rayOrig, rayDir, t)) {
        if (t < nearestHit.closestDistance && t > 0.0001f) {
            nearestHit.didHit = true;
            nearestHit.closestDistance = t;
            nearestHit.hitPoint = rayOrig + rayDir * t;
            nearestHit.mat = triangleArray[i].mat;

            vec3 edge1 = triangleArray[i].v1 - triangleArray[i].v0;
            vec3 edge2 = triangleArray[i].v2 - triangleArray[i].v0;
            nearestHit.hitNormal = edge1.cross(edge2).normalize();
        }
    }
}
    return nearestHit;
}
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
__device__ inline void testPrimitiveHit(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection, const sphere* spheres, const triangle* triangles, hitRecord& nearestHit);
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

    __device__ hitRecord traverseRay(const vec3& rayOrigin, const vec3& rayDirection, const sphere* spheres, const triangle* triangles) const {
        hitRecord nearestHit;

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
                testPrimitiveHit(primitiveType, primitiveIdx, rayOrigin, rayDirection, spheres, triangles, nearestHit);
            }
            if (overlapR && childRIsLeaf) {
                int primitiveIdx = d_leafNode[childRIdx].primitiveIndex;
                int primitiveType = d_primitiveTypes[childRIdx];
                testPrimitiveHit(primitiveType, primitiveIdx, rayOrigin, rayDirection, spheres, triangles, nearestHit);
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
    int d = (delta(sortedMortonCodes, numObjects, idx, idx+1) -
             delta(sortedMortonCodes, numObjects, idx, idx-1)) >= 0 ? 1 : -1;

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

__device__ inline int findSplit(const unsigned int* sortedMortonCodes, int first, int last){
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode  = sortedMortonCodes[last];

    if (firstCode == lastCode) return (first + last) >> 1;

    int commonPrefix = __clz(firstCode ^ lastCode);

    int split = first;
    int step  = last - first;
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if (newSplit < last){
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
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
        AABB leftBounds  = node.leftIsLeaf  ? d_leafNode[node.leftChild].bounds  : d_internalNodes[node.leftChild].bounds;
        AABB rightBounds = node.rightIsLeaf ? d_leafNode[node.rightChild].bounds : d_internalNodes[node.rightChild].bounds;
        node.bounds = merge(leftBounds, rightBounds);

        currentParent = node.parent;
    }
}

__global__ void buildInternalNodesKernel(BVH_Node* d_internalNodes, BVH_Node* d_leafNode,const unsigned int* sortedMortonCodes, int numObjects){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numObjects - 1) return;

    int2 range = determineRange(sortedMortonCodes, numObjects, idx);
    int first = range.x;
    int last  = range.y;
    int split = findSplit(sortedMortonCodes, first, last);

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

__device__ inline void testPrimitiveHit(int primitiveType, int primitiveIdx, const vec3& rayOrigin, const vec3& rayDirection, const sphere* spheres, const triangle* triangles, hitRecord& nearestHit){
    if (primitiveType == 0) {
        float intersectionDistance;
        bool frontFace;
        vec3 outwardNormal;
        if (hit_sphere(spheres[primitiveIdx], rayOrigin, rayDirection, intersectionDistance, frontFace, outwardNormal)) {
            if (intersectionDistance < nearestHit.closestDistance && intersectionDistance > 0.0001f) {
                nearestHit.didHit = true;
                nearestHit.closestDistance = intersectionDistance;
                nearestHit.hitPoint = rayOrigin + rayDirection * intersectionDistance;
                nearestHit.mat = spheres[primitiveIdx].mat;
                nearestHit.frontFace = frontFace;
                nearestHit.hitNormal = frontFace ? outwardNormal : -outwardNormal;
            }
        }
    } else {
        float intersectionDistance;
        if (hit_triangle(triangles[primitiveIdx], rayOrigin, rayDirection, intersectionDistance)) {
            if (intersectionDistance < nearestHit.closestDistance && intersectionDistance > 0.0001f) {
                nearestHit.didHit = true;
                nearestHit.closestDistance = intersectionDistance;
                nearestHit.hitPoint = rayOrigin + rayDirection * intersectionDistance;
                nearestHit.mat = triangles[primitiveIdx].mat;

                vec3 edge1 = triangles[primitiveIdx].v1 - triangles[primitiveIdx].v0;
                vec3 edge2 = triangles[primitiveIdx].v2 - triangles[primitiveIdx].v0;
                vec3 n = edge1.cross(edge2).normalize();

                bool front = rayDirection.dot(n) < 0.0f;     
                nearestHit.frontFace = front;
                nearestHit.hitNormal = front ? n : -n;
            }
        }
    }
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

    __device__ Ray getRay(int pixelX,int pixelY,curandState* rngState) const{
        vec3 pixelCenter = pixel00_location + (pixel_delta_u * float(pixelX)) + (pixel_delta_v * float(pixelY));
        
        vec3 randomPointOnLens = random_in_unit_disk(rngState) * lens_radius;
        vec3 lensOffSetInWorldSpace = u * randomPointOnLens.x + v * randomPointOnLens.y;
        vec3 rayOriginOnLens = center + lensOffSetInWorldSpace;

        vec3 rayDirection = (pixelCenter - rayOriginOnLens).normalize();
        return Ray(rayOriginOnLens,rayDirection);
    }
};
// ============================================================================



// ============================================================================
// === SCENE SETUP ===
// ============================================================================

struct hostScene {
    std::vector<sphere> spheres;
    std::vector<triangle> triangles;
    std::vector<PrimitiveRef> primitiveRefs;
    AABB bounds;
};

static void add_quad(hostScene& s, vec3 a, vec3 b, vec3 c, vec3 d, Material m){
    s.triangles.push_back(triangle(a, b, c, m));
    s.triangles.push_back(triangle(a, c, d, m));
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

__host__ hostScene setup_scene() {
    hostScene scene;

    
    Material white  = make_lambertian(vec3(0.73f, 0.73f, 0.73f));
    Material red = make_lambertian(vec3(0.65f, 0.05f, 0.05f));
    Material green  = make_lambertian(vec3(0.12f, 0.45f, 0.15f));
    Material mirror = make_conductor(vec3(0.9f, 0.9f, 0.9f), 0.02f);
    Material light  = make_emissive(vec3(15.0f, 15.0f, 15.0f));

    const float L = -2.f, R = 2.f, B = 0.f, T = 4.f, F = -4.f, N = 0.f;
    add_quad(scene, vec3(L,B,N), vec3(R,B,N), vec3(R,B,F), vec3(L,B,F), white);  // floor
    add_quad(scene, vec3(L,T,N), vec3(R,T,N), vec3(R,T,F), vec3(L,T,F), white);  // ceiling
    add_quad(scene, vec3(L,B,F), vec3(R,B,F), vec3(R,T,F), vec3(L,T,F), white);  // back
    add_quad(scene, vec3(L,B,N), vec3(L,B,F), vec3(L,T,F), vec3(L,T,N), red);    // left
    add_quad(scene, vec3(R,B,N), vec3(R,B,F), vec3(R,T,F), vec3(R,T,N), green);  // right
    
    add_quad(scene, vec3(-1.f,T-0.01f,-2.6f), vec3(1.f,T-0.01f,-2.6f),vec3(1.f,T-0.01f,-1.4f), vec3(-1.f,T-0.01f,-1.4f), light);

    vec3 boxCenter(-0.8f, 1.1f, -2.8f);
    add_box(scene, boxCenter, vec3(0.55f, 1.1f, 0.55f), 0.3f, white);
    vec3 bigCenter(0.85f, 0.7f, -1.7f);
    scene.spheres.push_back(sphere(bigCenter, 0.7f, mirror));

    
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
            Material m;
            if (chooseMat < 0.8f)       m = make_lambertian(vec3(rand()/(float)RAND_MAX, rand()/(float)RAND_MAX, rand()/(float)RAND_MAX));
            else if (chooseMat < 0.95f) m = make_conductor(vec3(0.5f + 0.5f*(rand()/(float)RAND_MAX), 0.5f + 0.5f*(rand()/(float)RAND_MAX), 0.5f + 0.5f*(rand()/(float)RAND_MAX)), 0.3f*(rand()/(float)RAND_MAX));
            else                        m = make_dielectric(1.5f);
            scene.spheres.push_back(sphere(c, r, m));
        }
    }

    for (size_t i = 0; i < scene.spheres.size(); i++) scene.spheres[i].calculate_bounding_box();
    for (size_t i = 0; i < scene.triangles.size(); i++) scene.triangles[i].calculate_bounding_box();

    AABB sceneBounds = scene.spheres[0].aabb;
    for (size_t i = 1; i < scene.spheres.size(); i++)   sceneBounds = merge(sceneBounds, scene.spheres[i].aabb);
    for (size_t i = 0; i < scene.triangles.size(); i++) sceneBounds = merge(sceneBounds, scene.triangles[i].aabb);
    scene.bounds = sceneBounds;

    for (size_t i = 0; i < scene.spheres.size(); i++) {
        PrimitiveRef ref;
        ref.bounds = scene.spheres[i].aabb;
        ref.centroid = scene.spheres[i].center;
        ref.primitiveType  = 0;
        ref.primitiveIndex = i;
        ref.mortonCode = morton3D(ref.centroid,scene.bounds);
        scene.primitiveRefs.push_back(ref);
        
    }
    for (size_t i = 0; i < scene.triangles.size(); i++) {
        PrimitiveRef ref;
        ref.bounds = scene.triangles[i].aabb;
        ref.centroid = scene.triangles[i].calculateCentroid();
        ref.primitiveType  = 1;
        ref.primitiveIndex = i;
        ref.mortonCode = morton3D(ref.centroid,scene.bounds);
        scene.primitiveRefs.push_back(ref);
        
    }
    return scene;
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
                                    int maximumX, int maximumY, curandState* rngStates, int sampleIndex){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= maximumX || j >= maximumY) return;
    int pixel_index = maximumX * j + i;

    curandState rngState = rngStates[pixel_index];

    Ray r = camera.getRay(i, j, &rngState);
    vec3 attenuation(1.0f, 1.0f, 1.0f);
    vec3 sampleColor(0.0f, 0.0f, 0.0f);

    for(int depth = 0 ; depth < camera.max_depth;depth++){
        hitRecord hit = bvh.traverseRay(r.origin, r.direction, spheres, triangles);
        
        if(!hit.didHit){
            sampleColor = sampleColor + attenuation * camera.backround_color;
            break;
        }
        sampleColor = sampleColor + attenuation * hit.mat.emittedColor;

        if (hit.mat.type == MaterialType::Emissive) {
            break;   
        }

        attenuation =  attenuation * hit.mat.albedo;
        
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
        r = Ray(hit.hitPoint, newDirection);
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
// === MAIN ===
// ============================================================================
int main() {

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
  
    cam.initialize();

    // scene setup
    hostScene scene = setup_scene();

    // === BVH: Morton koduna göre sort + cihaza kopyala ===
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

    auto renderStart = std::chrono::steady_clock::now();
    for (int s = 0; s < cam.samples_per_pixel; s++) {
        trace_sample_kernel<<<grid, block>>>(d_accumBuffer, cam, bvh, d_spheres, sphereCount, d_triangles, triangleCount, numberOfPixels_X, numberOfPixels_Y, d_rngStates, s);
        CUDA_CHECK_KERNEL();

        resolve_kernel<<<grid, block>>>(d_accumBuffer, device_frameBuffer,numberOfPixels_X, numberOfPixels_Y, s + 1);
        CUDA_CHECK_KERNEL();

        // PROGRESS BAR
        float percent = 100.0f * float(s + 1) / float(cam.samples_per_pixel);
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - renderStart).count();
        printf("\rSample %d / %d (%.1f%%) — %.3fs elapsed", s + 1, cam.samples_per_pixel, percent, elapsed);
        fflush(stdout);
    }
    printf("\n");

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
    bvh.free();
    delete[] host_frameBuffer;
    

    return 0;
}
// ============================================================================
