// ==============================================================================
// This project is fully written by Bora Yalçın (Undergraduate CSE Student at Yeditepe University)
// Checkout out my website for related blogs : https://borayalcinn.github.io/
// Checkout the related repository           : https://github.com/BoraYalcinn/MonoCUDA-PT 
// ==============================================================================
// === INCLUDE AND MACROS ===
// ==============================================================================
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
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
            float cosTheta1 = fminf((-direction).dot(surfaceNormal), 1.0f);   // acos'a güvenli argüman
            float theta1 = acosf(cosTheta1);

            float sinTheta2 = sinf(theta1) / refraction_cof; 
            if (sinTheta2 > 1.0f) {
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
};

// ============================================================================

// ============================================================================
// === MATERIALS ===
// ============================================================================
__device__ float schlick_approx(float cosine,float refraction_cof){
    float r0 = (1.f - refraction_cof) / (1.f + refraction_cof);
    r0 *= r0;
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
        schlick_approx(cosTheta, refraction_cof);
    }
    if (curand_uniform(rngState) < reflectProbability) {
        return unitDirection.reflect(surfaceNormal);
    }
    return unitDirection.refract(surfaceNormal, refraction_cof);
}
// ============================================================================



// ============================================================================
// === Mesh & Primitive Types ===
// ============================================================================
__device__ float edge_function(const vec3& a, const vec3& b, const vec3& c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

struct triangle {
    vec3 v0, v1, v2;
    Material mat;

    __host__ __device__ triangle() {}
    __host__ __device__ triangle(vec3 v0_, vec3 v1_, vec3 v2_) : v0(v0_), v1(v1_), v2(v2_) {}
    __host__ __device__ triangle(vec3 v0_, vec3 v1_, vec3 v2_,Material mat_) : v0(v0_), v1(v1_), v2(v2_), mat(mat_) {}
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

    __host__ __device__ sphere(){}
    __host__ __device__ sphere(vec3 center_,double radius_) : center(center_), radius(radius_){}
    __host__ __device__ sphere(vec3 center_,double radius_,Material mat_) : center(center_), radius(radius_), mat(mat_){} 
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
    int samples_per_pixel = 10;
    int max_depth = 10;

    float vfov = 90.f;
    vec3 look_from = vec3(0,0,0);
    vec3 look_at = vec3(0,0,-1);
    vec3 up = vec3(0,1,0);

    vec3 center;
    vec3 pixel00_location;
    vec3 pixel_delta_u, pixel_delta_v;
    vec3 u, v, w;

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
};

__host__ hostScene setup_scene() {
    hostScene scene;

    Material redDiffuse{MaterialType::Lambertian, vec3(0.8f,0.3f,0.3f), vec3(0,0,0), 0.f, 0.f};
    Material groundMat{MaterialType::Lambertian, vec3(0.5f,1.0f,0.5f), vec3(0,0,0), 0.f, 0.f};
    Material mirrorMat{MaterialType::Conductor, vec3(0.9f,0.9f,0.9f), vec3(0,0,0), 0.f, 0.f};
    Material glassMat{MaterialType::Dielectric, vec3(1.0f,1.0f,1.0f), vec3(0,0,0), 0.f, 1.5f};

    scene.spheres.push_back(sphere(vec3(0,0,-1), 0.5f, redDiffuse));
    scene.spheres.push_back(sphere(vec3(1,0,-1), 0.3f, mirrorMat));
    scene.spheres.push_back(sphere(vec3(-1,0,-1), 0.4f, glassMat));
    scene.spheres.push_back(sphere(vec3(0, -100.5f, -1), 100.0f, groundMat));


    return scene;
}

// ============================================================================

// ============================================================================
// === RENDER ===
// ============================================================================

__global__ void trace_sample_kernel(vec3* accumBuffer, Camera camera,sphere* spheres,int sphereCount,triangle* triangles,int triangleCount,
                                    int maximumX,int maximumY,unsigned long long seed,int sampleIndex){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= maximumX || j >= maximumY) return;
    int pixel_index = maximumX * j + i;

    curandState rngState;
    curand_init(seed,pixel_index,sampleIndex,&rngState);

    Ray r = camera.getRay(i, j, &rngState);
    vec3 attenuation(1.0f, 1.0f, 1.0f);
    vec3 sampleColor(0.0f, 0.0f, 0.0f);

    for(int depth = 0 ; depth < camera.max_depth;depth++){
        hitRecord hit = find_nearest_hit(r.origin, r.direction,spheres,sphereCount,triangles,triangleCount);
        
        if(!hit.didHit){
            vec3 unitDir = r.direction.normalize();
            float t = 0.5f * (unitDir.y + 1.0f);
            vec3 skyColor = vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
            sampleColor = attenuation * skyColor;
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
            case MaterialType::Conductor:
                newDirection = specular_scatter_direction(r.direction, hit.hitNormal);
                break;
            case MaterialType::Dielectric:
                float ratio;
                if(hit.frontFace){
                    ratio = 1.0f / hit.mat.refractionIndex;
                }else{
                    ratio = hit.mat.refractionIndex;
                }
                newDirection = dielectric_scatter_direction(r.direction, hit.hitNormal, ratio, &rngState);
                break;
            case MaterialType::Lambertian:
            default:
                newDirection = lambertian_scatter_direction(hit.hitNormal, &rngState);
                break;
        }
        r = Ray(hit.hitPoint, newDirection);
    }
    accumBuffer[pixel_index] = accumBuffer[pixel_index] + sampleColor;
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

    // initialize camera
    Camera cam;
    cam.aspect_ratio = numberOfPixels_X / float(numberOfPixels_Y);
    cam.image_width = numberOfPixels_X;
    cam.samples_per_pixel = 50;
    cam.max_depth = 20;
    cam.vfov = 50.0f;
    cam.look_from = vec3(0, 0.5f, 1.6f);   
    cam.look_at = vec3(0, 0, -1);           
    cam.up = vec3(0, 1, 0);
    cam.aperture = 0.1f;
    cam.focus_distance = 1.8f;
    cam.initialize();

    // scene setup
    hostScene scene = setup_scene();

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
    
    auto renderStart = std::chrono::steady_clock::now();
    for (int s = 0; s < cam.samples_per_pixel; s++) {
        trace_sample_kernel<<<grid, block>>>(d_accumBuffer, cam, d_spheres, sphereCount,d_triangles, triangleCount,numberOfPixels_X, numberOfPixels_Y,1234ULL, s);
        CUDA_CHECK_KERNEL();

        resolve_kernel<<<grid, block>>>(d_accumBuffer, device_frameBuffer,numberOfPixels_X, numberOfPixels_Y, s + 1);
        CUDA_CHECK_KERNEL();

        // PROGRESS BAR
        float percent = 100.0f * float(s + 1) / float(cam.samples_per_pixel);
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - renderStart).count();
        printf("\rSample %d / %d (%.1f%%) — %.1fs elapsed", s + 1, cam.samples_per_pixel, percent, elapsed);
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
            int r = int(255.99 * host_frameBuffer[pixel_index].x);
            int g = int(255.99 * host_frameBuffer[pixel_index].y);
            int b = int(255.99 * host_frameBuffer[pixel_index].z);
            fprintf(out, "%d %d %d\n", r, g, b);
        }
    }
    fclose(out);

    // free the memory
    CUDA_CHECK(cudaFree(device_frameBuffer));
    CUDA_CHECK(cudaFree(d_spheres));
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_accumBuffer));   
    delete[] host_frameBuffer;

    return 0;
}
// ============================================================================