    // ==============================================================================
    // === INCLUDE AND MACROS ===
    // ==============================================================================
    #include <cstdio>
    #include <cstdlib>
    #include <iostream>
    #include "cuda_runtime.h"
    #include <vector>

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
            __host__ __device__ vec3 operator-() const {return vec3{-x,-y,-z};}
            __host__ __device__ vec3 operator*(float n) const { return vec3{x*n,y*n,z*n};}
            __host__ __device__ vec3 operator/(float n) const { return vec3{x/n,y/n,z/n};}


            __host__ __device__ float length_squared() const {
                return x*x + y*y + z*z;
            }

            __host__ __device__ float length() const {
                return __sqrtf(this->length_squared());
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
        

        __host__ __device__ sphere(){}
        __host__ __device__ sphere(vec3 center_,double radius_) : center(center_), radius(radius_){}
    };

    __device__ bool hit_sphere(const sphere& targetSphere,const vec3& rayOrigin, const vec3& rayDirection,float& intersectionDistance) {
        vec3 vectorFromRayOriginToSphereCenter = targetSphere.center - rayOrigin;
        float distanceToClosestPointOnRay =  vectorFromRayOriginToSphereCenter.dot(rayDirection) / rayDirection.dot(rayDirection);
        vec3 closestPointOnRay = rayOrigin + rayDirection * distanceToClosestPointOnRay;

        float distanceSquaredFromCenterToClosest = (targetSphere.center - closestPointOnRay).length_squared();
        float discriminant = targetSphere.radius * targetSphere.radius - distanceSquaredFromCenterToClosest;

        if (discriminant < 0.0f) {
            return false;   
        }
        float halfChordLength = sqrtf(discriminant);
        intersectionDistance = distanceToClosestPointOnRay - halfChordLength;
        return true;
    }
    // ============================================================================


    // ============================================================================
    // === Ray & HitRecord ===
    // ============================================================================
    struct ray{
        vec3 origin;
        vec3 direction;

        __device__ ray(){}
        __device__ ray(vec3 origin_,vec3 direction_) : origin(origin_),direction(direction_) {}

        __device__ vec3 isAt(double t) const {
            return origin + direction * t; 
        }
    };

    struct hitRecord{
        bool didHit =  false;
        float closestDistance = 1e30f;
        vec3 hitPoint;
        vec3 hitNormal;
    };

    __device__ hitRecord find_nearest_hit(const vec3& rayOrig,const vec3& rayDir,
                                            const sphere* sphereArray,const int sphereCount,
                                            const triangle* triangleArray,const int triangleCount){
        hitRecord nearestHit;

        for(int i = 0; i < sphereCount;i++ ){
            float t;
            if(hit_sphere(sphereArray[i],rayOrig,rayDir,t)){
                if(t < nearestHit.closestDistance && t > 0.0001f){
                    nearestHit.didHit = true;
                    nearestHit.closestDistance = t;
                    nearestHit.hitPoint = rayOrig + rayDir * t;
                    nearestHit.hitNormal = (nearestHit.hitPoint - sphereArray[i].center).normalize();
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
    // === Materials ===
    // ============================================================================

    // ============================================================================

    // ============================================================================
    // === CAMERA ===
    // ============================================================================
    __host__ __device__ float degrees_to_radians(float degree) {
        return degree * (3.14159265358979323846f / 180.0f);
    }

    class Camera {
    public:
        float aspec_ratio = 1.f;
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

        float focus_distance = 10;

        __host__ __device__ Camera(){}

        __host__ __device__ void initialize() {
            w = (look_from - look_at).normalize();
            u = up.cross(w).normalize();
            v = w.cross(u);

            image_height = int(image_width / aspec_ratio);
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

        __device__ ray getRay(int i,int j) const{
            vec3 pixelCenter = pixel00_location + (pixel_delta_u * float(i)) + (pixel_delta_v * float(j));
            vec3 rayDirection = (pixelCenter - center).normalize();
            return ray(center,rayDirection);
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

        scene.spheres.push_back(sphere(vec3(0,0,-1), 0.5f));
        scene.spheres.push_back(sphere(vec3(1,0,-1), 0.3f));
        scene.spheres.push_back(sphere(vec3(-1,0,-1), 0.4f));

        scene.triangles.push_back(triangle(vec3(-2,-0.5,-2), vec3(2,-0.5,-2), vec3(0,2,-2)));

        return scene;
    }

    // ============================================================================

    // ============================================================================
    // === RENDER ===
    // ============================================================================

    __global__ void render_kernel(vec3* h_fb,Camera camera,sphere* spheres,int sphereCount,triangle* triangles,int triangleCount,int maximumX, int maximumY ){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        if((i >= maximumX) || (j >= maximumY)) return;
        ray r = camera.getRay(i,j);
        hitRecord hit = find_nearest_hit(r.origin, r.direction,spheres, sphereCount,triangles, triangleCount);
        int pixel_index = maximumX * j + i;
        if (hit.didHit) {
            h_fb[pixel_index] = (hit.hitNormal + vec3(1.0f, 1.0f, 1.0f)) * 0.5f;
        } else {
            vec3 unitDir = r.direction.normalize();
            float t = 0.5f * (unitDir.y + 1.0f);
            h_fb[pixel_index] = vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
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

        // initialize camera
        Camera cam;
        cam.aspec_ratio = numberOfPixels_X / float(numberOfPixels_Y);
        cam.image_width = numberOfPixels_X;
        cam.samples_per_pixel = 50;
        cam.max_depth = 20;
        cam.vfov = 90.0f;
        cam.look_from = vec3(0, 1, 3);
        cam.look_at = vec3(0, 0, 0);
        cam.up = vec3(0, 1, 0);
        cam.initialize();

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

        render_kernel<<<grid, block>>>(device_frameBuffer, cam,d_spheres, sphereCount,d_triangles, triangleCount,numberOfPixels_X, numberOfPixels_Y);
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
        CUDA_CHECK(cudaFree(device_frameBuffer));
        CUDA_CHECK(cudaFree(d_spheres));
        CUDA_CHECK(cudaFree(d_triangles));
        delete[] host_frameBuffer;

        return 0;
    }
    // ============================================================================