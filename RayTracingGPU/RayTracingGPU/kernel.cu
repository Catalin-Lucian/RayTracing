#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "camera_cuda.h"
#include "hitable_cuda.h"
#include "world_cuda.h"
#include "image_cuda.h"
#include "interval_cuda.h"
#include "material_cuda.h"
#include "ray_cuda.h"
#include "sphere_cuda.h"
#include "vec3_cuda.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define RND (curand_uniform(&local_rand_state))


void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// =============================== constant memory ===============================
__constant__ camera d_camera;
__constant__ world d_world;

// ===============================================================================

__device__ vec3 get_color(const ray& r,const world& world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = make_vec3(1.0f, 1.0f, 1.0f);
   
    // 50 iterations for ray bounce
    for (int i = 0; i < 1; i++) {
        record rec;

        if (hit(world, cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;

            if (scatter(rec.material, cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return make_vec3(0.0f, 0.0f, 1.0f);
            }
        }
        else {
            vec3 unit_direction = make_unit(cur_ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            vec3 c = (1.0f - t) * make_vec3(1.0f, 1.0f, 1.0f) + t * make_vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
}

__global__ void rand_pixels_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(2024 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(
    vec3* image,
    int max_x, int max_y,
    int ns,
    int startRow, int endRow,
    int startCol, int endCol,
    curandState* rand_state
) {
    // Global thread indices
    int i_global = threadIdx.x + blockIdx.x * blockDim.x;
    int j_global = threadIdx.y + blockIdx.y * blockDim.y;

    // Local thread indices within the segment
    int i = startCol + i_global;
    int j = startRow + j_global;

    if (i >= startCol && i < endCol && j >= startRow && j < endRow) {
        int pixel_index = j * max_x + i;
        curandState local_rand_state = rand_state[pixel_index];

        color col = make_vec3(0.f, 0.f, 0.f);
        for (int s = 0; s < 1; s++) {
            float u = (i + curand_uniform(&local_rand_state)) / float(max_x);
            float v = (j + curand_uniform(&local_rand_state)) / float(max_y);
            ray r = get_ray(d_camera, u, v, &local_rand_state);
            col += get_color(r, d_world, &local_rand_state);
        }

        //col /= float(ns);
        image[pixel_index] = col;
    }
}

void create_world(int nx, int ny, camera& h_camera, world& h_world) {

    h_world.objects[0] = make_sphere(make_vec3(0.f, -1000.0f, -1.f), 1000.f, make_lambertian(make_vec3(0.5f, 0.5f, 0.5f)));
    int idx = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = random_float();
            vec3 center = make_vec3(a + random_float(), 0.2f, b + random_float());
            color material_color;
            material mat;
            if (choose_mat < 0.8f) {
                material_color = make_color(random_float() * random_float(), random_float() * random_float(), random_float() * random_float());
                mat = make_lambertian(material_color);
            }
            else if (choose_mat < 0.95f) {
                material_color = make_color(random_float(0.5f, 1.f), random_float(0.5f, 1.f), random_float(0.5f, 1.f));
                mat = make_metal(material_color, random_float(0.f, 0.5f));
            }
            else {
                material_color = make_color(1.0f, 1.0f, 1.0f);
                mat = make_dielectric(1.5f);
            }
            h_world.objects[idx++]= make_sphere(center, 0.2f, mat);
        }
    }

    set_sphere(h_world.objects[idx++], make_vec3(0.f, 1.f, 0.f), 1.0f, make_dielectric(1.5f));
    set_sphere(h_world.objects[idx++], make_vec3(-4.f, 1.f, 0.f), 1.0f, make_lambertian(make_vec3(0.4f, 0.2f, 0.1f)));
    set_sphere(h_world.objects[idx++], make_vec3(4.f, 1.f, 0.f), 1.0f, make_metal(make_vec3(0.7f, 0.6f, 0.5f), 0.0f));
    h_world.size = idx;


    vec3 lookfrom = make_vec3(13.f, 2.f, 3.f);
    vec3 lookat = make_vec3(0.f, 0.f, 0.f);
    float dist_to_focus = length(lookfrom - lookat);
    float aperture = 0.1f;
    init_camera(
        h_camera,
        lookfrom,
        lookat,
        make_vec3(0.f, 1.f, 0.f),
        30.f,
        float(nx) / float(ny),
        aperture,
        dist_to_focus
    );
}

int main() {
    int nx = 1920; // width
    int ny = 1080; // heigth
    int ns = 500; // numar de sample uri
    int tile_size_x = 16; 
    int tile_size_y = 16;

    Image image(nx, ny);

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tile_size_x << "x" << tile_size_y << " blocks.\n";

    int num_pixels = nx * ny;
    size_t pixels_size = num_pixels * sizeof(color);
    color* d_image_pixels;
    checkCudaErrors(cudaMalloc((void**) & d_image_pixels, pixels_size));
    
    // allocate random state
    curandState* d_rand_state_pixels;
    checkCudaErrors(cudaMalloc(&d_rand_state_pixels, num_pixels * sizeof(curandState)));

    //// clock to measure time
    clock_t start, stop;
    start = clock();
    
    // initialize word and camera 
    camera h_camera;
    world h_world;
    create_world(nx, ny, h_camera, h_world);

    //  copy camera and world to constant memory
    //! ignore the warning: the symbol is passed corectly
    checkCudaErrors(cudaMemcpyToSymbol(d_camera, &h_camera, sizeof(camera)));
    checkCudaErrors(cudaMemcpyToSymbol(d_world, &h_world, sizeof(world)));

    // calculate blocks and threads
    dim3 blocks((nx + tile_size_x - 1) / tile_size_x, (ny + tile_size_y - 1) / tile_size_y);
    dim3 threads(tile_size_x, tile_size_y);

    // init random state for each pixel
    rand_pixels_init <<< blocks, threads >> > (nx, ny, d_rand_state_pixels);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Create streams
    int num_streams_height = 1; // Number of streams along height
    int num_streams_width = 3;  // Number of streams along width
    int total_streams = num_streams_height * num_streams_width;
    cudaStream_t* streams = new cudaStream_t[total_streams];

    for (int i = 0; i < total_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Determine segment sizes
    int segmentSizeHeight = ny / num_streams_height;
    int segmentSizeWidth = nx / num_streams_width;

    // Launch kernels for each segment
    for (int h = 0; h < num_streams_height; ++h) {
        for (int w = 0; w < num_streams_width; ++w) {
            int streamIdx = h * num_streams_width + w;

            int startRow = h * segmentSizeHeight;
            int endRow = (h + 1) * segmentSizeHeight;
            if (h == num_streams_height - 1) endRow = ny;

            int startCol = w * segmentSizeWidth;
            int endCol = (w + 1) * segmentSizeWidth;
            if (w == num_streams_width - 1) endCol = nx;

            // Calculate blocks for this segment
            dim3 segment_blocks(
                (endCol - startCol + tile_size_x - 1) / tile_size_x,
                (endRow - startRow + tile_size_y - 1) / tile_size_y);

            render <<< segment_blocks, threads, 0, streams[streamIdx] >> > (
                d_image_pixels, nx, ny, ns, startRow, endRow, startCol, endCol, d_rand_state_pixels);
        }
    }

    // Așteptarea finalizării tuturor streams
    for (int i = 0; i < total_streams; ++i) {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }

    color* h_image_pixels = new color[num_pixels];
    // copy the image pixels from device to host
    checkCudaErrors(cudaMemcpy(h_image_pixels, d_image_pixels, pixels_size, cudaMemcpyDeviceToHost));


    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";



    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            image.setPixel(ny - 1 - j, i, h_image_pixels[pixel_index]);
        }
    }

    image.displayImage();

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_rand_state_pixels));
    checkCudaErrors(cudaFree(d_image_pixels));
    delete [] h_image_pixels;

    for (int i = 0; i < total_streams; ++i) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    delete [] streams;

    cudaDeviceReset();
}
