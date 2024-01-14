﻿#include <iostream>
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

__device__ vec3 get_color(const ray& r, world& world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = make_vec3(1.0f, 1.0f, 1.0f);
   
    // 50 iterations for ray bounce
    for (int i = 0; i < 50; i++) {
        record rec;

        if (hit(world, cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (scatter(rec.material, cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return make_vec3(0.0f, 0.0f, 0.0f);
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

__global__ void rand_world_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
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
    camera* cam,
    world* worldd,
    int startRow, int endRow,
    curandState* rand_state
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (j >= startRow && j < endRow && i < max_x) {
        int pixel_index = j * max_x + i;
        curandState local_rand_state = rand_state[pixel_index];

        // shared camera
        __shared__ camera shared_cam;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            copy_camera(&shared_cam, cam);
        }
        __syncthreads();

        color col = make_vec3(0.f, 0.f, 0.f);
        for (int s = 0; s < ns; s++) {
            float u = (i + curand_uniform(&local_rand_state)) / float(max_x);
            float v = (j + curand_uniform(&local_rand_state)) / float(max_y);
            ray r = get_ray(shared_cam, u, v, &local_rand_state);
            col += get_color(r, *worldd, &local_rand_state);

        }

        col /= float(ns);
        image[pixel_index] = col;
    }
}

__global__ void create_world(world* world, sphere* d_objects, camera* camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        d_objects[0] = make_sphere(make_vec3(0.f, -1000.0f, -1.f), 1000.f, make_lambertian(make_vec3(0.5f, 0.5f, 0.5f)));
        int idx = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center = make_vec3(a + RND, 0.2f, b + RND);
                vec3 material_color;
                if (choose_mat < 0.8f) {
                    material_color = make_vec3(RND * RND, RND * RND, RND * RND);
                }
                else if (choose_mat < 0.95f) {
                    material_color = make_vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND));
                }
                else {
                    material_color = make_vec3(1.0f, 1.0f, 1.0f);
                }
                d_objects[idx++] = make_sphere(center, 0.2f, make_lambertian(material_color));
            }
        }

        d_objects[idx++] = make_sphere(make_vec3(0.f, 1.f, 0.f), 1.0f, make_dielectric(1.5f));
        d_objects[idx++] = make_sphere(make_vec3(-4.f, 1.f, 0.f), 1.0f, make_lambertian(make_vec3(0.4f, 0.2f, 0.1f)));
        d_objects[idx++] = make_sphere(make_vec3(4.f, 1.f, 0.f), 1.0f, make_metal(make_vec3(0.7f, 0.6f, 0.5f), 0.0f));

        init_world(world, d_objects, idx);

        //*rand_state = local_rand_state;

        vec3 lookfrom = make_vec3(13.f, 2.f, 3.f);
        vec3 lookat = make_vec3(0.f, 0.f, 0.f);
        float dist_to_focus = length(lookfrom - lookat);
        float aperture = 0.1;
        init_camera(
            camera,
            lookfrom,
            lookat,
            make_vec3(0.f, 1.f, 0.f),
            30.0f,
            float(nx) / float(ny),
            aperture,
            dist_to_focus
        );
    }
}


__global__ void free_world(world* d_world, camera* d_camera) {
    delete [] d_world->objects;
    delete d_world;
    delete d_camera;
}

int main() {
    int nx = 800; // width
    int ny = 450; // heigth
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
    curandState* d_rand_state_world;
    checkCudaErrors(cudaMalloc(&d_rand_state_world, 1 * sizeof(curandState)));

    // list of objects
    sphere* d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, 488 * sizeof(sphere)));

    //world
    world* d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(world)));

    // camera
    camera* d_camera;
    checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera)));

    //// clock to measure time
    clock_t start, stop;
    start = clock();

    // we need that 2nd random state to be initialized for the world creation
    rand_world_init <<< 1, 1 >>> (d_rand_state_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // create world
    create_world <<<1, 1 >>> (d_world, d_objects, d_camera, nx, ny, d_rand_state_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // calculate blocks and threads
    dim3 blocks((nx + tile_size_x - 1) / tile_size_x, (ny + tile_size_y - 1) / tile_size_y);
    dim3 threads(tile_size_x, tile_size_y);

    // init random state for each pixel
    rand_pixels_init <<< blocks, threads >> > (nx, ny, d_rand_state_pixels);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Crearea stream-urilor
    int num_streams = 4;
    cudaStream_t streams[4];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int segmentSize = ny / num_streams; // Împărțiți imaginea în segmente orizontale

    // Procesarea imaginii în porțiuni, folosind stream-uri diferite
    for (int s = 0; s < num_streams; ++s) {
        int startRow = s * segmentSize;
        int endRow = (s + 1) * segmentSize;
        if (s == num_streams - 1) {
            endRow = ny; // Asigurați-vă că ultimul segment acoperă restul imaginii
        }

        // Lansarea kernel-ului cu stream-ul specific
        render <<< blocks, threads, 0, streams[s] >> > (
            d_image_pixels, 
            nx, ny, ns, 
            d_camera, 
            d_world, 
            startRow, endRow,
            d_rand_state_pixels);
    }

    // Așteptarea finalizării tuturor streams
    for (int i = 0; i < num_streams; ++i) {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    // render the scene
    /*render <<< blocks, threads >>> (d_image_pixels, nx, ny, ns, d_camera, d_world, d_rand_state_pixels);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());*/

    color* h_image_pixels = new color[num_pixels];
    // copy the image pixels from device to host
    checkCudaErrors(cudaMemcpy(h_image_pixels,d_image_pixels, pixels_size, cudaMemcpyDeviceToHost));


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
    free_world <<<1, 1 >>> (d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state_pixels));
    checkCudaErrors(cudaFree(d_rand_state_world));
    checkCudaErrors(cudaFree(d_image_pixels));

    delete [] h_image_pixels;
    for (int i = 0; i < num_streams; ++i) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    cudaDeviceReset();
}
