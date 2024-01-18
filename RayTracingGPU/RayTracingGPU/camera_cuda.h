#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "vec3_cuda.h"
#include "ray_cuda.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct __align__(16) camera
{
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

//__device__ void copy_camera(camera* to, camera* from)
//{
//    to->origin = from->origin;
//    to->lower_left_corner = from->lower_left_corner;
//    to->horizontal = from->horizontal;
//    to->vertical = from->vertical;
//    to->u = from->u;
//    to->v = from->v;
//    to->w = from->w;
//    to->lens_radius = from->lens_radius;
//}

void init_camera(camera &cam, vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist)
{
    // vfov is top to bottom in degrees
    cam.lens_radius = aperture / 2.0f;
    float theta = vfov * ((float)M_PI) / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;
    cam.origin = lookfrom;
    cam.w = make_unit(lookfrom - lookat);
    cam.u = make_unit(cross(vup, cam.w));
    cam.v = cross(cam.w, cam.u);
    cam.lower_left_corner = cam.origin - half_width * focus_dist * cam.u - half_height * focus_dist * cam.v - focus_dist * cam.w;
    cam.horizontal = 2.0f * half_width * focus_dist * cam.u;
    cam.vertical = 2.0f * half_height * focus_dist * cam.v;
}

__device__ vec3 random_in_unit_disk(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * make_vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.f) - make_vec3(1.f, 1.f, 0.f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ ray get_ray(const camera &cam, float s, float t, curandState *local_rand_state)
{
    vec3 rd = cam.lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = cam.u * rd.x + cam.v * rd.y;
    return make_ray(
        cam.origin + offset,
        cam.lower_left_corner + s * cam.horizontal + t * cam.vertical - cam.origin - offset);
}
#endif