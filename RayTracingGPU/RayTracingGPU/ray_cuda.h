#ifndef RAYH
#define RAYH
#include <cuda_runtime.h>

struct __align__(16) ray {
    vec3 origin;
    vec3 direction;
};

__device__  inline
ray make_ray(vec3 origin, vec3 direction) {
	ray d;
	d.origin = origin;
	d.direction = direction;
	return d;
}

__device__ inline
vec3 operator+(const ray& r, float t) {
	return r.origin + t * r.direction;
}

#endif
