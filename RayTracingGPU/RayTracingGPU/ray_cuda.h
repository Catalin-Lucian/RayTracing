#ifndef RAYH
#define RAYH
#include <cuda_runtime.h>
#include "vec3_cuda.h"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3& _origin, const vec3& _direction) { origin = _origin; direction = _direction; }
    __device__ vec3 at(float t) const { return origin + t * direction; }

    vec3 origin;
    vec3 direction;
};

#endif
