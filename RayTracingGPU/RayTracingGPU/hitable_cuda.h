#ifndef HITABLEH
#define HITABLEH

#include "ray_cuda.h"

// Forward declaration of material
struct material;

namespace hittable {
    struct record {
        float t;
        vec3 p;
        vec3 normal;
        material material; // This line requires the complete type of material
    };

	__device__ inline
	record make_record(float t, vec3 p, vec3 normal, material mat) {
		record rec;
		rec.t = t;
		rec.p = p;
		rec.normal = normal;
		rec.material = mat;
		return rec;
	}
}

#include "material_cuda.h"

#endif
