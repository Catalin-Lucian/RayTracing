#ifndef HITABLEH
#define HITABLEH

#include "ray_cuda.h"

class material;

namespace hittable {
    struct record {
		float t;
		vec3 p;
		vec3 normal;
		material* material;
	};

	__device__ inline
	record make_record(float t, vec3 p, vec3 normal, material* mat) {
		record rec;
		rec.t = t;
		rec.p = p;
		rec.normal = normal;
		rec.material = mat;
		return rec;
	}
}

#endif