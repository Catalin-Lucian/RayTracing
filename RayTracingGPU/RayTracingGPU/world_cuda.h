#ifndef HITABLELISTH
#define HITABLELISTH

#include "sphere_cuda.h"

struct world
{
    sphere objects[488];
    int size;
};

__device__
bool hit(const world& w, const ray& r, float t_min, float t_max, record& rec) {
	record temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;
	for (int i = 0; i < w.size; i++) {
		if (hit(w.objects[i], r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}
#endif