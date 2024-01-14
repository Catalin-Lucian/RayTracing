#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable_cuda.h"

namespace hittable {
    struct world
    {
        sphere* objects;
        int size;
    };

    __device__ inline
    world* make_world(sphere* objects, int size) {
		world* w = new world;
		w->objects = objects;
		w->size = size;
		return w;
	}

	__device__
	world clone(const world& w) {
		world w2;
		w2.objects = new sphere[w.size];
		w2.size = w.size;
		for (int i = 0; i < w.size; i++) {
			w2.objects[i] = w.objects[i];
		}
		return w2;
	}

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

}

#endif