#ifndef HITABLELISTH
#define HITABLELISTH

#include "sphere_cuda.h"

struct world
{
    sphere* objects;
    int size;
};

__device__ inline
world init_world(world* world, sphere* objects, int size) {
	world->objects = objects;
	world->size = size;
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
void copy_world(world* newWorld, world* w) {
	newWorld->objects = new sphere[w->size];
	newWorld->size = w->size;
	memcpy(newWorld->objects, w->objects, w->size * sizeof(sphere));
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



#endif