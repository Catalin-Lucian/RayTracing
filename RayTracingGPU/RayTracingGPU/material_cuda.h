#ifndef MATERIALH
#define MATERIALH

#include <curand_kernel.h>
#include "ray_cuda.h"

struct material {
    enum type {
		LAMBERTIAN,
		METAL,
		DIELECTRIC
	} type;
	
	vec3 albedo;  // For LAMBERTIAN / METAL

    union {
        float fuzz;      // For METAL
        float ref_idx;   // For DIELECTRIC
    };
};

struct record {
    float t;
    vec3 p;
    vec3 normal;
    material material; // This line requires the complete type of material
};

__device__ __host__ 
material make_lambertian(vec3 albedo) {
	material m;
	m.type = material::LAMBERTIAN;
	m.albedo = albedo;
	return m;
}

__device__ __host__ 
material make_metal(vec3 albedo, float fuzz) {
	material m;
	m.type = material::METAL;
	m.albedo = albedo;
	if (fuzz < 1) 
        m.fuzz = fuzz; 
    else 
        m.fuzz = 1;
	return m;
}

__device__ __host__ 
material make_dielectric(float ref_idx) {
	material m;
	m.type = material::DIELECTRIC;
	m.ref_idx = ref_idx;
	return m;
}

#define RANDVEC3 make_vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ 
vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - make_vec3(1.f, 1.f, 1.f);
    } while (squared_length(p) >= 1.0f);
    return p;
}

__device__
float schlick(float cosine, float ref_idx) {
    float r0 = (1.f - ref_idx) / (1.f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.f - r0) * pow((1.f - cosine), 5.f);
}

__device__
bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = make_unit(v);
    float dt = dot(uv, n);
    float discriminant = 1.f - ni_over_nt * ni_over_nt * (1.f - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else {
        return false;
    }
}

__device__
vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.f * dot(v, n) * n;
}

__device__ 
bool scatter_lambertian(
    const material& mat, 
    const ray& r_in, 
    const record& rec, 
    vec3& attenuation, 
    ray& scattered, 
    curandState* local_rand_state
) {
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = make_ray(rec.p, target - rec.p);
    attenuation = mat.albedo;
    return true;
}

__device__ 
bool scatter_metal(
	const material& mat, 
	const ray& r_in, 
	const record& rec, 
	vec3& attenuation, 
	ray& scattered, 
	curandState* local_rand_state
) {
	vec3 reflected = reflect(make_unit(r_in.direction), rec.normal);
	scattered = make_ray(rec.p, reflected + mat.fuzz * random_in_unit_sphere(local_rand_state));
	attenuation = mat.albedo;
	return (dot(scattered.direction, rec.normal) > 0.0f);
}

__device__ 
bool scatter_dielectric(
    const material& mat,
    const ray& r_in,
    const record& rec,
    vec3& attenuation,
    ray& scattered,
    curandState* local_rand_state
) {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction, rec.normal);
    float ni_over_nt;
    attenuation = make_vec3(1.f, 1.f, 1.f);
    vec3 refracted;
    float reflect_prob;
    float cosine;

    if (dot(r_in.direction, rec.normal) > 0.f) {
        outward_normal = -rec.normal;
        ni_over_nt = mat.ref_idx;
        cosine = dot(r_in.direction, rec.normal) / length(r_in.direction);
        cosine = sqrt(1.f - mat.ref_idx * mat.ref_idx * (1.f - cosine * cosine));
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.f / mat.ref_idx;
        cosine = -dot(r_in.direction, rec.normal) / length(r_in.direction);
    }

    if (refract(r_in.direction, outward_normal, ni_over_nt, refracted))
        reflect_prob = schlick(cosine, mat.ref_idx);
    else
        reflect_prob = 1.f;

    if (curand_uniform(local_rand_state) < reflect_prob)
        scattered = make_ray(rec.p, reflected);
    else
        scattered = make_ray(rec.p, refracted);
    return true;
}

__device__
bool scatter(
    const material& mat,
    const ray& r_in,
    const record& rec,
    vec3& attenuation,
    ray& scattered,
    curandState* local_rand_state
) {
    switch (mat.type) {
        case material::LAMBERTIAN:
            return scatter_lambertian(mat, r_in, rec, attenuation, scattered, local_rand_state);
        case material::METAL:
            return scatter_metal(mat, r_in, rec, attenuation, scattered, local_rand_state);
        case material::DIELECTRIC:
            return scatter_dielectric(mat, r_in, rec, attenuation, scattered, local_rand_state);
    }
    return false;
}
#endif