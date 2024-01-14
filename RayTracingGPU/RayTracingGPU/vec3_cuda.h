#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>


struct vec3 {
    union {
        struct {
            float x, y, z;
        };
        struct {
            float r, g, b;
        };
    };
};

using point3 = vec3;
using color = vec3;

__host__ __device__ inline
vec3 make_vec3(float x, float y, float z) {
	vec3 d;
	d.x = x;
	d.y = y;
	d.z = z;
	return d;
}

__host__ __device__ inline
color make_color(float r, float g, float b) {
    color d;
    d.r = r;
    d.g = g;
    d.b = b;
    return d;
}

__host__ __device__ inline
vec3 operator+(vec3 a, vec3 b) {
	return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline
vec3 operator+=(vec3& a, vec3 b) {
	a.x += b.x; 
	a.y += b.y; 
	a.z += b.z;
	return a;
}

__host__ __device__ inline
vec3 operator-(vec3 a, vec3 b) {
	return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline
vec3 operator-=(vec3& a, vec3 b) {
	a.x -= b.x; 
	a.y -= b.y; 
	a.z -= b.z;
	return a;
}

__host__ __device__ inline
vec3 operator*(vec3 a, vec3 b) {
	return make_vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline
vec3 operator*=(vec3& a, vec3 b) {
	a.x *= b.x; 
	a.y *= b.y; 
	a.z *= b.z;
	return a;
}

__host__ __device__ inline
vec3 operator/(vec3 a, vec3 b) {
	return make_vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline
vec3 operator/=(vec3& a, vec3 b) {
	a.x /= b.x; 
	a.y /= b.y; 
	a.z /= b.z;
	return a;
}

__host__ __device__ inline
vec3 operator*(vec3 a, float b) {
	return make_vec3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline
vec3 operator*=(vec3& a, float b) {
	a.x *= b; 
	a.y *= b; 
	a.z *= b;
	return a;
}

__host__ __device__ inline
vec3 operator/(vec3 a, float b) {
	return make_vec3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ inline
vec3 operator/=(vec3& a, float b) {
	a.x /= b; 
	a.y /= b; 
	a.z /= b;
	return a;
}

__host__ __device__ inline
vec3 operator*(float a, vec3 b) {
	return make_vec3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ inline
vec3 operator/(float a, vec3 b) {
	return make_vec3(a / b.x, a / b.y, a / b.z);
}

__host__ __device__ inline
vec3 operator-(vec3 a) {
	return make_vec3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline
float dot(vec3 a, vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline
vec3 cross(vec3 a, vec3 b) {
	return make_vec3(a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x);
}

__host__ __device__ inline
float length(vec3 a) {
	return sqrtf(dot(a, a));
}

__host__ __device__ inline
float squared_length(vec3 a) {
	return dot(a, a);
}

__host__ __device__ inline
vec3 make_unit(vec3 a) {
	return a / length(a);
}
	
inline std::istream& operator>>(std::istream& is, vec3& t) {
	is >> t.x >> t.y >> t.z;
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
	os << t.x << " " << t.y << " " << t.z;
	return os;
}

#endif