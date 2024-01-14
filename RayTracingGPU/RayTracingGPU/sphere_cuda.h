#ifndef SPHEREH
#define SPHEREH


struct __align__(16) sphere
{
    vec3 center;
    float radius;
    material material;
};

__device__ inline
sphere make_sphere(vec3 center, float radius,material mat){
	sphere s;
	s.center = center;
	s.radius = radius;
	s.material = mat;
	return s;
}
    

__device__ bool hit(const sphere& s, const ray& r, float t_min, float t_max, record& rec) {
	vec3 oc = r.origin - s.center;
	float a = dot(r.direction, r.direction);
	float b = dot(oc, r.direction);
	float c = dot(oc, oc) - s.radius * s.radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r + rec.t;
			rec.normal = (rec.p - s.center) / s.radius;
			rec.material = s.material;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r + rec.t;
			rec.normal = (rec.p - s.center) / s.radius;
			rec.material = s.material;
			return true;
		}
	}
	return false;
}

#endif