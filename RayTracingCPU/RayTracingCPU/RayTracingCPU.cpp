#include "ray_tracing.h"

#include "camera.h"
#include "image.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"



int main()
{
    HittableList world;

    auto material_ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
    auto material_center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
    auto material_left = make_shared<Dielectric>(1.5);
    auto material_right = make_shared<Metal>(Color(0.8, 0.6, 0.2), 0.0);

    world.add(make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<Sphere>(Point3(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, material_right));

    Camera camera;

    camera.aspect_ratio = 16.0 / 9.0;
    camera.image_width = 800;
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;

    camera.render(world);

    return 0;
}