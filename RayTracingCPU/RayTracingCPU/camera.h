#ifndef CAMERA_H
#define CAMERA_H

#include "ray_tracing.h"

#include "image.h"
#include "hittable.h"
#include "material.h"

#include <iostream>

class Camera {
public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth = 10;   // Maximum number of ray bounces into scene

    double vfov = 90;  // Vertical view angle (field of view)
    Point3 lookfrom = Point3(0, 0, -1);  // Point camera is looking from
    Point3 lookat = Point3(0, 0, 0);   // Point camera is looking at
    Vec3   vup = Vec3(0, 1, 0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    void render(const Hittable& world) {
        initialize();

        std::cout << "Start rendering\n";
        Image image(image_width, image_height);
        for (int j = 0; j < image_height; ++j) {
            std::clog << "\rremaing rows: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; ++i) {
                Color pixel_color(0, 0, 0);
                for (int sample = 0; sample < samples_per_pixel; ++sample) {
                    Ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
				}
                image.setPixel(j, i, pixel_color, samples_per_pixel);
            }
        }

        std::clog << "\rDone.                 \n";
        image.displayImage();
    }

private:
    int    image_height;   // Rendered image height
    Point3 center;         // Camera center
    Point3 pixel00_loc;    // Location of pixel 0, 0
    Vec3   pixel_delta_u;  // Offset to pixel to the right
    Vec3   pixel_delta_v;  // Offset to pixel below
    Vec3   u, v, w;        // Camera frame basis vectors
    Vec3   defocus_disk_u;  // Defocus disk horizontal radius
    Vec3   defocus_disk_v;  // Defocus disk vertical radius

    void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        Vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    Color ray_color(const Ray& r,int depth, const Hittable& world) const {
        HitRecord rec;

        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return Color(0, 0, 0);

        if (world.hit(r, Interval(0.001, infinity), rec)) {
            Ray scattered;
            Color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth - 1, world);
            return Color(0, 0, 0);
        }

        // blue gradient
        Vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
    }

    Ray get_ray(int i, int j) const {
        // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
        // the camera defocus disk.

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square();

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }

    Point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    Vec3 pixel_sample_square() const {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + random_double();
        auto py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }
};

#endif