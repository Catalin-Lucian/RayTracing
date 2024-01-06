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

    void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = Point3(0, 0, 0);

        // Determine viewport dimensions.
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = Vec3(viewport_width, 0, 0);
        auto viewport_v = Vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            center - Vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    Color ray_color(const Ray& r,int depth, const Hittable& world) const {
        HitRecord rec;

        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return Color(0, 0, 0);

        if (world.hit(r, Interval(0.001, infinity), rec)) {
            Ray scattered;
            Color attenuation;
            // !!!!!!!! problem with the material
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
        // Get a randomly sampled camera ray for the pixel at location i,j.

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square();

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }

    Vec3 pixel_sample_square() const {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + random_double();
        auto py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }
};

#endif