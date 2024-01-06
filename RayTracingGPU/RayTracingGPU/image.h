#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <fstream>
#include <iostream>
#include <vector>
#include "interval.h"

using Color = Vec3;
using namespace std;


class Image {
public:
    int width, height;

    Image(int _width, int _height) : width(_width), height(_height) {
        image.resize(height);
        for (int i = 0; i < height; i++) {
            image[i].resize(width);
        }
    }

    void displayImage()
    {
        const int paddingAmount = (4 - (width * 3) % 4) % 4;
        const int scanlineSize = width * 3 + paddingAmount;
        const int fileSize = 54 + (scanlineSize * height);

        unsigned char bmpFileHeader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
        unsigned char bmpInfoHeader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };

        bmpFileHeader[2] = (unsigned char)(fileSize);
        bmpFileHeader[3] = (unsigned char)(fileSize >> 8);
        bmpFileHeader[4] = (unsigned char)(fileSize >> 16);
        bmpFileHeader[5] = (unsigned char)(fileSize >> 24);

        bmpInfoHeader[4] = (unsigned char)(width);
        bmpInfoHeader[5] = (unsigned char)(width >> 8);
        bmpInfoHeader[6] = (unsigned char)(width >> 16);
        bmpInfoHeader[7] = (unsigned char)(width >> 24);
        bmpInfoHeader[8] = (unsigned char)(height);
        bmpInfoHeader[9] = (unsigned char)(height >> 8);
        bmpInfoHeader[10] = (unsigned char)(height >> 16);
        bmpInfoHeader[11] = (unsigned char)(height >> 24);

        std::ofstream file("image.bmp", std::ios::out | std::ios::binary);
        if (!file) {
            std::cerr << "Could not open the file for writing.\n";
            return;
        }

        file.write(reinterpret_cast<char*>(bmpFileHeader), 14);
        file.write(reinterpret_cast<char*>(bmpInfoHeader), 40);

        unsigned char bmpPad[3] = { 0, 0, 0 };

        for (int y = height - 1; y >= 0; y--) {
            for (int x = 0; x < width; x++) {
                Vec3 pixel = image[y][x] * 255.99;
                unsigned char r = static_cast<unsigned char>(pixel[0]);
                unsigned char g = static_cast<unsigned char>(pixel[1]);
                unsigned char b = static_cast<unsigned char>(pixel[2]);
                unsigned char color[] = { b, g, r };
                file.write(reinterpret_cast<char*>(color), 3);
            }
            file.write(reinterpret_cast<char*>(bmpPad), paddingAmount);
        }

        file.close();

        // Request operating system to display image
#ifdef _WIN32
        system("start image.bmp");
#else
        system("xdg-open image.bmp");
#endif
    }



    void setPixel(int i, int j, Color pixelColor, int samples_per_pixel)
    {
        // Divide the color by the number of samples
        auto scale = 1.0 / samples_per_pixel;
        pixelColor *= scale;

        // Gamma-correct for gamma=2.0
        pixelColor[0] = linear_to_gamma(pixelColor[0]);
        pixelColor[1] = linear_to_gamma(pixelColor[1]);
        pixelColor[2] = linear_to_gamma(pixelColor[2]);

        static const Interval color_range(0.0, 1.0);
        image[i][j][0] = color_range.clamp(pixelColor.x());
        image[i][j][1] = color_range.clamp(pixelColor.y());
        image[i][j][2] = color_range.clamp(pixelColor.z());
    }

    double linear_to_gamma(double linear_component)
    {
        return sqrt(linear_component);
    }

private:
    vector<vector<Color>> image;
};

#endif