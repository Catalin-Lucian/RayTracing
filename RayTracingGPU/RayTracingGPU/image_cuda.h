#ifndef COLOR_H
#define COLOR_H

#include "vec3_cuda.h"

#include <fstream>
#include <iostream>
#include <vector>
#include "interval_cuda.h"

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
                auto pixel = image[y][x] * 255.99;
                unsigned char r = static_cast<unsigned char>(pixel.r);
                unsigned char g = static_cast<unsigned char>(pixel.g);
                unsigned char b = static_cast<unsigned char>(pixel.b);
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



    void setPixel(int i, int j, color pixelColor)
    {
        // Gamma-correct for gamma=2.0
        pixelColor.r = linear_to_gamma(pixelColor.r);
        pixelColor.g = linear_to_gamma(pixelColor.g);
        pixelColor.b = linear_to_gamma(pixelColor.b);

        static const Interval color_range(0.0, 1.0);
        image[i][j].r = color_range.clamp(pixelColor.r);
        image[i][j].g = color_range.clamp(pixelColor.g);
        image[i][j].b = color_range.clamp(pixelColor.b);
    }

    double linear_to_gamma(double linear_component)
    {
        return sqrt(linear_component);
    }

private:
    vector<vector<color>> image;
};

#endif