#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using Color = Vec3;
using namespace cv;

// wrapper for opencv image
class Image {

	public:
	Image(Size imageSize) : image(imageSize, CV_64FC3, Scalar(0, 0, 0)),size(imageSize) {}
	Image(int width, int height) : image(Size(width, height), CV_64FC3, Scalar(0, 0, 0)), size(width,height) {}

	void displayImage()
	{
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", image);
		waitKey(0);
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
		// OpenCV uses BGR instead of RGB
		image.at<Vec3d>(i, j)[0] = color_range.clamp(pixelColor.z());
		image.at<Vec3d>(i, j)[1] = color_range.clamp(pixelColor.y());
		image.at<Vec3d>(i, j)[2] = color_range.clamp(pixelColor.x());
	}

	double linear_to_gamma(double linear_component)
	{
		return sqrt(linear_component);
	}

	private:
		Mat image;
	public:
		Size size;
};
#endif