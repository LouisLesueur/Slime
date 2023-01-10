#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

struct TrailMatrix {
	int width;
	int height;
	int n_species;
	float* elements;
};

void mat_t_ocv(const TrailMatrix &map, cv::Mat ocv_map, uint8_t* colors);
