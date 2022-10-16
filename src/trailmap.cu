#include "trailmap.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;

void mat_t_ocv(const TrailMatrix &map, cv::Mat ocv_map, cv::Vec3b color){
	for(int i=0; i<map.width; i++){
		for(int j=0; j < map.height; j++){
			int index = i*map.width + j;
			ocv_map.at<cv::Vec3b>(i,j) = cv::Vec3b(uint8_t(float(color[0])*map.elements[index]),
					                       uint8_t(float(color[1])*map.elements[index]),
							       uint8_t(float(color[2])*map.elements[index]));
		}
	}


}
