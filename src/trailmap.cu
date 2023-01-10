#include "trailmap.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;

void mat_t_ocv(const TrailMatrix &map, cv::Mat ocv_map, uint8_t* color){
	for(int i=0; i<map.height; i++){
		for(int j=0; j < map.width; j++){
			int max_k = 0;
			float max_val = 0;

			for(int k=0; k < map.n_species; k++){
				int index = i*map.width + j + k*map.width*map.height;
				if(map.elements[index] > max_val){
					max_val = map.elements[index];
					max_k = k;
				}
			}

			ocv_map.at<cv::Vec3b>(i,j) = cv::Vec3b(uint8_t(float(color[max_k*3])*max_val),
							       uint8_t(float(color[max_k*3+1])*max_val),
							       uint8_t(float(color[max_k*3+2])*max_val));
		}
	}


}
