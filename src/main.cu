#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <curand.h>
#include "agent.cuh"
#include "trailmap.cuh"

#define CVUI_IMPLEMENTATION
#include "../lib/cvui.h"

#define PI 3.14159265

#define WINDOW_NAME "Slime simulation"

using namespace std;

void init_positions(const Agents &agents, float x, float y, float radius) {

	for(int i=0; i<agents.n_agents; i++){
		float rdm_radius = radius*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
		float rdm_angle = (static_cast <float> (rand()%RAND_MAX) / static_cast <float> (RAND_MAX))*2*PI-PI;

		agents.pos[i].x = rdm_radius*cos(rdm_angle)+x;
		agents.pos[i].y = rdm_radius*sin(rdm_angle)+y;
		agents.angle[i] = -rdm_angle;
	}
}

int main(void) {

	int height = 1000;
	int width = 1000;
	int n_agents = 100000;

	Params params;

	params.speed = 1;
	params.dt = 1;
	params.evaporate_rate = 0.02;
	params.senseAngle = 0.5*PI; // strong impact on dispersion
	params.senseSize = 2; // strong impact on edge formation
	params.senseRadius = 5; // Strong impact on cell sizes
	params.turnspeed = 0.5;
	params.diff_decay = 0.5;

	if(sin(params.senseAngle/2)*params.senseRadius <= params.senseSize)
		cout << "WARNING !, in this configuration detection zones of ants are overlapping !"<<endl;

        // For random numbers	
	srand (time(NULL));
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	// Allocate agents on host
	Agents agents;
	agents.n_agents = n_agents;
	agents.pos = (float2 *) malloc(agents.n_agents*sizeof(float2));
	agents.angle = (float *) malloc(agents.n_agents*sizeof(float));

	// Random initial positions
	float x_center = (width/2.f);
	float y_center = (height/2.f);
	float radius = 150.f;
	init_positions(agents, x_center, y_center, radius);
	
	//Send agents to device
	Agents d_agents;
	d_agents.n_agents = agents.n_agents;
	cudaMalloc(&(d_agents.pos), agents.n_agents*sizeof(float2));
	cudaMalloc(&(d_agents.angle), agents.n_agents*sizeof(float));
	cudaMemcpy(d_agents.pos, agents.pos, agents.n_agents*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_agents.angle, agents.angle, agents.n_agents*sizeof(float), cudaMemcpyHostToDevice);

	//Allocate trailmap on host
	TrailMatrix map;
	map.height = height;
	map.width = width;
	map.elements = (float *) malloc(map.height*map.width*sizeof(float));
	memset(map.elements, 0, map.height*map.width*sizeof(float));
	
	//Send trailmap to device
	TrailMatrix d_map;
	d_map.height = map.height;
	d_map.width = map.width;
	cudaMalloc(&(d_map.elements), map.width*map.height*sizeof(float));
	cudaMemcpy(d_map.elements, map.elements, map.width*map.height*sizeof(float), cudaMemcpyHostToDevice);

	// openCV matrix for visualisation + random color
	cv::Mat ocv_map(width, height, CV_8UC3);
	cv::Vec3b color(0, 14, 252);
	
	// Allocate a vector on device for randomness
	float *rdm_num;
	cudaMalloc(&rdm_num, d_agents.n_agents*sizeof(float));

	cvui::init(WINDOW_NAME);

	char keyboard = ' ';
	int step = 0;
	
	while (keyboard != 'q') {
	
		move(d_agents, d_map, params, gen, rdm_num);
		cudaMemcpy(map.elements, d_map.elements, map.width*map.height*sizeof(float), cudaMemcpyDeviceToHost);

		mat_t_ocv(map, ocv_map, color);
		
		std::stringstream stream;
		stream << std::setw(10) << std::setfill('0') << step;
		std::string step_string = stream.str();
		cv::imwrite("out/out_"+step_string+".png", ocv_map);
		
		cvui::window(ocv_map, 10, 50, 180, 700, "Settings");

		cvui::printf(ocv_map, 15, 100, "Evaporation rate = %.2f", params.evaporate_rate);
		int int_ev = int(255*params.evaporate_rate);
		cvui::trackbar(ocv_map, 15, 130, 150, &int_ev, 0, 255);
		params.evaporate_rate = float(int_ev)/255.f;
		
		cvui::printf(ocv_map, 15, 200, "Sense angle = %.2f PI", params.senseAngle/PI);
		int int_sa = int(255*params.senseAngle/PI);
		cvui::trackbar(ocv_map, 15, 230, 150, &int_sa, 0, 255);
		params.senseAngle = (float(int_sa)*PI)/255.f;
		
		cvui::printf(ocv_map, 15, 300, "Sense size = %i", params.senseSize);
		cvui::trackbar(ocv_map, 15, 330, 150, &params.senseSize, 0, 10);
		
		cvui::printf(ocv_map, 15, 400, "Sense radius = %f", params.senseRadius);
		int int_sr = int(params.senseRadius);
		cvui::trackbar(ocv_map, 15, 430, 150, &int_sr, 0, 25);
		params.senseRadius = float(int_sr);
		
		cvui::printf(ocv_map, 15, 500, "Turn speed = %f", params.turnspeed);
		int int_ts = int(params.turnspeed*255);
		cvui::trackbar(ocv_map, 15, 530, 150, &int_ts, 0, 255);
		params.turnspeed = float(int_ts)/255.f;
		
		cvui::printf(ocv_map, 15, 600, "Diffusion decay = %f", params.diff_decay);
		int int_dd = int(params.diff_decay*255);
		cvui::trackbar(ocv_map, 15, 630, 150, &int_dd, 0, 255);
		params.diff_decay = float(int_dd)/255.f;

		cvui::update();
		cvui::imshow(WINDOW_NAME, ocv_map);

		step++;
		keyboard = cv::waitKey(1);

	}
	
	cudaFree(d_agents.pos);
	cudaFree(d_agents.angle);
	cudaFree(d_map.elements);
	cudaFree(rdm_num);

	cv::destroyWindow("map");
	return 0;
}
