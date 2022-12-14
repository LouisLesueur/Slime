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

void init_positions(const Agents &agents, float x, float y, float radius, int n_species) {

	for(int i=0; i<agents.n_agents; i++){
		float rdm_radius = radius*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
		float rdm_angle = (static_cast <float> (rand()%RAND_MAX) / static_cast <float> (RAND_MAX))*2*PI-PI;

		agents.pos[i].x = rdm_radius*cos(rdm_angle)+x;
		agents.pos[i].y = rdm_radius*sin(rdm_angle)+y;
		agents.angle[i] = -rdm_angle;
		agents.species[i] = int(i%n_species);

	}

        for(int i=0; i<n_species; i++)	
		for(int j=0; j<3; j++)
			agents.colors[i*3+j] = uint8_t(rand()%255);
}

int main(void) {

	int height = 1080;
	int width = 1920;
	int n_species = 4;
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
	params.col_speed = 10;


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
	agents.species = (int *) malloc(agents.n_agents*sizeof(int));
	agents.colors = (uint8_t *) malloc(n_species*3*sizeof(uint8_t));

	// Random initial positions
	float x_center = (height/2.f);
	float y_center = (width/2.f);
	float radius = 500.f;
	init_positions(agents, x_center, y_center, radius, n_species);
	
	//Send agents to device
	Agents d_agents;
	d_agents.n_agents = agents.n_agents;
	cudaMalloc(&(d_agents.pos), agents.n_agents*sizeof(float2));
	cudaMalloc(&(d_agents.angle), agents.n_agents*sizeof(float));
	cudaMalloc(&(d_agents.species), agents.n_agents*sizeof(int));
	cudaMemcpy(d_agents.pos, agents.pos, agents.n_agents*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_agents.angle, agents.angle, agents.n_agents*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_agents.species, agents.species, agents.n_agents*sizeof(int), cudaMemcpyHostToDevice);

	//Allocate trailmap on host
	TrailMatrix map;
	map.height = height;
	map.width = width;
	map.n_species = n_species;
	map.elements = (float *) malloc(map.height*map.width*map.n_species*sizeof(float));
	memset(map.elements, 0, map.height*map.width*map.n_species*sizeof(float));
	
	//Send trailmap to device
	TrailMatrix d_map;
	d_map.height = map.height;
	d_map.width = map.width;
	d_map.n_species = map.n_species;
	cudaMalloc(&(d_map.elements), map.width*map.height*map.n_species*sizeof(float));
	cudaMemcpy(d_map.elements, map.elements, map.width*map.height*map.n_species*sizeof(float), cudaMemcpyHostToDevice);

	// openCV matrix for visualisation + random color
	cv::Mat ocv_map(height, width, CV_8UC3);
	
	// Allocate a vector on device for randomness
	float *rdm_num;
	cudaMalloc(&rdm_num, d_agents.n_agents*sizeof(float));

	cvui::init(WINDOW_NAME);

	char keyboard = ' ';
	int step = 0;

	bool show_menu = false;
	
	while (keyboard != 'q') {

		move(d_agents, d_map, params, gen, rdm_num);
		cudaMemcpy(map.elements, d_map.elements, map.width*map.height*map.n_species*sizeof(float), cudaMemcpyDeviceToHost);

		mat_t_ocv(map, ocv_map, agents.colors);
		
		if(show_menu==true){
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
		}

		cvui::update();
		cvui::imshow(WINDOW_NAME, ocv_map);

		step++;
		keyboard = cv::waitKey(1);
		if(keyboard == 'm')
			show_menu = !show_menu;

	}
	
	cudaFree(d_agents.pos);
	cudaFree(d_agents.angle);
	cudaFree(d_agents.colors);
	cudaFree(d_map.elements);
	cudaFree(rdm_num);

	cv::destroyWindow("map");
	return 0;
}
