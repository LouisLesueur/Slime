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
#define PI 3.14159265

using namespace std;

int main(void) {

	int height = 1000;
	int width = 1000;
	int n_agents = 100000;

	Params params;

	params.speed = 1;
	params.dt = 1;
	params.evaporate_rate = 0.05;
	params.senseAngle = 0.5*PI;
	params.senseSize = 10;
	params.senseRadius = 15;
	params.turnspeed = 0.5;

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
	float x_start = (width/2.f)-10.f;
	float x_stop = (width/2.f)+10.f;
	float y_start = (height/2.f)-10.f;
	float y_stop = (height/2.f)+10.f;

	for(int i=0; i<agents.n_agents; i++){
		agents.pos[i].x = (x_stop-x_start)*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX))+x_start;
		agents.pos[i].y = (y_stop-y_start)*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX))+y_start;
		agents.angle[i] = (static_cast <float> (rand()%RAND_MAX) / static_cast <float> (RAND_MAX))*2*PI;
	}
	
	//Send agents to device -> Not outside of the function in order to apply Gaussian blur
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
	for(int i=0; i<width; i++)
		for(int j=0; j<height; j++)
			map.elements[i*width+j] = 0;
	
	//Send trailmap to device
	TrailMatrix d_map;
	d_map.height = map.height;
	d_map.width = map.width;
	cudaMalloc(&(d_map.elements), map.width*map.height*sizeof(float));
	cudaMemcpy(d_map.elements, map.elements, map.width*map.height*sizeof(float), cudaMemcpyHostToDevice);

	cv::Mat ocv_map(width, height, CV_8UC3);
	cv::Vec3b color(rand()%255, rand()%255, rand()%255);
	
	char keyboard = ' ';
	int step = 0;
	while (keyboard != 'q') {
	
		move(d_agents, d_map, params, gen);
		cudaMemcpy(map.elements, d_map.elements, map.width*map.height*sizeof(float), cudaMemcpyDeviceToHost);

		mat_t_ocv(map, ocv_map, color);
		cv::imshow("map", ocv_map);

		std::stringstream stream;
		stream << std::setw(10) << std::setfill('0') << step;
		std::string step_string = stream.str();
		cv::imwrite("out/out_"+step_string+".png", ocv_map);

		//std::cout << "step "<<step_string<<std::endl;
		step++;
		keyboard = cv::waitKey(1);

	}
	
	cudaFree(d_agents.pos);
	cudaFree(d_agents.angle);
	cudaFree(d_map.elements);

	cv::destroyWindow("map");
	return 0;

}
