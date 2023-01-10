#pragma once
#include "trailmap.cuh"
#include <curand.h>

struct Agents {
	int n_agents;
	int* species;
	float2* pos;
	float* angle;
        uint8_t* colors;	
};

struct Params {
	float speed;
	float dt;
	float evaporate_rate;
	float senseAngle;
	float senseRadius; 
	int senseSize;
	float turnspeed;
	float diff_decay;
	int col_speed;
};

void move(const Agents &d_agents, const TrailMatrix &d_map, Params params, curandGenerator_t gen, float* rdm_num);
