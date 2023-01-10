#include "agent.cuh"
#include <cstdio>
#include <curand.h>
#define PI 3.14159265


__global__ void cuda_move(Agents agents, TrailMatrix map, float speed, float dt, float* rdm_num){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < agents.n_agents){
		float new_x = agents.pos[i].x + cos(agents.angle[i]) * speed * dt;
		float new_y = agents.pos[i].y + sin(agents.angle[i]) * speed * dt;

                // collisions		
		if (new_x < 0 || new_x >= map.height || new_y < 0 || new_y >= map.width) {
			// Margin for the hand-coded Gaussian kernel !
			new_x = min(float(map.height-3), max(0.f, new_x));
			new_y = min(float(map.width-3), max(0.f, new_y));
			agents.angle[i] += 2*PI*rdm_num[i];
		}

		agents.pos[i].x = new_x;
		agents.pos[i].y = new_y;
		

		int index = int(new_x)*map.width + int(new_y) + map.width*map.height*int(agents.species[i]);
		map.elements[index] = 1;
	}
}

__global__ void cuda_evaporate(TrailMatrix map, float evaporate_rate, float dt){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < map.width*map.height*map.n_species){
		
		if(map.elements[index] - evaporate_rate*dt < 0)
			map.elements[index] = 0;
		else
			map.elements[index] -= evaporate_rate*dt;
	}
}

	
__global__ void cuda_gauss(TrailMatrix map, TrailMatrix new_map, float decay){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int i = (int(index) / int(map.width))%int(map.height);
	int j = int(index) % int(map.width);
	
	
	//Gaussian blur 5x5 kernel white noise, 0 padding
	if(2 <= i && i < map.height-2 && 2 <= j && j < map.width-2){
		
		int indexes[25] = {
			(i-2)*map.width + (j-2), (i-1)*map.width + (j-2), i*map.width + (j-2), (i+1)*map.width + (j-2), (i+2)*map.width + (j-2),
			(i-2)*map.width + (j-1), (i-1)*map.width + (j-1), i*map.width + (j-1), (i+1)*map.width + (j-1), (i+2)*map.width + (j-1),
			(i-2)*map.width +     j, (i-1)*map.width +     j, i*map.width +     j, (i+1)*map.width +     j, (i+2)*map.width +     j,
			(i-2)*map.width + (j+1), (i-1)*map.width + (j+1), i*map.width + (j+1), (i+1)*map.width + (j+1), (i+2)*map.width + (j+1),
			(i-2)*map.width + (j+2), (i-1)*map.width + (j+2), i*map.width + (j+2), (i+1)*map.width + (j+2), (i+2)*map.width + (j+2)

		};

		float values[6] = {1.f, 4.f, 6.f, 16.f, 24.f, 36.f};

		for(int k=0; k<map.n_species; k++){
			float sum = 0;

			sum += values[0]*(map.elements[indexes[0 ]+k*map.width*map.height] + map.elements[indexes[4 ] +k*map.width*map.height] + 
					  map.elements[indexes[20]+k*map.width*map.height] + map.elements[indexes[24]+k*map.height*map.width]);
			sum += values[1]*(map.elements[indexes[1 ]+k*map.width*map.height] + map.elements[indexes[3 ] +k*map.width*map.height] + 
					  map.elements[indexes[5 ]+k*map.width*map.height] + map.elements[indexes[9 ] +k*map.height*map.width] + 
					  map.elements[indexes[15]+k*map.width*map.height] + map.elements[indexes[19]+k*map.width*map.height] + 
					  map.elements[indexes[21]+k*map.width*map.height] + map.elements[indexes[23]+k*map.height*map.width]);
			sum += values[2]*(map.elements[indexes[2 ]+k*map.width*map.height] + map.elements[indexes[10]+k*map.width*map.height] + 
					  map.elements[indexes[14]+k*map.width*map.height] + map.elements[indexes[16]+k*map.height*map.width] + 
					  map.elements[indexes[18]+k*map.width*map.height]);
			sum += values[3]*(map.elements[indexes[6 ]+k*map.width*map.height] + map.elements[indexes[8 ] +k*map.width*map.height] + 
					  map.elements[indexes[16]+k*map.width*map.height] + map.elements[indexes[18]+k*map.height*map.width]);
			sum += values[4]*(map.elements[indexes[7 ]+k*map.width*map.height] + map.elements[indexes[11]+k*map.width*map.height] + 
					  map.elements[indexes[13]+k*map.width*map.height] + map.elements[indexes[17]+k*map.width*map.height]);
			sum += values[5]* map.elements[indexes[12]+k*map.width*map.height];

			new_map.elements[indexes[12]+k*map.width*map.height] = decay * min(sum/256.f, 1.f) + (1-decay)*map.elements[indexes[12]+k*map.width*map.height];

		}
	}

}

__global__ void cuda_sense(Agents agents, TrailMatrix map, float senseAngle, float senseRadius, int senseSize, float turnspeed, float *rdm_num){
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < agents.n_agents){
		
		float w[3] = {0, 0, 0};
		float angles[3] = {-senseAngle, 0, senseAngle};

		for(int ii=0; ii<3; ii++){

			float angle = agents.angle[i] + angles[ii];
			float dir_x = 2*senseRadius*cos(angle);
			float dir_y = 2*senseRadius*sin(angle);

			float x = agents.pos[i].x + dir_x;
			float y = agents.pos[i].y + dir_y;

			int species = agents.species[i];

			int posx, posy;

			for(int k=-senseSize; k<senseSize; k++){
				for(int j=-senseSize; j<senseSize; j++){
					posx = int(x) + k;
					posy = int(y) + j;

					if(0<=posx && posx<map.height && 0 <=posy && posy<map.width){
						int index = posx*map.width + posy;
						w[ii] += map.elements[index+species*map.width*map.height];
						for(int s=0; s<map.n_species; s++)
							if(s != species)
								w[ii] -= map.elements[index+s*map.width*map.height];
					}
				}
			}
		}

		if(w[1] > w[0] && w[1] > w[2])
			agents.angle[i] += 0;
		else if(w[1] < w[0] && w[1] < w[2])
			agents.angle[i] += (2*rdm_num[i] - 1)*turnspeed; 
		else if(w[2] > w[0])
			agents.angle[i] -= rdm_num[i]*turnspeed;
		else if(w[0] > w[2])
		        agents.angle[i]	+= rdm_num[i]*turnspeed;

	}

}

void move(const Agents &d_agents, const TrailMatrix &d_map, Params params, curandGenerator_t gen, float* rdm_num){

	int threadsPerBlock = 512;
	int numBlocks(d_agents.n_agents / threadsPerBlock + 1);

	curandGenerateUniform(gen, rdm_num, d_agents.n_agents);
	cuda_move<<<numBlocks, threadsPerBlock>>>(d_agents, d_map, params.speed, params.dt, rdm_num);
	
	TrailMatrix d_n_map;
	d_n_map.height = d_map.height;
	d_n_map.width = d_map.width;
	d_n_map.n_species = d_map.n_species;

	cudaMalloc(&(d_n_map.elements), d_map.width*d_map.height*d_map.n_species*sizeof(float));
	numBlocks = (d_map.width*d_map.height) / threadsPerBlock + 1;
	cuda_gauss<<<numBlocks, threadsPerBlock>>>(d_map, d_n_map, params.diff_decay);
	cudaMemcpy(d_map.elements, d_n_map.elements, d_map.width*d_map.height*d_map.n_species*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(d_n_map.elements);
	
	numBlocks = (d_map.width*d_map.height*d_map.n_species) / threadsPerBlock + 1;
	cuda_evaporate<<<numBlocks, threadsPerBlock>>>(d_map, params.evaporate_rate, params.dt);
	
	curandGenerateUniform(gen, rdm_num, d_agents.n_agents);
	numBlocks = d_agents.n_agents / threadsPerBlock + 1;
	cuda_sense<<<numBlocks, threadsPerBlock>>>(d_agents, d_map, params.senseAngle, params.senseRadius, params.senseSize, params.turnspeed, rdm_num);
}
