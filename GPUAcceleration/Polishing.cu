#include "Functions.h"
#include <device_functions.h>
using namespace gtom;

#define SHIFT_THREADS 128

__global__ void PolishingGetDiffKernel(float2* d_particleframe, float2* d_refproj, float2* d_shiftfactors, float* d_ctf, float* d_invsigma, uint length, float2* d_shifts, float* d_diff, float* d_debugdiff);


__declspec(dllexport) void CreatePolishing(float* d_particles, float2* d_particlesft, float* d_masks, int2 dims, int2 dimscropped, int nparticles, int nframes)
{
	float* d_temp;
	cudaMalloc((void**)&d_temp, ElementsFFT2(dims) * nparticles * sizeof(float2));

	for(int z = 0; z < nframes / 3; z++)
	{
		cudaMemcpy(d_temp, d_particles + Elements2(dims) * nparticles * (z * 3 + 0), Elements2(dims) * nparticles * sizeof(float), cudaMemcpyDeviceToDevice);
		d_AddVector(d_temp, d_particles + Elements2(dims) * nparticles * (z * 3 + 1), d_temp, Elements2(dims) * nparticles);
		d_AddVector(d_temp, d_particles + Elements2(dims) * nparticles * (z * 3 + 2), d_temp, Elements2(dims) * nparticles);
		
		d_RemapFull2FullFFT(d_temp, d_temp, toInt3(dims), nparticles);
		d_FFTR2C(d_temp, (float2*)d_temp, 2, toInt3(dims), nparticles);
		d_FFTCrop((float2*)d_temp, d_particlesft + ElementsFFT2(dimscropped) * nparticles * z, toInt3(dims), toInt3(dimscropped), nparticles);
	}

	cudaFree(d_temp);
}

__declspec(dllexport) void PolishingGetDiff(float2* d_particleframe, 
												float2* d_refproj, 
												float2* d_shiftfactors, 
												float* d_ctf,
												float* d_invsigma,
												int2 dims, 
												float2* d_shifts,
												float* h_diff, 
												float* h_diffall,
												uint npositions, 
												uint nframes)
{
	int TpB = SHIFT_THREADS;
	dim3 grid = dim3(1, nframes, npositions);

	float* d_diff;
	cudaMalloc((void**)&d_diff, npositions * nframes * grid.x * sizeof(float));
	float* d_diffreduced;
	cudaMalloc((void**)&d_diffreduced, npositions * nframes * sizeof(float));

	float* d_debugdiff = NULL;
	//cudaMalloc((void**)&d_debugdiff, npositions * nframes * ElementsFFT2(dims) * sizeof(float));

	PolishingGetDiffKernel <<<grid, TpB>>> (d_particleframe, d_refproj, d_shiftfactors, d_ctf, d_invsigma, ElementsFFT2(dims), d_shifts, d_diff, d_debugdiff);

	//d_WriteMRC(d_debugdiff, toInt3(dims.x / 2 + 1, dims.y, npositions * nframes), "d_debugdiff.mrc");

	d_SumMonolithic(d_diff, d_diffreduced, nframes, npositions);
	//d_ReduceMean(d_diff, d_diffreduced, npositions, nframes);
	cudaMemcpy(h_diff, d_diffreduced, npositions * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_diffall, d_diff, npositions * nframes * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_diffreduced);
	cudaFree(d_diff);
}

__global__ void PolishingGetDiffKernel(float2* d_particleframe, float2* d_refproj, float2* d_shiftfactors, float* d_ctf, float* d_invsigma, uint length, float2* d_shifts, float* d_diff, float* d_debugdiff)
{
	__shared__ float s_diff[SHIFT_THREADS];
	s_diff[threadIdx.x] = 0.0f;

	uint specid = blockIdx.z * gridDim.y + blockIdx.y;
	d_particleframe += specid * length;
	d_refproj += specid * length;
	d_ctf += specid * length;
	d_debugdiff += specid * length;

	float2 shift = d_shifts[specid];
	float diffsum = 0.0f;

	for (uint id = threadIdx.x; 
		 id < length; 
		 id += SHIFT_THREADS)
	{
		float2 value = d_particleframe[id];
		float2 average = d_refproj[id];
		float ctf = d_ctf[id];
		average *= ctf;

		float2 shiftfactors = d_shiftfactors[id];

		float phase = shiftfactors.x * shift.x + shiftfactors.y * shift.y;
		float2 change = make_float2(__cosf(phase), __sinf(phase));
		value = cuCmulf(value, change);
		
		float2 diff = value - average;
		diffsum += dotp2(diff, diff) * d_invsigma[id];
	}
	
	s_diff[threadIdx.x] = diffsum;
	__syncthreads();

	for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			diffsum += s_diff[threadIdx.x + lim];
			s_diff[threadIdx.x] = diffsum;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		diffsum += s_diff[1];

		d_diff[specid * gridDim.x] = diffsum;
	}
}