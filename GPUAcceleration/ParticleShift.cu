#include "Functions.h"
#include <device_functions.h>
using namespace gtom;

#define SHIFT_THREADS 128

__global__ void ParticleShiftGetDiffKernel(float2* d_phase, float2* d_average, float2* d_shiftfactors, float* d_invsigma, uint length, uint probelength, float2* d_shifts, float* d_diff, float* d_debugdiff);
__global__ void ParticleShiftGetGradKernel(float2* d_phase, float2* d_average, float2* d_shiftfactors, float* d_invsigma, uint length, uint probelength, float2* d_shifts, float2* d_grad);

/*

Supplied with a stack of frames, extraction positions for sub-regions, and a mask of relevant pixels in Fspace, 
this method extracts portions of each frame, computes the FT, and returns the relevant pixels.

*/

__declspec(dllexport) void CreateParticleShift(float* d_frame,
												int2 dimsframe,
												int nframes,
												float2* h_positions,
												float2* h_shifts,
												int npositions,
												int2 dimsregion,
												size_t* h_indices,
												uint indiceslength,
												float* d_masks,
												float2* d_projections,
												CTFParams* h_ctfparams,
												float2* d_ctfcoords,
												float* d_invsigma,
												float pixelmajor,
												float pixelminor,
												float pixelangle,
												float2* d_outputparticles,
												float2* d_outputprojections,
												float* d_outputinvsigma)
{
	int2 dimspadded = toInt2(dimsregion.x + 64, dimsregion.y + 64);

	size_t* d_indices = (size_t*)CudaMallocFromHostArray(h_indices, indiceslength * sizeof(size_t));
	tfloat* d_temp;
	cudaMalloc((void**)&d_temp, npositions * ElementsFFT2(dimsregion) * sizeof(tcomplex));
	tcomplex* d_tempft;
	cudaMalloc((void**)&d_tempft, npositions * ElementsFFT2(dimsregion) * sizeof(tcomplex));
	tfloat* d_extracts;
	cudaMalloc((void**)&d_extracts, npositions * Elements2(dimspadded) * sizeof(float));
	int3* d_origins;
	cudaMalloc((void**)&d_origins, npositions * sizeof(int3));

	tcomplex* d_dense;
	cudaMalloc((void**)&d_dense, npositions * indiceslength * sizeof(tcomplex));

	tfloat* d_sums = CudaMallocValueFilled(npositions * Elements2(dimsregion), (tfloat)0);
	tfloat* d_sumamps = CudaMallocValueFilled(npositions * indiceslength, (tfloat)0);

	for (uint z = 0; z < nframes; z++)
	{
		// Get closest origins for extractions, add residuals to the shifts

		int3* h_origins = (int3*)malloc(npositions * sizeof(int3));
		float3* h_overallshifts = (float3*)malloc(npositions * sizeof(float3));
		for (uint i = 0; i < npositions; i++)
		{
			float2 position = h_positions[i];
			float2 shift = h_shifts[z * npositions + i];
		    h_origins[i] = toInt3((int)position.x - dimspadded.x / 2, (int)position.y - dimspadded.y / 2, 0);
			h_overallshifts[i] = make_float3(shift.x - (position.x - (int)position.x), 
											shift.y - (position.y - (int)position.y), 
											0.0f);
		}		
		cudaMemcpy(d_origins, h_origins, npositions * sizeof(int3), cudaMemcpyHostToDevice);
		free(h_origins);

		d_ExtractMany(d_frame + Elements2(dimsframe) * z, d_extracts, toInt3(dimsframe), toInt3(dimspadded), d_origins, npositions);
		d_Shift(d_extracts, d_extracts, toInt3(dimspadded), (tfloat3*)h_overallshifts, NULL, NULL, NULL, npositions);
		d_Pad(d_extracts, d_temp, toInt3(dimspadded), toInt3(dimsregion), T_PAD_VALUE, (tfloat)0, npositions);

		d_MagAnisotropyCorrect(d_temp, dimsregion, d_extracts, dimsregion, pixelmajor, pixelminor, pixelangle, 4, npositions);

		//float* h_temp;
		//ReadMRC("refs.mrc", (void**)&h_temp);
		//cudaMemcpy(d_temp, h_temp, Elements2(dimsregion) * npositions * sizeof(float), cudaMemcpyHostToDevice);
		//cudaFreeHost(h_temp);

		d_NormBackground(d_extracts, d_temp, toInt3(dimsregion), (uint)(100.0f / 1.057f), true, npositions);
		//d_WriteMRC(d_temp, toInt3(dimsregion.x, dimsregion.y, npositions), "d_extracts.mrc");

		//d_NormMonolithic(d_temp, d_temp, Elements2(dimsregion), d_masks, T_NORM_MEAN01STD, npositions);
		//d_MultiplyByVector(d_temp, d_masks, d_temp, Elements2(dimsregion) * npositions);
		//if (z == 1)
			//d_WriteMRC(d_temp, toInt3(dimsregion.x, dimsregion.y, npositions), "d_extractsmasked.mrc");
		d_AddVector(d_sums, d_temp, d_sums, npositions * Elements2(dimsregion));

		d_FFTR2C(d_temp, d_tempft, 2, toInt3(dimsregion), npositions);
		d_RemapHalfFFT2Half(d_tempft, d_tempft, toInt3(dimsregion), npositions);
		d_Remap(d_tempft, d_indices, d_outputparticles + indiceslength * npositions * z, indiceslength, ElementsFFT2(dimsregion), make_cuComplex(0, 0), npositions);

		d_Abs(d_outputparticles + indiceslength * npositions * z, d_temp, indiceslength * npositions);
		d_AddVector(d_sumamps, d_temp, d_sumamps, indiceslength * npositions);

		free(h_overallshifts);
	}
	//d_WriteMRC(d_sums, toInt3(dimsregion.x, dimsregion.y, npositions), "d_extracts.mrc");
	//d_WriteMRC(d_sumamps, toInt3(indiceslength, npositions, 1), "d_extractsamps.mrc");

	d_CTFSimulate(h_ctfparams, d_ctfcoords, d_temp, ElementsFFT2(dimsregion), false, npositions);
	//d_WriteMRC(d_temp, toInt3(dimsregion.x / 2 + 1, dimsregion.y, npositions), "d_ctf.mrc");
	d_ComplexMultiplyByVector(d_projections, d_temp, d_projections, ElementsFFT2(dimsregion) * npositions);

	d_IFFTC2R(d_projections, d_temp, 2, toInt3(dimsregion), npositions);
	d_RemapFullFFT2Full(d_temp, d_temp, toInt3(dimsregion), npositions);
	d_NormBackground(d_temp, d_temp, toInt3(dimsregion), (uint)(100.0f / 1.057f), false, npositions);
	//d_WriteMRC(d_temp, toInt3(dimsregion.x, dimsregion.y, npositions), "d_projections.mrc");
	d_FFTR2C(d_temp, d_projections, 2, toInt3(dimsregion), npositions);
	d_RemapHalfFFT2Half(d_projections, d_projections, toInt3(dimsregion), npositions);
	d_Remap(d_projections, d_indices, d_outputprojections, indiceslength, ElementsFFT2(dimsregion), make_cuComplex(0, 0), npositions);

	d_RemapHalfFFT2Half(d_invsigma, d_invsigma, toInt3(dimsregion));
	d_Remap(d_invsigma, d_indices, d_outputinvsigma, indiceslength, ElementsFFT2(dimsregion), (float)0, 1);

	cudaFree(d_sums);
	cudaFree(d_sumamps);

	cudaFree(d_dense);
	cudaFree(d_tempft);
	cudaFree(d_temp);
	cudaFree(d_indices);
	cudaFree(d_extracts);
	cudaFree(d_origins);
}

__declspec(dllexport) void ParticleShiftGetDiff(float2* d_phase, 
											float2* d_average, 
											float2* d_shiftfactors, 
											float* d_invsigma,
											uint length, 
											uint probelength,
											float2* d_shifts,
											float* h_diff, 
											uint npositions, 
											uint nframes)
{
	int TpB = tmin(SHIFT_THREADS, NextMultipleOf(probelength, 32));
	dim3 grid = dim3(tmin(128, (probelength + TpB - 1) / TpB), npositions, nframes);

	float* d_diff;
	cudaMalloc((void**)&d_diff, npositions * nframes * grid.x * sizeof(float));
	float* d_diffreduced;
	cudaMalloc((void**)&d_diffreduced, npositions * nframes * sizeof(float));

	float* d_debugdiff = NULL;
	//cudaMalloc((void**)&d_debugdiff, npositions * nframes * length * sizeof(float));

	ParticleShiftGetDiffKernel <<<grid, TpB>>> (d_phase, d_average, d_shiftfactors, d_invsigma, length, probelength, d_shifts, d_diff, d_debugdiff);

	//d_WriteMRC(d_debugdiff, toInt3(129, 256, npositions), "d_debugdiff.mrc");

	d_SumMonolithic(d_diff, d_diffreduced, grid.x, npositions * nframes);
	d_DivideByScalar(d_diffreduced, d_diffreduced, npositions * nframes, (float)grid.x);
	cudaMemcpy(h_diff, d_diffreduced, npositions * nframes * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_diffreduced);
	cudaFree(d_diff);
}

__global__ void ParticleShiftGetDiffKernel(float2* d_phase, float2* d_average, float2* d_shiftfactors, float* d_invsigma, uint length, uint probelength, float2* d_shifts, float* d_diff, float* d_debugdiff)
{
	__shared__ float s_diff[SHIFT_THREADS];
	s_diff[threadIdx.x] = 0.0f;

	uint specid = blockIdx.z * gridDim.y + blockIdx.y;
	d_phase += specid * length;
	d_average += blockIdx.y * length;
	//d_debugdiff += specid * length;

	float2 shift = d_shifts[specid];
	float diffsum = 0.0f;

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; 
		 id < probelength; 
		 id += gridDim.x * blockDim.x)
	{
		float2 value = d_phase[id];
		float2 average = d_average[id];

		float2 shiftfactors = d_shiftfactors[id];

		float phase = shiftfactors.x * shift.x + shiftfactors.y * shift.y;
		float2 change = make_float2(__cosf(phase), __sinf(phase));
		value = cuCmulf(value, change);

		float2 diff = value - average;

		diffsum += (diff.x * diff.x + diff.y * diff.y) * d_invsigma[id];
		//d_debugdiff[id] = (diff.x * diff.x + diff.y * diff.y) * d_invsigma[id];
	}

	s_diff[threadIdx.x] = diffsum;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (uint id = 1; id < blockDim.x; id++)
			diffsum += s_diff[id];

		d_diff[specid * gridDim.x + blockIdx.x] = diffsum / (float)probelength;
	}
}

__declspec(dllexport) void ParticleShiftGetGrad(float2* d_phase, 
										float2* d_average, 
										float2* d_shiftfactors, 
										float* d_invsigma,
										uint length, 
										uint probelength,
										float2* d_shifts,
										float2* h_grad, 
										uint npositions, 
										uint nframes)
{
	int TpB = tmin(SHIFT_THREADS, NextMultipleOf(probelength, 32));
	dim3 grid = dim3(tmin(128, (probelength + TpB - 1) / TpB), npositions, nframes);

	float2* d_grad;
	cudaMalloc((void**)&d_grad, npositions * nframes * grid.x * sizeof(float2));
	float2* d_gradreduced;
	cudaMalloc((void**)&d_gradreduced, npositions * nframes * sizeof(float2));

	ParticleShiftGetGradKernel <<<grid, TpB>>> (d_phase, d_average, d_shiftfactors, d_invsigma, length, probelength, d_shifts, d_grad);

	float2* h_grad2 = (float2*)MallocFromDeviceArray(d_grad, npositions * nframes * grid.x * sizeof(float2));
	free(h_grad2);

	d_SumMonolithic(d_grad, d_gradreduced, grid.x, npositions * nframes);
	d_DivideByScalar((float*)d_gradreduced, (float*)d_gradreduced, npositions * nframes * 2, (float)grid.x);
	cudaMemcpy(h_grad, d_gradreduced, npositions * nframes * sizeof(float2), cudaMemcpyDeviceToHost);
	
	cudaFree(d_gradreduced);
	cudaFree(d_grad);
}

__global__ void ParticleShiftGetGradKernel(float2* d_phase, 
									float2* d_average, 
									float2* d_shiftfactors, 
									float* d_invsigma,
									uint length, 
									uint probelength, 
									float2* d_shifts, 
									float2* d_grad)
{
	__shared__ float2 s_grad[SHIFT_THREADS];
	s_grad[threadIdx.x] = make_float2(0.0f, 0.0f);

	uint specid = blockIdx.z * gridDim.y + blockIdx.y;
	d_phase += specid * length;
	d_average += blockIdx.y * length;

	float2 shift = d_shifts[specid];
	float2 gradsum = make_float2(0.0f, 0.0f);

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; 
		 id < probelength; 
		 id += gridDim.x * blockDim.x)
	{
		float2 value = d_phase[id];
		float2 average = d_average[id];

		float2 shiftfactors = d_shiftfactors[id];

		{
			float phase = shiftfactors.x * (shift.x + 0.0001f) + shiftfactors.y * shift.y;
			float2 change = make_float2(__cosf(phase), __sinf(phase));
			float2 valueplus = cmul(value, change);
			float2 diffplus = valueplus - average;
			float scoreplus = diffplus.x * diffplus.x + diffplus.y * diffplus.y;

			phase = shiftfactors.x * (shift.x - 0.0001f) + shiftfactors.y * shift.y;
			change = make_float2(__cosf(phase), __sinf(phase));
			float2 valueminus = cmul(value, change);
			float2 diffminus = valueminus - average;
			float scoreminus = diffminus.x * diffminus.x + diffminus.y * diffminus.y;
		
			gradsum.x += (scoreplus - scoreminus) / 0.0002f * d_invsigma[id];
		}

		{
			float phase = shiftfactors.x * shift.x + shiftfactors.y * (shift.y + 0.0001f);
			float2 change = make_float2(__cosf(phase), __sinf(phase));
			float2 valueplus = cmul(value, change);
			float2 diffplus = valueplus - average;
			float scoreplus = diffplus.x * diffplus.x + diffplus.y * diffplus.y;

			phase = shiftfactors.x * shift.x + shiftfactors.y * (shift.y - 0.0001f);
			change = make_float2(__cosf(phase), __sinf(phase));
			float2 valueminus = cmul(value, change);
			float2 diffminus = valueminus - average;
			float scoreminus = diffminus.x * diffminus.x + diffminus.y * diffminus.y;
		
			gradsum.y += (scoreplus - scoreminus) / 0.0002f * d_invsigma[id];
		}
	}

	s_grad[threadIdx.x] = gradsum;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (uint id = 1; id < blockDim.x; id++)
			gradsum = gradsum + s_grad[id];

		d_grad[specid * gridDim.x + blockIdx.x] = gradsum / (float)probelength;
	}
}