#include "Functions.h"
using namespace gtom;

#define TOMO_THREADS 128

__global__ void TomoRefineGetDiffKernel(float2* d_experimental, float2* d_reference, float2* d_shiftfactors, float* d_ctf, uint length, float2* d_shifts, float* d_diff, float* d_weights, float* d_debugdiff);
__global__ void TomoRealspaceCorrelateKernel(float* d_projections, float* d_experimental, float* d_mask, uint elements, uint ntilts, float* d_weights, float* d_result);
__global__ void TomoGlobalAlignKernel(float2* d_experimental, float2* d_reference, float2* d_shiftfactors, float* d_ctf, uint length, uint ntilts, float2* d_shifts, float* d_diff, float* d_weights, float* d_debugdiff);


__declspec(dllexport) void TomoRefineGetDiff(float2* d_experimental, 
												float2* d_reference, 
												float2* d_shiftfactors, 
												float* d_ctf,
												float* d_weights,
												int2 dims, 
												float2* h_shifts,
												float* h_diff,
												uint nparticles)
{
	int TpB = TOMO_THREADS;
	dim3 grid = dim3(nparticles);

	float* d_diff;
	cudaMalloc((void**)&d_diff, nparticles * sizeof(float));

	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, nparticles * sizeof(float2));
	
	float* d_debugdiff = NULL;
	//cudaMalloc((void**)&d_debugdiff, npositions * nframes * ElementsFFT2(dims) * sizeof(float));

	TomoRefineGetDiffKernel <<<grid, TpB>>> (d_experimental, d_reference, d_shiftfactors, d_ctf, ElementsFFT2(dims), d_shifts, d_diff, d_weights, d_debugdiff);
	
	cudaMemcpy(h_diff, d_diff, nparticles * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_shifts);
	cudaFree(d_diff);
}

__global__ void TomoRefineGetDiffKernel(float2* d_experimental, float2* d_reference, float2* d_shiftfactors, float* d_ctf, uint length, float2* d_shifts, float* d_diff, float* d_weights, float* d_debugdiff)
{
	__shared__ float s_num[TOMO_THREADS];
	s_num[threadIdx.x] = 0.0f;
	__shared__ float s_denom1[TOMO_THREADS];
	s_denom1[threadIdx.x] = 0.0f;
	__shared__ float s_denom2[TOMO_THREADS];
	s_denom2[threadIdx.x] = 0.0f;

	uint specid = blockIdx.x;
	d_experimental += specid * length;
	d_reference += specid * length;
	d_ctf += specid * length;
	d_debugdiff += specid * length;

	float2 shift = d_shifts[specid];
	float numsum = 0.0f, denomsum1 = 0.0f, denomsum2 = 0.0f;

	for (uint id = threadIdx.x; 
		 id < length; 
		 id += TOMO_THREADS)
	{
		float2 experimental = d_experimental[id];
		float2 reference = d_reference[id];
		reference *= d_ctf[id];

		float2 shiftfactors = d_shiftfactors[id];

		float phase = shiftfactors.x * shift.x + shiftfactors.y * shift.y;
		float2 change = make_float2(__cosf(phase), __sinf(phase));
		experimental = cuCmulf(experimental, change);

		float weight = sqrt(abs(d_ctf[id]));
		experimental *= weight;
		reference *= weight;

		numsum += experimental.x * reference.x + experimental.y * reference.y;
		denomsum1 += dotp2(experimental, experimental);
		denomsum2 += dotp2(reference, reference);
	}
	
	s_num[threadIdx.x] = numsum;
	s_denom1[threadIdx.x] = denomsum1;
	s_denom2[threadIdx.x] = denomsum2;
	__syncthreads();

	for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			numsum += s_num[threadIdx.x + lim];
			s_num[threadIdx.x] = numsum;
			
			denomsum1 += s_denom1[threadIdx.x + lim];
			s_denom1[threadIdx.x] = denomsum1;
			
			denomsum2 += s_denom2[threadIdx.x + lim];
			s_denom2[threadIdx.x] = denomsum2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		numsum += s_num[1];
		denomsum1 += s_denom1[1];
		denomsum2 += s_denom2[1];

		d_diff[specid] = numsum / tmax(1e-6f, sqrt(denomsum1 * denomsum2)) * d_weights[specid];
	}
}

__declspec(dllexport) void TomoRealspaceCorrelate(float* d_projections, int2 dims, uint nprojections, uint ntilts, float* d_experimental, float* d_ctf, float* d_mask, float* d_weights, float* h_shifts, float* h_result)
{
	uint batchsize = 1024;

	float* d_result;
	cudaMalloc((void**)&d_result, nprojections * sizeof(float));

    float* d_experimentalshifted;
	cudaMalloc((void**)&d_experimentalshifted, Elements2(dims) * ntilts * sizeof(float));

	// Shift experimental data
	d_Shift(d_experimental, d_experimentalshifted, toInt3(dims), (tfloat3*)h_shifts, NULL, NULL, NULL, ntilts);
	//d_MultiplyByScalar(d_experimentalshifted, d_experimentalshifted, Elements2(dims) * ntilts, -1.0f);
	d_NormMonolithic(d_experimentalshifted, d_experimentalshifted, Elements2(dims), d_mask, T_NORM_MEAN01STD, ntilts);
	//d_MultiplyByVector(d_experimentalshifted, d_mask, d_experimentalshifted, Elements2(dims), ntilts);
	//d_WriteMRC(d_experimentalshifted, toInt3(dims.x, dims.y, ntilts), "d_experimental.mrc");

	//for (uint b = 0; b < nprojections; b += batchsize)
	{
	    //uint curbatch = tmin(batchsize, nprojections - b);

		uint TpB = 128;
		dim3 grid = dim3(nprojections, 1, 1);
		TomoRealspaceCorrelateKernel <<<grid, TpB>>> (d_projections, d_experimentalshifted, d_mask, Elements2(dims), ntilts, d_weights, d_result);
	}

	cudaMemcpy(h_result, d_result, nprojections * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_experimentalshifted);
	cudaFree(d_result);
}

__global__ void TomoRealspaceCorrelateKernel(float* d_projections, float* d_experimental, float* d_mask, uint elements, uint ntilts, float* d_weights, float* d_result)
{
    d_projections += elements * ntilts * blockIdx.x;

	__shared__ float s_sums1[128];
	__shared__ float s_samples[128];
	__shared__ float s_corrsum;

	if (threadIdx.x == 0)
		s_corrsum = 0;
	
	for (uint t = 0; t < ntilts; t++)
	{
		float tiltcorr = 0;
		float samples = 0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
		{
			float mask = d_mask[i];
			float projection = d_projections[i];
			float experimental = d_experimental[i];

			tiltcorr += projection * experimental * mask;
			samples += mask;
		}
		s_sums1[threadIdx.x] = tiltcorr;
		s_samples[threadIdx.x] = samples;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < 128; i++)
			{
				tiltcorr += s_sums1[i];
				samples += s_samples[i];
			}

			s_corrsum += tiltcorr / samples * d_weights[t];
		}
		__syncthreads();

		d_experimental += elements;
		d_projections += elements;
	}

	if (threadIdx.x == 0)
	{
		d_result[blockIdx.x] = s_corrsum;
	}
}

__declspec(dllexport) void TomoGlobalAlign(float2* d_experimental, 
												float2* d_shiftfactors, 
												float* d_ctf,
												float* d_weights,
												int2 dims, 
												float2* d_ref,
												int3 dimsref, 
												int refsupersample,
												float3* h_angles,
												uint nangles,
												float2* h_shifts,
												uint nshifts,
												uint nparticles,
												uint ntilts,
												int* h_bestangles,
												int* h_bestshifts,
												float* h_bestscores)
{
	uint batchangles = 128;

	float2* d_proj;
	cudaMalloc((void**)&d_proj, ElementsFFT2(dims) * batchangles * ntilts * sizeof(float2));

	float* d_scores;
	cudaMalloc((void**)&d_scores, nparticles * batchangles * nshifts * sizeof(float));

	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, nshifts * ntilts * sizeof(float2));

	for (uint b = 0; b < nangles; b += batchangles)
	{
		uint curbatch = tmin(batchangles, nangles - b);

		d_rlnProject(d_ref, dimsref, d_proj, toInt3(dims), (tfloat3*)h_angles + b * ntilts, refsupersample, curbatch);

		int TpB = TOMO_THREADS;
		dim3 grid = dim3(nshifts, curbatch, nparticles);
	
		float* d_debugdiff = NULL;
		//cudaMalloc((void**)&d_debugdiff, npositions * nframes * ElementsFFT2(dims) * sizeof(float));

		TomoGlobalAlignKernel <<<grid, TpB>>> (d_experimental, d_proj, d_shiftfactors, d_ctf, ElementsFFT2(dims), ntilts, d_shifts, d_scores, d_weights, d_debugdiff);
	
		float* h_scores = (float*)MallocFromDeviceArray(d_scores, nparticles * curbatch * nshifts * sizeof(float));

		for (uint p = 0; p < nparticles; p++)
			for (uint a = 0; a < curbatch; a++)
				for (uint s = 0; s < nshifts; s++)
				{
				    uint scoreid = (p * curbatch + a) * nshifts + s;
					if (h_bestscores[p] < h_scores[scoreid])
					{
					    h_bestscores[p] = h_scores[scoreid];
						h_bestangles[p] = b + a;
						h_bestshifts[p] = s;
					}
				}
	}
	
	cudaFree(d_shifts);
	cudaFree(d_scores);
	cudaFree(d_proj);
}

__global__ void TomoGlobalAlignKernel(float2* d_experimental, float2* d_reference, float2* d_shiftfactors, float* d_ctf, uint length, uint ntilts, float2* d_shifts, float* d_diff, float* d_weights, float* d_debugdiff)
{
	__shared__ float s_num[TOMO_THREADS];
	s_num[threadIdx.x] = 0.0f;
	__shared__ float s_denom1[TOMO_THREADS];
	s_denom1[threadIdx.x] = 0.0f;
	__shared__ float s_denom2[TOMO_THREADS];
	s_denom2[threadIdx.x] = 0.0f;

	uint shiftid = blockIdx.x;
	uint angleid = blockIdx.y;
	uint partid = blockIdx.z;

	d_experimental += partid * length * ntilts;
	d_reference += angleid * length * ntilts;
	d_shifts += shiftid * ntilts;
	d_ctf += partid * length * ntilts;
	d_weights += partid * ntilts;
	d_debugdiff += partid * length * ntilts;

	float partsum = 0;

	for (uint n = 0; n < ntilts; n++)
	{
		float2 shift = d_shifts[n];
		float numsum = 0, denomsum1 = 0, denomsum2 = 0;

		for (uint id = threadIdx.x; 
			 id < length; 
			 id += TOMO_THREADS)
		{
			float2 experimental = d_experimental[id];
			float2 reference = d_reference[id];
			reference *= d_ctf[id];

			float2 shiftfactors = d_shiftfactors[id];

			float phase = shiftfactors.x * shift.x + shiftfactors.y * shift.y;
			float2 change = make_float2(__cosf(phase), __sinf(phase));
			experimental = cuCmulf(experimental, change);

			float weight = abs(d_ctf[id]);
			experimental *= weight;
			reference *= weight;

			numsum += experimental.x * reference.x + experimental.y * reference.y;
			denomsum1 += dotp2(experimental, experimental);
			denomsum2 += dotp2(reference, reference);
		}
	
		s_num[threadIdx.x] = numsum;
		s_denom1[threadIdx.x] = denomsum1;
		s_denom2[threadIdx.x] = denomsum2;
		__syncthreads();

		for (uint lim = 64; lim > 1; lim >>= 1)
		{
			if (threadIdx.x < lim)
			{
				numsum += s_num[threadIdx.x + lim];
				s_num[threadIdx.x] = numsum;
			
				denomsum1 += s_denom1[threadIdx.x + lim];
				s_denom1[threadIdx.x] = denomsum1;
			
				denomsum2 += s_denom2[threadIdx.x + lim];
				s_denom2[threadIdx.x] = denomsum2;
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			numsum += s_num[1];
			denomsum1 += s_denom1[1];
			denomsum2 += s_denom2[1];

			partsum += numsum / tmax(1e-15f, sqrt(denomsum1 * denomsum2)) * d_weights[n];
		}

		d_experimental += length;
		d_reference += length;
		d_ctf += length;
	}

	__syncthreads();

	if (threadIdx.x == 0)
		d_diff[(partid * gridDim.y + angleid) * gridDim.x + shiftid] = partsum;
}