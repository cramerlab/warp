#include "Functions.h"
using namespace gtom;

#define TOMO_THREADS 128

__global__ void MultiParticleDiffKernel(float* hp_result,
										float2* hp_experimental,
										int dimdata,
										uint dimdataft,
										int elementsdata,
										float2* d_shifts,
										float3* d_angles,
										float* d_defoci,
										CTFParamsLean ctfparams,
										float* d_weights,
										cudaTex d_reference1Re,
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										int dimprojector,
										int* d_subsets,
										float relativeweight,
										int ntilts,
										int itilt);
__global__ void MultiParticleSimulateKernel(float2* d_result,
											int dimdata,
											uint dimdataft,
											int elementsdata,
											float2* d_shifts,
											float3* d_angles,
											float* d_defoci,
											CTFParamsLean ctfparams,
											float* d_weights,
											cudaTex d_reference1Re,
											cudaTex d_reference1Im,
											cudaTex d_reference2Re,
											cudaTex d_reference2Im,
											int dimprojector,
											int* d_subsets,
											int ntilts,
											int itilt);
__global__ void MultiParticleCorr2DKernel(float* d_resultab,
										float* d_resulta2,
										float* d_resultb2,
										float2* hp_experimental,
										int dimdata,
										uint dimdataft,
										int elementsdata,
										float2* d_shifts,
										float3* d_angles,
										float* d_defoci,
										CTFParamsLean ctfparams,
										float* d_weights,
										cudaTex d_reference1Re,
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										int dimprojector,
										int* d_subsets,
										int ntilts,
										int itilt,
										bool getdivisor);


__declspec(dllexport) void MultiParticleDiff(float* hp_result,
											 float2** hp_experimental, 
											 int dimdata,
											 float2* h_shifts,
											 float3* h_angles,
											 float* h_defoci,
											 float* d_weights,
											 CTFParams* h_ctfparams,
											 cudaTex* h_volumeRe,
											 cudaTex* h_volumeIm,
											 int dimprojector,
											 int* d_subsets,
											 float* h_relativeweights,
											 int nparticles,
											 int ntilts)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));
	float* d_defoci = (float*)CudaMallocFromHostArray(h_defoci, ntilts * nparticles * sizeof(float));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	for (int i = 0; i < ntilts; i++)
	{
		MultiParticleDiffKernel <<<Grid, TpB>>> (hp_result,
												 hp_experimental[i],
												 dimdata,
												 ElementsFFT1(dimdata),
												 elementsdata,
												 d_shifts,
												 d_angles,
												 d_defoci,
												 CTFParamsLean(h_ctfparams[i], toInt3(1, 1, 1)),
												 d_weights + i * elementsdata,
												 h_volumeRe[0],
												 h_volumeIm[0],
												 h_volumeRe[1],
												 h_volumeIm[1],
												 dimprojector,
												 d_subsets,
												 h_relativeweights[i],
												 ntilts,
												 i);
	}

	cudaDeviceSynchronize();

	cudaFree(d_defoci);
	cudaFree(d_angles);
	cudaFree(d_shifts);
}


__declspec(dllexport) void MultiParticleCorr2D(float* d_resultab,
	    									   float* d_resulta2,
											   float* d_resultb2,
									 		   float2** hp_experimental,
									 		   int dimdata,
										   	   float2* h_shifts,
											   float3* h_angles,
											   float* h_defoci,
											   float* d_weights,
											   CTFParams* h_ctfparams,
											   cudaTex* h_volumeRe,
											   cudaTex* h_volumeIm,
											   int dimprojector,
											   int* d_subsets,
											   int nparticles,
											   int ntilts,
											   bool getdivisor)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));
	float* d_defoci = (float*)CudaMallocFromHostArray(h_defoci, ntilts * nparticles * sizeof(float));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	for (int i = 0; i < ntilts; i++)
	{
		MultiParticleCorr2DKernel << <Grid, TpB >> > (d_resultab + ElementsFFT1(dimdata) * dimdata * i,
														d_resulta2 + ElementsFFT1(dimdata) * dimdata * i,
														d_resultb2 + ElementsFFT1(dimdata) * dimdata * i,
														hp_experimental[i],
														dimdata,
														ElementsFFT1(dimdata),
														elementsdata,
														d_shifts,
														d_angles,
														d_defoci,
														CTFParamsLean(h_ctfparams[i], toInt3(1, 1, 1)),
														d_weights + i * elementsdata,
														h_volumeRe[0],
														h_volumeIm[0],
														h_volumeRe[1],
														h_volumeIm[1],
														dimprojector,
														d_subsets,
														ntilts,
														i,
														getdivisor);
	}

	cudaDeviceSynchronize();

	cudaFree(d_defoci);
	cudaFree(d_angles);
	cudaFree(d_shifts);
}


__declspec(dllexport) void MultiParticleSimulate(float* d_result,
												 int2 dimsmic,
												 int dimdata,
												 float2* h_positions,
												 float2* h_shifts,
												 float3* h_angles,
												 float* h_defoci,
												 float* d_weights,
												 CTFParams* h_ctfparams,
												 cudaTex* h_volumeRe,
												 cudaTex* h_volumeIm,
												 int dimprojector,
												 int* d_subsets,
												 int nparticles,
												 int ntilts)
{
	int3* h_insertorigins = (int3*)malloc(nparticles * ntilts * sizeof(int3));
	float2 particlecenter = make_float2(dimdata / 2, dimdata / 2);
	for (int p = 0; p < nparticles; p++)
	{
		for (int t = 0; t < ntilts; t++)
		{
			float2 pos = h_positions[p * ntilts + t];
			float2 shift = h_shifts[p * ntilts + t];
			pos += shift;
			shift = pos;
			pos = make_float2(floor(pos.x), floor(pos.y));
			shift = shift - pos + particlecenter;

			h_insertorigins[t * nparticles + p] = toInt3((int)pos.x - dimdata / 2, (int)pos.y - dimdata / 2, 0);
			h_shifts[p * ntilts + t] = shift;
		}
	}
	int3* d_insertorigins = (int3*)CudaMallocFromHostArray(h_insertorigins, nparticles * ntilts * sizeof(int3));
	free(h_insertorigins);

	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));
	float* d_defoci = (float*)CudaMallocFromHostArray(h_defoci, ntilts * nparticles * sizeof(float));

	float2* d_projft;
	cudaMalloc((void**)&d_projft, ElementsFFT1(dimdata) * dimdata * nparticles * sizeof(float2));
	float* d_proj;
	cudaMalloc((void**)&d_proj, dimdata * dimdata * nparticles * sizeof(float));

	cufftHandle planback = d_IFFTC2RGetPlan(2, toInt3(dimdata, dimdata, 1), nparticles);

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	for (int i = 0; i < ntilts; i++)
	{
		MultiParticleSimulateKernel << <Grid, TpB >> > (d_projft,
														dimdata,
														ElementsFFT1(dimdata),
														elementsdata,
														d_shifts,
														d_angles,
														d_defoci,
														CTFParamsLean(h_ctfparams[i], toInt3(1, 1, 1)),
														d_weights + i * elementsdata,
														h_volumeRe[0],
														h_volumeIm[0],
														h_volumeRe[1],
														h_volumeIm[1],
														dimprojector,
														d_subsets,
														ntilts,
														i);
		d_IFFTC2R(d_projft, d_proj, &planback);

		d_InsertAdditive(d_proj, d_result + Elements2(dimsmic) * i, toInt3(dimsmic), toInt3(dimdata, dimdata, 1), d_insertorigins + nparticles * i, nparticles);
	}

	cudaDeviceSynchronize();

	cufftDestroy(planback);
	cudaFree(d_proj);
	cudaFree(d_projft);

	cudaFree(d_defoci);
	cudaFree(d_angles);
	cudaFree(d_shifts);
	cudaFree(d_insertorigins);
}

__device__ float2 d_GetProjectionSlice(cudaTex t_volumeRe, cudaTex t_volumeIm, int dim, glm::vec3 pos, glm::mat3 rotation, float2 shift, int dimproj)
{
	float shiftfactor = -(shift.x * pos.x + shift.y * pos.y) / dimproj * (tfloat)PI2;
	float2 shiftmultiplicator = make_cuComplex(cos(shiftfactor), sin(shiftfactor));

	float2 val;

	pos = pos * rotation;	// vector * matrix uses the transposed version, which is exactly what is needed here

	// Only asymmetric half is stored
	float is_neg_x = 1.0f;
	if (pos.x < -1e-5f)
	{
		// Get complex conjugated hermitian symmetry pair
		pos.x = abs(pos.x);
		pos.y = -pos.y;
		pos.z = -pos.z;
		is_neg_x = -1.0f;
	}

	// Trilinear interpolation (with physical coords)
	float x0 = floor(pos.x + 1e-5f);
	pos.x -= x0;
	x0 += 0.5f;
	float x1 = x0 + 1.0f;

	float y0 = floor(pos.y);
	pos.y -= y0;
	float y1 = y0 + 1;
	if (y0 < 0)
		y0 += dim;
	y0 += 0.5f;
	if (y1 < 0)
		y1 += dim;
	y1 += 0.5f;

	float z0 = floor(pos.z);
	pos.z -= z0;
	float z1 = z0 + 1;
	if (z0 < 0)
		z0 += dim;
	z0 += 0.5f;
	if (z1 < 0)
		z1 += dim;
	z1 += 0.5f;

	float2 d000 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y0, z0), tex3D<float>(t_volumeIm, x0, y0, z0));
	float2 d001 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y0, z0), tex3D<float>(t_volumeIm, x1, y0, z0));
	float2 d010 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y1, z0), tex3D<float>(t_volumeIm, x0, y1, z0));
	float2 d011 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y1, z0), tex3D<float>(t_volumeIm, x1, y1, z0));
	float2 d100 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y0, z1), tex3D<float>(t_volumeIm, x0, y0, z1));
	float2 d101 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y0, z1), tex3D<float>(t_volumeIm, x1, y0, z1));
	float2 d110 = make_cuComplex(tex3D<float>(t_volumeRe, x0, y1, z1), tex3D<float>(t_volumeIm, x0, y1, z1));
	float2 d111 = make_cuComplex(tex3D<float>(t_volumeRe, x1, y1, z1), tex3D<float>(t_volumeIm, x1, y1, z1));

	float2 dx00 = lerp(d000, d001, pos.x);
	float2 dx01 = lerp(d010, d011, pos.x);
	float2 dx10 = lerp(d100, d101, pos.x);
	float2 dx11 = lerp(d110, d111, pos.x);

	float2 dxy0 = lerp(dx00, dx01, pos.y);
	float2 dxy1 = lerp(dx10, dx11, pos.y);

	val = lerp(dxy0, dxy1, pos.z);

	val.y *= is_neg_x;

	return cmul(val, shiftmultiplicator);
}

__global__ void MultiParticleDiffKernel(float* hp_result, 
										float2* hp_experimental, 
										int dimdata, 
										uint dimdataft,
										int elementsdata,
										float2* d_shifts, 
										float3* d_angles, 
										float* d_defoci, 
										CTFParamsLean ctfparams, 
										float* d_weights,
										cudaTex d_reference1Re, 
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										int dimprojector, 
										int* d_subsets,
										float relativeweight,
										int ntilts,
										int itilt)
{
	__shared__ float s_diff2[128];
	__shared__ float s_ref2[128];
	__shared__ float s_part2[128];
	//__shared__ float s_weight[128];
	
	hp_experimental += blockIdx.x * elementsdata;
	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);
	float defocus = d_defoci[blockIdx.x * ntilts + itilt];

	cudaTex referenceRe = d_subsets[blockIdx.x] == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = d_subsets[blockIdx.x] == 0 ? d_reference1Im : d_reference2Im;

	ctfparams.defocus = d_defoci[blockIdx.x * ntilts + itilt] * (-1e4f);

	float diff2 = 0, ref2 = 0, part2 = 0, weight = 0;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
		{
			//hp_experimental[id] = make_float2(0, 0);
			continue;
		}

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(x, y, 0), angles, shift, dimdata);

		float angle = atan2f((float)y, (float)x);
		float r = sqrt((float)x * x + y * y) / dimdata;
		float pixelsize = ctfparams.pixelsize + ctfparams.pixeldelta * cosf(2.0f * (angle - ctfparams.pixelangle));
		r /= pixelsize;
		//float ctf = d_GetCTF<false, false>(r, angle, ctfparams);
		//float weight = d_weights[id];

		val *= d_weights[id];// *ctf;
		//weight += d_weights[id];

		//float2 diff = hp_experimental[id] - (val * ctf);
		//diff2 += dotp2(diff, diff) * d_weights[id];
		//hp_experimental[id] = val;

		float2 part = hp_experimental[id];
		float cc = dotp2(part, val);
		if (cc != 0)
		{
			diff2 += cc;
			ref2 += dotp2(val, val);
			part2 += dotp2(part, part);
		}
	}

	s_diff2[threadIdx.x] = diff2;
	s_ref2[threadIdx.x] = ref2;
	s_part2[threadIdx.x] = part2;
	//s_weight[threadIdx.x] = weight;
	__syncthreads();

	/*for (uint lim = 64; lim > 1; lim >>= 1)
	{
		if (threadIdx.x < lim)
		{
			diff2 += s_diff2[threadIdx.x + lim];
			s_diff2[threadIdx.x] = diff2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		diff2 += s_diff2[1];

		hp_result[blockIdx.x * ntilts + itilt] = blockIdx.x;
	}*/

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			diff2 += s_diff2[i];
			ref2 += s_ref2[i];
			part2 += s_part2[i];
			//weight += s_weight[i];
		}

		diff2 /= sqrt(ref2 * part2);
		diff2 *= relativeweight;

		hp_result[blockIdx.x * ntilts + itilt] = diff2;
	}
}

__global__ void MultiParticleSimulateKernel(float2* d_result,
											int dimdata,
											uint dimdataft,
											int elementsdata,
											float2* d_shifts,
											float3* d_angles,
											float* d_defoci,
											CTFParamsLean ctfparams,
											float* d_weights,
											cudaTex d_reference1Re,
											cudaTex d_reference1Im,
											cudaTex d_reference2Re,
											cudaTex d_reference2Im,
											int dimprojector,
											int* d_subsets,
											int ntilts,
											int itilt)
{
	d_result += blockIdx.x * elementsdata;
	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);
	float defocus = d_defoci[blockIdx.x * ntilts + itilt];

	cudaTex referenceRe = d_subsets[blockIdx.x] == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = d_subsets[blockIdx.x] == 0 ? d_reference1Im : d_reference2Im;

	ctfparams.defocus = d_defoci[blockIdx.x * ntilts + itilt] * (-1e4f);

	float diff2 = 0, ref2 = 0;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
		{
			d_result[id] = make_float2(0, 0);
			continue;
		}

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(x, y, 0), angles, shift, dimdata);

		float angle = atan2f((float)y, (float)x);
		float r = sqrt((float)x * x + y * y) / dimdata;
		float pixelsize = ctfparams.pixelsize + ctfparams.pixeldelta * cosf(2.0f * (angle - ctfparams.pixelangle));
		r /= pixelsize;
		float ctf = d_GetCTF<false, false>(r, angle, ctfparams);

		val *= ctf;// *d_weights[id];

		d_result[id] = val;
	}
}

__global__ void MultiParticleCorr2DKernel(float* d_resultab,
										float* d_resulta2,
										float* d_resultb2,
										float2* hp_experimental,
										int dimdata,
										uint dimdataft,
										int elementsdata,
										float2* d_shifts,
										float3* d_angles,
										float* d_defoci,
										CTFParamsLean ctfparams,
										float* d_weights,
										cudaTex d_reference1Re,
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										int dimprojector,
										int* d_subsets,
										int ntilts,
										int itilt,
										bool getdivisor)
{
	hp_experimental += blockIdx.x * elementsdata;
	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);
	float defocus = d_defoci[blockIdx.x * ntilts + itilt];

	cudaTex referenceRe = d_subsets[blockIdx.x] == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = d_subsets[blockIdx.x] == 0 ? d_reference1Im : d_reference2Im;

	ctfparams.defocus = d_defoci[blockIdx.x * ntilts + itilt] * (-1e4f);
	
	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
			continue;

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(x, y, 0), angles, shift, dimdata);

		float angle = atan2f((float)y, (float)x);
		float r = sqrt((float)x * x + y * y) / dimdata;
		float pixelsize = ctfparams.pixelsize + ctfparams.pixeldelta * cosf(2.0f * (angle - ctfparams.pixelangle));
		r /= pixelsize;
		float ctf = d_GetCTF<false, false>(r, angle, ctfparams);
		//float weight = d_weights[id];

		val *= ctf * d_weights[id];

		//float2 diff = hp_experimental[id] - (val * ctf);
		//diff2 += dotp2(diff, diff) * d_weights[id];
		//hp_experimental[id] = val;

		float2 part = hp_experimental[id];
		float cc = dotp2(part, val);
		
		{
			float part2 = dotp2(part, part);
			float val2 = dotp2(val, val);

			if (!getdivisor)
				atomicAdd((float*)(d_resultab + id), cc);
			else
			{
				if (part2 != 0)
					atomicAdd((float*)(d_resultab + id), sqrt(val2 / part2));
			}
			atomicAdd((float*)(d_resulta2 + id), val2);
			atomicAdd((float*)(d_resultb2 + id), part2);
		}
	}
}