#include "Functions.h"
#include "../../gtom/include/DeviceFunctions.cuh"
using namespace gtom;

#define TOMO_THREADS 128

__global__ void MultiParticleDiffKernel(float3* d_result,
										float2* hp_experimental,
										int dimdata,
										int dimdatafull,
										uint dimdataft,
										int elementsdata,
										float2* d_shifts,
										float3* d_angles,
										glm::mat2 magnification,
										float2* d_beamtilt,
										float* d_weights,
										float2* d_beamtiltcoords,
										cudaTex d_reference1Re,
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										int dimprojector,
										int* d_subsets,
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
											int dimdatafull,
											uint dimdataft,
											int elementsdata,
											float2* d_shifts,
											float3* d_angles,
											glm::mat2 magnification,
											cudaTex d_reference1Re,
											cudaTex d_reference1Im,
											cudaTex d_reference2Re,
											cudaTex d_reference2Im,
											int dimprojector,
											int* d_subsets,
											int ntilts,
											int itilt);
__global__ void MultiParticleResidualKernel(float2* d_result,
											float2* hp_experimental,
											int dimdata,
											uint dimdataft,
											int elementsdata,
											float2* d_shifts,
											float3* d_angles,
											glm::mat2 magnification,
											float* d_weights,
											cudaTex d_reference1Re,
											cudaTex d_reference1Im,
											cudaTex d_reference2Re,
											cudaTex d_reference2Im,
											int dimprojector,
											int* d_subsets,
											int ntilts,
											int itilt);
__global__ void MultiParticleSumAmplitudesKernel(float* d_result,
												int dimdata,
												uint dimdataft,
												int elementsdata,
												float3* d_angles,
												cudaTex t_referenceRe,
												cudaTex t_referenceIm,
												int dimprojector);


__declspec(dllexport) void MultiParticleDiff(float3* h_result,
											float2** hp_experimental,
											int dimdata,
											int* h_relevantdims,
											float2* h_shifts,
											float3* h_angles,
											float3 magnification,
											float2* h_beamtilt,
											float* d_weights,
											float2* d_beamtiltcoords,
											cudaTex* h_volumeRe,
											cudaTex* h_volumeIm,
											int dimprojector,
											int* d_subsets,
											int nparticles,
											int ntilts)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));
	float2* d_beamtilt = NULL;
	if (h_beamtilt != NULL)
		d_beamtilt = (float2*)CudaMallocFromHostArray(h_beamtilt, nparticles * sizeof(float2));
	float3* d_result;
	cudaMalloc((void**)&d_result, nparticles * ntilts * sizeof(float3));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	glm::mat2 m_magnification = Matrix2Rotation(-magnification.z) * Matrix2Scale(tfloat2(magnification.x, magnification.y)) * Matrix2Rotation(magnification.z);

	for (int i = 0; i < ntilts; i++)
	{
		int relevantdim = dimdata;
		if (h_relevantdims != NULL)
			relevantdim = h_relevantdims[i];

		MultiParticleDiffKernel << <Grid, TpB >> > (d_result,
													hp_experimental[i],
													relevantdim,
													dimdata,
													ElementsFFT1(relevantdim),
													ElementsFFT1(relevantdim) * relevantdim,
													d_shifts,
													d_angles,
													m_magnification,
													d_beamtilt,
													(d_weights != NULL) ? (d_weights + i * elementsdata) : (float*)NULL,
													(h_relevantdims == NULL) ? d_beamtiltcoords : (d_beamtiltcoords + i * elementsdata),
													h_volumeRe[0],
													h_volumeIm[0],
													h_volumeRe[1],
													h_volumeIm[1],
													dimprojector,
													d_subsets,
													ntilts,
													i);
	}

	cudaMemcpy(h_result, d_result, nparticles * ntilts * sizeof(float3), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	if (d_beamtilt != NULL)
		cudaFree(d_beamtilt);
	cudaFree(d_result);

	cudaFree(d_angles);
	cudaFree(d_shifts);
}


__declspec(dllexport) void MultiParticleCorr2D(float* d_resultab,
												float* d_resulta2,
												float* d_resultb2,
												float2** hp_experimental,
												int dimdata,
												int* h_relevantdims,
												float2* h_shifts,
												float3* h_angles,
												float3 magnification,
												cudaTex* h_volumeRe,
												cudaTex* h_volumeIm,
												int dimprojector,
												int* d_subsets,
												int nparticles,
												int ntilts)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	glm::mat2 m_magnification = Matrix2Rotation(-magnification.z) * Matrix2Scale(tfloat2(magnification.x, magnification.y)) * Matrix2Rotation(magnification.z);

	for (int i = 0; i < ntilts; i++)
	{
		int relevantdim = dimdata;
		if (h_relevantdims != NULL)
			relevantdim = h_relevantdims[i];

		MultiParticleCorr2DKernel << <Grid, TpB >> > (d_resultab + elementsdata * i,
														d_resulta2 + elementsdata * i,
														d_resultb2 + elementsdata * i,
														hp_experimental[i],
														relevantdim,
														dimdata,
														ElementsFFT1(relevantdim),
														ElementsFFT1(relevantdim)* relevantdim,
														d_shifts,
														d_angles,
														m_magnification,
														h_volumeRe[0],
														h_volumeIm[0],
														h_volumeRe[1],
														h_volumeIm[1],
														dimprojector,
														d_subsets,
														ntilts,
														i);
	}

	cudaDeviceSynchronize();

	cudaFree(d_angles);
	cudaFree(d_shifts);
}


__declspec(dllexport) void MultiParticleResidual(float2* d_result,
												float2** hp_experimental,
												int dimdata,
												float2* h_shifts,
												float3* h_angles,
												float3 magnification,
												float* d_weights,
												cudaTex* h_volumeRe,
												cudaTex* h_volumeIm,
												int dimprojector,
												int* d_subsets,
												int nparticles,
												int ntilts)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, ntilts * nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, ntilts * nparticles * sizeof(float3));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	glm::mat2 m_magnification = Matrix2Rotation(-magnification.z) * Matrix2Scale(tfloat2(magnification.x, magnification.y)) * Matrix2Rotation(magnification.z);

	for (int i = 0; i < ntilts; i++)
	{
		MultiParticleResidualKernel << <Grid, TpB >> > (d_result + ElementsFFT1(dimdata) * dimdata * i,
			hp_experimental[i],
			dimdata,
			ElementsFFT1(dimdata),
			elementsdata,
			d_shifts,
			d_angles,
			m_magnification,
			d_weights,
			h_volumeRe[0],
			h_volumeIm[0],
			h_volumeRe[1],
			h_volumeIm[1],
			dimprojector,
			d_subsets,
			ntilts,
			i);
	}

	cudaDeviceSynchronize();

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


__declspec(dllexport) void MultiParticleSumAmplitudes(float* hp_result,
														int dimdata,
														float3* h_angles,
														cudaTex t_volumeRe,
														cudaTex t_volumeIm,
														int dimprojector,
														int nparticles)
{
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, nparticles * sizeof(float3));
	float* d_result;
	cudaMalloc((void**)& d_result, nparticles * sizeof(float));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	
	{
		MultiParticleSumAmplitudesKernel << <Grid, TpB >> > (d_result,
															dimdata,
															ElementsFFT1(dimdata),
															elementsdata,
															d_angles,
															t_volumeRe,
															t_volumeIm,
															dimprojector);
	}

	cudaMemcpy(hp_result, d_result, nparticles * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(d_result);
	cudaFree(d_angles);
}

__global__ void MultiParticleDiffKernel(float3* d_result,
										float2* hp_experimental,
										int dimdata,
										int dimdatafull,
										uint dimdataft,
										int elementsdata,
										float2* d_shifts,
										float3* d_angles,
										glm::mat2 magnification,
										float2* d_beamtilt,
										float* d_weights,
										float2* d_beamtiltcords,
										cudaTex d_reference1Re,
										cudaTex d_reference1Im,
										cudaTex d_reference2Re,
										cudaTex d_reference2Im,
										int dimprojector,
										int* d_subsets,
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
	float2 beamtilt;
	if (d_beamtilt != NULL)
		beamtilt = d_beamtilt[blockIdx.x];

	int subset = d_subsets == NULL ? 0 : d_subsets[blockIdx.x];

	cudaTex referenceRe = subset == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = subset == 0 ? d_reference1Im : d_reference2Im;

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

		glm::vec2 posmag = magnification * glm::vec2(x, y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, shift, dimdatafull);
				
		float weight = d_weights == NULL ? 1 : d_weights[id];

		val *= weight;

		float2 part = hp_experimental[id];

		if (d_beamtilt != NULL)
		{
			float tiltphase = dotp2(d_beamtiltcords[id], beamtilt);
			part = cmul(part, make_cuComplex(cos(tiltphase), sin(tiltphase)));
		}

		float cc = dotp2(part, val);

		diff2 += cc;
		ref2 += dotp2(val, val);
		part2 += dotp2(part, part);
	}
	//return;

	s_diff2[threadIdx.x] = diff2;
	s_ref2[threadIdx.x] = ref2;
	s_part2[threadIdx.x] = part2;

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
		}
		
		d_result[blockIdx.x * ntilts + itilt] = make_float3(diff2, ref2, part2);
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
											int dimdatafull,
											uint dimdataft,
											int elementsdata,
											float2* d_shifts,
											float3* d_angles,
											glm::mat2 magnification,
											cudaTex d_reference1Re,
											cudaTex d_reference1Im,
											cudaTex d_reference2Re,
											cudaTex d_reference2Im,
											int dimprojector,
											int* d_subsets,
											int ntilts,
											int itilt)
{
	hp_experimental += blockIdx.x * elementsdata;

	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);

	cudaTex referenceRe = d_subsets[blockIdx.x] == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = d_subsets[blockIdx.x] == 0 ? d_reference1Im : d_reference2Im;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
			continue;

		glm::vec2 posmag = magnification * glm::vec2(x, y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, shift, dimdatafull);

		//float2 diff = hp_experimental[id] - (val * ctf);
		//diff2 += dotp2(diff, diff) * d_weights[id];
		//hp_experimental[id] = val;

		float2 part = hp_experimental[id];
		float cc = dotp2(part, val);

		{
			float part2 = dotp2(part, part);
			float val2 = dotp2(val, val);

			atomicAdd((float*)(d_resultab + id), cc);
			atomicAdd((float*)(d_resulta2 + id), val2);
			atomicAdd((float*)(d_resultb2 + id), part2);
		}
	}
}

__global__ void MultiParticleResidualKernel(float2* d_result,
	float2* hp_experimental,
	int dimdata,
	uint dimdataft,
	int elementsdata,
	float2* d_shifts,
	float3* d_angles,
	glm::mat2 magnification,
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
	hp_experimental += blockIdx.x * elementsdata;

	float2 shift = d_shifts[blockIdx.x * ntilts + itilt];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x * ntilts + itilt]);

	cudaTex referenceRe = d_subsets[blockIdx.x] == 0 ? d_reference1Re : d_reference2Re;
	cudaTex referenceIm = d_subsets[blockIdx.x] == 0 ? d_reference1Im : d_reference2Im;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
			continue;

		glm::vec2 posmag = magnification * glm::vec2(x, y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, shift, dimdata);
		val *= d_weights[id];

		float2 part = hp_experimental[id];
		float2 res = cmul(part, cconj(val));

		{
			atomicAdd((float*)(d_result + id) + 0, res.x);
			atomicAdd((float*)(d_result + id) + 1, res.y);
		}
	}
}

__global__ void MultiParticleSumAmplitudesKernel(float* d_result,
												int dimdata,
												uint dimdataft,
												int elementsdata,
												float3* d_angles,
												cudaTex t_referenceRe,
												cudaTex t_referenceIm,
												int dimprojector)
{
	__shared__ float s_sum2[128];
	__shared__ float s_samples[128];

	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x]);

	cudaTex referenceRe = t_referenceRe;
	cudaTex referenceIm = t_referenceIm;

	float sum2 = 0;
	int samples = 0;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
			continue;

		glm::vec2 posmag = glm::vec2(x, y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, make_float2(0, 0), dimdata);
		
		sum2 += dotp2(val, val);
		samples++;
	}

	s_sum2[threadIdx.x] = sum2;
	s_samples[threadIdx.x] = (float)samples;

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			sum2 += s_sum2[i];
			samples += s_samples[i];
		}

		d_result[blockIdx.x] = sqrt(sum2 / tmax(1e-5f, samples));
	}
}