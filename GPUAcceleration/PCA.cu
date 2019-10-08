#include "Functions.h"
#include "../../gtom/include/DeviceFunctions.cuh"
using namespace gtom;

#define TOMO_THREADS 128

__global__ void PCAKernel(float* d_result,
						float2* d_experimental,
						float2* d_experimentalunmasked,
						float* d_ctf,
						float* d_spectral,
						int dimdata,
						uint dimdataft,
						int elementsdata,
						float2* d_shifts,
						float3* d_angles,
						glm::mat2 magnification,
						cudaTex t_reference1Re,
						cudaTex t_reference1Im,
						int dimprojector,
						bool performsubtraction);


__declspec(dllexport) void PCA(float* h_result,
								float2* d_experimental,
								float2* d_experimentalunmasked,
								float* d_ctf,
								float* d_spectral,
								int dimdata,
								float2* h_shifts,
								float3* h_angles,
								float3 magnification,
								cudaTex t_volumeRe,
								cudaTex t_volumeIm,
								int dimprojector,
								int nparticles,
								bool performsubtraction)
{
	float2* d_shifts = (float2*)CudaMallocFromHostArray(h_shifts, nparticles * sizeof(float2));
	float3* d_angles = (float3*)CudaMallocFromHostArray(h_angles, nparticles * sizeof(float3));
	float* d_result;
	cudaMalloc((void**)&d_result, nparticles * sizeof(float));

	int elementsdata = ElementsFFT1(dimdata) * dimdata;
	dim3 TpB = dim3(128, 1, 1);
	dim3 Grid = dim3(nparticles, 1, 1);

	glm::mat2 m_magnification = Matrix2Rotation(-magnification.z) * Matrix2Scale(tfloat2(magnification.x, magnification.y)) * Matrix2Rotation(magnification.z);

	PCAKernel << <Grid, TpB >> > (d_result,
									d_experimental,
									d_experimentalunmasked,
									d_ctf,
									d_spectral,
									dimdata,
									ElementsFFT1(dimdata),
									elementsdata,
									d_shifts,
									d_angles,
									m_magnification,
									t_volumeRe,
									t_volumeIm,
									dimprojector,
									performsubtraction);

	cudaMemcpy(h_result, d_result, nparticles * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(d_result);
	cudaFree(d_angles);
	cudaFree(d_shifts);
}

__global__ void PCAKernel(float* d_result,
							float2* d_experimental,
							float2* d_experimentalunmasked,
							float* d_ctf,
							float* d_spectral,
							int dimdata,
							uint dimdataft,
							int elementsdata,
							float2* d_shifts,
							float3* d_angles,
							glm::mat2 magnification,
							cudaTex t_reference1Re,
							cudaTex t_reference1Im,
							int dimprojector,
							bool performsubtraction)
{
	__shared__ float s_cc2[128];
	__shared__ float s_ref2[128];
	__shared__ float s_part2[128];
	//__shared__ float s_weight[128];

	d_experimental += blockIdx.x * elementsdata;
	d_experimentalunmasked += blockIdx.x * elementsdata;
	d_ctf += blockIdx.x * elementsdata;
	d_spectral += blockIdx.x * elementsdata;
	float2 shift = d_shifts[blockIdx.x];
	glm::mat3 angles = d_Matrix3Euler(d_angles[blockIdx.x]);

	cudaTex referenceRe = t_reference1Re;
	cudaTex referenceIm = t_reference1Im;

	float cc2 = 0, ref2 = 0, part2 = 0;

	for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
	{
		uint idx = id % dimdataft;
		uint idy = id / dimdataft;
		int x = idx;
		int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

		if (x * x + y * y >= dimdata * dimdata / 4)
		{
			//d_experimental[id] = make_float2(0, 0);
			continue;
		}

		glm::vec2 posmag = magnification * glm::vec2(x, y);

		float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, shift, dimdata);
		
		val *= d_ctf[id];
		val *= d_spectral[id];

		float2 part = d_experimental[id];
		//part *= d_spectral[id];

		float cc = dotp2(part, val);
		//if (cc != 0)
		{
			cc2 += cc;
			ref2 += dotp2(val, val);
			part2 += dotp2(part, part);
		}
	}
	//return;

	s_cc2[threadIdx.x] = cc2;
	s_ref2[threadIdx.x] = ref2;
	s_part2[threadIdx.x] = part2;

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			cc2 += s_cc2[i];
			ref2 += s_ref2[i];
			part2 += s_part2[i];
			//weight += s_weight[i];
		}

		//cc2 /= tmax(1e-16f, sqrt(ref2 * part2));
		cc2 /= tmax(1e-16f, sqrt(ref2));

		d_result[blockIdx.x] = cc2;
		s_ref2[0] = ref2;
	}

	if (performsubtraction)
	{
		__syncthreads();

		cc2 = d_result[blockIdx.x];
		ref2 = 1.0f / sqrt(s_ref2[0]);

		for (uint id = threadIdx.x; id < elementsdata; id += blockDim.x)
		{
			uint idx = id % dimdataft;
			uint idy = id / dimdataft;
			int x = idx;
			int y = (idy <= dimdata / 2 ? idy : ((int)idy - (int)dimdata));

			if (x * x + y * y >= dimdata * dimdata / 4)
			{
				//d_experimental[id] = make_float2(0, 0);
				continue;
			}

			glm::vec2 posmag = magnification * glm::vec2(x, y);

			float2 val = d_GetProjectionSlice(referenceRe, referenceIm, dimprojector, glm::vec3(posmag.x, posmag.y, 0), angles, shift, dimdata);

			val *= d_ctf[id];
			val *= d_spectral[id];
			val *= ref2;
			val *= cc2;
			
			//d_experimental[id] = val;
			//d_experimental[id] -= val;
			d_experimentalunmasked[id] -= val;
		}
	}
}