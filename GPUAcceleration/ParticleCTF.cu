#include "Functions.h"
using namespace gtom;

__global__ void SpectrumCompareKernel(float2* d_ps, float2* d_pscoords, float2* d_ref, float* d_invsigma, CTFParamsLean* d_params, float* d_scores, uint length);//, float* d_debugref, float* d_debugps);

/*

Supplied with a stack of frames, and extraction positions for particles, this method 
extracts particles, masks them, computes the FT, and averages the results as follows:

-3D full fitting: d_output contains all individual spectra from each frame
-2D spatial fitting: d_output contains averages for every position over all frames

*/

__declspec(dllexport) void CreateParticleSpectra(float* d_frame, 
												int2 dimsframe, 
												int nframes, 
												int3* h_origins, 
												int norigins, 
												float* d_masks,
												int2 dimsregion, 
												bool ctftime,
												int framegroupsize, 
												float majorpixel, 
												float minorpixel, 
												float majorangle,
												float2* d_outputall)
{
	int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, norigins * nframes * sizeof(int3));
	tcomplex* d_tempspectra;
	cudaMalloc((void**)&d_tempspectra, norigins * ElementsFFT2(dimsregion) * sizeof(tcomplex));
	tfloat* d_tempextracts;
	cudaMalloc((void**)&d_tempextracts, norigins * Elements2(dimsregion) * sizeof(tfloat));

	tfloat* d_tempsum = CudaMallocValueFilled(Elements2(dimsregion) * norigins, 0.0f);

	int nspectra = norigins * (ctftime ? nframes : 1);

	// Temp spectra will be summed up to be averaged later in case of only spatial resolution
	if (!ctftime)
	{
		d_ValueFill(d_outputall, ElementsFFT2(dimsregion) * norigins, make_cuComplex(0.0f, 0.0f));
	}
	else
	{
		d_ValueFill(d_outputall, ElementsFFT2(dimsregion) * norigins * (nframes / framegroupsize), make_cuComplex(0.0f, 0.0f));
	}

	for (int z = 0; z < nframes; z++)
	{
		d_ExtractMany(d_frame + Elements2(dimsframe) * z, d_tempextracts, toInt3(dimsframe), toInt3(dimsregion), d_origins + norigins * z, norigins);
		if (abs(majorpixel - minorpixel) > 0)
		{
			d_MultiplyByScalar(d_tempextracts, d_tempextracts, Elements2(dimsregion) * norigins, -1.0f);
			d_MagAnisotropyCorrect(d_tempextracts, dimsregion, (float*)d_tempspectra, dimsregion, majorpixel, minorpixel, majorangle, 4, norigins);
		}
		else
		{
			d_MultiplyByScalar(d_tempextracts, (float*)d_tempspectra, Elements2(dimsregion) * norigins, -1.0f);
		}

		d_AddVector(d_tempsum, (float*)d_tempspectra, d_tempsum, norigins * Elements2(dimsregion));
		d_RemapFull2FullFFT((float*)d_tempspectra, d_tempextracts, toInt3(dimsregion), norigins);
		//d_MultiplyByVector(d_tempextracts, d_masks, d_tempextracts, norigins * Elements2(dimsregion));

		//d_WriteMRC(d_tempextracts, toInt3(dimsregion.x, dimsregion.y, norigins), "d_tempextracts.mrc");

		// Full precision, just write everything to output which is big enough
		if (ctftime)
		{
			d_FFTR2C(d_tempextracts, d_tempspectra, 2, toInt3(dimsregion), norigins);		
			d_AddVector((float*)d_tempspectra, 
						(float*)(d_outputall + (z / framegroupsize) * norigins * ElementsFFT2(dimsregion)), 
						(float*)(d_outputall + (z / framegroupsize) * norigins * ElementsFFT2(dimsregion)), 
						norigins * ElementsFFT2(dimsregion) * 2);	
		}
		else // Spatial precision
		{
			d_FFTR2C(d_tempextracts, d_tempspectra, 2, toInt3(dimsregion), norigins);
			d_AddVector((float*)d_tempspectra, (float*)d_outputall, (float*)d_outputall, norigins * ElementsFFT2(dimsregion) * 2);
		}
	}

	d_WriteMRC(d_tempsum, toInt3(dimsregion.x, dimsregion.y, norigins), "d_tempsum.mrc");

	if (!ctftime)
		d_ComplexMultiplyByScalar(d_outputall, d_outputall, norigins * ElementsFFT2(dimsregion), 1.0f / nframes);
	else
		d_ComplexMultiplyByScalar(d_outputall, d_outputall, norigins * (nframes / framegroupsize) * ElementsFFT2(dimsregion), 1.0f / framegroupsize);

	cudaFree(d_tempsum);
	cudaFree(d_origins);
	cudaFree(d_tempspectra);
	cudaFree(d_tempextracts);
}

__declspec(dllexport) void ParticleCTFMakeAverage(float2* d_ps, float2* d_pscoords, uint length, uint sidelength, CTFParams* h_sourceparams, CTFParams targetparams, uint minbin, uint maxbin, uint batch, float* d_output)
{
	uint nbins = maxbin - minbin;
	d_CTFRotationalAverageToTarget((tcomplex*)d_ps, d_pscoords, length, sidelength, h_sourceparams, targetparams, d_output, minbin, maxbin, 1);
}

__declspec(dllexport) void ParticleCTFCompareToSim(float2* d_ps, float2* d_pscoords, float2* d_ref, float* d_invsigma, uint length, CTFParams* h_sourceparams, float* h_scores, uint nframes, uint batch)
{
	float* d_scores;
	cudaMalloc((void**)&d_scores, batch * nframes * sizeof(float));

	CTFParamsLean* h_lean;
	cudaMallocHost((void**)&h_lean, batch * nframes * sizeof(CTFParamsLean));
	#pragma omp parallel for
	for (int i = 0; i < batch * nframes; i++)
		h_lean[i] = CTFParamsLean(h_sourceparams[i], toInt3(1, 1, 1));	// Sidelength and pixelsize are already included in d_addresses
	CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * nframes * sizeof(CTFParamsLean));
	cudaFreeHost(h_lean);

	//float* d_debugref, *d_debugps;
	//cudaMalloc((void**)&d_debugref, length * batch * sizeof(float));
	//cudaMalloc((void**)&d_debugps, length * batch * sizeof(float));

	int TpB = 128;
	dim3 grid = dim3(batch, nframes, 1);
	SpectrumCompareKernel <<<grid, TpB>>> (d_ps, d_pscoords, d_ref, d_invsigma, d_lean, d_scores, length);//, d_debugref, d_debugps);

	//d_ReduceMean(d_debugref, d_debugref, length, batch);
	//d_WriteMRC(d_debugref, toInt3(129, 256, batch), "d_ref.mrc");
	//d_ReduceMean(d_debugps, d_debugps, length, batch);
	//d_WriteMRC(d_debugps, toInt3(129, 256, 1), "d_ps.mrc");
	
	//cudaFree(d_debugref);
	//cudaFree(d_debugps);

	cudaMemcpy(h_scores, d_scores, batch * nframes * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_lean);
	cudaFree(d_scores);
}

__global__ void SpectrumCompareKernel(float2* d_ps, float2* d_pscoords, float2* d_ref, float* d_invsigma, CTFParamsLean* d_params, float* d_scores, uint length)//, float* d_debugref, float* d_debugps)
{
	__shared__ float s_num[128];
	__shared__ float s_denom1[128];
	__shared__ float s_denom2[128];

	d_ps += (blockIdx.y * gridDim.x + blockIdx.x) * length;
	d_ref += blockIdx.x * length;
	//d_debugref += blockIdx.x * length;
	//d_debugps += blockIdx.x * length;

	CTFParamsLean params = d_params[blockIdx.y * gridDim.x + blockIdx.x];

	float num = 0.0, denom1 = 0.0, denom2 = 0.0;
	for (uint i = threadIdx.x; i < length; i += blockDim.x)
	{
		float2 simcoords = d_pscoords[i];

		float2 refval = d_ref[i] * d_GetCTF<false, false>(simcoords.x / params.pixelsize, simcoords.y, 0, params);
		float2 psval = d_ps[i];
		float invsigma = d_invsigma[i];
		refval *= invsigma;
		psval *= invsigma;
		
		num += dotp2(refval, psval);
		denom1 += dotp2(refval, refval);
		denom2 += dotp2(psval, psval);
		
		//d_debugref[i] = sqrt(refval.x * refval.x + refval.y * refval.y);
		//d_debugps[i] = sqrt(psval.x * psval.x + psval.y * psval.y);
	}
	s_num[threadIdx.x] = num;
	s_denom1[threadIdx.x] = denom1;
	s_denom2[threadIdx.x] = denom2;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < 128; i++)
		{
			num += s_num[i];
			denom1 += s_denom1[i];
			denom2 += s_denom2[i];
		}

		d_scores[blockIdx.y * gridDim.x + blockIdx.x] = num / tmax(1e-6f, sqrt(denom1 * denom2));
	}
}