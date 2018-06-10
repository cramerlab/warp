#include "Functions.h"
using namespace gtom;


__declspec(dllexport) void CompareParticles(float* d_particles, 
											float* d_masks, 
											float* d_projections, 
											int2 dims, 
											float2* d_ctfcoords, 
											CTFParams* h_ctfparams, 
											float highpass, 
											float lowpass, 
											float* d_scores, 
											uint nparticles)
{
	float* d_ctf;
	cudaMalloc((void**)&d_ctf, ElementsFFT2(dims) * nparticles * sizeof(float));
	d_CTFSimulate(h_ctfparams, d_ctfcoords, d_ctf, ElementsFFT2(dims), false, nparticles);
	//d_WriteMRC(d_ctf, toInt3(dims.x / 2 + 1, dims.y, nparticles), "d_ctf.mrc");

	d_Bandpass(d_particles, d_particles, toInt3(dims), highpass, lowpass, 1.0f, NULL, NULL, NULL, nparticles);
	d_NormMonolithic(d_particles, d_particles, Elements2(dims), d_masks, T_NORM_MEAN01STD, nparticles);
	d_MultiplyByVector(d_particles, d_masks, d_particles, Elements2(dims) * nparticles);
	d_NormMonolithic(d_particles, d_particles, Elements2(dims), T_NORM_MEAN01STD, nparticles);
	//d_WriteMRC(d_particles, toInt3(dims.x, dims.y, nparticles), "d_particles.mrc");
	
	float2* d_projectionsft;
	cudaMalloc((void**)&d_projectionsft, ElementsFFT2(dims) * nparticles * sizeof(float2));
	d_FFTR2C(d_projections, d_projectionsft, 2, toInt3(dims), nparticles);
	d_ComplexMultiplyByVector(d_projectionsft, d_ctf, d_projectionsft, ElementsFFT2(dims) * nparticles);
	d_IFFTC2R(d_projectionsft, d_projections, 2, toInt3(dims), nparticles);

	d_RemapFullFFT2Full(d_projections, d_projections, toInt3(dims), nparticles);
	d_Bandpass(d_projections, d_projections, toInt3(dims), highpass, lowpass, 1.0f, NULL, NULL, NULL, nparticles);
	d_NormMonolithic(d_projections, d_projections, Elements2(dims), d_masks, T_NORM_MEAN01STD, nparticles);
	d_MultiplyByVector(d_projections, d_masks, d_projections, Elements2(dims) * nparticles);
	d_NormMonolithic(d_projections, d_projections, Elements2(dims), T_NORM_MEAN01STD, nparticles);
	//d_WriteMRC(d_projections, toInt3(dims.x, dims.y, nparticles), "d_projections.mrc");

	d_MultiplyByVector(d_particles, d_projections, d_projections, Elements2(dims) * nparticles);
	d_SumMonolithic(d_projections, d_scores, Elements2(dims), nparticles);
	d_MultiplyByScalar(d_scores, d_scores, nparticles, 1.0f / Elements2(dims));

	cudaFree(d_projectionsft);
	cudaFree(d_ctf);
}