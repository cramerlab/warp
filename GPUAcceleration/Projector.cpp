#include "Functions.h"
#include "liblion.h"
using namespace gtom;

__declspec(dllexport) void __stdcall InitProjector(int3 dims, int oversampling, float* h_data, float* h_initialized, int projdim)
{
    relion::MultidimArray<float> dummy;
    relion::MultidimArray<float> vol;
    vol.initZeros(dims.z, dims.y, dims.x);

    for (uint i = 0; i < Elements(dims); i++)
        vol.data[i] = h_data[i];

    relion::Projector projector(dims.x, TRILINEAR, oversampling, 10, projdim);
    projector.computeFourierTransformMap(vol, dummy, -1, 1, true, false, false);

    int3 projectordims = toInt3(XSIZE(projector.data), YSIZE(projector.data), ZSIZE(projector.data));
    for (uint i = 0; i < Elements(projectordims); i++)
        ((float2*)h_initialized)[i] = make_float2(projector.data.data[i].real, projector.data.data[i].imag);
}

__declspec(dllexport) void __stdcall BackprojectorReconstruct(int3 dimsori, int oversampling, float* h_data, float* h_weights, char* c_symmetry, bool do_reconstruct_ctf, float* h_reconstruction)
{
    relion::FileName fn_symmetry(c_symmetry);

    relion::FourierTransformer transformer;
    transformer.setThreadsNumber(16);

    relion::BackProjector backprojector(dimsori.x, 3, fn_symmetry, TRILINEAR, oversampling, 10, 0, 1.9, 15, 2);
    backprojector.initZeros(dimsori.x);

    int3 projectordims = toInt3(XSIZE(backprojector.data), YSIZE(backprojector.data), ZSIZE(backprojector.data));
    memcpy(backprojector.data.data, h_data, Elements(projectordims) * sizeof(float2));
    memcpy(backprojector.weight.data, h_weights, Elements(projectordims) * sizeof(float));

    relion::MultidimArray<float> vol, dummy;
    relion::MultidimArray<relion::Complex > F2D;
    relion::MultidimArray<float> fsc;
    fsc.resize(dimsori.x / 2 + 1);

    backprojector.reconstruct(vol, 10, false, 1., dummy, dummy, dummy, fsc, 1., false, true, 1, -1);

    if (do_reconstruct_ctf)
    {
        F2D.clear();
        transformer.FourierTransform(vol, F2D);
        
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F2D)
        {
            h_reconstruction[n] = sqrt(F2D.data[n].real * F2D.data[n].real + F2D.data[n].imag * F2D.data[n].imag) * (DOUBLE)dimsori.x;
        }
    }
    else
    {
        memcpy(h_reconstruction, vol.data, Elements(dimsori) * sizeof(float));
    }
}

__declspec(dllexport) void __stdcall BackprojectorReconstructGPU(int3 dimsori, int3 dimspadded, int oversampling, float2* d_dataft, float* d_weights, bool do_reconstruct_ctf, float* d_result, cufftHandle pre_planforw, cufftHandle pre_planback, cufftHandle pre_planforwctf)
{
    float* d_reconstructed;
    cudaMalloc((void**)&d_reconstructed, ElementsFFT(dimsori) * sizeof(float2));

    d_ReconstructGridding(d_dataft, d_weights, d_reconstructed, dimsori, dimspadded, oversampling, pre_planforw, pre_planback);

    if (do_reconstruct_ctf)
    {
		float2* d_reconstructedft;
		cudaMalloc((void**)&d_reconstructedft, ElementsFFT(dimsori) * sizeof(float2));

        if (pre_planforwctf > NULL)
            d_FFTR2C(d_reconstructed, d_reconstructedft, &pre_planforwctf);
        else
            d_FFTR2C(d_reconstructed, d_reconstructedft, 3, dimsori);
        d_Abs(d_reconstructedft, d_result, ElementsFFT(dimsori));

		cudaFree(d_reconstructedft);
    }
    else
    {
        cudaMemcpy(d_result, d_reconstructed, Elements(dimsori) * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_reconstructed);
}