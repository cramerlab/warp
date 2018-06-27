#include "Functions.h"

__declspec(dllexport) float* __stdcall MallocDevice(long elements)
{
	float* d_memory;
	cudaMalloc((void**)&d_memory, elements * sizeof(float));

	return d_memory;
}

__declspec(dllexport) float* __stdcall MallocDeviceFromHost(float* h_data, long elements)
{
	float* d_memory = (float*)gtom::CudaMallocFromHostArray(h_data, elements * sizeof(float));
	return d_memory;
}

__declspec(dllexport) int* __stdcall MallocDeviceFromHostInt(int* h_data, long elements)
{
    int* d_memory = (int*)gtom::CudaMallocFromHostArray(h_data, elements * sizeof(int));
    return d_memory;
}

__declspec(dllexport) void* __stdcall MallocDeviceHalf(long elements)
{
	half* d_memory;
	cudaMalloc((void**)&d_memory, elements * sizeof(half));

	return d_memory;
}

__declspec(dllexport) void* __stdcall MallocDeviceHalfFromHost(float* h_data, long elements)
{
	half* d_memory;
	cudaMalloc((void**)&d_memory, elements * sizeof(half));

	CopyHostToDeviceHalf(h_data, d_memory, elements);

	return d_memory;
}

__declspec(dllexport) void __stdcall FreeDevice(void* d_data)
{
	cudaFree(d_data);
}

__declspec(dllexport) void __stdcall CopyDeviceToHost(float* d_source, float* h_dest, long elements)
{
	cudaMemcpy(h_dest, d_source, elements * sizeof(float), cudaMemcpyDeviceToHost);
}

__declspec(dllexport) void __stdcall CopyDeviceToHostPinned(float* d_source, float* hp_dest, long elements)
{
    cudaMemcpy(hp_dest, d_source, elements * sizeof(float), cudaMemcpyDeviceToHost);
}

__declspec(dllexport) void __stdcall CopyDeviceHalfToHost(half* d_source, float* h_dest, long elements)
{
	float* d_source32;
	cudaMallocHost((void**)&d_source32, elements * sizeof(float));

	gtom::d_ConvertToTFloat(d_source, d_source32, elements);
	cudaMemcpy(h_dest, d_source32, elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_source32);
}

__declspec(dllexport) void __stdcall CopyDeviceToDevice(float* d_source, float* d_dest, long elements)
{
	cudaMemcpy(d_dest, d_source, elements * sizeof(float), cudaMemcpyDeviceToDevice);
}

__declspec(dllexport) void __stdcall CopyDeviceHalfToDeviceHalf(half* d_source, half* d_dest, long elements)
{
	cudaMemcpy(d_dest, d_source, elements * sizeof(half), cudaMemcpyDeviceToDevice);
}

__declspec(dllexport) void __stdcall CopyHostToDevice(float* h_source, float* d_dest, long elements)
{
	cudaMemcpy(d_dest, h_source, elements * sizeof(float), cudaMemcpyHostToDevice);
}

__declspec(dllexport) void __stdcall CopyHostToHost(float* h_source, float* h_dest, long elements)
{
    cudaMemcpy(h_dest, h_source, elements * sizeof(float), cudaMemcpyHostToHost);
}

__declspec(dllexport) void __stdcall CopyHostPinnedToDevice(float* hp_source, float* d_dest, long elements)
{
    cudaMemcpy(d_dest, hp_source, elements * sizeof(float), cudaMemcpyHostToDevice);
}

__declspec(dllexport) void __stdcall CopyHostToDeviceHalf(float* h_source, half* d_dest, long elements)
{
	float* d_source = (float*)gtom::CudaMallocFromHostArray(h_source, elements * sizeof(float));

	gtom::d_ConvertTFloatTo(d_source, d_dest, elements);

	cudaFree(d_source);
}

__declspec(dllexport) void __stdcall SingleToHalf(float* d_source, half* d_dest, long elements)
{
	gtom::d_ConvertTFloatTo(d_source, d_dest, elements);
}

__declspec(dllexport) void __stdcall HalfToSingle(half* d_source, float* d_dest, long elements)
{
	gtom::d_ConvertToTFloat(d_source, d_dest, elements);
}

__declspec(dllexport) float* __stdcall MallocHostPinned(long elements)
{
    float* hp_memory;
    cudaMallocHost((void**)&hp_memory, elements * sizeof(float));

    return hp_memory;
}

__declspec(dllexport) void __stdcall FreeHostPinned(void* hp_data)
{
    cudaFreeHost(hp_data);
}

__declspec(dllexport) void __stdcall HostFree(void* h_pointer)
{
    free(h_pointer);
}

__declspec(dllexport) void __stdcall DeviceReset()
{
	cudaDeviceReset();
}