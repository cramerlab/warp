#include "Functions.h"
#include "einspline/bspline.h"

using namespace gtom;

__declspec(dllexport) void* __stdcall CreateEinspline3(float* h_values, int3 dims, float3 margins)
{
    Ugrid gridx, gridy, gridz;
    gridx.start = margins.x; gridx.end = 1 - margins.x; gridx.num = dims.x;
    gridy.start = margins.y; gridy.end = 1 - margins.y; gridy.num = dims.y;
    gridz.start = margins.z; gridz.end = 1 - margins.z; gridz.num = dims.z;

    BCtype_s bcx, bcy, bcz;
    bcx.lCode = bcx.rCode = NATURAL;
    bcy.lCode = bcy.rCode = NATURAL;
    bcz.lCode = bcz.rCode = NATURAL;

    UBspline_3d_s* spline = create_UBspline_3d_s(gridz, gridy, gridx, bcz, bcy, bcx, h_values);

    return spline;
}

__declspec(dllexport) void* __stdcall CreateEinspline2(float* h_values, int2 dims, float2 margins)
{
    Ugrid gridx, gridy;
    gridx.start = margins.x; gridx.end = 1 - margins.x; gridx.num = dims.x;
    gridy.start = margins.y; gridy.end = 1 - margins.y; gridy.num = dims.y;

    BCtype_s bcx, bcy;
    bcx.lCode = bcx.rCode = NATURAL;
    bcy.lCode = bcy.rCode = NATURAL;

    UBspline_2d_s* spline = create_UBspline_2d_s(gridy, gridx, bcy, bcx, h_values);

    return spline;
}

__declspec(dllexport) void* __stdcall CreateEinspline1(float* h_values, int dims, float margins)
{
    Ugrid gridx, gridy;
    gridx.start = margins; gridx.end = 1 - margins; gridx.num = dims;

    BCtype_s bcx;
    bcx.lCode = bcx.rCode = NATURAL;

    UBspline_1d_s* spline = create_UBspline_1d_s(gridx, bcx, h_values);

    return spline;
}

__declspec(dllexport) void __stdcall EvalEinspline3(void* spline, float3* h_pos, int npos, float* h_output)
{
//#pragma omp parallel for
    for (int i = 0; i < npos; i++)
        eval_UBspline_3d_s((UBspline_3d_s*)spline, h_pos[i].z, h_pos[i].y, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline2XY(void* spline, float3* h_pos, int npos, float* h_output)
{
//#pragma omp parallel for
    for (int i = 0; i < npos; i++)
        eval_UBspline_2d_s((UBspline_2d_s*)spline, h_pos[i].y, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline2XZ(void* spline, float3* h_pos, int npos, float* h_output)
{
//#pragma omp parallel for
    for (int i = 0; i < npos; i++)
        eval_UBspline_2d_s((UBspline_2d_s*)spline, h_pos[i].z, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline2YZ(void* spline, float3* h_pos, int npos, float* h_output)
{
//#pragma omp parallel for
    for (int i = 0; i < npos; i++)
        eval_UBspline_2d_s((UBspline_2d_s*)spline, h_pos[i].z, h_pos[i].y, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline1X(void* spline, float3* h_pos, int npos, float* h_output)
{
//#pragma omp parallel for
    for (int i = 0; i < npos; i++)
        eval_UBspline_1d_s((UBspline_1d_s*)spline, h_pos[i].x, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline1Y(void* spline, float3* h_pos, int npos, float* h_output)
{
//#pragma omp parallel for
    for (int i = 0; i < npos; i++)
        eval_UBspline_1d_s((UBspline_1d_s*)spline, h_pos[i].y, h_output + i);
}

__declspec(dllexport) void __stdcall EvalEinspline1Z(void* spline, float3* h_pos, int npos, float* h_output)
{
//#pragma omp parallel for
    for (int i = 0; i < npos; i++)
        eval_UBspline_1d_s((UBspline_1d_s*)spline, h_pos[i].z, h_output + i);
}

__declspec(dllexport) void __stdcall DestroyEinspline(void* spline)
{
    destroy_Bspline(spline);
}