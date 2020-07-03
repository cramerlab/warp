using System;
using System.Runtime.InteropServices;
using System.Security;
using Warp.Tools;

namespace Warp
{
    [SuppressUnmanagedCodeSecurity]
    public static class CPU
    {
        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "HostMalloc")]
        public static extern IntPtr HostMalloc(long elements);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "HostFree")]
        public static extern void HostFree(IntPtr h_pointer);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "CubicInterpOnGrid")]
        public static extern void CubicInterpOnGrid(int3 dimensions, float[] values, float3 spacing, int3 valueGrid, float3 step, float3 offset, float[] output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "CubicInterpIrregular")]
        public static extern void CubicInterpIrregular(int3 dimensions, float[] values, float[] positions, int npositions, float3 spacing, float3 margin, float3 marginscale, float[] output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "InitProjector")]
        public static extern void InitProjector(int3 dims, int oversampling, float[] data, float[] initialized, int projdim);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "BackprojectorReconstruct")]
        public static extern void BackprojectorReconstruct(int3 dimsori, int oversampling, IntPtr d_data, IntPtr d_weights, string c_symmetry, bool do_reconstruct_ctf, float[] h_reconstruction);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetAnglesCount")]
        public static extern int GetAnglesCount(int healpixorder, string c_symmetry = "C1", float limittilt = -91);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetAngles")]
        public static extern void GetAngles(float[] h_angles, int healpixorder, string c_symmetry = "C1", float limittilt = -91);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "SymmetryGetNumberOfMatrices")]
        public static extern int SymmetryGetNumberOfMatrices(string c_symmetry);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "SymmetryGetMatrices")]
        public static extern void SymmetryGetMatrices(string c_symmetry, float[] h_matrices);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "OptimizeWeights")]
        public static extern void OptimizeWeights(int nrecs,
                                                  float[] h_recft,
                                                  float[] h_recweights,
                                                  float[] h_r2,
                                                  int elements,
                                                  int[] h_subsets,
                                                  float[] h_bfacs,
                                                  float[] h_weightfactors,
                                                  float[] h_recsum1,
                                                  float[] h_recsum2,
                                                  float[] h_weightsum1,
                                                  float[] h_weightsum2);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "SparseEigen")]
        public static extern void SparseEigen(int[] sparsei,
                                              int[] sparsej,
                                              double[] sparsevalues,
                                              int nsparse,
                                              int sidelength,
                                              int nvectors,
                                              double[] eigenvectors,
                                              double[] eigenvalues);

        // Bessel.cpp:
        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "KaiserBessel")]
        public static extern float KaiserBessel(float r, float a, float alpha, int m);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "KaiserBesselFT")]
        public static extern float KaiserBesselFT(float w, float a, float alpha, int m);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "KaiserBesselProj")]
        public static extern float KaiserBesselProj(float r, float a, float alpha, int m);

        // Einspline.cpp:

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "CreateEinspline3")]
        public static extern IntPtr CreateEinspline3(float[] h_values, int3 dims, float3 margins);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "CreateEinspline2")]
        public static extern IntPtr CreateEinspline2(float[] h_values, int2 dims, float2 margins);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "CreateEinspline1")]
        public static extern IntPtr CreateEinspline1(float[] h_values, int dims, float margins);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EvalEinspline3")]
        public static extern void EvalEinspline3(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EvalEinspline2XY")]
        public static extern void EvalEinspline2XY(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EvalEinspline2XZ")]
        public static extern void EvalEinspline2XZ(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EvalEinspline2YZ")]
        public static extern void EvalEinspline2YZ(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EvalEinspline1X")]
        public static extern void EvalEinspline1X(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EvalEinspline1Y")]
        public static extern void EvalEinspline1Y(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EvalEinspline1Z")]
        public static extern void EvalEinspline1Z(IntPtr spline, float[] h_pos, int npos, float[] h_output);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DestroyEinspline")]
        public static extern void DestroyEinspline(IntPtr spline);

        // FFT.cpp:

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "FFT_CPU")]
        public static extern void FFT_CPU(float[] data, float[] result, int3 dims, int nthreads);

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "IFFT_CPU")]
        public static extern void IFFT_CPU(float[] data, float[] result, int3 dims, int nthreads);

        // FSC.cpp:

        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "ConicalFSC")]
        public static extern void ConicalFSC(float[] volume1ft, float[] volume2ft, int3 dims, float[] directions, int ndirections, float anglestep, int minshell, float threshold, float particlefraction, float[] result);
    }
}
