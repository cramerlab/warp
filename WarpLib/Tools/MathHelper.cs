using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Accord;

namespace Warp.Tools
{
    public static class MathHelper
    {
        public static float Sinc(float v)
        {
            if (Math.Abs(v) < 1e-8f)
                return 1;

            return (float)(Math.Sin(v * Math.PI) / (v * Math.PI));
        }
        public static float Sinc2(float v)
        {
            if (Math.Abs(v) < 1e-8f)
                return 1;

            float Result = (float)(Math.Sin(v * Math.PI) / (v * Math.PI));
            return Result * Result;
        }

        public static float Mean(IEnumerable<float> data)
        {
            double Sum = data.Sum(i => i);
            return (float)Sum / data.Count();
        }

        public static float2 Mean(IEnumerable<float2> data)
        {
            float2 Sum = new float2(0, 0);
            foreach (var p in data)
                Sum += p;

            return Sum / data.Count();
        }

        public static float3 Mean(IEnumerable<float3> data)
        {
            float3 Sum = new float3(0, 0, 0);
            foreach (var p in data)
                Sum += p;

            return Sum / data.Count();
        }

        public static float4 Mean(IEnumerable<float4> data)
        {
            float4 Sum = new float4(0, 0, 0, 0);
            foreach (var p in data)
                Sum += p;

            return Sum / data.Count();
        }

        public static float StdDev(IEnumerable<float> data)
        {
            double Sum = 0f, Sum2 = 0f;
            foreach (var i in data)
            {
                Sum += i;
                Sum2 += i * i;
            }

            return (float)Math.Sqrt(data.Count() * Sum2 - Sum * Sum) / data.Count();
        }

        public static float2 MeanAndStd(IEnumerable<float> data)
        {
            double Sum = 0f, Sum2 = 0f;
            foreach (var i in data)
            {
                Sum += i;
                Sum2 += i * i;
            }

            return new float2((float)Sum / data.Count(), (float)Math.Sqrt(data.Count() * Sum2 - Sum * Sum) / data.Count());
        }

        public static float2 MeanAndStdNonZero(IEnumerable<float> data)
        {
            double Sum = 0f, Sum2 = 0f;
            long Samples = 0;

            foreach (var i in data)
            {
                if (i == 0)
                    continue;

                Sum += i;
                Sum2 += i * i;
                Samples++;
            }

            return new float2((float)Sum / Samples, (float)Math.Sqrt(Samples * Sum2 - Sum * Sum) / Samples);
        }

        public static float[] Normalize(float[] data)
        {
            double Sum = 0f, Sum2 = 0f;
            foreach (var i in data)
            {
                Sum += i;
                Sum2 += i * i;
            }

            float Std = (float)Math.Sqrt(data.Length * Sum2 - Sum * Sum) / data.Count();
            float Avg = (float) Sum / data.Length;

            float[] Result = new float[data.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = (data[i] - Avg) / Std;

            return Result;
        }

        public static void NormalizeInPlace(float[] data)
        {
            double Sum = 0f, Sum2 = 0f;
            foreach (var i in data)
            {
                Sum += i;
                Sum2 += i * i;
            }

            float Std = (float)Math.Sqrt(data.Length * Sum2 - Sum * Sum) / data.Length;
            float Avg = (float)Sum / data.Length;
            
            for (int i = 0; i < data.Length; i++)
                data[i] = (data[i] - Avg) / Std;
        }

        public static void NormalizeL2InPlace(float[] data)
        {
            double Sum = 0;
            for (int i = 0; i < data.Length; i++)
                Sum += data[i] * data[i];
            Sum = Math.Sqrt(Sum);

            for (int i = 0; i < data.Length; i++)
                data[i] /= (float)Sum;
        }

        public static float CrossCorrelate(float[] data1, float[] data2)
        {
            return Mult(data1, data2).Sum() / data1.Length;
        }

        public static float CrossCorrelateNormalized(float[] data1, float[] data2)
        {
            return CrossCorrelate(Normalize(data1), Normalize(data2));
        }

        public static float Min(IEnumerable<float> data)
        {
            float Min = float.MaxValue;
            return data.Aggregate(Min, (start, i) => Math.Min(start, i));
        }

        public static float[] Min(float[] data, float val)
        {
            float[] Result = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                Result[i] = Math.Min(data[i], val);

            return Result;
        }

        public static float[] Min(float[] data1, float[] data2)
        {
            if (data1.Length != data2.Length)
                throw new DimensionMismatchException();

            float[] Result = new float[data1.Length];
            for (int i = 0; i < data1.Length; i++)
                Result[i] = Math.Min(data1[i], data2[i]);

            return Result;
        }

        public static int Min(IEnumerable<int> data)
        {
            int Min = int.MaxValue;
            return data.Aggregate(Min, (start, i) => Math.Min(start, i));
        }

        public static float Max(IEnumerable<float> data)
        {
            float Max = -float.MaxValue;
            return data.Aggregate(Max, (start, i) => Math.Max(start, i));
        }

        public static float2 Max(IEnumerable<float2> data)
        {
            float2 Max = new float2(-float.MaxValue);
            foreach (var val in data)
            {
                Max.X = Math.Max(Max.X, val.X);
                Max.Y = Math.Max(Max.Y, val.Y);
            }

            return Max;
        }

        public static float3 Max(IEnumerable<float3> data)
        {
            float3 Max = new float3(-float.MaxValue);
            foreach (var val in data)
            {
                Max.X = Math.Max(Max.X, val.X);
                Max.Y = Math.Max(Max.Y, val.Y);
                Max.Z = Math.Max(Max.Z, val.Z);
            }

            return Max;
        }

        public static float[] Max(float[] data, float val)
        {
            float[] Result = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                Result[i] = Math.Max(data[i], val);

            return Result;
        }

        public static float[] Max(float[] data1, float[] data2)
        {
            if (data1.Length != data2.Length)
                throw new DimensionMismatchException();

            float[] Result = new float[data1.Length];
            for (int i = 0; i < data1.Length; i++)
                Result[i] = Math.Max(data1[i], data2[i]);

            return Result;
        }

        public static double[] Max(double[] data1, double[] data2)
        {
            if (data1.Length != data2.Length)
                throw new DimensionMismatchException();

            double[] Result = new double[data1.Length];
            for (int i = 0; i < data1.Length; i++)
                Result[i] = Math.Max(data1[i], data2[i]);

            return Result;
        }

        public static int Max(IEnumerable<int> data)
        {
            int Max = -int.MaxValue;
            return data.Aggregate(Max, (start, i) => Math.Max(start, i));
        }

        public static float[] Plus(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float[] Minus(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float[] Mult(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] * data2[i];

            return Result;
        }

        public static float[] Div(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] / data2[i];

            return Result;
        }

        public static float[] Subtract(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float2[] Subtract(float2[] data1, float2[] data2)
        {
            float2[] Result = new float2[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float3[] Subtract(float3[] data1, float3[] data2)
        {
            float3[] Result = new float3[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] - data2[i];

            return Result;
        }

        public static float[] Add(float[] data1, float[] data2)
        {
            float[] Result = new float[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float2[] Add(float2[] data1, float2[] data2)
        {
            float2[] Result = new float2[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float3[] Add(float3[] data1, float3[] data2)
        {
            float3[] Result = new float3[data1.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = data1[i] + data2[i];

            return Result;
        }

        public static float[] Diff(float[] data)
        {
            float[] D = new float[data.Length - 1];
            for (int i = 0; i < data.Length - 1; i++)
                D[i] = data[i + 1] - data[i];

            return D;
        }

        public static double[] Diff(double[] data)
        {
            double[] D = new double[data.Length - 1];
            for (int i = 0; i < data.Length - 1; i++)
                D[i] = data[i + 1] - data[i];

            return D;
        }

        public static float2[] Diff(float2[] data)
        {
            float2[] D = new float2[data.Length - 1];
            for (int i = 0; i < data.Length - 1; i++)
                D[i] = data[i + 1] - data[i];

            return D;
        }

        public static float DotProduct(float[] data1, float[] data2)
        {
            double Sum = 0;
            for (int i = 0; i < data1.Length; i++)
                Sum += data1[i] * data2[i];

            return (float)Sum;
        }

        public static int NextMultipleOf(int value, int factor)
        {
            return ((value + factor - 1) / factor) * factor;
        }

        public static float ReduceWeighted(float[] data, float[] weights)
        {
            float Sum = 0f;
            float Weightsum = 0f;
            unsafe
            {
                fixed (float* dataPtr = data)
                fixed (float* weightsPtr = weights)
                {
                    float* dataP = dataPtr;
                    float* weightsP = weightsPtr;

                    for (int i = 0; i < data.Length; i++)
                    {
                        Sum += *dataP++ * *weightsP;
                        Weightsum += *weightsP++;
                    }
                }
            }

            return Sum;
        }

        public static float MeanWeighted(float[] data, float[] weights)
        {
            float Sum = 0f;
            float Weightsum = 0f;
            unsafe
            {
                fixed (float* dataPtr = data)
                fixed (float* weightsPtr = weights)
                {
                    float* dataP = dataPtr;
                    float* weightsP = weightsPtr;

                    for (int i = 0; i < data.Length; i++)
                    {
                        Sum += *dataP++ * *weightsP;
                        Weightsum += *weightsP++;
                    }
                }
            }

            if (Math.Abs(Weightsum) > 1e-6f)
                return Sum / Weightsum;
            else
                return 0;
        }

        public static float3 MeanWeighted(float3[] data, float[] weights)
        {
            float3 Sum = new float3();
            float Weightsum = 0f;
            unsafe
            {
                fixed (float3* dataPtr = data)
                fixed (float* weightsPtr = weights)
                {
                    float3* dataP = dataPtr;
                    float* weightsP = weightsPtr;

                    for (int i = 0; i < data.Length; i++)
                    {
                        Sum += *dataP++ * *weightsP;
                        Weightsum += *weightsP++;
                    }
                }
            }

            if (Math.Abs(Weightsum) > 1e-6f)
                return Sum / Weightsum;
            else
                return new float3();
        }

        public static float[] MeanWeighted(float[][] data, float[] weights)
        {
            float[] Mean = new float[data[0].Length];
            float WeightSum = weights.Sum();

            for (int i = 0; i < data.Length; i++)
                for (int j = 0; j < Mean.Length; j++)
                    Mean[j] += data[i][j];

            for (int i = 0; i < Mean.Length; i++)
                Mean[i] /= WeightSum;

            return Mean;
        }

        public static void UnNaN(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                if (float.IsNaN(data[i]))
                    data[i] = 0;
        }

        public static void UnNaN(float2[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (float.IsNaN(data[i].X))
                    data[i].X = 0;
                if (float.IsNaN(data[i].Y))
                    data[i].Y = 0;
            }
        }

        public static float ResidualFraction(float value)
        {
            return value - (int)value;
        }

        public static float Median(IEnumerable<float> data)
        {
            List<float> Sorted = new List<float>(data);
            Sorted.Sort();

            return Sorted[Sorted.Count / 2];
        }

        public static float[] WithinNStd(float[] data, float nstd)
        {
            float Mean = MathHelper.Mean(data);
            float Std = StdDev(data) * nstd;

            List<float> Result = data.Where(t => Math.Abs(t - Mean) <= Std).ToList();

            return Result.ToArray();
        }

        public static float[] WithinNStdFromMedian(float[] data, float nstd)
        {
            float Mean = Median(data);
            float Std = StdDev(data) * nstd;

            List<float> Result = data.Where(t => Math.Abs(t - Mean) <= Std).ToList();

            return Result.ToArray();
        }

        public static int[] WithinNStdFromMedianIndices(float[] data, float nstd)
        {
            float Mean = Median(data);
            float Std = StdDev(data) * nstd;

            List<int> Result = new List<int>();

            for (int i = 0; i < data.Length; i++)
                if (Math.Abs(data[i] - Mean) <= Std)
                    Result.Add(i);

            return Result.ToArray();
        }

        public static float[] TakeNLowest(float[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            float[] Result = new float[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                float LowestVal = float.MaxValue;
                int LowestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] < LowestVal && !ResultIndices.Contains(i))
                    {
                        LowestVal = data[i];
                        LowestIndex = i;
                    }

                Result[n] = LowestVal;
                ResultIndices.Add(LowestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static double[] TakeNLowest(double[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            double[] Result = new double[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                double LowestVal = double.MaxValue;
                int LowestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] < LowestVal && !ResultIndices.Contains(i))
                    {
                        LowestVal = data[i];
                        LowestIndex = i;
                    }

                Result[n] = LowestVal;
                ResultIndices.Add(LowestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static float[] TakeNHighest(float[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            float[] Result = new float[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                float HighestVal = -float.MaxValue;
                int HighestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] > HighestVal && !ResultIndices.Contains(i))
                    {
                        HighestVal = data[i];
                        HighestIndex = i;
                    }

                Result[n] = HighestVal;
                ResultIndices.Add(HighestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static double[] TakeNHighest(double[] data, int N, out int[] indices)
        {
            N = Math.Min(N, data.Length);
            double[] Result = new double[N];
            List<int> ResultIndices = new List<int>();

            for (int n = 0; n < N; n++)
            {
                double HighestVal = -double.MaxValue;
                int HighestIndex = 0;

                for (int i = 0; i < data.Length; i++)
                    if (data[i] > HighestVal && !ResultIndices.Contains(i))
                    {
                        HighestVal = data[i];
                        HighestIndex = i;
                    }

                Result[n] = HighestVal;
                ResultIndices.Add(HighestIndex);
            }

            indices = ResultIndices.ToArray();
            return Result;
        }

        public static float[] TakeAllBelow(float[] data, float threshold, out int[] indices)
        {
            List<float> Result = new List<float>();
            List<int> ResultIndices = new List<int>();

            for (int i = 0; i < data.Length; i++)
                if (data[i] < threshold)
                {
                    Result.Add(data[i]);
                    ResultIndices.Add(i);
                }

            indices = ResultIndices.ToArray();
            return Result.ToArray();
        }

        public static float[] TakeAllAbove(float[] data, float threshold, out int[] indices)
        {
            List<float> Result = new List<float>();
            List<int> ResultIndices = new List<int>();

            for (int i = 0; i < data.Length; i++)
                if (data[i] > threshold)
                {
                    Result.Add(data[i]);
                    ResultIndices.Add(i);
                }

            indices = ResultIndices.ToArray();
            return Result.ToArray();
        }

        public static float Lerp(float a, float b, float x)
        {
            return a + (b - a) * x;
        }

        public static float3 FitPlane(float3[] points)
        {
            double D = 0;
            double E = 0;
            double F = 0;
            double G = 0;
            double H = 0;
            double I = 0;
            double J = 0;
            double K = 0;
            double L = 0;
            double W2 = 0;
            double error = 0;
            double denom = 0;

            for (int i = 0; i < points.Length; i++)
            {
                D += points[i].X * points[i].X;
                E += points[i].X * points[i].Y;
                F += points[i].X;
                G += points[i].Y * points[i].Y;
                H += points[i].Y;
                I += 1;
                J += points[i].X * points[i].Z;
                K += points[i].Y * points[i].Z;
                L += points[i].Z;
            }

            denom = F * F * G - 2 * E * F * H + D * H * H + E * E * I - D * G * I;

            // X axis slope
            double plane_a = (H * H * J - G * I * J + E * I * K + F * G * L - H * (F * K + E * L)) / denom;
            // Y axis slope
            double plane_b = (E * I * J + F * F * K - D * I * K + D * H * L - F * (H * J + E * L)) / denom;
            // Z axis intercept
            double plane_c = (F * G * J - E * H * J - E * F * K + D * H * K + E * E * L - D * G * L) / denom;

            return new float3((float)plane_a, (float)plane_b, (float)plane_c);
        }

        public static float[] FitAndGeneratePlane(float[] intensities, int2 dims)
        {
            float3[] Points = new float3[dims.Elements()];
            for (int y = 0; y < dims.Y; y++)
            {
                for (int x = 0; x < dims.X; x++)
                {
                    Points[y * dims.X + x] = new float3(x, y, intensities[y * dims.X + x]);
                }
            }

            float3 Plane = FitPlane(Points);

            float[] Result = new float[dims.Elements()];

            for (int y = 0; y < dims.Y; y++)
            {
                for (int x = 0; x < dims.X; x++)
                {
                    Result[y * dims.X + x] = x * Plane.X + y * Plane.Y + Plane.Z;
                }
            }

            return Result;
        }

        public static void FitAndSubtractPlane(float[] intensities, int2 dims)
        {
            float[] Plane = FitAndGeneratePlane(intensities, dims);
            for (int i = 0; i < intensities.Length; i++)
                intensities[i] -= Plane[i];
        }

        public static string GetSHA1(byte[] data)
        {
            using (SHA1 hasher = new SHA1CryptoServiceProvider())
            {
                byte[] HashBytes = hasher.ComputeHash(data);

                return Convert.ToBase64String(HashBytes).Replace("=", "").Replace("/", "-").Replace("+", "_");
            }
        }

        public static float3 FitLineWeighted(float3[] points)
        {
            float ss_xy = 0;
            float ss_xx = 0;
            float ss_yy = 0;
            float ave_x = 0;
            float ave_y = 0;
            float sum_w = 0;
            for (int i = 0; i < points.Length; i++)
            {
                ave_x += points[i].Z * points[i].X;
                ave_y += points[i].Z * points[i].Y;
                sum_w += points[i].Z;
                ss_xx += points[i].Z * points[i].X * points[i].X;
                ss_yy += points[i].Z * points[i].Y * points[i].Y;
                ss_xy += points[i].Z * points[i].X * points[i].Y;
            }
            ave_x /= sum_w;
            ave_y /= sum_w;
            ss_xx -= sum_w * ave_x * ave_x;
            ss_yy -= sum_w * ave_y * ave_y;
            ss_xy -= sum_w * ave_x * ave_y;

            float Slope = 0;
            float Intercept = 0;
            float Quality = 0;

            if (ss_xx > 0)
            {
                Slope = ss_xy / ss_xx;
                Intercept = ave_y - Slope * ave_x;
                Quality = ss_xy * ss_xy / (ss_xx * ss_yy);
            }

            return new float3(Slope, Intercept, Quality);
        }
    }

    public class KaiserTable
    {
        public float[] Values;
        public float Sampling;
        public float Radius;

        public KaiserTable(int samples, float radius, float alpha, int order)
        {
            Sampling = radius / samples;
            Values = new float[samples];
            Radius = radius;

            for (int i = 0; i < samples; i++)
                Values[i] = CPU.KaiserBessel(i * Sampling, radius, alpha, order);
        }

        public float GetValue(float r)
        {
            int Sample = (int)(r / Sampling);
            if (Sample >= Values.Length)
                return 0;

            return Values[Sample];
        }
    }

    public class KaiserFTTable
    {
        public float[] Values;
        public float Sampling;
        public float Radius;

        public KaiserFTTable(int samples, float radius, float alpha, int order)
        {
            Sampling = 0.5f / samples;
            Values = new float[samples];
            Radius = radius;

            for (int i = 0; i < samples; i++)
                Values[i] = CPU.KaiserBessel(i * Sampling, radius, alpha, order);
        }

        public float GetValue(float r)
        {
            int Sample = (int)(r / Sampling);
            if (Sample >= Values.Length)
                return 0;

            return Values[Sample];
        }
    }
}
