using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using BitMiracle.LibTiff.Classic;
using Warp.Tools;

namespace Warp
{
    public static class FSC
    {
        public static float[] GetFSC(Image volume1, Image volume2)
        {
            Image HalfFT1 = volume1.IsComplex ? volume1 : volume1.AsFFT(true);
            Image HalfFT2 = volume2.IsComplex ? volume2 : volume2.AsFFT(true);

            int NShells = volume1.Dims.X / 2;

            double[] Nums = new double[NShells];
            double[] Denoms1 = new double[NShells];
            double[] Denoms2 = new double[NShells];

            float[] Vol1 = HalfFT1.GetHostContinuousCopy();
            float[] Vol2 = HalfFT2.GetHostContinuousCopy();

            int3 DimsFT = HalfFT1.DimsFT;

            Parallel.For(0, 20, p =>
            {
                double[] ThreadNums = new double[NShells];
                double[] ThreadDenoms1 = new double[NShells];
                double[] ThreadDenoms2 = new double[NShells];

                for (int z = p; z < DimsFT.Z; z += 20)
                {
                    int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                    zz *= zz;

                    for (int y = 0; y < DimsFT.Y; y++)
                    {
                        int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                        yy *= yy;

                        for (int x = 0; x < DimsFT.X; x++)
                        {
                            int xx = x;
                            xx *= x;

                            float R = (float)Math.Sqrt(zz + yy + xx);
                            if (R >= NShells)
                                continue;

                            int ID = (int)R;
                            float W1 = R - ID;
                            float W0 = 1f - W1;

                            int i = (z * DimsFT.Y + y) * DimsFT.X + x;

                            float Nom = Vol1[i * 2] * Vol2[i * 2] + Vol1[i * 2 + 1] * Vol2[i * 2 + 1];
                            float Denom1 = Vol1[i * 2] * Vol1[i * 2] + Vol1[i * 2 + 1] * Vol1[i * 2 + 1];
                            float Denom2 = Vol2[i * 2] * Vol2[i * 2] + Vol2[i * 2 + 1] * Vol2[i * 2 + 1];

                            ThreadNums[ID] += W0 * Nom;
                            ThreadDenoms1[ID] += W0 * Denom1;
                            ThreadDenoms2[ID] += W0 * Denom2;

                            if (ID < NShells - 1)
                            {
                                ThreadNums[ID + 1] += W1 * Nom;
                                ThreadDenoms1[ID + 1] += W1 * Denom1;
                                ThreadDenoms2[ID + 1] += W1 * Denom2;
                            }
                        }
                    }
                }

                lock (Nums)
                    for (int i = 0; i < Nums.Length; i++)
                    {
                        Nums[i] += ThreadNums[i];
                        Denoms1[i] += ThreadDenoms1[i];
                        Denoms2[i] += ThreadDenoms2[i];
                    }
            });

            float[] Result = new float[NShells];

            for (int i = 0; i < Result.Length; i++)
                Result[i] = (float)(Nums[i] / Math.Max(1e-6f, Math.Sqrt(Denoms1[i] * Denoms2[i])));

            if (HalfFT1 != volume1)
                HalfFT1.Dispose();
            if (HalfFT2 != volume2)
                HalfFT2.Dispose();

            return Result;
        }

        public static float[] GetAmps1D(Image volume)
        {
            Image VolumeFT = volume.IsComplex ? volume : volume.AsFFT(true);
            Image VolumeAbs = VolumeFT.AsAmplitudes();
            if (VolumeFT != volume)
                VolumeFT.Dispose();

            int NShells = volume.Dims.X / 2;

            float[] PS1D = new float[NShells];
            float[] Samples = new float[NShells];

            float[] VolumeAbsData = VolumeAbs.GetHostContinuousCopy();

            int3 DimsFT = VolumeAbs.DimsFT;

            Parallel.For(0, 20, p =>
            {
                float[] ThreadPS1D = new float[NShells];
                float[] ThreadSamples = new float[NShells];

                for (int z = p; z < DimsFT.Z; z += 20)
                {
                    int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                    zz *= zz;

                    for (int y = 0; y < DimsFT.Y; y++)
                    {
                        int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                        yy *= yy;

                        for (int x = 0; x < DimsFT.X; x++)
                        {
                            int xx = x;
                            xx *= x;

                            float R = (float)Math.Sqrt(zz + yy + xx);
                            if (R >= NShells)
                                continue;

                            int ID = (int)R;
                            float W1 = R - ID;
                            float W0 = 1f - W1;

                            int i = (z * DimsFT.Y + y) * DimsFT.X + x;

                            float Amp = VolumeAbsData[i];

                            ThreadPS1D[ID] += W0 * Amp;
                            ThreadSamples[ID] += W0;

                            if (ID < NShells - 1)
                            {
                                ThreadPS1D[ID + 1] += W1 * Amp;
                                ThreadSamples[ID + 1] += W1;
                            }
                        }
                    }
                }

                lock (PS1D)
                    for (int i = 0; i < PS1D.Length; i++)
                    {
                        PS1D[i] += ThreadPS1D[i];
                        Samples[i] += ThreadSamples[i];
                    }
            });

            float[] Result = new float[NShells];

            for (int i = 0; i < Result.Length; i++)
                Result[i] = PS1D[i] / Math.Max(1e-6f, Samples[i]);
            
            return Result;
        }

        public static float GetCutoffShell(float[] fsc, float threshold)
        {
            if (fsc[0] <= threshold)
                return 0;

            int Shell = -1;

            for (int i = 0; i < fsc.Length; i++, Shell++)
                if (fsc[i] <= threshold)
                    break;

            if (Shell >= fsc.Length - 1)
                return Shell;

            float Frac = (fsc[Shell] - threshold) / Math.Max(1e-5f, fsc[Shell] - fsc[Shell + 1]);

            return Shell + Frac;
        }

        public static Image ApplyRamp(Image volume, float[] ramp)
        {
            Image VolumeFT = volume.IsComplex ? volume : volume.AsFFT(true);
            Image Filter = new Image(VolumeFT.Dims, true);
            float[][] FilterData = Filter.GetHost(Intent.Write);

            int3 DimsFT = Filter.DimsFT;
            for (int z = 0; z < DimsFT.Z; z++)
            {
                int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                zz *= zz;

                for (int y = 0, i = 0; y < DimsFT.Y; y++)
                {
                    int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                    yy *= yy;

                    for (int x = 0; x < DimsFT.X; x++, i++)
                    {
                        int xx = x;
                        xx *= x;

                        float R = (float)Math.Sqrt(zz + yy + xx);

                        int ID = (int)R;
                        float W1 = R - ID;
                        float W0 = 1f - W1;
                        
                        FilterData[z][i] = W0 * ramp[Math.Min(ramp.Length - 1, ID)] + W1 * ramp[Math.Min(ramp.Length - 1, ID + 1)];
                    }
                }
            }

            VolumeFT.Multiply(Filter);
            Image Result = VolumeFT.AsIFFT(true);

            Filter.Dispose();
            if (volume != VolumeFT)
                VolumeFT.Dispose();

            return Result;
        }

        public static void AverageLowFrequencies(Image volume1, Image volume2, int maxshell, out Image averaged1, out Image averaged2)
        {
            Image VolumeFT1 = volume1.AsFFT(true);
            volume1.FreeDevice();
            Image VolumeFT2 = volume2.AsFFT(true);
            volume2.FreeDevice();

            float[][] VolumeFT1Data = VolumeFT1.GetHost(Intent.ReadWrite);
            float[][] VolumeFT2Data = VolumeFT2.GetHost(Intent.ReadWrite);
            int3 DimsFT = VolumeFT1.DimsFT;
            Parallel.For(0, DimsFT.Z, z =>
            {
                int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                zz *= zz;

                for (int y = 0, i = 0; y < DimsFT.Y; y++)
                {
                    int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                    yy *= yy;

                    for (int x = 0; x < DimsFT.X; x++, i++)
                    {
                        int xx = x;
                        xx *= x;

                        if (xx + yy + zz < maxshell * maxshell)
                        {
                            float2 Average = new float2(VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 0] + VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 0],
                                                        VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 1] + VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 1]) * 0.5f;

                            VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 0] = Average.X;
                            VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 0] = Average.X;
                            VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 1] = Average.Y;
                            VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 1] = Average.Y;
                        }
                    }
                }
            });

            averaged1 = VolumeFT1.AsIFFT(true);
            averaged2 = VolumeFT2.AsIFFT(true);

            VolumeFT1.Dispose();
            VolumeFT2.Dispose();
        }

        public static Image MakeSoftMask(Image binaryMask, int expand, int smooth)
        {
            Image BinaryExpanded;
            if (expand > 0)
            {
                BinaryExpanded = binaryMask.AsDistanceMapExact(expand);
                BinaryExpanded.Multiply(-1);
                BinaryExpanded.Binarize(-expand + 1e-6f);
            }
            else
            {
                BinaryExpanded = binaryMask.GetCopyGPU();
            }
            binaryMask.FreeDevice();

            Image ExpandedSmooth;
            if (smooth > 0)
            {
                ExpandedSmooth = BinaryExpanded.AsDistanceMapExact(smooth);
                ExpandedSmooth.Multiply((float)Math.PI / smooth);
                ExpandedSmooth.Cos();
                ExpandedSmooth.Add(1);
                ExpandedSmooth.Multiply(0.5f);
            }
            else
            {
                ExpandedSmooth = BinaryExpanded.GetCopyGPU();
            }
            BinaryExpanded.Dispose();

            return ExpandedSmooth;
        }

        public static void GetCorrectedFSC(Image volume1,
                                           Image volume2,
                                           Image mask,
                                           float threshold,
                                           float thresholdRandomize,
                                           out float globalShell,
                                           out float[] fscUnmasked,
                                           out float[] fscMasked,
                                           out float[] fscRandomized,
                                           out float[] fscCorrected)
        {
            int RandomizeAfter = 1;

            // FSC unmasked
            {
                fscUnmasked = GetFSC(volume1, volume2);
                while (fscUnmasked[RandomizeAfter] > thresholdRandomize && RandomizeAfter < fscUnmasked.Length - 1)
                    RandomizeAfter++;
            }

            // FSC masked
            {
                Image Masked1 = volume1.GetCopyGPU();
                volume1.FreeDevice();
                Masked1.Multiply(mask);

                Image Masked2 = volume2.GetCopyGPU();
                volume2.FreeDevice();
                Masked2.Multiply(mask);

                fscMasked = GetFSC(Masked1, Masked2);

                Masked1.Dispose();
                Masked2.Dispose();
            }

            // FSC randomized masked
            {
                Random Rand = new Random(RandomizeAfter);

                Image Fuzz1 = new Image(volume1.Dims, true, true);
                Image Fuzz2 = new Image(volume2.Dims, true, true);
                float[][] Fuzz1Data = Fuzz1.GetHost(Intent.Write);
                float[][] Fuzz2Data = Fuzz2.GetHost(Intent.Write);
                int3 DimsFT = Fuzz1.DimsFT;
                Helper.ForEachElementFT(volume1.Dims, (x, y, z, xx, yy, zz, r) =>
                {
                    double Phase1 = Rand.NextDouble() * Math.PI * 2;
                    double Phase2 = Rand.NextDouble() * Math.PI * 2;
                    long i = y * DimsFT.X + x;

                    Fuzz1Data[z][i * 2 + 0] = r < RandomizeAfter ? 1f : (float)Math.Cos(Phase1);
                    Fuzz1Data[z][i * 2 + 1] = r < RandomizeAfter ? 0f : (float)Math.Sin(Phase1);

                    Fuzz2Data[z][i * 2 + 0] = r < RandomizeAfter ? 1f : (float)Math.Cos(Phase2);
                    Fuzz2Data[z][i * 2 + 1] = r < RandomizeAfter ? 0f : (float)Math.Sin(Phase2);
                });

                Image Randomized1FT = volume1.AsFFT(true);
                volume1.FreeDevice();
                Randomized1FT.Multiply(Fuzz1);
                Fuzz1.Dispose();
                Image Randomized1 = Randomized1FT.AsIFFT(true);
                Randomized1FT.Dispose();
                Randomized1.Multiply(mask);

                Image Randomized2FT = volume2.AsFFT(true);
                volume2.FreeDevice();
                Randomized2FT.Multiply(Fuzz2);
                Fuzz2.Dispose();
                Image Randomized2 = Randomized2FT.AsIFFT(true);
                Randomized2FT.Dispose();
                Randomized2.Multiply(mask);

                fscRandomized = GetFSC(Randomized1, Randomized2);
                Randomized1.Dispose();
                Randomized2.Dispose();
            }

            // FSC corrected
            {
                fscMasked[0] = 1;
                fscUnmasked[0] = 1;
                fscCorrected = new float[fscUnmasked.Length];

                for (int i = 0; i < fscCorrected.Length; i++)
                {
                    if (i < RandomizeAfter + 2)
                        fscCorrected[i] = fscMasked[i];
                    else
                        fscCorrected[i] = (fscMasked[i] - fscRandomized[i]) / (1 - fscRandomized[i]);
                }
            }

            globalShell = GetCutoffShell(fscCorrected, threshold);
        }

        public static void GetWeightedSharpened(Image volume,
                                                float[] fsc,
                                                float angpix,
                                                float fitWorstRes,
                                                float fitBestRes,
                                                int lowpassShell,
                                                out float globalBFactor,
                                                out float fitQuality,
                                                out Image volumeWeighted,
                                                out Image volumeSharpened)
        {
            Image VolumeFT = volume.AsFFT(true);
            VolumeFT.Multiply(1f / volume.Dims.Elements());
            volume.FreeDevice();
            
            float[] Amps1D = GetAmps1D(VolumeFT);
            int NShells = Amps1D.Length;

            // Apply weighting
            int ResMax = NShells - 1;
            float[] FSCWeighting = new float[NShells];
            for (int i = 0; i < NShells; i++)
            {
                if (fsc[i] < 1e-4f)
                    ResMax = Math.Min(i, ResMax);
                FSCWeighting[i] = i >= ResMax ? 0 : (float)Math.Sqrt(Math.Max(0, 2 * fsc[i] / (1 + fsc[i])));
                Amps1D[i] *= FSCWeighting[i];
            }

            // Make weighted Guinier plot
            float[] ResInv = Helper.ArrayOfFunction(i => i / ((float)volume.Dims.X * angpix), NShells);
            float3[] WeightedPoints = Helper.ArrayOfFunction(i => new float3(ResInv[i] * ResInv[i],
                                                                             (float)Math.Log(Math.Max(1e-20f, Amps1D[i])),
                                                                             1f / ResInv[i] < fitWorstRes &&
                                                                             1f / ResInv[i] >= fitBestRes &&
                                                                             i < ResMax
                                                                                 ? 1
                                                                                 : 0), NShells);

            // Fit line through plot
            {
                float ss_xy = 0;
                float ss_xx = 0;
                float ss_yy = 0;
                float ave_x = 0;
                float ave_y = 0;
                float sum_w = 0;
                for (int i = 1; i < ResMax; i++)
                {
                    ave_x += WeightedPoints[i].Z * WeightedPoints[i].X;
                    ave_y += WeightedPoints[i].Z * WeightedPoints[i].Y;
                    sum_w += WeightedPoints[i].Z;
                    ss_xx += WeightedPoints[i].Z * WeightedPoints[i].X * WeightedPoints[i].X;
                    ss_yy += WeightedPoints[i].Z * WeightedPoints[i].Y * WeightedPoints[i].Y;
                    ss_xy += WeightedPoints[i].Z * WeightedPoints[i].X * WeightedPoints[i].Y;
                }
                ave_x /= sum_w;
                ave_y /= sum_w;
                ss_xx -= sum_w * ave_x * ave_x;
                ss_yy -= sum_w * ave_y * ave_y;
                ss_xy -= sum_w * ave_x * ave_y;

                if (ss_xx > 0)
                {
                    globalBFactor = 4 * ss_xy / ss_xx;
                    fitQuality = ss_xy * ss_xy / (ss_xx * ss_yy);
                }
                else
                {
                    globalBFactor = fitQuality = 0;
                }
            }

            // Calculate ramp as FSCweight * -Bfac * lowpass
            float[] FilterWeight = new float[NShells];
            float[] FilterSharpen = new float[NShells];
            for (int i = 0; i < NShells; i++)
            {
                FilterWeight[i] = FSCWeighting[i];
                FilterWeight[i] *= 1f - Math.Max(0, Math.Min(1, (i - (lowpassShell - 1)) / 2f));

                FilterSharpen[i] = FilterWeight[i] * (float)Math.Exp(-globalBFactor / 4 * ResInv[i] * ResInv[i]);
            }

            volumeWeighted = ApplyRamp(VolumeFT, FilterWeight);
            volumeSharpened = ApplyRamp(VolumeFT, FilterSharpen);

            VolumeFT.Dispose();
        }

        public static Image GetAnisotropicFSC(Image volume1,
                                                    Image volume2,
                                                    Image mask,
                                                    float pixelsize,
                                                    float threshold,
                                                    int healpixOrder,
                                                    float scaleToGlobalShell)
        {
            float MaskFraction = mask.GetHostContinuousCopy().Sum() / mask.ElementsReal;

            Image Masked1 = volume1.GetCopyGPU();
            volume1.FreeDevice();
            Masked1.Multiply(mask);
            Image Volume1FT = Masked1.AsFFT(true);
            Masked1.Dispose();

            Image Masked2 = volume2.GetCopyGPU();
            Masked2.Multiply(mask);
            volume2.FreeDevice();
            Image Volume2FT = Masked2.AsFFT(true);
            Masked2.Dispose();

            mask.FreeDevice();

            float2[] Angles = Helper.GetHealpixRotTilt(healpixOrder, "C1", 91f);
            float3[] Directions = Angles.Select(a => Matrix3.Euler(a.X * Helper.ToRad, a.Y * Helper.ToRad, 0).Transposed() * float3.UnitZ).ToArray();

            float AngleSpacing = (float)(60.0 / Math.Pow(2, Math.Min(2, healpixOrder)));

            float[] ConeShells = new float[Directions.Length];

            CPU.ConicalFSC(Volume1FT.GetHostContinuousCopy(),
                           Volume2FT.GetHostContinuousCopy(),
                           volume1.Dims,
                           Helper.ToInterleaved(Directions),
                           Directions.Length,
                           AngleSpacing,
                           10,
                           threshold,
                           MaskFraction,
                           ConeShells);

            Volume1FT.Dispose();
            Volume2FT.Dispose();

            float MeanShell = MathHelper.MeanWeighted(ConeShells, MathHelper.Mult(ConeShells, ConeShells));
            ConeShells = ConeShells.Select(v => v / MeanShell * scaleToGlobalShell).ToArray();

            int NRot = (int)Math.Ceiling(360f / AngleSpacing * 2);
            float StepRot = 360f / NRot;
            int NTilt = (int)Math.Ceiling(90f / AngleSpacing * 2);
            float StepTilt = 90f / NTilt;
            float[] PolarValues = new float[NRot * NTilt];

            for (int t = 0; t < NTilt; t++)
            {
                float Tilt = t * StepTilt;
                for (int r = 0; r < NRot; r++)
                {
                    float Rot = r * StepRot;

                    float3 Direction = Matrix3.Euler(Rot * Helper.ToRad, Tilt * Helper.ToRad, 0).Transposed() * float3.UnitZ;

                    float WeightedShell = 0;
                    float Weights = 0;
                    for (int d = 0; d < Directions.Length; d++)
                    {
                        float AngleDiff = (float)Math.Acos(Math.Min(1, Math.Abs(float3.Dot(Directions[d], Direction)))) * Helper.ToDeg;
                        float Weight = 1 - Math.Min(1, AngleDiff / AngleSpacing);
                        WeightedShell += ConeShells[d] * Weight;
                        Weights += Weight;
                    }

                    PolarValues[t * NRot + r] = volume1.Dims.X * pixelsize / (WeightedShell / Weights);
                }
            }

            return new Image(PolarValues, new int3(NRot, NTilt, 1));
        }
    }
}
