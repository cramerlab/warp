using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Accord.Math;
using Accord.Math.Optimization;
using CommandLine;
using Warp;
using Warp.Tools;

namespace PCA3d
{
    class Program
    {
        static int DeviceID = 0;
        static int NThreads = 5;

        static string WorkingDirectory = @"";

        static int DimOri = -1;
        static int Dim = -1;
        static float AngPixOri = 1.05f;
        static float AngPix = 8.0f;

        static float Diameter = 300f;

        static string ParticlesStarPath = "run_ct1_data.star";
        static string MaskPath = "mask.mrc";

        static string Symmetry = "C1";
        static Matrix3[] SymmetryMatrices;
        static int NSymmetry = 1;

        static int BatchLimit = 1024;

        static int NBatches = 0;
        static List<int> BatchSizes = new List<int>();
        static List<int[]> BatchOriginalRows = new List<int[]>();
        static List<Image> BatchParticlesOri = new List<Image>();
        static List<Image> BatchParticlesOriUnsubtracted = new List<Image>();
        static List<Image> BatchParticlesMasked = new List<Image>();
        static List<Image> BatchParticlesMaskedUnsubtracted = new List<Image>();
        static List<Image> BatchCTFs = new List<Image>();
        static List<Image> BatchSpectralWeights = new List<Image>();
        static List<Matrix3[]> BatchRotations = new List<Matrix3[]>();
        static List<float[]> BatchStratificationWeights = new List<float[]>();
        static List<int[]> BatchSubsets = new List<int[]>();

        static Image DummySpectralWeights;

        static int NParticles = 0;
        static int NComponents = 8;
        static int NIterations = 60;

        static bool PerformPolish = false;
        static bool PerformAlignment = false;

        static int PolishingInterpolationSteps = 50;
        static int PolishingBinSize = 15000;

        static void Main(string[] args)
        {
            Options Options = new Options();

            #region Command line parsing

            if (!Debugger.IsAttached)
            {
                Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);
                WorkingDirectory = Environment.CurrentDirectory + "/";

                DeviceID = Options.DeviceID;
                NThreads = Options.NThreads;

                AngPixOri = Options.AngPixOri;
                AngPix = Options.AngPix;

                Diameter = Options.Diameter;

                ParticlesStarPath = Options.StarPath;
                MaskPath = Options.MaskPath;

                Symmetry = Options.Symmetry;

                BatchLimit = Options.BatchSize;

                NComponents = Options.NComponents;
                NIterations = Options.NIterations;
            }

            #endregion

            GPU.SetDevice(DeviceID);

            Symmetry S = new Symmetry(Symmetry);
            SymmetryMatrices = S.GetRotationMatrices();
            NSymmetry = SymmetryMatrices.Length;

            #region Read STAR

            Console.WriteLine("Reading table...");

            Star TableIn = new Star(WorkingDirectory + ParticlesStarPath);
            //TableIn.SortByKey(TableIn.GetColumn("rlnImageName").Select(s => s.Substring(s.IndexOf('@') + 1)).ToArray());
            //TableIn.RemoveRows(Helper.Combine(Helper.ArrayOfSequence(0, 10000, 1), Helper.ArrayOfSequence(20000, TableIn.RowCount, 1)));
            NParticles = 0;
            float3[] ParticleAngles = TableIn.GetRelionAngles().Select(a => a * Helper.ToRad).ToArray();
            float3[] ParticleShifts = TableIn.GetRelionOffsets();

            CTF[] ParticleCTFParams = TableIn.GetRelionCTF();
            {
                float MeanNorm = MathHelper.Mean(ParticleCTFParams.Select(p => (float)p.Scale));
                for (int p = 0; p < ParticleCTFParams.Length; p++)
                    ParticleCTFParams[p].Scale /= (decimal)MeanNorm;
            }

            int[] ParticleSubset = TableIn.GetColumn("rlnRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

            float[] ParticleNormCorrection = TableIn.HasColumn("rlnNormCorrection") ? 
                                             TableIn.GetColumn("rlnNormCorrection").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : 
                                             Helper.ArrayOfConstant(1f, ParticleShifts.Length);

            string[] ParticleNames = TableIn.GetColumn("rlnImageName");
            string[] UniqueMicrographs = Helper.GetUniqueElements(ParticleNames.Select(s => s.Substring(s.IndexOf('@') + 1))).ToArray();

            Console.WriteLine("Done.\n");

            #endregion

            #region Prepare data

            Console.WriteLine("Loading and preparing data...");
            Console.Write("0/0");

            int NDone = 0;
            Helper.ForCPU(0, UniqueMicrographs.Length, 8, threadID => GPU.SetDevice(DeviceID),
                (imic, threadID) =>
            {
                int[] RowIndices = Helper.GetIndicesOf(ParticleNames, (s) => s.Substring(s.IndexOf('@') + 1) == UniqueMicrographs[imic]);
                string StackPath = WorkingDirectory + ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1);

                if (!File.Exists(StackPath))
                {
                    throw new Exception($"No data found for {UniqueMicrographs[imic]}!");
                }

                Image OriginalStack = Image.FromFile(StackPath);

                lock (TableIn)
                {
                    if (Dim <= 0)   // Figure out dimensions using the first stack
                    {
                        DimOri = OriginalStack.Dims.X;
                        Dim = (int)Math.Round(DimOri * AngPixOri / AngPix / 2) * 2;
                        AngPix = (float)DimOri / Dim * AngPixOri;   // Adjust pixel size to match rounded box size
                    }
                }

                int[] SliceIndices = Helper.IndexedSubset(ParticleNames, RowIndices).Select(s => int.Parse(s.Split(new[] { '@' })[0]) - 1).ToArray();

                float3[] MicShifts = Helper.IndexedSubset(ParticleShifts, RowIndices);

                Image OriginalStackFT = OriginalStack.AsFFT();

                #region Calculate invsigma

                Image OriginalAmps = OriginalStackFT.AsAmplitudes();
                OriginalAmps.Multiply(OriginalAmps);
                Image AmpsReduced = OriginalAmps.AsReducedAlongZ();
                OriginalAmps.Dispose();

                Image OriginalReduced = OriginalStack.AsReducedAlongZ();
                OriginalStack.Dispose();
                Image OriginalReducedFT = OriginalReduced.AsFFT();
                OriginalReduced.Dispose();
                Image ReducedAmps = OriginalReducedFT.AsAmplitudes();
                OriginalReducedFT.Dispose();
                ReducedAmps.Multiply(ReducedAmps);

                // Otherwise we're left with 0 everywhere
                if (OriginalStack.Dims.Z > 1)
                    AmpsReduced.Subtract(ReducedAmps);
                ReducedAmps.Dispose();

                float[] Amps1D = new float[DimOri / 2];
                float[] Samples1D = new float[DimOri / 2];
                float[] Amps2D = AmpsReduced.GetHost(Intent.Read)[0];
                AmpsReduced.FreeDevice();

                Helper.ForEachElementFT(new int2(DimOri), (x, y, xx, yy, r, angle) =>
                {
                    int idx = (int)Math.Round(r);
                    if (idx < DimOri / 2)
                    {
                        float W1 = r - (int)r;
                        float W0 = 1 - W1;
                        Amps1D[idx] += Amps2D[y * (DimOri / 2 + 1) + x] * W0;
                        Samples1D[idx] += W0;
                        Amps1D[Math.Min(Amps1D.Length - 1, idx + 1)] += Amps2D[y * (DimOri / 2 + 1) + x] * W1;
                        Samples1D[Math.Min(Amps1D.Length - 1, idx + 1)] += W1;
                    }
                });

                for (int i = 0; i < Amps1D.Length; i++)
                    Amps1D[i] = 1 / (float)Math.Sqrt(Amps1D[i] / Samples1D[i]);
                Amps1D[0] = 0f;
                float MeanAmps = MathHelper.Mean(Amps1D);
                for (int i = 0; i < Amps1D.Length; i++)
                    Amps1D[i] /= MeanAmps;
                //Amps1D = Helper.ArrayOfConstant(1f, Amps1D.Length);
                //Amps1D[0] = 0f;

                AmpsReduced = new Image(new int3(Dim, Dim, 1), true);
                Amps2D = AmpsReduced.GetHost(Intent.Write)[0];
                Helper.ForEachElementFT(new int2(Dim), (x, y, xx, yy, r, angle) =>
                {
                    Amps2D[y * (Dim / 2 + 1) + x] = MathHelper.Lerp(Amps1D[Math.Min((int)r, DimOri / 2 - 1)],
                                                                    Amps1D[Math.Min((int)r + 1, DimOri / 2 - 1)],
                                                                    r - (int)r);
                });

                Image InvSigma = new Image(Amps2D, new int3(Dim, Dim, 1), true);
                AmpsReduced.Dispose();
                //InvSigma.WriteMRC("d_invsigma.mrc", true);

                #endregion

                float[][] RelevantStackData = Helper.IndexedSubset(OriginalStackFT.GetHost(Intent.Read), SliceIndices);
                Image RelevantStack = new Image(RelevantStackData, new int3(DimOri, DimOri, SliceIndices.Length), true, true);
                OriginalStackFT.Dispose();

                RelevantStack.Multiply(1f / RelevantStack.Dims.ElementsSlice());

                RelevantStack.ShiftSlices(MicShifts.Select(v => new float3(v.X + DimOri / 2, v.Y + DimOri / 2, 0)).ToArray());  // Shift and de-center

                Image RelevantStackScaled = RelevantStack.AsPadded(new int2(Dim));
                RelevantStack.Dispose();
                //RelevantStackScaled.AsIFFT().WriteMRC("d_stackscaled.mrc", true);

                #region Create CTF

                Image CTFCoords = CTF.GetCTFCoords(Dim, Dim);
                CTFStruct[] CTFParams = new CTFStruct[RowIndices.Length];
                for (int p = 0; p < RowIndices.Length; p++)
                {
                    int R = RowIndices[p];
                    ParticleCTFParams[R].PixelSize = (decimal)AngPix;
                    CTFParams[p] = ParticleCTFParams[R].ToStruct();
                }

                Image StackCTF = new Image(new int3(Dim, Dim, RowIndices.Length), true);
                GPU.CreateCTF(StackCTF.GetDevice(Intent.Write),
                              CTFCoords.GetDevice(Intent.Read),
                              IntPtr.Zero,
                              (uint)CTFCoords.ElementsComplex,
                              CTFParams,
                              false,
                              (uint)CTFParams.Length);
                //StackCTF.WriteMRC("d_ctf.mrc", true);
                CTFCoords.Dispose();

                #endregion

                lock (BatchSizes)
                {
                    for (int p = 0; p < RowIndices.Length; p++)
                    {
                        if (BatchSizes.Count == 0 || BatchSizes[BatchSizes.Count - 1] >= BatchLimit)
                        {
                            BatchSizes.Add(0);
                            BatchOriginalRows.Add(new int[BatchLimit]);
                            BatchParticlesOri.Add(new Image(new int3(Dim, Dim, BatchLimit), true, true));
                            BatchCTFs.Add(new Image(new int3(Dim, Dim, BatchLimit), true));
                            BatchSpectralWeights.Add(new Image(new int3(Dim, Dim, BatchLimit), true));
                            BatchRotations.Add(Helper.ArrayOfFunction(i => new Matrix3(), BatchLimit));
                            BatchSubsets.Add(new int[BatchLimit]);
                        }

                        int BatchID = BatchSizes.Count - 1;
                        int BatchEnd = BatchSizes[BatchID];

                        BatchOriginalRows[BatchID][BatchEnd] = RowIndices[p];
                        BatchParticlesOri[BatchID].GetHost(Intent.Write)[BatchEnd] = RelevantStackScaled.GetHost(Intent.Read)[p];
                        BatchCTFs[BatchID].GetHost(Intent.Write)[BatchEnd] = StackCTF.GetHost(Intent.Read)[p];
                        BatchSpectralWeights[BatchID].GetHost(Intent.Write)[BatchEnd] = InvSigma.GetHostContinuousCopy();
                        BatchRotations[BatchID][BatchEnd] = Matrix3.Euler(ParticleAngles[RowIndices[p]]);
                        BatchSubsets[BatchID][BatchEnd] = ParticleSubset[RowIndices[p]];

                        BatchSizes[BatchID]++;
                    }

                    NParticles += RowIndices.Length;

                    ClearCurrentConsoleLine();
                    Console.Write($"{++NDone}/{UniqueMicrographs.Length}, {GPU.GetFreeMemory(DeviceID)} MB");
                }

                RelevantStackScaled.Dispose();
                StackCTF.Dispose();
                InvSigma.Dispose();
            }, null);
            Console.WriteLine("");

            NBatches = BatchSizes.Count;

            Console.WriteLine("Done.\n");

            #endregion

            #region Mask

            Console.WriteLine("Loading mask...");

            Image Mask = Image.FromFile(WorkingDirectory + MaskPath);
            Image MaskHard = null;
            int[] MaskIndices, MaskIndicesHard;
            {
                if (Mask.Dims.X != DimOri)
                {
                    Image MaskPadded = Mask.AsPadded(new int3(DimOri));
                    Mask.Dispose();
                    Mask = MaskPadded;
                }

                Image MaskScaled = Mask.AsScaled(new int3(Dim));
                Mask.Dispose();
                MaskScaled.Binarize(0.5f);

                MaskHard = FSC.MakeSoftMask(MaskScaled, 2, 0);
                //Mask = MaskScaled;
                MaskScaled.Dispose();

                Mask = FSC.MakeSoftMask(MaskHard, 0, 3);
                Mask.WriteMRC("d_mask.mrc", true);
            }

            {
                float[] MaskData = Mask.GetHostContinuousCopy();
                List<int> MaskIndicesList = new List<int>();
                for (int i = 0; i < MaskData.Length; i++)
                    if (MaskData[i] > 0)
                        MaskIndicesList.Add(i);
                MaskIndices = MaskIndicesList.ToArray();
            }

            {
                float[] MaskData = MaskHard.GetHostContinuousCopy();
                List<int> MaskIndicesList = new List<int>();
                for (int i = 0; i < MaskData.Length; i++)
                    if (MaskData[i] > 0)
                        MaskIndicesList.Add(i);
                MaskIndicesHard = MaskIndicesList.ToArray();
            }

            Console.WriteLine("Done.\n");

            Dim = Mask.Dims.X;

            #endregion

            Console.WriteLine(GPU.GetFreeMemory(0));

            #region Reconstruct and subtract average

            {
                Console.WriteLine("Reconstructing and subtracting average...");

                {
                    Projector Reconstructor = new Projector(new int3(Dim), 2);

                    Helper.ForCPU(0, NBatches, NThreads, threadID => GPU.SetDevice(DeviceID),
                        (batchID, threadID) =>
                    {
                        Image ParticlesCopy = BatchParticlesOri[batchID].GetCopyGPU();
                        Image CTFCopy = BatchCTFs[batchID].GetCopyGPU();

                        ParticlesCopy.Multiply(CTFCopy);
                        ParticlesCopy.Multiply(BatchSpectralWeights[batchID]);
                        //ParticlesCopy.Multiply(Helper.Combine(BatchStratificationWeights[batchID],
                        //                                      Helper.ArrayOfConstant(1f, BatchLimit - BatchStratificationWeights[batchID].Length)));

                        CTFCopy.Multiply(CTFCopy);
                        CTFCopy.Multiply(BatchSpectralWeights[batchID]);
                        //CTFCopy.Multiply(Helper.Combine(BatchStratificationWeights[batchID],
                        //                                Helper.ArrayOfConstant(1f, BatchLimit - BatchStratificationWeights[batchID].Length)));

                        foreach (var m in SymmetryMatrices)
                            Reconstructor.BackProject(ParticlesCopy,
                                                      CTFCopy,
                                                      BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                                                      new float3(1, 1, 0));

                        ParticlesCopy.Dispose();
                        CTFCopy.Dispose();

                        BatchParticlesOri[batchID].FreeDevice();
                        BatchCTFs[batchID].FreeDevice();
                        BatchSpectralWeights[batchID].FreeDevice();
                    }, null);

                    Image VolumeAverage = Reconstructor.Reconstruct(false, "C1");
                    //VolumeAverage.Multiply(1f / VolumeAverage.Dims.X / 8);
                    Reconstructor.Dispose();

                    GPU.SphereMask(VolumeAverage.GetDevice(Intent.Read),
                                   VolumeAverage.GetDevice(Intent.Write),
                                   new int3(Dim),
                                   Diameter / AngPix / 2,
                                   Dim / 8f,
                                   false,
                                   1);
                    VolumeAverage.WriteMRC("d_recavg.mrc", true);
                    VolumeAverage.WriteMRC(WorkingDirectory + $"pc_{0:D2}.mrc", true);

                    // Subtract average projections from raw particles, and make a real space-masked copy for comparisons later

                    {
                        Projector ProjectorAverage = new Projector(VolumeAverage, 2);
                        VolumeAverage.Dispose();

                        Projector ReconstructorVariance = new Projector(new int3(Dim), 1);

                        for (int batchID = 0; batchID < BatchSizes.Count; batchID++)
                        {
                            // Leave unsubtracted data for later
                            if (PerformPolish)
                            {
                                BatchParticlesOriUnsubtracted.Add(BatchParticlesOri[batchID].GetCopy());

                                BatchParticlesOriUnsubtracted[batchID].Multiply(BatchCTFs[batchID]);
                                //BatchParticlesOriUnsubtracted[batchID].Multiply(BatchSpectralWeights[batchID]);

                                Image IFT = BatchParticlesOriUnsubtracted[batchID].AsIFFT(false, 0);
                                BatchParticlesOriUnsubtracted[batchID].FreeDevice();

                                GPU.SphereMask(IFT.GetDevice(Intent.Read),
                                                IFT.GetDevice(Intent.Write),
                                                new int3(Dim, Dim, 1),
                                                Diameter / AngPix / 2,
                                                Dim / 8f,
                                                true,
                                                (uint)BatchLimit);
                                //IFT.WriteMRC("d_particlesmasked.mrc", true);

                                Image FT = IFT.AsFFT();
                                FT.Multiply(1f / FT.Dims.ElementsSlice());
                                BatchParticlesMaskedUnsubtracted.Add(FT);
                                IFT.Dispose();
                                BatchParticlesMaskedUnsubtracted[batchID].FreeDevice();
                            }

                            // Subtract average projections
                            {
                                Image Projections = ProjectorAverage.Project(new int2(Dim), BatchRotations[batchID].Select(a => Matrix3.EulerFromMatrix(a)).ToArray());
                                Projections.Multiply(BatchCTFs[batchID]);

                                BatchParticlesOri[batchID].Subtract(Projections);
                                Projections.Dispose();

                                Image ParticlesCopyAmps = BatchParticlesOri[batchID].AsAmplitudes();
                                Image ParticlesCopy = ParticlesCopyAmps.AsComplex();
                                ParticlesCopyAmps.Dispose();

                                Image CTFCopy = BatchCTFs[batchID].GetCopyGPU();


                                ParticlesCopy.Multiply(BatchSpectralWeights[batchID]);
                                ParticlesCopy.Multiply(ParticlesCopy);
                                ParticlesCopy.Multiply(CTFCopy);
                                ParticlesCopy.Abs();

                                CTFCopy.Multiply(CTFCopy);
                                CTFCopy.Multiply(BatchCTFs[batchID]);
                                CTFCopy.Abs();
                                //CTFCopy.Multiply(BatchSpectralWeights[batchID]);

                                foreach (var m in SymmetryMatrices)
                                    ReconstructorVariance.BackProject(ParticlesCopy,
                                                              CTFCopy,
                                                              BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                                                              new float3(1, 1, 0));

                                ParticlesCopy.Dispose();
                                CTFCopy.Dispose();

                                BatchParticlesOri[batchID].FreeDevice();
                                BatchCTFs[batchID].FreeDevice();
                                BatchSpectralWeights[batchID].FreeDevice();
                            }
                        }

                        Image VolumeVarianceIFT = ReconstructorVariance.Reconstruct(false);
                        ReconstructorVariance.Dispose();
                        VolumeVarianceIFT.MaskSpherically(Diameter / AngPix, Dim / 8f, true);

                        float[] Variance1D = VolumeVarianceIFT.AsAmplitudes1D();
                        VolumeVarianceIFT.Dispose();
                        Variance1D[0] = Variance1D[1];
                        float VarianceMean = MathHelper.Mean(Variance1D);
                        Variance1D = Variance1D.Select(v => (float)Math.Sqrt(v / VarianceMean)).ToArray();

                        Image Variance2D = new Image(new int3(Dim, Dim, 1), true);
                        float[] Variance2DData = Variance2D.GetHost(Intent.Write)[0];
                        Helper.ForEachElementFT(new int2(Dim), (x, y, xx, yy, r, angle) =>
                        {
                            Variance2DData[y * (Dim / 2 + 1) + x] = 1f / MathHelper.Lerp(Variance1D[Math.Min((int)r, Variance1D.Length - 1)],
                                                                                        Variance1D[Math.Min((int)r + 1, Variance1D.Length - 1)],
                                                                                        r - (int)r);
                        });
                        Variance2D.WriteMRC("d_variance2d.mrc", true);

                        for (int batchID = 0; batchID < BatchSizes.Count; batchID++)
                            BatchSpectralWeights[batchID].MultiplySlices(Variance2D);

                        Variance2D.Dispose();
                    }

                    //{
                    //    Projector ReconstructorVariance = new Projector(new int3(Dim), 1);

                    //    for (int batchID = 0; batchID < BatchSizes.Count; batchID++)
                    //    {
                    //        // Subtract average projections
                    //        {
                    //            Image ParticlesCopyAmps = BatchParticlesOri[batchID].AsAmplitudes();
                    //            Image ParticlesCopy = ParticlesCopyAmps.AsComplex();
                    //            ParticlesCopyAmps.Dispose();

                    //            Image CTFCopy = BatchCTFs[batchID].GetCopyGPU();


                    //            ParticlesCopy.Multiply(BatchSpectralWeights[batchID]);
                    //            ParticlesCopy.Multiply(ParticlesCopy);
                    //            ParticlesCopy.Multiply(CTFCopy);
                    //            ParticlesCopy.Abs();

                    //            CTFCopy.Multiply(CTFCopy);
                    //            CTFCopy.Multiply(BatchCTFs[batchID]);
                    //            CTFCopy.Abs();
                    //            //CTFCopy.Multiply(BatchSpectralWeights[batchID]);

                    //            foreach (var m in SymmetryMatrices)
                    //                ReconstructorVariance.BackProject(ParticlesCopy,
                    //                                          CTFCopy,
                    //                                          BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                    //                                          new float3(1, 1, 0));

                    //            ParticlesCopy.Dispose();
                    //            CTFCopy.Dispose();
                    //        }
                    //    }

                    //    Image VolumeVarianceIFT = ReconstructorVariance.Reconstruct(false);
                    //    ReconstructorVariance.Dispose();
                    //    VolumeVarianceIFT.MaskSpherically(Diameter / AngPix, Dim / 8f, true);

                    //    Image VolumeVariance = VolumeVarianceIFT.AsFFT(true);
                    //    Image VolumeVarianceAmps = VolumeVariance.AsAmplitudes();
                    //    VolumeVarianceAmps.WriteMRC("d_variance3d.mrc", true);

                    //    float[] Variance1D = VolumeVarianceIFT.AsAmplitudes1D();
                    //    VolumeVarianceIFT.Dispose();
                    //    Variance1D[0] = Variance1D[1];

                    //    Image Variance2D = new Image(new int3(Dim, Dim, 1), true);
                    //    float[] Variance2DData = Variance2D.GetHost(Intent.Write)[0];
                    //    Helper.ForEachElementFT(new int2(Dim), (x, y, xx, yy, r, angle) =>
                    //    {
                    //        Variance2DData[y * (Dim / 2 + 1) + x] = 1f / MathHelper.Lerp(Variance1D[Math.Min((int)r, Variance1D.Length - 1)],
                    //                                                                    Variance1D[Math.Min((int)r + 1, Variance1D.Length - 1)],
                    //                                                                    r - (int)r);
                    //    });
                    //    Variance2D.WriteMRC("d_variance2d.mrc", true);

                    //    for (int batchID = 0; batchID < BatchSizes.Count; batchID++)
                    //        BatchSpectralWeights[batchID].MultiplySlices(Variance2D);

                    //    Variance2D.Dispose();
                    //}

                    {
                        for (int batchID = 0; batchID < BatchSizes.Count; batchID++)
                        {
                            // Mask particles after pre-multiplying by spectral weights
                            // CTF is not pre-multiplied here

                            {
                                BatchParticlesOri[batchID].Multiply(BatchSpectralWeights[batchID]);

                                Image IFT = BatchParticlesOri[batchID].AsIFFT(false);

                                GPU.SphereMask(IFT.GetDevice(Intent.Read),
                                               IFT.GetDevice(Intent.Write),
                                               new int3(Dim, Dim, 1),
                                               Diameter / AngPix / 2,
                                               Dim / 8f,
                                               true,
                                               (uint)BatchLimit);
                                //IFT.WriteMRC("d_particlesmasked.mrc", true);

                                Image FT = IFT.AsFFT();
                                FT.Multiply(1f / FT.Dims.ElementsSlice());
                                BatchParticlesMasked.Add(FT);
                                FT.FreeDevice();
                                IFT.Dispose();
                            }
                        }
                    }
                }
                
                #region Make reconstructions from mean-subtracted particles, unmasked and masked

                {
                    Projector Reconstructor = new Projector(new int3(Dim), 2);
                    Projector ReconstructorMasked = new Projector(new int3(Dim), 2);

                    for (int batchID = 0; batchID < BatchSizes.Count; batchID++)
                    {
                        Image ParticlesCopy = BatchParticlesOri[batchID].GetCopyGPU();
                        Image CTFCopy = BatchCTFs[batchID].GetCopyGPU();

                        ParticlesCopy.Multiply(CTFCopy);
                        CTFCopy.Multiply(CTFCopy);
                        CTFCopy.Multiply(BatchSpectralWeights[batchID]);

                        foreach (var m in SymmetryMatrices)
                            Reconstructor.BackProject(ParticlesCopy,
                                                      CTFCopy,
                                                      BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                                                      new float3(1, 1, 0));

                        ParticlesCopy.Dispose();
                        CTFCopy.Dispose();

                        BatchParticlesOri[batchID].FreeDevice();
                        BatchCTFs[batchID].FreeDevice();
                        BatchSpectralWeights[batchID].FreeDevice();
                    }

                    for (int batchID = 0; batchID < BatchSizes.Count; batchID++)
                    {
                        Image ParticlesCopy = BatchParticlesMasked[batchID].GetCopyGPU();
                        Image CTFCopy = BatchCTFs[batchID].GetCopyGPU();

                        ParticlesCopy.Multiply(CTFCopy);
                        CTFCopy.Multiply(CTFCopy);
                        CTFCopy.Multiply(BatchSpectralWeights[batchID]);

                        foreach (var m in SymmetryMatrices)
                            ReconstructorMasked.BackProject(ParticlesCopy,
                                                            CTFCopy,
                                                            BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                                                            new float3(1, 1, 0));

                        ParticlesCopy.Dispose();
                        CTFCopy.Dispose();

                        BatchParticlesMasked[batchID].FreeDevice();
                        BatchCTFs[batchID].FreeDevice();
                        BatchSpectralWeights[batchID].FreeDevice();
                    }

                    {
                        Image VolumeAverage = Reconstructor.Reconstruct(false, "C1");
                        Reconstructor.Dispose();

                        GPU.SphereMask(VolumeAverage.GetDevice(Intent.Read),
                                       VolumeAverage.GetDevice(Intent.Write),
                                       new int3(Dim),
                                       Diameter / AngPix / 2,
                                       Dim / 8f,
                                       false,
                                       1);
                        VolumeAverage.WriteMRC("d_recavg_subtracted.mrc", true);
                    }

                    {
                        Image VolumeAverage = ReconstructorMasked.Reconstruct(false, "C1");
                        ReconstructorMasked.Dispose();

                        GPU.SphereMask(VolumeAverage.GetDevice(Intent.Read),
                                       VolumeAverage.GetDevice(Intent.Write),
                                       new int3(Dim),
                                       Diameter / AngPix / 2,
                                       Dim / 8f,
                                       false,
                                       1);
                        VolumeAverage.WriteMRC("d_recavg_subtracted_masked.mrc", true);
                    }

                    //BatchParticlesMasked[0].AsIFFT().WriteMRC("d_particlesmasked.mrc", true);
                }

                #endregion

                Console.WriteLine("Done.\n");
            }

            #endregion

            Star SettingsTable = new Star(new[] { "wrpParticleTable", "wrpSymmetry", "wrpPixelSize", "wrpDiameter", "wrpMask" });
            SettingsTable.AddRow(new List<string>() { ParticlesStarPath, Symmetry, AngPixOri.ToString(), Diameter.ToString(), MaskPath });

            Star WeightsTable = new Star(new string[0]);
            WeightsTable.AddColumn($"wrpPCA{(0):D2}", Helper.ArrayOfConstant("1.000", NParticles * NSymmetry));

            float[] VolumeData = new float[Dim * Dim * Dim];

            List<float[]> AllComponents = new List<float[]>();
            //List<float[]> AllComponentsHard = new List<float[]>();
            RandomNormal RandN = new RandomNormal(123);

            List<float[]> AllNormalizedScores = new List<float[]>();

            //Image[] BatchParticles = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(Dim, Dim, BatchSize)), NThreads);
            Image[] BatchParticlesCopy = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true, true), NThreads);
            Image[] BatchCTFsCopy = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true), NThreads);
            Image[] BatchSpectralCopy = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true), NThreads);
            //Image[] BatchProj = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(Dim, Dim, BatchSize)), NThreads);
            //Image[] BatchProjFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(Dim, Dim, BatchSize), true, true), NThreads);
            //Image[] BatchWeights = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(BatchSize, 1, 1)), NThreads);

            int PlanForwRec, PlanBackRec, PlanForwCTF;
            Projector.GetPlans(new int3(Dim), 2, out PlanForwRec, out PlanBackRec, out PlanForwCTF);

            // Since masked particles are already pre-whitened, use dummy weights filled with 1 for correlation
            DummySpectralWeights = new Image(new int3(Dim, Dim, 1), true);
            DummySpectralWeights.Fill(1);

            for (int icomponent = 0; icomponent < NComponents; icomponent++)
            {
                Console.WriteLine($"PC {icomponent}:");

                float[] PC = Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), MaskIndices.Length);
                //float[] PCHard = new float[MaskIndicesHard.Length];

                float[] AllScores = new float[NParticles * NSymmetry];

                float[] FSCCurve = Helper.ArrayOfFunction(i => i < Math.Max(Dim / 8, 4) ? 1f : 0f, Dim / 2);

                Console.Write($"0/{NIterations}");

                //for (int i = 0; i < NBatches; i++)
                //    BatchStratificationWeights[i] = Helper.ArrayOfConstant(1f, BatchStratificationWeights[i].Length);

                for (int iiter = 0; iiter < NIterations; iiter++)
                {
                    //if (i < Iterations - 1)
                        MathHelper.NormalizeL2InPlace(PC);

                    for (int j = 0; j < MaskIndices.Length; j++)
                        VolumeData[MaskIndices[j]] = PC[j];

                    Image Volume = new Image(VolumeData, new int3(Dim));
                    //Image VolumeRamped = FSC.ApplyRamp(Volume, FSCCurve);
                    //Volume.Dispose();
                    //Volume = VolumeRamped;
                    Volume.Multiply(Mask);
                    Volume.WriteMRC($"d_pc{icomponent:D2}.mrc", true);

                    Projector VolumeProjector = new Projector(Volume, 2);
                    VolumeProjector.PutTexturesOnDevice();
                    Volume.Dispose();

                    Projector[] NextReconstructor = { new Projector(new int3(Dim), 2),
                                                      new Projector(new int3(Dim), 2) };

                    Helper.ForCPU(0, NBatches, NThreads, null,
                                  (b, threadID) =>
                                  {
                                      int CurBatch = BatchSizes[b];

                                      float[] Scores = new float[BatchLimit];

                                      for (int im = 0; im < NSymmetry; im++)
                                      {
                                          Matrix3 MSym = SymmetryMatrices[im];
                                          float3[] Angles = BatchRotations[b].Take(CurBatch).Select(a => Matrix3.EulerFromMatrix(a * MSym)).ToArray();

                                          GPU.PCA(Scores,
                                                  BatchParticlesMasked[b].GetHostPinned(Intent.Read),
                                                  BatchParticlesOri[b].GetHostPinned(Intent.Read),
                                                  BatchCTFs[b].GetHostPinned(Intent.Read),
                                                  BatchSpectralWeights[b].GetHostPinned(Intent.Read),
                                                  Dim,
                                                  new float[CurBatch * 2],
                                                  Helper.ToInterleaved(Angles),
                                                  new float3(1, 1, 0) * VolumeProjector.Oversampling,
                                                  VolumeProjector.t_DataRe,
                                                  VolumeProjector.t_DataIm,
                                                  VolumeProjector.DimsOversampled.X,
                                                  CurBatch,
                                                  false);

                                          for (int p = 0; p < CurBatch; p++)
                                              AllScores[(b * BatchLimit + p) * NSymmetry + im] = Scores[p];
                                      }
                                  }, null);

                    VolumeProjector.Dispose();

                    //foreach (var prevScores in AllNormalizedScores)
                    //{
                    //    float DotP = MathHelper.DotProduct(AllScores, prevScores);
                    //    for (int j = 0; j < AllScores.Length; j++)
                    //        AllScores[j] -= DotP * prevScores[j];
                    //}

                    for (int ihalf = 0; ihalf < 2; ihalf++)
                        Helper.ForCPU(0, NBatches, NThreads, null,
                                      (b, threadID) =>
                                      {
                                          int CurBatch = BatchSizes[b];

                                          float[] Scores = new float[BatchLimit];

                                          for (int im = 0; im < NSymmetry; im++)
                                          {
                                              GPU.CopyDeviceToDevice(BatchParticlesOri[b].GetHostPinned(Intent.Read), BatchParticlesCopy[threadID].GetDevice(Intent.Write), BatchParticlesOri[b].ElementsReal);
                                              GPU.CopyDeviceToDevice(BatchCTFs[b].GetHostPinned(Intent.Read), BatchCTFsCopy[threadID].GetDevice(Intent.Write), BatchCTFs[b].ElementsReal);
                                              GPU.CopyDeviceToDevice(BatchSpectralWeights[b].GetHostPinned(Intent.Read), BatchSpectralCopy[threadID].GetDevice(Intent.Write), BatchCTFs[b].ElementsReal);

                                              Matrix3 MSym = SymmetryMatrices[im];
                                              float3[] Angles = BatchRotations[b].Take(CurBatch).Select(a => Matrix3.EulerFromMatrix(a * MSym)).ToArray();

                                              for (int p = 0; p < CurBatch; p++)
                                                  Scores[p] = AllScores[(b * BatchLimit + p) * NSymmetry + im] * (ihalf == BatchSubsets[b][p] ? 1f : 0f);

                                              //BatchParticlesOriCopy[threadID].Multiply(Scores);
                                              //BatchCTFsCopy[threadID].Multiply(Scores);
                                              //BatchCTFsCopy[threadID].Multiply(Scores);

                                              BatchCTFsCopy[threadID].Multiply(Scores);

                                              BatchParticlesCopy[threadID].Multiply(BatchCTFsCopy[threadID]);
                                              //BatchParticlesCopy[threadID].Multiply(BatchSpectralCopy[threadID]);

                                              BatchCTFsCopy[threadID].Multiply(BatchCTFsCopy[threadID]);
                                              BatchCTFsCopy[threadID].Multiply(BatchSpectralCopy[threadID]);

                                              lock (NextReconstructor)
                                                  NextReconstructor[ihalf].BackProject(BatchParticlesCopy[threadID], BatchCTFsCopy[threadID], Angles, new float3(1, 1, 0));
                                          }
                                      }, null);

                    Image[] Halves = NextReconstructor.Select(r => r.Reconstruct(false, "C1", PlanForwRec, PlanBackRec, PlanForwCTF)).ToArray();
                    foreach (var rec in NextReconstructor)
                        rec.Dispose();

                    Image NextVolume = Halves[0].GetCopyGPU();
                    NextVolume.Add(Halves[1]);
                    NextVolume.Multiply(0.5f);

                    foreach (var half in Halves)
                        half.Multiply(Mask);

                    //if (iiter >= 5)
                    //{
                    //    FSCCurve = FSC.GetFSC(Halves[0], Halves[1]);
                    //    float MaxShell = FSC.GetCutoffShell(FSCCurve, 0.5f);
                    //    FSCCurve = Helper.ArrayOfFunction(i => i <= MaxShell ? (FSCCurve[i] - 0.5f) * 2 : 0f, FSCCurve.Length);

                    //    Image NextVolumeRamped = FSC.ApplyRamp(NextVolume, FSCCurve);
                    //    NextVolume.Dispose();
                    //    NextVolume = NextVolumeRamped;
                    //}

                    foreach (var half in Halves)
                        half.Dispose();

                    GPU.SphereMask(NextVolume.GetDevice(Intent.Read),
                                   NextVolume.GetDevice(Intent.Write),
                                   new int3(Dim),
                                   Diameter / AngPix / 2,
                                   Dim / 8f,
                                   false,
                                   1);
                    NextVolume.WriteMRC($"d_nextvolume{icomponent:D2}.mrc", true);

                    float[] NextData = Helper.IndexedSubset(NextVolume.GetHostContinuousCopy(), MaskIndices);
                    //float[] NextDataHard = Helper.IndexedSubset(NextVolume.GetHostContinuousCopy(), MaskIndicesHard);

                    //float NextDataMean = MathHelper.Mean(NextData);
                    //for (int i = 0; i < NextData.Length; i++)
                    //    NextData[i] -= NextDataMean;
                    MathHelper.NormalizeL2InPlace(NextData);
                    //MathHelper.NormalizeL2InPlace(NextDataHard);

                    float Orthogonality = 0;
                    for (int c = 0; c < AllComponents.Count; c++)
                    {
                        float[] prevComponent = AllComponents[c];
                        //float[] prevComponentHard = AllComponentsHard[c];

                        float DotP = MathHelper.DotProduct(NextData, prevComponent);

                        for (int j = 0; j < NextData.Length; j++)
                            NextData[j] -= DotP * prevComponent[j];

                        Orthogonality += DotP;
                    }
                    if (AllComponents.Count > 1)
                        Orthogonality /= AllComponents.Count;
                    Orthogonality = 1 - Math.Abs(Orthogonality);

                    float DotWithPreviousIteration = MathHelper.DotProduct(PC, NextData);

                    for (int j = 0; j < PC.Length; j++)
                        PC[j] = MathHelper.Lerp(PC[j], NextData[j], 1f);

                    //for (int j = 0; j < PCHard.Length; j++)
                    //    PCHard[j] = MathHelper.Lerp(PCHard[j], NextDataHard[j], 1f);

                    //Console.WriteLine(GPU.GetFreeMemory(1));
                    ClearCurrentConsoleLine();
                    Console.Write($"{iiter + 1}/{NIterations}, convergence = {DotWithPreviousIteration:F7}, orthogonality = {Orthogonality:F7}");
                }
                Console.Write("\n");

                #region Subtract the new component from data

                if (false)
                {
                    MathHelper.NormalizeL2InPlace(PC);

                    for (int i = 0; i < MaskIndices.Length; i++)
                        VolumeData[MaskIndices[i]] = PC[i];

                    Image Volume = new Image(VolumeData, new int3(Dim));
                    Volume.Multiply(Mask);

                    Projector VolumeProjector = new Projector(Volume, 2);
                    VolumeProjector.PutTexturesOnDevice();
                    Volume.Dispose();

                    Helper.ForCPU(0, NBatches, NThreads, null,
                                  (b, threadID) =>
                                  {
                                      int CurBatch = BatchSizes[b];

                                      float[] Scores = new float[BatchLimit];

                                      for (int im = 0; im < NSymmetry; im++)
                                      {
                                          //GPU.CopyDeviceToDevice(BatchParticlesMasked[b].GetDevice(Intent.Read), BatchParticlesCopy[threadID].GetDevice(Intent.Write), BatchParticlesOri[b].ElementsReal);
                                          GPU.CopyDeviceToDevice(BatchCTFs[b].GetDevice(Intent.Read), BatchCTFsCopy[threadID].GetDevice(Intent.Write), BatchCTFs[b].ElementsReal);
                                          GPU.CopyDeviceToDevice(BatchSpectralWeights[b].GetDevice(Intent.Read), BatchSpectralCopy[threadID].GetDevice(Intent.Write), BatchSpectralWeights[b].ElementsReal);

                                          Matrix3 MSym = SymmetryMatrices[im];
                                          float3[] Angles = BatchRotations[b].Take(CurBatch).Select(a => Matrix3.EulerFromMatrix(a * MSym)).ToArray();

                                          GPU.PCA(Scores,
                                                  BatchParticlesMasked[b].GetDevice(Intent.ReadWrite),
                                                  BatchParticlesOri[b].GetDevice(Intent.ReadWrite),
                                                  BatchCTFs[b].GetDevice(Intent.Read),
                                                  BatchSpectralWeights[b].GetDevice(Intent.Read),
                                                  Dim,
                                                  new float[CurBatch * 2],
                                                  Helper.ToInterleaved(Angles),
                                                  new float3(1, 1, 0) * VolumeProjector.Oversampling,
                                                  VolumeProjector.t_DataRe,
                                                  VolumeProjector.t_DataIm,
                                                  VolumeProjector.DimsOversampled.X,
                                                  CurBatch,
                                                  true);
                                          

                                          Image IFT = BatchParticlesOri[b].AsIFFT(false);

                                          GPU.SphereMask(IFT.GetDevice(Intent.Read),
                                                         IFT.GetDevice(Intent.Write),
                                                         new int3(Dim, Dim, 1),
                                                         Diameter / AngPix / 2,
                                                         Dim / 8f,
                                                         true,
                                                         (uint)BatchLimit);
                                          //IFT.WriteMRC("d_particlesmasked.mrc", true);

                                          Image FT = IFT.AsFFT();
                                          FT.Multiply(1f / FT.Dims.ElementsSlice());

                                          BatchParticlesMasked[b].Dispose();
                                          BatchParticlesMasked[b] = FT;

                                          IFT.Dispose();
                                      }
                                  }, null);

                    VolumeProjector.Dispose();
                }

                #endregion

                float AllScoresMean = MathHelper.Mean(AllScores);
                AllNormalizedScores.Add(MathHelper.NormalizeL2(AllScores.Select(v => v - AllScoresMean).ToArray()));

                float[] InterpolatedScores = new float[NParticles * NSymmetry];

                if (PerformPolish)
                {
                    int NInterpolants = PolishingInterpolationSteps;
                    int PoolSize = PolishingBinSize;

                    List<float> AllScoresSorted = AllScores.ToList();
                    AllScoresSorted.Sort();
                    float Min = MathHelper.Mean(AllScoresSorted.Take(PoolSize / 4));
                    float Max = MathHelper.Mean(AllScoresSorted.Skip(AllScoresSorted.Count - PoolSize / 4));

                    float[][] InterpolantScores = Helper.ArrayOfFunction(i => new float[NParticles * NSymmetry], NInterpolants);
                    float[] InterpStops = new float[NInterpolants];

                    Console.Write($"Polishing: 0/{NInterpolants}");

                    for (int iinterp = 0; iinterp < NInterpolants; iinterp++)
                    {
                        float Center = MathHelper.Lerp(Min, Max, (float)iinterp / (NInterpolants - 1));
                        InterpStops[iinterp] = Center;

                        var Sorted = new List<(float diff, int id)>();
                        for (int j = 0; j < AllScores.Length; j++)
                        {
                            float Diff = Center - AllScores[j];
                            Diff *= Diff;
                            Sorted.Add((Diff, j));
                        }
                        Sorted.Sort((a, b) => a.diff.CompareTo(b.diff));

                        Sorted = Sorted.Take(PoolSize).ToList();

                        Projector InterpolantReconstructor = new Projector(new int3(Dim), 2);

                        for (int batchStart = 0; batchStart < Sorted.Count; batchStart += BatchLimit)
                        {
                            int CurBatch = Math.Min(BatchLimit, Sorted.Count - batchStart);

                            float3[] Angles = new float3[CurBatch];

                            for (int ib = 0; ib < CurBatch; ib++)
                            {
                                int ID = Sorted[batchStart + ib].id;
                                int s = ID % NSymmetry;
                                ID /= NSymmetry;
                                int b = ID / BatchLimit;
                                int bp = ID % BatchLimit;

                                GPU.CopyDeviceToDevice(BatchParticlesOriUnsubtracted[b].GetDeviceSlice(bp, Intent.Read), BatchParticlesCopy[0].GetDeviceSlice(ib, Intent.Write), BatchParticlesOri[b].ElementsSliceReal);
                                GPU.CopyDeviceToDevice(BatchCTFs[b].GetDeviceSlice(bp, Intent.Read), BatchCTFsCopy[0].GetDeviceSlice(ib, Intent.Write), BatchCTFs[b].ElementsSliceReal);

                                Matrix3 MSym = SymmetryMatrices[s];
                                Angles[ib] = Matrix3.EulerFromMatrix(BatchRotations[b][bp] * MSym);
                            }

                            InterpolantReconstructor.BackProject(BatchParticlesCopy[0], BatchCTFsCopy[0], Angles, new float3(1, 1, 0));
                        }

                        Image Interpolant = InterpolantReconstructor.Reconstruct(false, "C1");
                        InterpolantReconstructor.Dispose();

                        Interpolant.Multiply(Mask);
                        Interpolant.WriteMRC($"d_pc{icomponent:D2}_interp{iinterp:D2}.mrc", true);

                        Projector InterpolantProjector = new Projector(Interpolant, 2);
                        InterpolantProjector.PutTexturesOnDevice();

                        Helper.ForCPU(0, NBatches, NThreads, null,
                                  (b, threadID) =>
                                  {
                                      int CurBatch = BatchSizes[b];

                                      float[] Scores = new float[BatchLimit];

                                      for (int im = 0; im < NSymmetry; im++)
                                      {
                                          Matrix3 MSym = SymmetryMatrices[im];
                                          float3[] Angles = BatchRotations[b].Take(CurBatch).Select(a => Matrix3.EulerFromMatrix(a * MSym)).ToArray();

                                          if (!PerformAlignment)
                                              GPU.MultiParticleDiff(Scores,
                                                                      new[] { BatchParticlesMaskedUnsubtracted[b].GetDevice(Intent.Read) },
                                                                      Dim,
                                                                      null,
                                                                      new float[CurBatch * 2],
                                                                      Helper.ToInterleaved(Angles),
                                                                      new float3(1, 1, 0) * InterpolantProjector.Oversampling,
                                                                      DummySpectralWeights.GetDevice(Intent.Read),
                                                                      IntPtr.Zero,
                                                                      0,
                                                                      InterpolantProjector.Dims.X / 2,
                                                                      new[] { InterpolantProjector.t_DataRe, InterpolantProjector.t_DataRe },
                                                                      new[] { InterpolantProjector.t_DataIm, InterpolantProjector.t_DataIm },
                                                                      InterpolantProjector.Oversampling,
                                                                      InterpolantProjector.DimsOversampled.X,
                                                                      IntPtr.Zero,
                                                                      CurBatch,
                                                                      1);
                                          else
                                              (Scores, _, _) = OptimizePoses(BatchParticlesMaskedUnsubtracted[b], InterpolantProjector, InterpolantProjector, Angles, new float2[CurBatch]);
                                          
                                          for (int p = 0; p < CurBatch; p++)
                                              InterpolantScores[iinterp][(b * BatchLimit + p) * NSymmetry + im] = Scores[p];
                                      }
                                  }, null);

                        InterpolantProjector.Dispose();

                        ClearCurrentConsoleLine();
                        Console.Write($"Polishing: {iinterp + 1}/{NInterpolants}");
                    }
                    Console.Write("\n");

                    float[] GaussKernel = MathHelper.GetGaussianKernel1D(NInterpolants / 10, NInterpolants / 30f, false);

                    for (int p = 0; p < InterpolatedScores.Length; p++)
                    {
                        float[] Weights = InterpolantScores.Select(a => a[p]).ToArray();
                        //float MinWeight = MathHelper.Min(Weights);
                        //Weights = Weights.Select(v => v - MinWeight).ToArray();
                        Weights = MathHelper.ConvolveWithKernel1D(Weights, GaussKernel);

                        var MaxElement = MathHelper.MaxElement(Weights);

                        //float Coord = 0;
                        //float WeightSum = 0;
                        //for (int i = Math.Max(0, MaxElement.id - 1); i <= Math.Min(Weights.Length - 1, MaxElement.id + 1); i++)
                        //{
                        //    Coord += InterpStops[i] * Weights[i];
                        //    WeightSum += Weights[i];
                        //}

                        InterpolatedScores[p] = InterpStops[MaxElement.id];// Coord / WeightSum;
                    }
                }

                {                    
                    MathHelper.NormalizeL2InPlace(PC);
                    AllComponents.Add(PC);

                    //MathHelper.NormalizeL2InPlace(PCHard);
                    //AllComponentsHard.Add(PCHard);

                    for (int i = 0; i < MaskIndices.Length; i++)
                        VolumeData[MaskIndices[i]] = PC[i];

                    Image Volume = new Image(VolumeData, new int3(Dim));
                    Volume.Bandpass(0, 1, true);
                    Image VolumeFT = Volume.AsFFT(true);
                    Volume.Dispose();
                    float VolumeL2 = (float)Math.Sqrt(VolumeFT.GetHostContinuousCopy().Select(v => v * v).Sum() * 2);
                    VolumeFT.Multiply(1 / VolumeL2 / Dim);
                    Volume = VolumeFT.AsIFFT(true);
                    VolumeFT.Dispose();

                    Volume.WriteMRC(WorkingDirectory + $"pc_{(icomponent + 1):D2}.mrc", true);
                    Volume.Dispose();

                    float NormFactor = 1f / Dim;// /= Dim * Dim;


                    if (!PerformPolish)
                    {
                        float[] OrderedWeights = new float[NParticles * NSymmetry];
                        for (int p = 0; p < NParticles; p++)
                        {
                            int b = p / BatchLimit;
                            int bp = p % BatchLimit;
                            int r = BatchOriginalRows[b][bp];

                            for (int s = 0; s < NSymmetry; s++)
                                OrderedWeights[r * NSymmetry + s] = AllScores[p * NSymmetry + s];
                        }

                        WeightsTable.AddColumn($"wrpPCA{(icomponent + 1):D2}", OrderedWeights.Select(v => (v / NormFactor).ToString(CultureInfo.InvariantCulture)).ToArray());
                    }
                    else
                    {
                        float[] OrderedWeights = new float[NParticles * NSymmetry];
                        for (int p = 0; p < NParticles; p++)
                        {
                            int b = p / BatchLimit;
                            int bp = p % BatchLimit;
                            int r = BatchOriginalRows[b][bp];

                            for (int s = 0; s < NSymmetry; s++)
                                OrderedWeights[r * NSymmetry + s] = InterpolatedScores[p * NSymmetry + s];
                        }

                        WeightsTable.AddColumn($"wrpPCA{(icomponent + 1):D2}", OrderedWeights.Select(v => (v / NormFactor).ToString(CultureInfo.InvariantCulture)).ToArray());
                    }

                    Star.SaveMultitable(WorkingDirectory + "3dpca.star", new Dictionary<string, Star>{
                        { "settings", SettingsTable},
                        { "weights", WeightsTable}
                    });

                    Console.WriteLine(GPU.GetFreeMemory(DeviceID) + " MB");
                }
            }


            //Console.Read();
        }

        public static (float[] scores, float2[] shifts, float3[] angles) OptimizePoses(Image particlesFT, 
                                                                                       Projector ref1, 
                                                                                       Projector ref2, 
                                                                                       float3[] angles, 
                                                                                       float2[] shifts)
        {
            int NPoses = angles.Length;
            int IterationsLeft = 10;

            double[] Params = new double[NPoses * 5];
            float3[] CurrentAngles = new float3[NPoses];
            float2[] CurrentShifts = new float2[NPoses];
            float[] CurrentScores = new float[NPoses];

            Action<double[]> SetFromVector = input =>
            {
                for (int p = 0; p < NPoses; p++)
                {
                    CurrentAngles[p] = angles[p] + new float3((float)input[p * 5 + 0],
                                                              (float)input[p * 5 + 1],
                                                              (float)input[p * 5 + 2]);
                    CurrentShifts[p] = shifts[p] + new float2((float)input[p * 5 + 3],
                                                              (float)input[p * 5 + 4]);
                }
            };

            Action<double[]> UpdateScores = input =>
            {
                SetFromVector(input);

                GPU.MultiParticleDiff(CurrentScores,
                                        new[] { particlesFT.GetDevice(Intent.Read) },
                                        Dim,
                                        null,
                                        Helper.ToInterleaved(CurrentShifts),
                                        Helper.ToInterleaved(CurrentAngles),
                                        new float3(1, 1, 0),
                                        DummySpectralWeights.GetDevice(Intent.Read),
                                        IntPtr.Zero,
                                        0,
                                        ref1.Dims.X / 2,
                                        new[] { ref1.t_DataRe, ref2.t_DataRe },
                                        new[] { ref1.t_DataIm, ref2.t_DataIm },
                                        ref1.Oversampling,
                                        ref1.DimsOversampled.X,
                                        IntPtr.Zero,
                                        NPoses,
                                        1);
            };

            Func<double[], double> Eval = input =>
            {
                UpdateScores(input);

                double Sum = 0;
                for (int p = 0; p < NPoses; p++)
                    Sum += CurrentScores[p];

                Sum *= 100;
                //Debug.WriteLine(Sum);

                return Sum;
            };

            Func<double[], double[]> Grad = input =>
            {
                double Delta = 0.001;
                double Delta2 = Delta * 2;

                double[] Result = new double[input.Length];

                if (IterationsLeft-- < 0)
                    return Result;

                for (int i = 0; i < 5; i++)
                {
                    double[] InputPlus = input.ToList().ToArray();
                    double[] InputMinus = input.ToList().ToArray();
                    for (int p = 0; p < NPoses; p++)
                    {
                        InputPlus[p * 5 + i] += Delta;
                        InputMinus[p * 5 + i] -= Delta;
                    }

                    UpdateScores(InputPlus);
                    float[] ScoresPlus = CurrentScores.ToList().ToArray();

                    UpdateScores(InputMinus);

                    for (int p = 0; p < NPoses; p++)
                        Result[p * 5 + i] = (ScoresPlus[p] - CurrentScores[p]) / Delta2 * 100;
                }

                return Result;
            };

            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(Params.Length, Eval, Grad);
            Optimizer.Maximize(Params);

            UpdateScores(Optimizer.Solution);

            return (CurrentScores, CurrentShifts, CurrentAngles);
        }

        public static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }
    }
}
