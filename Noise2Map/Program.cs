using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    class Program
    {
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            #region Command line options

            Options Options = new Options();
            string WorkingDirectory;

            string ProgramFolder = System.Reflection.Assembly.GetEntryAssembly().Location;
            ProgramFolder = ProgramFolder.Substring(0, Math.Max(ProgramFolder.LastIndexOf('\\'), ProgramFolder.LastIndexOf('/')) + 1);

            if (!Debugger.IsAttached)
            {
                Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);
                WorkingDirectory = Environment.CurrentDirectory + "/";
            }
            else
            {
                Options.Observation1Path = @"E:\sara_debug\TS_for_M\reconstruction\odd";
                Options.Observation2Path = @"E:\sara_debug\TS_for_M\reconstruction\even";
                Options.ObservationCombinedPath = @"E:\sara_debug\TS_for_M\reconstruction\even";
                Options.DenoiseSeparately = false;
                Options.MaskPath = @"";
                Options.OldModelName = "";
                Options.DontAugment = false;
                Options.DontFlatten = true;
                Options.Overflatten = 1.0f;
                Options.PixelSize = 15f;
                Options.Upsample = 1.0f;
                Options.Lowpass = -1;
                Options.KeepDimensions = true;
                Options.MaskOutput = false;
                Options.NIterations = 600;
                Options.BatchSize = 4;
                Options.GPUNetwork = 0;
                Options.GPUPreprocess = 1;
                WorkingDirectory = @"G:\localsharpen\";
            }

            if (!Options.DontFlatten && Options.PixelSize < 0)
                throw new Exception("Flattening requested, but pixel size not specified.");

            #endregion

            GPU.SetDevice(Options.GPUPreprocess);

            #region Mask

            Console.Write("Loading mask... ");

            Image Mask = null;
            int3 BoundingBox = new int3(-1);
            if (!string.IsNullOrEmpty(Options.MaskPath))
            {
                Mask = Image.FromFile(Options.MaskPath);

                Mask.TransformValues((x, y, z, v) =>
                {
                    if (v > 1e-3f)
                    {
                        BoundingBox.X = Math.Max(BoundingBox.X, Math.Abs(x - Mask.Dims.X / 2) * 2);
                        BoundingBox.Y = Math.Max(BoundingBox.Y, Math.Abs(y - Mask.Dims.Y / 2) * 2);
                        BoundingBox.Z = Math.Max(BoundingBox.Z, Math.Abs(z - Mask.Dims.Z / 2) * 2);
                    }

                    return v;
                });

                if (BoundingBox.X < 2)
                    throw new Exception("Mask does not seem to contain any non-zero values.");

                BoundingBox += 64;

                BoundingBox.X = Math.Min(BoundingBox.X, Mask.Dims.X);
                BoundingBox.Y = Math.Min(BoundingBox.Y, Mask.Dims.Y);
                BoundingBox.Z = Math.Min(BoundingBox.Z, Mask.Dims.Z);
            }

            Console.WriteLine("done.\n");

            #endregion

            #region Load and prepare data

            Console.WriteLine("Preparing data:");

            List<Image> Maps1 = new List<Image>();
            List<Image> Maps2 = new List<Image>();
            List<Image> MapsForDenoising = new List<Image>();
            List<Image> MapsForDenoising2 = new List<Image>();
            List<string> NamesForDenoising = new List<string>();
            List<int3> DimensionsForDenoising = new List<int3>();
            List<int3> OriginalBoxForDenoising = new List<int3>();
            List<float2> MeanStdForDenoising = new List<float2>();
            List<float> PixelSizeForDenoising = new List<float>();

            foreach (var file in Directory.EnumerateFiles(Options.Observation1Path, "*.mrc"))
            {
                string MapName = Helper.PathToName(file);
                string[] Map2Paths = Directory.EnumerateFiles(Options.Observation2Path, MapName + ".mrc").ToArray();
                if (Map2Paths == null || Map2Paths.Length == 0)
                    continue;
                string MapCombinedPath = null;
                if (!string.IsNullOrEmpty(Options.ObservationCombinedPath))
                {
                    string[] MapCombinedPaths = Directory.EnumerateFiles(Options.ObservationCombinedPath, MapName + ".mrc").ToArray();
                    if (MapCombinedPaths == null || MapCombinedPaths.Length == 0)
                        continue;
                    MapCombinedPath = MapCombinedPaths.First();
                }

                Console.Write($"Preparing {MapName}... ");

                Image Map1 = Image.FromFile(file);
                Image Map2 = Image.FromFile(Map2Paths.First());
                Image MapCombined = MapCombinedPath == null ? null : Image.FromFile(MapCombinedPath);

                float MapPixelSize = Map1.PixelSize / (Options.KeepDimensions ? 1 : Options.Upsample);

                if (!Options.DontFlatten)
                {
                    Image Average = Map1.GetCopy();
                    Average.Add(Map2);

                    if (Mask != null)
                        Average.Multiply(Mask);

                    float[] Spectrum = Average.AsAmplitudes1D(true, 1, (Average.Dims.X + Average.Dims.Y + Average.Dims.Z) / 6);
                    Average.Dispose();

                    int i10A = (int)(Options.PixelSize * 2 / 10 * Spectrum.Length);
                    float Amp10A = Spectrum[i10A];

                    for (int i = 0; i < Spectrum.Length; i++)
                        Spectrum[i] = i < i10A ? 1 : (float)Math.Pow(Amp10A / Spectrum[i], Options.Overflatten);

                    Image Map1Flat = Map1.AsSpectrumMultiplied(true, Spectrum);
                    Map1.Dispose();
                    Map1 = Map1Flat;
                    Map1.FreeDevice();

                    Image Map2Flat = Map2.AsSpectrumMultiplied(true, Spectrum);
                    Map2.Dispose();
                    Map2 = Map2Flat;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedFlat = MapCombined.AsSpectrumMultiplied(true, Spectrum);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedFlat;
                        MapCombined.FreeDevice();
                    }
                }

                if (Options.Lowpass > 0)
                {
                    Map1.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                    Map2.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                    MapCombined?.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                }

                //{
                //    int NShells = Map1.Dims.X / 2;
                //    float[] ResInv = Helper.ArrayOfFunction(i => Math.Min((int)(0.45 * NShells), i) / (Map1.Dims.X * MapPixelSize), NShells);
                //    float[] FilterSharpen = new float[NShells];
                //    for (int i = 0; i < NShells; i++)
                //        FilterSharpen[i] = (float)Math.Exp(100 / 4 * ResInv[i] * ResInv[i]);

                //    Image Map1Sharp = FSC.ApplyRamp(Map1, FilterSharpen);
                //    Map1.Dispose();
                //    Map1 = Map1Sharp;

                //    Image Map2Sharp = FSC.ApplyRamp(Map2, FilterSharpen);
                //    Map2.Dispose();
                //    Map2 = Map2Sharp;
                //}

                OriginalBoxForDenoising.Add(Map1.Dims);

                if (BoundingBox.X > 0)
                {
                    Image Map1Cropped = Map1.AsPadded(BoundingBox);
                    Map1.Dispose();
                    Map1 = Map1Cropped;
                    Map1.FreeDevice();

                    Image Map2Cropped = Map2.AsPadded(BoundingBox);
                    Map2.Dispose();
                    Map2 = Map2Cropped;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedCropped = MapCombined.AsPadded(BoundingBox);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedCropped;
                        MapCombined.FreeDevice();
                    }
                }

                DimensionsForDenoising.Add(Map1.Dims);

                if (Options.Upsample != 1f)
                {
                    Image Map1Scaled = Map1.AsScaled(Map1.Dims * Options.Upsample / 2 * 2);
                    Map1.Dispose();
                    Map1 = Map1Scaled;
                    Map1.FreeDevice();

                    Image Map2Scaled = Map2.AsScaled(Map2.Dims * Options.Upsample / 2 * 2);
                    Map2.Dispose();
                    Map2 = Map2Scaled;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedScaled = MapCombined.AsScaled(Map2.Dims * Options.Upsample / 2 * 2);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedScaled;
                        MapCombined.FreeDevice();
                    }
                }

                float2 MeanStd = MathHelper.MeanAndStd(Helper.Combine(Map1.GetHostContinuousCopy(), Map2.GetHostContinuousCopy()));
                MeanStdForDenoising.Add(MeanStd);

                Map1.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                Map2.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                MapCombined?.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);

                Image ForDenoising = (MapCombined == null || Options.DenoiseSeparately) ? Map1.GetCopy() : MapCombined;
                Image ForDenoising2 = Options.DenoiseSeparately ? Map2.GetCopy() : null;

                GPU.PrefilterForCubic(Map1.GetDevice(Intent.ReadWrite), Map1.Dims);
                Map1.FreeDevice();
                Maps1.Add(Map1);

                if (!Options.DenoiseSeparately)
                {
                    ForDenoising.Add(Map2);
                    ForDenoising.Multiply(0.5f);
                }

                GPU.PrefilterForCubic(Map2.GetDevice(Intent.ReadWrite), Map2.Dims);
                Map2.FreeDevice();
                Maps2.Add(Map2);

                ForDenoising.FreeDevice();
                MapsForDenoising.Add(ForDenoising);
                NamesForDenoising.Add(MapName);

                PixelSizeForDenoising.Add(MapPixelSize);

                if (Options.DenoiseSeparately)
                {
                    ForDenoising2.FreeDevice();
                    MapsForDenoising2.Add(ForDenoising2);
                }

                Console.WriteLine(" Done.");
            }

            Mask?.FreeDevice();

            if (Maps1.Count == 0)
                throw new Exception("No maps were found. Please make sure the paths are correct and the names are consistent between the two observations.");

            Console.WriteLine("");

            #endregion

            NoiseNet3D TrainModel = null;
            string NameTrainedModel = Options.OldModelName;
            int Dim = 64;

            if (Options.BatchSize != 4 || Maps1.Count > 1)
            {
                if (Options.BatchSize < 1)
                    throw new Exception("Batch size must be at least 1.");

                Options.NIterations = Options.NIterations * 4 / Options.BatchSize / Maps1.Count;
                Console.WriteLine($"Adjusting the number of iterations to {Options.NIterations} to match batch size and number of maps.\n");
            }

            if (string.IsNullOrEmpty(Options.OldModelName))
            {
                #region Load model

                string ModelPath = Options.StartModelName;
                if (!Directory.Exists(ModelPath))
                    ModelPath = Path.Combine(ProgramFolder, Options.StartModelName);
                if (!Directory.Exists(ModelPath))
                    throw new Exception($"Could not find initial model '{Options.StartModelName}'. Please make sure it can be found either here, or in the installation directory.");

                Console.WriteLine("Loading model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB free.");
                TrainModel = new NoiseNet3D(ModelPath, new int3(Dim), 1, Options.BatchSize, true, Options.GPUNetwork);
                Console.WriteLine("Loaded model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB remaining.\n");

                #endregion

                GPU.SetDevice(Options.GPUPreprocess);

                #region Training

                Random Rand = new Random(123);

                int NMaps = Maps1.Count;
                int NMapsPerBatch = Math.Min(128, NMaps);
                int MapSamples = Options.BatchSize;

                Image[] ExtractedSource = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Dim * MapSamples)), NMapsPerBatch);
                Image[] ExtractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Dim * MapSamples)), NMapsPerBatch);

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                Queue<float> Losses = new Queue<float>();

                for (int iter = 0; iter < Options.NIterations; iter++)
                {
                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMaps, 1), NMapsPerBatch, Rand.Next(9999));

                    for (int m = 0; m < NMapsPerBatch; m++)
                    {
                        int MapID = ShuffledMapIDs[m];

                        Image Map1 = Maps1[MapID];
                        Image Map2 = Maps2[MapID];

                        int3 DimsMap = Map1.Dims;

                        int3 Margin = new int3((int)(Dim / 2 * 1.5f));
                        //Margin.Z = 0;
                        float3[] Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Z - Margin.Z * 2) + Margin.Z), MapSamples);

                        float3[] Angle;
                        if (Options.DontAugment)
                            Angle = Helper.ArrayOfFunction(i => new float3((float)Math.Round(Rand.NextDouble()) * 180,
                                                                           (float)Math.Round(Rand.NextDouble()) * 180,
                                                                           (float)Math.Round(Rand.NextDouble()) * 180) * Helper.ToRad, MapSamples);
                        else
                            Angle = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 360,
                                                                           (float)Rand.NextDouble() * 360,
                                                                           (float)Rand.NextDouble() * 360) * Helper.ToRad, MapSamples);

                        {
                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            GPU.CreateTexture3D(Map1.GetDevice(Intent.Read), Map1.Dims, Texture, TextureArray, true);
                            //Map1.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture[0],
                                                  Map1.Dims,
                                                  ExtractedSource[m].GetDevice(Intent.Write),
                                                  new int3(Dim),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedSource[MapID].WriteMRC("d_extractedsource.mrc", true);

                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        {
                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            GPU.CreateTexture3D(Map2.GetDevice(Intent.Read), Map2.Dims, Texture, TextureArray, true);
                            //Map2.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture[0],
                                                  Map2.Dims,
                                                  ExtractedTarget[m].GetDevice(Intent.Write),
                                                  new int3(Dim),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedTarget.WriteMRC("d_extractedtarget.mrc", true);

                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        //Map1.FreeDevice();
                        //Map2.FreeDevice();
                    }

                    float[] PredictedData = null, Loss = null;

                    {
                        double CurrentLearningRate = Math.Exp(MathHelper.Lerp((float)Math.Log(Options.LearningRateStart),
                                                                              (float)Math.Log(Options.LearningRateFinish),
                                                                              iter / (float)Options.NIterations));

                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
                        {
                            int MapID = m;

                            bool Twist = Rand.Next(2) == 0;

                            if (Twist)
                                TrainModel.Train(ExtractedSource[MapID].GetDevice(Intent.Read),
                                                 ExtractedTarget[MapID].GetDevice(Intent.Read),
                                                 (float)CurrentLearningRate,
                                                 0,
                                                 out PredictedData,
                                                 out Loss);
                            else
                                TrainModel.Train(ExtractedTarget[MapID].GetDevice(Intent.Read),
                                                 ExtractedSource[MapID].GetDevice(Intent.Read),
                                                 (float)CurrentLearningRate,
                                                 0,
                                                 out PredictedData,
                                                 out Loss);

                            Losses.Enqueue(Loss[0]);
                            if (Losses.Count > 100)
                                Losses.Dequeue();
                        }
                    }

                    double TicksPerIteration = Watch.ElapsedTicks / (double)(iter + 1);
                    TimeSpan TimeRemaining = new TimeSpan((long)(TicksPerIteration * (Options.NIterations - 1 - iter)));

                    ClearCurrentConsoleLine();
                    Console.Write($"{iter + 1}/{Options.NIterations}, {TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}");

                    if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");
                }

                Watch.Stop();

                NameTrainedModel = Options.StartModelName + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss");
                TrainModel.Export(NameTrainedModel);
                TrainModel.Dispose();

                TFHelper.TF_FreeAllMemory();

                Console.WriteLine("\nDone training!\n");

                #endregion
            }

            #region Denoise

            Console.WriteLine("Loading trained model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB free.");
            TrainModel = new NoiseNet3D(NameTrainedModel, new int3(Dim), 1, Options.BatchSize, false, Options.GPUNetwork);
            //TrainModel = new NoiseNet3D(@"H:\denoise_refine\noisenet3d_64_20180808_010023", new int3(Dim), 1, Options.BatchSize, false, Options.GPUNetwork);
            Console.WriteLine("Loaded trained model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB remaining.\n");

            //Directory.Delete(NameTrainedModel, true);

            Directory.CreateDirectory("denoised");

            GPU.SetDevice(Options.GPUPreprocess);

            for (int imap = 0; imap < MapsForDenoising.Count; imap++)
            {
                Console.Write($"Denoising {NamesForDenoising[imap]}... ");

                Image Map1 = MapsForDenoising[imap];
                NoiseNet3D.Denoise(Map1, new NoiseNet3D[] { TrainModel });

                float2 MeanStd = MeanStdForDenoising[imap];

                Map1.TransformValues(v => v * MeanStd.Y);

                if (Options.KeepDimensions)
                {
                    if (DimensionsForDenoising[imap] != Map1.Dims)
                    {
                        Image Scaled = Map1.AsScaled(DimensionsForDenoising[imap]);
                        Map1.Dispose();
                        Map1 = Scaled;
                    }
                    if (OriginalBoxForDenoising[imap] != Map1.Dims)
                    {
                        Image Padded = Map1.AsPadded(OriginalBoxForDenoising[imap]);
                        Map1.Dispose();
                        Map1 = Padded;
                    }
                }
                Map1.PixelSize = PixelSizeForDenoising[imap];

                Map1.TransformValues(v => v + MeanStd.X);

                if (Options.Lowpass > 0)
                    Map1.Bandpass(0, Map1.PixelSize * 2 / Options.Lowpass, true, 0.01f);

                if (Options.KeepDimensions && Options.MaskOutput)
                    Map1.Multiply(Mask);

                string SavePath1 = "denoised/" + NamesForDenoising[imap] + (Options.DenoiseSeparately ? "_1" : "") + ".mrc";
                Map1.WriteMRC(SavePath1, true);
                Map1.Dispose();

                Console.WriteLine("Done. Saved to " + SavePath1);

                if (Options.DenoiseSeparately)
                {
                    Console.Write($"Denoising {NamesForDenoising[imap]} (2nd observation)... ");

                    Image Map2 = MapsForDenoising2[imap];
                    NoiseNet3D.Denoise(Map2, new NoiseNet3D[] { TrainModel });

                    Map2.TransformValues(v => v * MeanStd.Y);

                    if (Options.KeepDimensions)
                    {
                        if (DimensionsForDenoising[imap] != Map2.Dims)
                        {
                            Image Scaled = Map2.AsScaled(DimensionsForDenoising[imap]);
                            Map2.Dispose();
                            Map2 = Scaled;
                        }
                        if (OriginalBoxForDenoising[imap] != Map2.Dims)
                        {
                            Image Padded = Map2.AsPadded(OriginalBoxForDenoising[imap]);
                            Map2.Dispose();
                            Map2 = Padded;
                        }
                    }
                    Map2.PixelSize = PixelSizeForDenoising[imap];

                    Map2.TransformValues(v => v + MeanStd.X);

                    if (Options.Lowpass > 0)
                        Map2.Bandpass(0, Map2.PixelSize * 2 / Options.Lowpass, true, 0.01f);

                    if (Options.KeepDimensions && Options.MaskOutput)
                        Map2.Multiply(Mask);

                    string SavePath2 = "denoised/" + NamesForDenoising[imap] + "_2" + ".mrc";
                    Map2.WriteMRC(SavePath2, true);
                    Map2.Dispose();

                    Console.WriteLine("Done. Saved to " + SavePath2);
                }
            }

            Console.WriteLine("\nAll done!");

            #endregion
        }
        
        private static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }
    }
}
