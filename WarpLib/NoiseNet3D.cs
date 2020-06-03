using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using Warp.Tools;

namespace Warp
{
    public class NoiseNet3D : IDisposable
    {
        public readonly int3 BoxDimensions;
        public readonly float PixelSize = 8;
        public readonly int BatchSize = 8;
        public readonly int Dim = 128;
        public readonly string ModelDir;
        public readonly int MaxThreads;
        public readonly int DeviceID;

        private bool ForTraining;
        private TFSession Session;
        private TFGraph Graph;

        private TFTensor[] TensorSource;
        private TFTensor[] TensorTarget;
        private TFTensor[] TensorLearningRate;

        private TFOutput NodeInputSource, NodeInputTarget, NodeLearningRate;
        private TFOutput NodeOutputPredicted, NodeOutputLoss;
        private TFOutput NodeOpTrain;
        private float[][] ResultPredicted;
        private float[][] ResultLoss;

        private TFSession.Runner[] RunnerPrediction;
        private TFSession.Runner[] RunnerTraining;

        private bool IsDisposed = false;

        public NoiseNet3D(string modelDir, int3 boxDimensions, int nThreads = 1, int batchSize = 8, bool forTraining = true, int deviceID = 0)
        {
            lock (TFHelper.DeviceSync[deviceID])
            {
                DeviceID = deviceID;
                BoxDimensions = boxDimensions;
                ForTraining = forTraining;
                ModelDir = modelDir;
                MaxThreads = nThreads;
                BatchSize = batchSize;

                TFSessionOptions SessionOptions = TFHelper.CreateOptions();
                TFSession Dummy = new TFSession(new TFGraph(), SessionOptions);

                Session = TFHelper.FromSavedModel(SessionOptions, null, ModelDir, new[] { forTraining ? "train" : "serve" }, new TFGraph(), $"/device:GPU:{deviceID}");
                Graph = Session.Graph;

                NodeInputSource = Graph["volume_source"][0];
                if (forTraining)
                {
                    NodeInputTarget = Graph["volume_target"][0];
                    NodeLearningRate = Graph["training_learning_rate"][0];
                    NodeOpTrain = Graph["train_momentum"][0];
                    NodeOutputLoss = Graph["l2_loss"][0];
                }

                NodeOutputPredicted = Graph["volume_predict"][0];

                TensorSource = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensions.X, BoxDimensions.Y, boxDimensions.Z, 1),
                                                                             new float[BatchSize * BoxDimensions.Elements()],
                                                                             0,
                                                                             BatchSize * (int)BoxDimensions.Elements()),
                                                    nThreads);

                if (ForTraining)
                {
                    TensorTarget = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensions.X, BoxDimensions.Y, boxDimensions.Z, 1),
                                                                                   new float[BatchSize * BoxDimensions.Elements()],
                                                                                   0,
                                                                                   BatchSize * (int)BoxDimensions.Elements()),
                                                          nThreads);

                    TensorLearningRate = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(1),
                                                                                         new float[1],
                                                                                         0,
                                                                                         1),
                                                                nThreads);
                }

                ResultPredicted = Helper.ArrayOfFunction(i => new float[BatchSize * BoxDimensions.Elements()], nThreads);
                ResultLoss = Helper.ArrayOfFunction(i => new float[1], nThreads);

                //if (!ForTraining)
                RunnerPrediction = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                               AddInput(NodeInputSource, TensorSource[i]).
                                                               Fetch(NodeOutputPredicted),
                                                          nThreads);
                if (ForTraining)
                    RunnerTraining = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                                 AddInput(NodeInputSource, TensorSource[i]).
                                                                 AddInput(NodeInputTarget, TensorTarget[i]).
                                                                 AddInput(NodeLearningRate, TensorLearningRate[i]).
                                                                 Fetch(NodeOutputPredicted, NodeOutputLoss, NodeOpTrain),
                                                            nThreads);
            }

            // Run prediction or training for one batch to claim all the memory needed
            float[] InitDecoded;
            float[] InitLoss;
            //if (!ForTraining)
            {
                Predict(new float[BoxDimensions.Elements() * BatchSize],
                        0,
                        out InitDecoded);
            }
            if (ForTraining)
            {
                RandomNormal RandN = new RandomNormal();
                Train(Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BatchSize * (int)BoxDimensions.Elements()),
                      Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BatchSize * (int)BoxDimensions.Elements()),
                      1e-10f,
                      0,
                      out InitDecoded,
                      out InitLoss);
            }
        }

        public void Predict(float[] data, int threadID, out float[] prediction)
        {
            lock (TFHelper.DeviceSync[DeviceID])
            {
                //if (ForTraining)
                //    throw new Exception("Network was loaded in training mode, but asked to predict.");

                Marshal.Copy(data, 0, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
                var Output = RunnerPrediction[threadID].Run();

                Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.Elements());

                prediction = ResultPredicted[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();
            }
        }

        public void Predict(IntPtr d_data, int threadID, out float[] prediction)
        {
            lock (TFHelper.DeviceSync[DeviceID])
            {
                //if (ForTraining)
                //    throw new Exception("Network was loaded in training mode, but asked to predict.");

                GPU.CopyDeviceToHostPinned(d_data, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
                var Output = RunnerPrediction[threadID].Run();

                Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.Elements());

                prediction = ResultPredicted[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();
            }
        }

        public void Train(float[] source,
                          float[] target,
                          float learningRate,
                          int threadID,
                          out float[] prediction,
                          out float[] loss)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            lock (TFHelper.DeviceSync[DeviceID])
            {
                Marshal.Copy(source, 0, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
                Marshal.Copy(target, 0, TensorTarget[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
                Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

                var Output = RunnerTraining[threadID].Run();

                Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.Elements());

                Marshal.Copy(Output[1].Data, ResultLoss[threadID], 0, 1);

                prediction = ResultPredicted[threadID];
                loss = ResultLoss[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();
            }
        }

        public void Train(IntPtr d_source,
                          IntPtr d_target,
                          float learningRate,
                          int threadID,
                          out float[] prediction,
                          out float[] loss)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            lock (TFHelper.DeviceSync[DeviceID])
            {
                GPU.CopyDeviceToHostPinned(d_source, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToHostPinned(d_target, TensorTarget[threadID].Data, BatchSize * (int)BoxDimensions.Elements());

                Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

                var Output = RunnerTraining[threadID].Run();

                Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.Elements());

                Marshal.Copy(Output[1].Data, ResultLoss[threadID], 0, 1);

                prediction = ResultPredicted[threadID];
                loss = ResultLoss[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();
            }
        }

        public void Export(string newModelDir)
        {
            if (newModelDir.Last() != '/' && newModelDir.Last() != '\\')
                newModelDir += "/";

            TFOutput NodeSaver, NodeSaverPath;
            //if (!ForTraining)
            {
                NodeSaver = Graph["save_1/control_dependency"][0];
                NodeSaverPath = Graph["save_1/Const"][0];
            }
            //else
            //{
            //    NodeSaver = Graph["save_2/control_dependency"][0];
            //    NodeSaverPath = Graph["save_2/Const"][0];
            //}

            Directory.CreateDirectory(newModelDir);
            if (Directory.Exists(newModelDir + "variables"))
                Directory.Delete(newModelDir + "variables", true);
            Directory.CreateDirectory(newModelDir + "variables");

            foreach (var fileName in Directory.EnumerateFiles(ModelDir))
            {
                string Source = fileName;
                string Destination = newModelDir + Helper.PathToNameWithExtension(fileName);

                bool AreSame = false;
                try
                {
                    AreSame = Helper.NormalizePath(Source) == Helper.NormalizePath(Destination);
                }
                catch { }

                if (!AreSame)
                    File.Copy(fileName, newModelDir + Helper.PathToNameWithExtension(fileName), true);
            }

            TFTensor TensorPath = TFTensor.CreateString(Encoding.ASCII.GetBytes(newModelDir + "variables/variables"));
            var Runner = Session.GetRunner().AddInput(NodeSaverPath, TensorPath);
            Runner.Run(NodeSaver);

            if (Directory.EnumerateDirectories(newModelDir + "variables", "variables_temp*").Count() > 0)
            {
                string TempName = Directory.EnumerateDirectories(newModelDir + "variables", "variables_temp*").First();
                foreach (var oldPath in Directory.EnumerateFiles(TempName))
                {
                    string OldName = Helper.PathToNameWithExtension(oldPath);
                    string NewName = "variables" + OldName.Substring(OldName.IndexOf("."));
                    string NewPath = newModelDir + "variables/" + NewName;

                    File.Move(oldPath, NewPath);
                }
                Directory.Delete(TempName, true);
            }

            TensorPath.Dispose();
        }

        ~NoiseNet3D()
        {
            Dispose();
        }

        public void Dispose()
        {
            lock (this)
            {
                if (!IsDisposed)
                {
                    IsDisposed = true;

                    foreach (var tensor in TensorSource)
                        tensor.Dispose();
                    if (TensorTarget != null)
                        foreach (var tensor in TensorTarget)
                            tensor.Dispose();

                    Session.DeleteSession();
                }
            }
        }

        public static void Denoise(Image noisy, NoiseNet3D[] networks)
        {
            int GPUID = GPU.GetDevice();
            int NThreads = networks[0].MaxThreads;

            int3 Dims = noisy.Dims;
            int Dim = networks[0].BoxDimensions.X;
            int BatchSize = networks[0].BatchSize;

            int3 DimsValid = new int3(Dim) / 2;

            int3 DimsPositions = (Dims + DimsValid - 1) / DimsValid;
            float3 PositionStep = new float3(Dims - new int3(Dim)) / new float3(Math.Max(DimsPositions.X - 1, 1),
                                                                                Math.Max(DimsPositions.Y - 1, 1),
                                                                                Math.Max(DimsPositions.Z - 1, 1));

            int NPositions = (int)DimsPositions.Elements();

            int3[] Positions = new int3[NPositions];
            for (int p = 0; p < NPositions; p++)
            {
                int X = p % DimsPositions.X;
                int Y = (p % (int)DimsPositions.ElementsSlice()) / DimsPositions.X;
                int Z = p / (int)DimsPositions.ElementsSlice();
                Positions[p] = new int3((int)(X * PositionStep.X + Dim / 2),
                                        (int)(Y * PositionStep.Y + Dim / 2),
                                        (int)(Z * PositionStep.Z + Dim / 2));
            }

            float[][] PredictionTiles = new float[Positions.Length][];

            Image[] Extracted = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Dim * BatchSize)), NThreads);

            Helper.ForCPU(0, (Positions.Length + BatchSize - 1) / BatchSize, NThreads, 
                (threadID) => GPU.SetDevice(GPUID),
                (ib, threadID) =>
                //for (int b = 0; b < Positions.Length; b += BatchSize)
                {
                    int b = ib * BatchSize;
                    int CurBatch = Math.Min(BatchSize, Positions.Length - b);

                    int3[] CurPositions = Positions.Skip(b).Take(CurBatch).ToArray();
                    GPU.Extract(noisy.GetDevice(Intent.Read),
                                Extracted[threadID].GetDevice(Intent.Write),
                                noisy.Dims,
                                new int3(Dim),
                                Helper.ToInterleaved(CurPositions.Select(p => p - new int3(Dim / 2)).ToArray()),
                                (uint)CurBatch);

                    float[] PredictionData = null;
                    networks[0].Predict(Extracted[threadID].GetDevice(Intent.Read), threadID, out PredictionData);

                    for (int i = 0; i < CurBatch; i++)
                        PredictionTiles[b + i] = PredictionData.Skip(i * Dim * Dim * Dim).Take(Dim * Dim * Dim).ToArray();
                }, null);

            foreach (var item in Extracted)
                item.Dispose();

            noisy.FreeDevice();

            float[][] Denoised = noisy.GetHost(Intent.Write);
            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - Dim / 2) / PositionStep.X + 0.5f)));
                        int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - Dim / 2) / PositionStep.Y + 0.5f)));
                        int ClosestZ = (int)Math.Max(0, Math.Min(DimsPositions.Z - 1, (int)(((float)z - Dim / 2) / PositionStep.Z + 0.5f)));
                        int ClosestID = (ClosestZ * DimsPositions.Y + ClosestY) * DimsPositions.X + ClosestX;

                        int3 Position = Positions[ClosestID];
                        int LocalX = Math.Max(0, Math.Min(Dim - 1, x - Position.X + Dim / 2));
                        int LocalY = Math.Max(0, Math.Min(Dim - 1, y - Position.Y + Dim / 2));
                        int LocalZ = Math.Max(0, Math.Min(Dim - 1, z - Position.Z + Dim / 2));

                        Denoised[z][y * Dims.X + x] = PredictionTiles[ClosestID][(LocalZ * Dim + LocalY) * Dim + LocalX];
                    }
                }
            }
        }

        public static (Image[] Halves1, Image[] Halves2, float2[] Stats) TrainOnVolumes(NoiseNet3D network, 
                                                                                        Image[] halves1, 
                                                                                        Image[] halves2, 
                                                                                        Image[] masks, 
                                                                                        float angpix, 
                                                                                        float lowpass, 
                                                                                        float upsample, 
                                                                                        bool dontFlatten,
                                                                                        bool performTraining,
                                                                                        int niterations,
                                                                                        float startFrom, 
                                                                                        int batchsize,
                                                                                        int gpuprocess,
                                                                                        Action<string> progressCallback)
        {
            GPU.SetDevice(gpuprocess);

            #region Mask

            Debug.Write("Preparing mask... ");
            progressCallback?.Invoke("Preparing mask... ");

            int3[] BoundingBox = Helper.ArrayOfFunction(i => new int3(-1), halves1.Length);
            if (masks != null)
            {
                for (int i = 0; i < masks.Length; i++)
                {
                    Image Mask = masks[i];

                    Mask.TransformValues((x, y, z, v) =>
                    {
                        if (v > 0.5f)
                        {
                            BoundingBox[i].X = Math.Max(BoundingBox[i].X, Math.Abs(x - Mask.Dims.X / 2) * 2);
                            BoundingBox[i].Y = Math.Max(BoundingBox[i].Y, Math.Abs(y - Mask.Dims.Y / 2) * 2);
                            BoundingBox[i].Z = Math.Max(BoundingBox[i].Z, Math.Abs(z - Mask.Dims.Z / 2) * 2);
                        }

                        return v;
                    });

                    if (BoundingBox[i].X < 2)
                        throw new Exception("Mask does not seem to contain any non-zero values.");

                    BoundingBox[i] += 64;

                    BoundingBox[i].X = Math.Min(BoundingBox[i].X, Mask.Dims.X);
                    BoundingBox[i].Y = Math.Min(BoundingBox[i].Y, Mask.Dims.Y);
                    BoundingBox[i].Z = Math.Min(BoundingBox[i].Z, Mask.Dims.Z);
                }
            }

            Console.WriteLine("done.\n");

            #endregion

            #region Load and prepare data

            Console.WriteLine("Preparing data:");

            List<Image> Maps1 = new List<Image>();
            List<Image> Maps2 = new List<Image>();

            List<Image> HalvesForDenoising1 = new List<Image>();
            List<Image> HalvesForDenoising2 = new List<Image>();
            List<float2> StatsForDenoising = new List<float2>();

            for (int imap = 0; imap < halves1.Length; imap++)
            {
                Debug.Write($"Preparing map {imap}... ");
                progressCallback?.Invoke($"Preparing map {imap}... ");

                Image Map1 = halves1[imap];
                Image Map2 = halves2[imap];

                float MapPixelSize = Map1.PixelSize / upsample;

                if (!dontFlatten)
                {
                    Image Average = Map1.GetCopy();
                    Average.Add(Map2);

                    if (masks != null)
                        Average.Multiply(masks[imap]);

                    float[] Spectrum = Average.AsAmplitudes1D(true, 1, (Average.Dims.X + Average.Dims.Y + Average.Dims.Z) / 6);
                    Average.Dispose();

                    int i10A = Math.Min((int)(angpix * 2 / 10 * Spectrum.Length), Spectrum.Length - 1);
                    float Amp10A = Spectrum[i10A];

                    for (int i = 0; i < Spectrum.Length; i++)
                        Spectrum[i] = i < i10A ? 1 : (Amp10A / Math.Max(1e-10f, Spectrum[i]));

                    Image Map1Flat = Map1.AsSpectrumMultiplied(true, Spectrum);
                    Map1.FreeDevice();
                    Map1 = Map1Flat;
                    Map1.FreeDevice();

                    Image Map2Flat = Map2.AsSpectrumMultiplied(true, Spectrum);
                    Map2.FreeDevice();
                    Map2 = Map2Flat;
                    Map2.FreeDevice();
                }

                if (lowpass > 0)
                {
                    Map1.Bandpass(0, angpix * 2 / lowpass, true, 0.01f);
                    Map2.Bandpass(0, angpix * 2 / lowpass, true, 0.01f);
                }

                if (upsample != 1f)
                {
                    Image Map1Scaled = Map1.AsScaled(Map1.Dims * upsample / 2 * 2);
                    Map1.FreeDevice();
                    Map1 = Map1Scaled;
                    Map1.FreeDevice();

                    Image Map2Scaled = Map2.AsScaled(Map2.Dims * upsample / 2 * 2);
                    Map2.FreeDevice();
                    Map2 = Map2Scaled;
                    Map2.FreeDevice();
                }

                Image ForDenoising1 = Map1.GetCopy();
                Image ForDenoising2 = Map2.GetCopy();

                if (BoundingBox[imap].X > 0)
                {
                    Image Map1Cropped = Map1.AsPadded(BoundingBox[imap]);
                    Map1.FreeDevice();
                    Map1 = Map1Cropped;
                    Map1.FreeDevice();

                    Image Map2Cropped = Map2.AsPadded(BoundingBox[imap]);
                    Map2.FreeDevice();
                    Map2 = Map2Cropped;
                    Map2.FreeDevice();
                }

                float2 MeanStd = MathHelper.MeanAndStd(Helper.Combine(Map1.GetHostContinuousCopy(), Map2.GetHostContinuousCopy()));

                Map1.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                Map2.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);

                ForDenoising1.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                ForDenoising2.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);

                HalvesForDenoising1.Add(ForDenoising1);
                HalvesForDenoising2.Add(ForDenoising2);
                StatsForDenoising.Add(MeanStd);

                GPU.PrefilterForCubic(Map1.GetDevice(Intent.ReadWrite), Map1.Dims);
                Map1.FreeDevice();
                Maps1.Add(Map1);
                
                GPU.PrefilterForCubic(Map2.GetDevice(Intent.ReadWrite), Map2.Dims);
                Map2.FreeDevice();
                Maps2.Add(Map2);
                
                Debug.WriteLine(" Done.");
            }

            if (masks != null)
                foreach (var mask in masks)
                    mask.FreeDevice();

            #endregion
                       
            if (batchsize != 4 || Maps1.Count > 1)
            {
                if (batchsize < 1)
                    throw new Exception("Batch size must be at least 1.");

                niterations = niterations * 4 / batchsize / Maps1.Count;
                Console.WriteLine($"Adjusting the number of iterations to {niterations} to match batch size and number of maps.\n");
            }

            int Dim = network.BoxDimensions.X;

            progressCallback?.Invoke($"0/{niterations}");

            if (performTraining)
            {
                GPU.SetDevice(gpuprocess);

                #region Training

                Random Rand = new Random(123);

                int NMaps = Maps1.Count;
                int NMapsPerBatch = Math.Min(128, NMaps);
                int MapSamples = batchsize;

                Image[] ExtractedSource = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Dim * MapSamples)), NMapsPerBatch);
                Image[] ExtractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Dim * MapSamples)), NMapsPerBatch);

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                Queue<float> Losses = new Queue<float>();

                for (int iter = (int)(startFrom * niterations); iter < niterations; iter++)
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

                        float3[] Angle = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 360,
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
                        float CurrentLearningRate = 0.0001f * (float)Math.Pow(10, -iter / (float)niterations * 2);

                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
                        {
                            int MapID = m;

                            bool Twist = Rand.Next(2) == 0;

                            if (Twist)
                                network.Train(ExtractedSource[MapID].GetDevice(Intent.Read),
                                              ExtractedTarget[MapID].GetDevice(Intent.Read),
                                              CurrentLearningRate,
                                              0,
                                              out PredictedData,
                                              out Loss);
                            else
                                network.Train(ExtractedTarget[MapID].GetDevice(Intent.Read),
                                              ExtractedSource[MapID].GetDevice(Intent.Read),
                                              CurrentLearningRate,
                                              0,
                                              out PredictedData,
                                              out Loss);

                            Losses.Enqueue(Loss[0]);
                            if (Losses.Count > 100)
                                Losses.Dequeue();
                        }
                    }


                    double TicksPerIteration = Watch.ElapsedTicks / (double)(iter + 1);
                    TimeSpan TimeRemaining = new TimeSpan((long)(TicksPerIteration * (niterations - 1 - iter)));

                    string ProgressText = $"{iter + 1}/{niterations}, {TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}";

                    if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    Debug.WriteLine(ProgressText);
                    progressCallback?.Invoke(ProgressText);
                }

                Debug.WriteLine("\nDone training!\n");

                #endregion
            }

            return (HalvesForDenoising1.ToArray(), HalvesForDenoising2.ToArray(), StatsForDenoising.ToArray());
        }
    }
}
