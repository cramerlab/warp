using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using Warp.Tools;

namespace Warp
{
    public class CubeNet : IDisposable
    {
        public  float PixelSize = 10;

        public static readonly int3 BoxDimensionsTrain = new int3(96);
        public static readonly int3 BoxDimensionsValidTrain = new int3(64);
        public static readonly int3 BoxDimensionsPredict = new int3(96);
        public static readonly int3 BoxDimensionsValidPredict = new int3(64);

        public readonly int BatchSize = 1;
        public readonly string ModelDir;
        public readonly int MaxThreads;
        public readonly int DeviceID;
        public readonly int NClasses = 2;

        private bool ForTraining;
        private TFSession Session;
        private TFGraph Graph;

        private TFTensor[] TensorMicTile, TensorMicTilePredict;
        private TFTensor[] TensorTrainingLabels;
        private TFTensor[] TensorTrainingWeights;
        private TFTensor[] TensorLearningRate;

        private TFOutput NodeInputMicTile, NodeInputMicTilePredict, NodeInputLabels, NodeInputWeights, NodeLearningRate;
        private TFOutput NodeOutputArgMax, NodeOutputSoftMax, NodeOutputLoss;
        private TFOutput NodeOpTrain;
        private long[][] ResultArgMax;
        private float[][] ResultSoftMax;
        private float[][] ResultLoss;

        private TFSession.Runner[] RunnerPrediction;
        private TFSession.Runner[] RunnerTraining;

        private bool IsDisposed = false;

        public CubeNet(string modelDir, int deviceID = 0, int nThreads = 1, int batchSize = 1, int nClasses = 2, bool forTraining = false)
        {
            lock (TFHelper.DeviceSync[deviceID])
            {
                DeviceID = deviceID;
                ForTraining = forTraining;
                ModelDir = modelDir;
                MaxThreads = nThreads;
                BatchSize = batchSize;
                NClasses = nClasses;

                TFSessionOptions SessionOptions = TFHelper.CreateOptions();
                TFSession Dummy = new TFSession(new TFGraph(), SessionOptions);

                Session = TFHelper.FromSavedModel(SessionOptions, null, ModelDir, new[] { forTraining ? "train" : "serve" }, new TFGraph(), $"/device:GPU:{deviceID}");
                Graph = Session.Graph;

                if (forTraining)
                {
                    NodeInputMicTile = Graph["images"][0];
                    NodeInputLabels = Graph["image_classes"][0];
                    NodeInputWeights = Graph["image_weights"][0];
                    NodeLearningRate = Graph["training_learning_rate"][0];
                    NodeOpTrain = Graph["train_momentum"][0];

                    NodeOutputLoss = Graph["cross_entropy"][0];
                }
                else
                {
                    NodeInputMicTilePredict = Graph["images_predict"][0];
                }

                NodeOutputArgMax = Graph["argmax_tensor"][0];
                NodeOutputSoftMax = Graph["softmax_tensor"][0];

                if (forTraining)
                {
                    TensorMicTile = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsTrain.X, BoxDimensionsTrain.Y, BoxDimensionsTrain.Z, 1),
                                                                                    new float[BatchSize * BoxDimensionsTrain.Elements()],
                                                                                    0,
                                                                                    BatchSize * (int)BoxDimensionsTrain.Elements()),
                                                           nThreads);

                    TensorTrainingLabels = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsTrain.X, BoxDimensionsTrain.Y, BoxDimensionsTrain.Z, NClasses),
                                                                                           new float[BatchSize * BoxDimensionsTrain.Elements() * NClasses],
                                                                                           0,
                                                                                           BatchSize * (int)BoxDimensionsTrain.Elements() * NClasses),
                                                                  nThreads);

                    TensorTrainingWeights = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsTrain.X, BoxDimensionsTrain.Y, BoxDimensionsTrain.Z, 1),
                                                                                            new float[BatchSize * BoxDimensionsTrain.Elements()],
                                                                                            0,
                                                                                            BatchSize * (int)BoxDimensionsTrain.Elements()),
                                                                   nThreads);

                    TensorLearningRate = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(1),
                                                                                         new float[1],
                                                                                         0,
                                                                                         1),
                                                                nThreads);
                }
                else
                {

                    TensorMicTilePredict = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsPredict.X, BoxDimensionsPredict.Y, BoxDimensionsPredict.Z, 1),
                                                                                           new float[BatchSize * BoxDimensionsPredict.Elements()],
                                                                                           0,
                                                                                           BatchSize * (int)BoxDimensionsPredict.Elements()),
                                                                  nThreads);
                }

                if (forTraining)
                {
                    ResultArgMax = Helper.ArrayOfFunction(i => new long[BatchSize * (int)BoxDimensionsTrain.Elements()], nThreads);
                    ResultSoftMax = Helper.ArrayOfFunction(i => new float[BatchSize * (int)BoxDimensionsTrain.Elements() * NClasses], nThreads);
                    ResultLoss = Helper.ArrayOfFunction(i => new float[BatchSize], nThreads);
                }
                else
                {
                    ResultArgMax = Helper.ArrayOfFunction(i => new long[BatchSize * (int)BoxDimensionsPredict.Elements()], nThreads);
                    ResultSoftMax = Helper.ArrayOfFunction(i => new float[BatchSize * (int)BoxDimensionsPredict.Elements() * NClasses], nThreads);
                }

                if (!ForTraining)
                    RunnerPrediction = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                                   AddInput(NodeInputMicTilePredict, TensorMicTilePredict[i]).
                                                                   Fetch(NodeOutputArgMax, NodeOutputSoftMax),
                                                              nThreads);
                if (ForTraining)
                    RunnerTraining = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                                 AddInput(NodeInputMicTile, TensorMicTile[i]).
                                                                 AddInput(NodeInputLabels, TensorTrainingLabels[i]).
                                                                 AddInput(NodeInputWeights, TensorTrainingWeights[i]).
                                                                 AddInput(NodeLearningRate, TensorLearningRate[i]).
                                                                 Fetch(NodeOpTrain, NodeOutputArgMax, NodeOutputSoftMax, NodeOutputLoss),
                                                            nThreads);
            }

            // Run prediction or training for one batch to claim all the memory needed
            long[] InitArgMax;
            float[] InitProb;
            if (!ForTraining)
                Predict(new float[BoxDimensionsPredict.Elements() * BatchSize],
                        0,
                        out InitArgMax,
                        out InitProb);
            if (ForTraining)
            {
                RandomNormal RandN = new RandomNormal();
                Train(Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BatchSize * (int)BoxDimensionsTrain.Elements()),
                      Helper.ArrayOfConstant(0.0f, BatchSize * (int)BoxDimensionsTrain.Elements() * NClasses),
                      Helper.ArrayOfConstant(0.0f, BatchSize * (int)BoxDimensionsTrain.Elements()),
                      1e-6f,
                      0,
                      out InitArgMax,
                      out InitProb);
            }
        }

        public void Predict(float[] data, int threadID, out long[] argmax, out float[] probability)
        {
            //if (ForTraining)
            //    throw new Exception("Network was loaded in training mode, but asked to predict.");

            lock (TFHelper.DeviceSync[DeviceID])
            {
                Marshal.Copy(data, 0, TensorMicTilePredict[threadID].Data, BatchSize * (int)BoxDimensionsPredict.Elements());
                var Output = RunnerPrediction[threadID].Run();

                Marshal.Copy(Output[0].Data, ResultArgMax[threadID], 0, BatchSize * (int)BoxDimensionsPredict.Elements());
                Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsPredict.Elements() * NClasses);

                argmax = ResultArgMax[threadID];
                probability = ResultSoftMax[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();
            }
        }

        public void Predict(IntPtr d_data, int threadID, out long[] argmax, out float[] probability)
        {
            //if (ForTraining)
            //    throw new Exception("Network was loaded in training mode, but asked to predict.");

            lock (TFHelper.DeviceSync[DeviceID])
            {
                GPU.CopyDeviceToHostPinned(d_data, TensorMicTilePredict[threadID].Data, BatchSize * (int)BoxDimensionsPredict.Elements());
                var Output = RunnerPrediction[threadID].Run();

                Marshal.Copy(Output[0].Data, ResultArgMax[threadID], 0, BatchSize * (int)BoxDimensionsPredict.Elements());
                Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsPredict.Elements() * NClasses);

                argmax = ResultArgMax[threadID];
                probability = ResultSoftMax[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();
            }
        }

        public float Train(float[] data, float[] labels, float[] weights, float learningRate, int threadID, out long[] argmax, out float[] probability)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            lock (TFHelper.DeviceSync[DeviceID])
            {
                Marshal.Copy(data, 0, TensorMicTile[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements());
                Marshal.Copy(labels, 0, TensorTrainingLabels[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements() * NClasses);
                Marshal.Copy(weights, 0, TensorTrainingWeights[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements());
                Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

                var Output = RunnerTraining[threadID].Run();

                Marshal.Copy(Output[1].Data, ResultArgMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements());
                Marshal.Copy(Output[2].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements() * NClasses);
                Marshal.Copy(Output[3].Data, ResultLoss[threadID], 0, BatchSize);

                argmax = ResultArgMax[threadID];
                probability = ResultSoftMax[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();

                return MathHelper.Mean(ResultLoss[threadID]);
            }
        }

        public float Train(IntPtr d_data, IntPtr d_labels, IntPtr d_weights, float learningRate, int threadID, out long[] argmax, out float[] probability)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            lock (TFHelper.DeviceSync[DeviceID])
            {
                GPU.CopyDeviceToHostPinned(d_data, TensorMicTile[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements());
                GPU.CopyDeviceToHostPinned(d_labels, TensorTrainingLabels[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements() * NClasses);
                GPU.CopyDeviceToHostPinned(d_weights, TensorTrainingWeights[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements());

                Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

                var Output = RunnerTraining[threadID].Run();

                Marshal.Copy(Output[1].Data, ResultArgMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements());
                Marshal.Copy(Output[2].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements() * NClasses);
                Marshal.Copy(Output[3].Data, ResultLoss[threadID], 0, BatchSize);

                argmax = ResultArgMax[threadID];
                probability = ResultSoftMax[threadID];

                foreach (var tensor in Output)
                    tensor.Dispose();

                return MathHelper.Mean(ResultLoss[threadID]);
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

            if (new DirectoryInfo(newModelDir).FullName != new DirectoryInfo(ModelDir).FullName)
            {
                if (Directory.Exists(newModelDir + "variables"))
                    Directory.Delete(newModelDir + "variables", true);
                Directory.CreateDirectory(newModelDir + "variables");

                foreach (var fileName in Directory.EnumerateFiles(ModelDir))
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

        ~CubeNet()
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

                    if (TensorMicTile != null)
                        foreach (var tensor in TensorMicTile)
                            tensor.Dispose();
                    if (TensorMicTilePredict != null)
                        foreach (var tensor in TensorMicTilePredict)
                            tensor.Dispose();
                    if (TensorTrainingLabels != null)
                        foreach (var tensor in TensorTrainingLabels)
                            tensor.Dispose();
                    if (TensorTrainingWeights != null)
                        foreach (var tensor in TensorTrainingWeights)
                            tensor.Dispose();

                    Session.DeleteSession();
                }
            }
        }

        public float4[] Match(Image volume, float diameterAngstrom, float threshold, Func<int3, int, string, bool> progressCallback)
        {
            float PixelSizeBN = PixelSize;
            int3 DimsRegionBN = CubeNet.BoxDimensionsPredict;
            int3 DimsRegionValidBN = CubeNet.BoxDimensionsValidPredict;
            int BorderBN = (DimsRegionBN.X - DimsRegionValidBN.X) / 2;

            int3 DimsBN = volume.Dims;

            GPU.Normalize(volume.GetDevice(Intent.Read),
                          volume.GetDevice(Intent.Write),
                          (uint)volume.ElementsReal,
                          1);

            volume.FreeDevice();
            
            float[] Predictions = new float[DimsBN.Elements()];
            float[] Mask = new float[DimsBN.Elements()];


            {
                int3 DimsPositions = (DimsBN + DimsRegionValidBN - 1) / DimsRegionValidBN;
                float3 PositionStep = new float3(DimsBN - DimsRegionBN) / new float3(Math.Max(DimsPositions.X - 1, 1),
                                                                                     Math.Max(DimsPositions.Y - 1, 1),
                                                                                     Math.Max(DimsPositions.Z - 1, 1));

                int NPositions = (int)DimsPositions.Elements();

                int3[] Positions = new int3[NPositions];
                for (int p = 0; p < NPositions; p++)
                {
                    int Z = p / (DimsPositions.X * DimsPositions.Y);
                    int Y = (p - Z * DimsPositions.X * DimsPositions.Y) / DimsPositions.X;
                    int X = p % DimsPositions.X;
                    Positions[p] = new int3((int)(X * PositionStep.X + DimsRegionBN.X / 2),
                                            (int)(Y * PositionStep.Y + DimsRegionBN.Y / 2),
                                            (int)(Z * PositionStep.Z + DimsRegionBN.Z / 2));
                }

                float[][] PredictionTiles = Helper.ArrayOfFunction(i => new float[DimsRegionBN.Elements()], NPositions);
                
                int NGPUThreads = MaxThreads;

                Image[] Extracted = new Image[NGPUThreads];
                int DeviceID = GPU.GetDevice();

                int BatchesDone = 0;

                Helper.ForCPU(0, NPositions, NGPUThreads,

                              threadID =>
                              {
                                  GPU.SetDevice(DeviceID);
                                  Extracted[threadID] = new Image(IntPtr.Zero, DimsRegionBN);
                              },

                              (b, threadID) =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  #region Extract and normalize windows

                                  GPU.Extract(volume.GetDevice(Intent.Read),
                                              Extracted[threadID].GetDevice(Intent.Write),
                                              volume.Dims,
                                              DimsRegionBN,
                                              new int[] { Positions[b].X - DimsRegionBN.X / 2, Positions[b].Y - DimsRegionBN.Y / 2, Positions[b].Z - DimsRegionBN.Z / 2 },
                                              1);

                                  //GPU.Normalize(Extracted[threadID].GetDevice(Intent.Read),
                                  //              Extracted[threadID].GetDevice(Intent.Write),
                                  //              (uint)Extracted[threadID].ElementsReal,
                                  //              1);

                                  #endregion

                                  //Extracted[threadID].WriteMRC("d_extracted.mrc", true);

                                  #region Predict
                                  
                                  long[] BatchArgMax;
                                  float[] BatchProbability;
                                  Predict(Extracted[threadID].GetDevice(Intent.Read),
                                          GPUThreadID,
                                          out BatchArgMax,
                                          out BatchProbability);

                                  //new Image(BatchArgMax.Select(v => (float)v).ToArray(), new int3(DimsRegionBN)).WriteMRC("d_labels.mrc", true);

                                  for (int i = 0; i < BatchArgMax.Length; i++)
                                  {
                                      int Label = (int)BatchArgMax[i];
                                      float Probability = BatchProbability[i * NClasses + Label];

                                      PredictionTiles[b][i] = (Label >= 1 && Probability >= threshold ? Probability : 0);
                                  }

                                  #endregion

                                  lock (volume)
                                      progressCallback?.Invoke(new int3(NPositions, 1, 1), ++BatchesDone, "");
                              },

                              threadID =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  Extracted[threadID].Dispose();
                              });

                for (int z = 0; z < DimsBN.Z; z++)
                {
                    for (int y = 0; y < DimsBN.Y; y++)
                    {
                        for (int x = 0; x < DimsBN.X; x++)
                        {
                            int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - DimsRegionBN.X / 2) / PositionStep.X + 0.5f)));
                            int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - DimsRegionBN.Y / 2) / PositionStep.Y + 0.5f)));
                            int ClosestZ = (int)Math.Max(0, Math.Min(DimsPositions.Z - 1, (int)(((float)z - DimsRegionBN.Z / 2) / PositionStep.Z + 0.5f)));
                            int ClosestID = (ClosestZ * DimsPositions.Y + ClosestY) * DimsPositions.X + ClosestX;

                            int3 Position = Positions[ClosestID];
                            int LocalX = Math.Max(0, Math.Min(DimsRegionBN.X - 1, x - Position.X + DimsRegionBN.X / 2));
                            int LocalY = Math.Max(0, Math.Min(DimsRegionBN.Y - 1, y - Position.Y + DimsRegionBN.Y / 2));
                            int LocalZ = Math.Max(0, Math.Min(DimsRegionBN.Z - 1, z - Position.Z + DimsRegionBN.Z / 2));

                            Predictions[(z * DimsBN.Y + y) * DimsBN.X + x] = PredictionTiles[ClosestID][(LocalZ * DimsRegionBN.Y + LocalY) * DimsRegionBN.X + LocalX];
                        }
                    }
                }
                
                volume.FreeDevice();
            }

            #region Apply Gaussian and find peaks

            Image PredictionsImage = new Image(Predictions, new int3(DimsBN));
            PredictionsImage.WriteMRC("d_predictions.mrc", true);

            //Image PredictionsConvolved = PredictionsImage.AsConvolvedGaussian((float)options.ExpectedDiameter / PixelSizeBN / 6);
            //PredictionsConvolved.Multiply(PredictionsImage);
            //PredictionsImage.Dispose();

            //PredictionsImage.WriteMRC(MatchingDir + RootName + "_boxnet.mrc", PixelSizeBN, true);

            int3[] Peaks = PredictionsImage.GetLocalPeaks((int)(diameterAngstrom / PixelSizeBN / 4 + 0.5f), 1e-6f);
            PredictionsImage.Dispose();

            int BorderDist = (int)(diameterAngstrom / PixelSizeBN * 0.8f + 0.5f);
            Peaks = Peaks.Where(p => p.X > BorderDist && 
                                     p.Y > BorderDist && 
                                     p.Z > BorderDist && 
                                     p.X < DimsBN.X - BorderDist && 
                                     p.Y < DimsBN.Y - BorderDist &&
                                     p.Z < DimsBN.Z - BorderDist).ToArray();

            #endregion

            #region Label connected components and get centroids

            List<float3> Centroids;
            List<int> Extents;
            {
                List<List<int3>> Components = new List<List<int3>>();
                int[] PixelLabels = Helper.ArrayOfConstant(-1, Predictions.Length);

                foreach (var peak in Peaks)
                {
                    if (PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                        continue;

                    List<int3> Component = new List<int3>() { peak };
                    int CN = Components.Count;

                    PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                    Queue<int3> Expansion = new Queue<int3>(100);
                    Expansion.Enqueue(peak);

                    while (Expansion.Count > 0)
                    {
                        int3 pos = Expansion.Dequeue();
                        int PosElement = (int)DimsBN.ElementFromPosition(pos);

                        if (pos.X > 0 && Predictions[PosElement - 1] > 0 && PixelLabels[PosElement - 1] < 0)
                        {
                            PixelLabels[PosElement - 1] = CN;
                            Component.Add(pos + new int3(-1, 0, 0));
                            Expansion.Enqueue(pos + new int3(-1, 0, 0));
                        }
                        if (pos.X < DimsBN.X - 1 && Predictions[PosElement + 1] > 0 && PixelLabels[PosElement + 1] < 0)
                        {
                            PixelLabels[PosElement + 1] = CN;
                            Component.Add(pos + new int3(1, 0, 0));
                            Expansion.Enqueue(pos + new int3(1, 0, 0));
                        }

                        if (pos.Y > 0 && Predictions[PosElement - DimsBN.X] > 0 && PixelLabels[PosElement - DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement - DimsBN.X] = CN;
                            Component.Add(pos + new int3(0, -1, 0));
                            Expansion.Enqueue(pos + new int3(0, -1, 0));
                        }
                        if (pos.Y < DimsBN.Y - 1 && Predictions[PosElement + DimsBN.X] > 0 && PixelLabels[PosElement + DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement + DimsBN.X] = CN;
                            Component.Add(pos + new int3(0, 1, 0));
                            Expansion.Enqueue(pos + new int3(0, 1, 0));
                        }

                        if (pos.Z > 0 && Predictions[PosElement - DimsBN.X * DimsBN.Y] > 0 && PixelLabels[PosElement - DimsBN.X * DimsBN.Y] < 0)
                        {
                            PixelLabels[PosElement - DimsBN.X * DimsBN.Y] = CN;
                            Component.Add(pos + new int3(0, 0, -1));
                            Expansion.Enqueue(pos + new int3(0, 0, -1));
                        }
                        if (pos.Z < DimsBN.Z - 1 && Predictions[PosElement + DimsBN.X * DimsBN.Y] > 0 && PixelLabels[PosElement + DimsBN.X * DimsBN.Y] < 0)
                        {
                            PixelLabels[PosElement + DimsBN.X * DimsBN.Y] = CN;
                            Component.Add(pos + new int3(0, 0, 1));
                            Expansion.Enqueue(pos + new int3(0, 0, 1));
                        }
                    }

                    Components.Add(Component);
                }

                Centroids = Components.Select(c => MathHelper.Mean(c.Select(v => new float3(v)))).ToList();
                Extents = Components.Select(c => c.Count).ToList();
            }

            List<int> ToDelete = new List<int>();

            // Hit test with crap mask
            //for (int c1 = 0; c1 < Centroids.Count; c1++)
            //{
            //    float2 P1 = Centroids[c1];
            //    if (MaskHitTest[(int)P1.Y * DimsBN.X + (int)P1.X] == 0)
            //    {
            //        ToDelete.Add(c1);
            //        continue;
            //    }
            //}

            for (int c1 = 0; c1 < Centroids.Count - 1; c1++)
            {
                float3 P1 = Centroids[c1];

                for (int c2 = c1 + 1; c2 < Centroids.Count; c2++)
                {
                    if ((P1 - Centroids[c2]).Length() < diameterAngstrom / PixelSizeBN / 1.5f)
                    {
                        int D = Extents[c1] < Extents[c2] ? c1 : c2;

                        if (!ToDelete.Contains(D))
                            ToDelete.Add(D);
                    }
                }
            }

            ToDelete.Sort();
            for (int i = ToDelete.Count - 1; i >= 0; i--)
            {
                Centroids.RemoveAt(ToDelete[i]);
                Extents.RemoveAt(ToDelete[i]);
            }

            #endregion

            //new Image(Predictions, DimsBN).WriteMRC("d_predictions.mrc", true);

            #region Write peak positions and angles into table

            return Centroids.Select((c, i) => new float4(c.X, c.Y, c.Z, Extents[i])).ToArray();

            #endregion
        }
    }
}
