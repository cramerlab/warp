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
    public class NoiseNet2D : IDisposable
    {
        public readonly int2 BoxDimensions;
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

        public NoiseNet2D(string modelDir, int2 boxDimensions, int nThreads = 1, int batchSize = 8, bool forTraining = true, int deviceID = 0)
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

                TensorSource = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensions.X, BoxDimensions.Y, 1),
                                                                             new float[BatchSize * BoxDimensions.Elements()],
                                                                             0,
                                                                             BatchSize * (int)BoxDimensions.Elements()),
                                                    nThreads);

                if (ForTraining)
                {
                    TensorTarget = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensions.X, BoxDimensions.Y, 1),
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
                File.Copy(fileName, newModelDir + Helper.PathToNameWithExtension(fileName), true);

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

        ~NoiseNet2D()
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

                    if (TensorSource != null)
                        foreach (var tensor in TensorSource)
                            tensor.Dispose();
                    if (TensorTarget != null)
                        foreach (var tensor in TensorTarget)
                            tensor.Dispose();

                    Session?.DeleteSession();
                }
            }
        }

        public static void Denoise(Image noisy, NoiseNet2D[] networks)
        {
            int3 Dims = noisy.Dims;
            int Dim = networks[0].BoxDimensions.X;
            int BatchSize = networks[0].BatchSize;

            int2 DimsValid = new int2(Dim) / 2;

            int2 DimsPositions = (new int2(Dims) + DimsValid - 1) / DimsValid;
            float2 PositionStep = new float2(new int2(Dims) - new int2(Dim)) / new float2(Math.Max(DimsPositions.X - 1, 1),
                                                                                          Math.Max(DimsPositions.Y - 1, 1));

            int NPositions = (int)DimsPositions.Elements();

            int3[] Positions = new int3[NPositions];
            for (int p = 0; p < NPositions; p++)
            {
                int X = p % DimsPositions.X;
                int Y = p / DimsPositions.X;
                Positions[p] = new int3((int)(X * PositionStep.X + Dim / 2),
                                        (int)(Y * PositionStep.Y + Dim / 2),
                                        0);
            }

            float[][] PredictionTiles = new float[Positions.Length][];

            Image Extracted = new Image(new int3(Dim, Dim, BatchSize));

            for (int b = 0; b < Positions.Length; b += BatchSize)
            {
                int CurBatch = Math.Min(BatchSize, Positions.Length - b);

                int3[] CurPositions = Positions.Skip(b).Take(CurBatch).ToArray();
                GPU.Extract(noisy.GetDevice(Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            noisy.Dims,
                            new int3(Dim, Dim, 1),
                            Helper.ToInterleaved(CurPositions.Select(p => p - new int3(Dim / 2, Dim / 2, 0)).ToArray()),
                            (uint)CurBatch);

                float[] PredictionData = null;
                networks[0].Predict(Extracted.GetDevice(Intent.Read), 0, out PredictionData);

                for (int i = 0; i < CurBatch; i++)
                    PredictionTiles[b + i] = PredictionData.Skip(i * Dim * Dim).Take(Dim * Dim).ToArray();
            }

            Extracted.Dispose();

            float[] Denoised = noisy.GetHost(Intent.Write)[0];
            for (int y = 0; y < Dims.Y; y++)
            {
                for (int x = 0; x < Dims.X; x++)
                {
                    int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - Dim / 2) / PositionStep.X + 0.5f)));
                    int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - Dim / 2) / PositionStep.Y + 0.5f)));
                    int ClosestID = ClosestY * DimsPositions.X + ClosestX;

                    int3 Position = Positions[ClosestID];
                    int LocalX = Math.Max(0, Math.Min(Dim - 1, x - Position.X + Dim / 2));
                    int LocalY = Math.Max(0, Math.Min(Dim - 1, y - Position.Y + Dim / 2));

                    Denoised[y * Dims.X + x] = PredictionTiles[ClosestID][LocalY * Dim + LocalX];
                }
            }
        }
    }
}
