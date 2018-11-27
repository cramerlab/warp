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
    public class BoxNet2 : IDisposable
    {
        public const float PixelSize = 8;

        public static readonly int2 BoxDimensionsTrain = new int2(256);
        public static readonly int2 BoxDimensionsValidTrain = new int2(128);
        public static readonly int2 BoxDimensionsPredict = new int2(256);
        public static readonly int2 BoxDimensionsValidPredict = new int2(128);

        public readonly int BatchSize = 1;
        public readonly string ModelDir;
        public readonly int MaxThreads;
        public readonly int DeviceID;

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

        public BoxNet2(string modelDir, int deviceID = 0, int nThreads = 1, int batchSize = 1, bool forTraining = false)
        {
            lock (TFHelper.DeviceSync[deviceID])
            {
                DeviceID = deviceID;
                ForTraining = forTraining;
                ModelDir = modelDir;
                MaxThreads = nThreads;
                BatchSize = batchSize;

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
                    TensorMicTile = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsTrain.X, BoxDimensionsTrain.Y, 1),
                                                                                    new float[BatchSize * BoxDimensionsTrain.Elements()],
                                                                                    0,
                                                                                    BatchSize * (int)BoxDimensionsTrain.Elements()),
                                                           nThreads);

                    TensorTrainingLabels = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsTrain.X, BoxDimensionsTrain.Y, 3),
                                                                                           new float[BatchSize * BoxDimensionsTrain.Elements() * 3],
                                                                                           0,
                                                                                           BatchSize * (int)BoxDimensionsTrain.Elements() * 3),
                                                                  nThreads);

                    TensorTrainingWeights = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsTrain.X, BoxDimensionsTrain.Y, 1),
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

                    TensorMicTilePredict = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BoxDimensionsPredict.X, BoxDimensionsPredict.Y, 1),
                                                                                           new float[BatchSize * BoxDimensionsPredict.Elements()],
                                                                                           0,
                                                                                           BatchSize * (int)BoxDimensionsPredict.Elements()),
                                                                  nThreads);
                }

                if (forTraining)
                {
                    ResultArgMax = Helper.ArrayOfFunction(i => new long[BatchSize * (int)BoxDimensionsTrain.Elements()], nThreads);
                    ResultSoftMax = Helper.ArrayOfFunction(i => new float[BatchSize * (int)BoxDimensionsTrain.Elements() * 3], nThreads);
                    ResultLoss = Helper.ArrayOfFunction(i => new float[BatchSize], nThreads);
                }
                else
                {
                    ResultArgMax = Helper.ArrayOfFunction(i => new long[BatchSize * (int)BoxDimensionsPredict.Elements()], nThreads);
                    ResultSoftMax = Helper.ArrayOfFunction(i => new float[BatchSize * (int)BoxDimensionsPredict.Elements() * 3], nThreads);
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
                      Helper.Combine(Helper.ArrayOfFunction(i => new[] { 1.0f, 0.0f, 0.0f }, BatchSize * (int)BoxDimensionsTrain.Elements())),
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
                Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsPredict.Elements() * 3);

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
                Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsPredict.Elements() * 3);

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
                Marshal.Copy(labels, 0, TensorTrainingLabels[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements() * 3);
                Marshal.Copy(weights, 0, TensorTrainingWeights[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements());
                Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

                var Output = RunnerTraining[threadID].Run();

                Marshal.Copy(Output[1].Data, ResultArgMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements());
                Marshal.Copy(Output[2].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements() * 3);
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
                GPU.CopyDeviceToHostPinned(d_labels, TensorTrainingLabels[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements() * 3);
                GPU.CopyDeviceToHostPinned(d_weights, TensorTrainingWeights[threadID].Data, BatchSize * (int)BoxDimensionsTrain.Elements());

                Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

                var Output = RunnerTraining[threadID].Run();

                Marshal.Copy(Output[1].Data, ResultArgMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements());
                Marshal.Copy(Output[2].Data, ResultSoftMax[threadID], 0, BatchSize * (int)BoxDimensionsTrain.Elements() * 3);
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

        ~BoxNet2()
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
    }
}
