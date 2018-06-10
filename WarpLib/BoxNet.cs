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
    public class BoxNet : IDisposable
    {
        public readonly int2 BoxDimensions = new int2(96);
        public readonly float PixelSize = 8;
        public readonly int BatchSize;
        public readonly string ModelDir;
        public readonly int MaxThreads;

        private bool ForTraining;
        private TFSession Session;
        private TFGraph Graph;

        private TFTensor[] TensorMicTile;
        private TFTensor[] TensorTrainingLabels;
        private TFTensor[] TensorLearningRate;

        private TFOutput NodeInputMicTile, NodeInputLabels, NodeLearningRate;
        private TFOutput NodeOutputArgMax, NodeOutputSoftMax;
        private TFOutput NodeOpTrain;
        private long[][] ResultArgMax;
        private float[][] ResultSoftMax;

        private TFSession.Runner[] RunnerPrediction;
        private TFSession.Runner[] RunnerTraining;

        private bool IsDisposed = false;

        public BoxNet(string modelDir, int gpuID = 0, int nThreads = 1, int batchSize = 128, bool forTraining = false)
        {
            ForTraining = forTraining;
            BatchSize = batchSize;
            ModelDir = modelDir;
            MaxThreads = nThreads;
            
            TFSessionOptions SessionOptions = TFHelper.CreateOptions();
            TFSession Dummy = new TFSession(new TFGraph(), SessionOptions);

            Session = TFHelper.FromSavedModel(SessionOptions, null, ModelDir, new[] { forTraining ? "train" : "serve" }, new TFGraph(), $"/device:GPU:{gpuID}");
            Graph = Session.Graph;

            NodeInputMicTile = Graph["mic_tiles"][0];
            if (forTraining)
            {
                NodeInputLabels = Graph["training_labels"][0];
                NodeLearningRate = Graph["training_learning_rate"][0];
                NodeOpTrain = Graph["train_momentum"][0];
            }

            NodeOutputArgMax = Graph["ArgMax"][0];
            NodeOutputSoftMax = Graph["softmax_tensor"][0];

            TensorMicTile = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, 1, BoxDimensions.Y, BoxDimensions.X),
                                                                            new float[BatchSize * BoxDimensions.Elements()],
                                                                            0,
                                                                            BatchSize * (int)BoxDimensions.Elements()),
                                                   nThreads);

            TensorTrainingLabels = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, 2),
                                                                                   new float[BatchSize * 2],
                                                                                   0,
                                                                                   BatchSize * 2),
                                                          nThreads);

            TensorLearningRate = Helper.ArrayOfFunction(i => new TFTensor(0.0f),
                                                        nThreads);

            ResultArgMax = Helper.ArrayOfFunction(i => new long[BatchSize], nThreads);
            ResultSoftMax = Helper.ArrayOfFunction(i => new float[BatchSize * 2], nThreads);

            if (!ForTraining)
                RunnerPrediction = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                               AddInput(NodeInputMicTile, TensorMicTile[i]).
                                                               Fetch(NodeOutputArgMax, NodeOutputSoftMax), 
                                                          nThreads);
            else
                RunnerTraining = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                             AddInput(NodeInputMicTile, TensorMicTile[i]).
                                                             AddInput(NodeInputLabels, TensorTrainingLabels[i]).
                                                             AddInput(NodeLearningRate, TensorLearningRate[i]).
                                                             Fetch(NodeOutputArgMax, NodeOutputSoftMax, NodeOpTrain), 
                                                        nThreads);

            // Run prediction or training for one batch to claim all the memory needed
            long[] InitArgMax;
            float[] InitProb;
            if (!ForTraining)
                Predict(new float[BoxDimensions.Elements() * BatchSize],
                        0,
                        out InitArgMax,
                        out InitProb);
            else
            {
                RandomNormal RandN = new RandomNormal();
                Train(Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BatchSize * (int)BoxDimensions.Elements()),
                      Helper.Combine(Helper.ArrayOfFunction(i => new[] { 1.0f, 0.0f }, 128)),
                      1e-6f,
                      0,
                      out InitArgMax,
                      out InitProb);
            }
        }

        public void Predict(float[] data, int threadID, out long[] argmax, out float[] probability)
        {
            if (ForTraining)
                throw new Exception("Network was loaded in training mode, but asked to predict.");

            Marshal.Copy(data, 0, TensorMicTile[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
            var Output = RunnerPrediction[threadID].Run();

            Marshal.Copy(Output[0].Data, ResultArgMax[threadID], 0, BatchSize);
            Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * 2);

            argmax = ResultArgMax[threadID];
            probability = ResultSoftMax[threadID];
            
            foreach (var tensor in Output)
                tensor.Dispose();
        }

        public void Predict(IntPtr d_data, int threadID, out long[] argmax, out float[] probability)
        {
            if (ForTraining)
                throw new Exception("Network was loaded in training mode, but asked to predict.");

            GPU.CopyDeviceToHostPinned(d_data, TensorMicTile[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
            var Output = RunnerPrediction[threadID].Run();

            Marshal.Copy(Output[0].Data, ResultArgMax[threadID], 0, BatchSize);
            Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * 2);

            argmax = ResultArgMax[threadID];
            probability = ResultSoftMax[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
        }

        public void Train(float[] data, float[] labels, float learningRate, int threadID, out long[] argmax, out float[] probability)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            Marshal.Copy(data, 0, TensorMicTile[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
            Marshal.Copy(labels, 0, TensorTrainingLabels[threadID].Data, BatchSize * 2);
            Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

            var Output = RunnerTraining[threadID].Run();

            Marshal.Copy(Output[0].Data, ResultArgMax[threadID], 0, BatchSize);
            Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * 2);

            argmax = ResultArgMax[threadID];
            probability = ResultSoftMax[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
        }

        public void Train(IntPtr d_data, float[] labels, float learningRate, int threadID, out long[] argmax, out float[] probability)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            GPU.CopyDeviceToHostPinned(d_data, TensorMicTile[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
            Marshal.Copy(labels, 0, TensorTrainingLabels[threadID].Data, BatchSize * 2);
            Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);

            var Output = RunnerTraining[threadID].Run();

            Marshal.Copy(Output[0].Data, ResultArgMax[threadID], 0, BatchSize);
            Marshal.Copy(Output[1].Data, ResultSoftMax[threadID], 0, BatchSize * 2);

            argmax = ResultArgMax[threadID];
            probability = ResultSoftMax[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
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

        ~BoxNet()
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

                    foreach (var tensor in TensorMicTile)
                        tensor.Dispose();
                    foreach (var tensor in TensorTrainingLabels)
                        tensor.Dispose();

                    Session.DeleteSession();
                }
            }
        }
    }
}
