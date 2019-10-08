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
    public class FlexNet3D : IDisposable
    {
        public readonly int3 BoxDimensions;
        public readonly float PixelSize = 8;
        public readonly int BatchSize = 512;
        public readonly int BottleneckWidth = 2;
        public readonly int NWeights0 = 64;
        public readonly int NLayers = 4;
        public readonly string ModelDir;
        public readonly int MaxThreads;

        private bool ForTraining;
        private TFSession Session;
        private TFGraph Graph;

        private TFTensor[] TensorSource;
        private TFTensor[] TensorTarget;
        private TFTensor[] TensorWeightSource;
        private TFTensor[] TensorWeightTarget;
        private TFTensor[] TensorCode;
        private TFTensor[] TensorLearningRate;
        private TFTensor[] TensorDropoutRate;
        private TFTensor[] TensorOrthogonalityRate;
        private TFTensor TensorWeights0;

        private TFOutput NodeInputSource, NodeInputTarget, NodeInputWeightSource, NodeInputWeightTarget, NodeLearningRate, NodeDropoutRate, NodeOrthogonalityRate;
        private TFOutput NodeBottleneck, NodeCode;
        private TFOutput NodeOutputPredicted, NodeOutputLoss, NodeOutputLossKL;
        private TFOutput NodeOpTrain;
        private TFOutput NodeWeights0, NodeWeights0Assign, NodeWeights0Input;
        private TFOutput NodeWeights1, NodeWeights1Assign, NodeWeights1Input;
        private float[][] ResultPredicted;
        private float[][] ResultBottleneck;
        private float[][] ResultLoss;
        private float[][] ResultLossKL;

        private float[] RetrievedWeights;

        private TFSession.Runner[] RunnerPrediction;
        private TFSession.Runner[] RunnerEncode;
        private TFSession.Runner[] RunnerTraining;
        private TFSession.Runner RunnerRetrieveWeights0, RunnerRetrieveWeights1;
        private TFSession.Runner RunnerAssignWeights0, RunnerAssignWeights1;

        private bool IsDisposed = false;

        public FlexNet3D(string modelDir, int3 boxDimensions, int gpuID = 0, int nThreads = 1, bool forTraining = true, int batchSize = 128, int bottleneckWidth = 2, int layerWidth = 64, int nlayers = 4)
        {
            BoxDimensions = boxDimensions;
            ForTraining = forTraining;
            BatchSize = batchSize;
            BottleneckWidth = bottleneckWidth;
            NWeights0 = layerWidth;
            NLayers = nlayers;
            ModelDir = modelDir;
            MaxThreads = nThreads;

            TFSessionOptions SessionOptions = TFHelper.CreateOptions();
            TFSession Dummy = new TFSession(new TFGraph(), SessionOptions);

            Session = TFHelper.FromSavedModel(SessionOptions, null, ModelDir, new[] { forTraining ? "train" : "serve" }, new TFGraph(), $"/device:GPU:{gpuID}");
            Graph = Session.Graph;

            NodeInputSource = Graph["volume_source"][0];
            NodeInputTarget = Graph["volume_target"][0];
            NodeInputWeightSource = Graph["volume_weight_source"][0];
            NodeInputWeightTarget = Graph["volume_weight_target"][0];
            NodeDropoutRate = Graph["training_dropout_rate"][0];
            if (forTraining)
            {
                NodeLearningRate = Graph["training_learning_rate"][0];
                NodeOrthogonalityRate = Graph["training_orthogonality"][0];
                NodeOpTrain = Graph["train_momentum"][0];
                NodeOutputLoss = Graph["l2_loss"][0];
                NodeOutputLossKL = Graph["kl_loss"][0];
                NodeBottleneck = Graph["bottleneck"][0];
            }

            NodeCode = Graph["volume_code"][0];

            NodeOutputPredicted = Graph["volume_predict"][0];

            NodeWeights0 = Graph["encoder_0/weights_0"][0];
            NodeWeights1 = Graph[$"decoder_{nlayers - 1}/weights_{nlayers - 1}"][0];
            if (forTraining)
            {
                NodeWeights0Assign = Graph["encoder_0/assign_layer0"][0];
                NodeWeights0Input = Graph["encoder_0/assign_layer0_values"][0];

                NodeWeights1Assign = Graph[$"decoder_{nlayers - 1}/assign_layer0"][0];
                NodeWeights1Input = Graph[$"decoder_{nlayers - 1}/assign_layer0_values"][0];
            }

            TensorSource = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, (BoxDimensions.X / 2 + 1), BoxDimensions.Y, BoxDimensions.Z, 2),
                                                                         new float[BatchSize * BoxDimensions.ElementsFFT() * 2],
                                                                         0,
                                                                         BatchSize * (int)BoxDimensions.ElementsFFT() * 2),
                                                nThreads);

            TensorTarget = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, (BoxDimensions.X / 2 + 1), BoxDimensions.Y, BoxDimensions.Z, 2),
                                                                         new float[BatchSize * BoxDimensions.ElementsFFT() * 2],
                                                                         0,
                                                                         BatchSize * (int)BoxDimensions.ElementsFFT() * 2),
                                                nThreads);

            TensorWeightSource = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, (BoxDimensions.X / 2 + 1), BoxDimensions.Y, BoxDimensions.Z, 1),
                                                                        new float[BatchSize * BoxDimensions.ElementsFFT()],
                                                                        0,
                                                                        BatchSize * (int)BoxDimensions.ElementsFFT()),
                                               nThreads);

            TensorWeightTarget = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, (BoxDimensions.X / 2 + 1), BoxDimensions.Y, BoxDimensions.Z, 1),
                                                                        new float[BatchSize * BoxDimensions.ElementsFFT()],
                                                                        0,
                                                                        BatchSize * (int)BoxDimensions.ElementsFFT()),
                                               nThreads);

            TensorCode = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(BatchSize, BottleneckWidth),
                                                                         new float[BatchSize * BottleneckWidth],
                                                                         0,
                                                                         BatchSize * BottleneckWidth),
                                                nThreads);

            TensorLearningRate = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(1),
                                                                                 new float[1],
                                                                                 0,
                                                                                 1),
                                                        nThreads);

            TensorDropoutRate = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(1),
                                                                                 new float[1],
                                                                                 0,
                                                                                 1),
                                                       nThreads);

            TensorOrthogonalityRate = Helper.ArrayOfFunction(i => TFTensor.FromBuffer(new TFShape(1),
                                                                                 new float[1],
                                                                                 0,
                                                                                 1),
                                                             nThreads);

            ResultPredicted = Helper.ArrayOfFunction(i => new float[BatchSize * BoxDimensions.ElementsFFT() * 2], nThreads);
            ResultBottleneck = Helper.ArrayOfFunction(i => new float[BatchSize * BottleneckWidth], nThreads);
            ResultLoss = Helper.ArrayOfFunction(i => new float[1], nThreads);
            ResultLossKL = Helper.ArrayOfFunction(i => new float[1], nThreads);

            RetrievedWeights = new float[boxDimensions.ElementsFFT() * 2 * NWeights0];

            //if (!ForTraining)
                RunnerPrediction = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                               AddInput(NodeCode, TensorCode[i]).
                                                               AddInput(NodeDropoutRate, TensorDropoutRate[i]).
                                                               Fetch(NodeOutputPredicted),
                                                          nThreads);
            //else
                RunnerTraining = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                             AddInput(NodeInputSource, TensorSource[i]).
                                                             AddInput(NodeInputTarget, TensorTarget[i]).
                                                             AddInput(NodeInputWeightSource, TensorWeightSource[i]).
                                                             AddInput(NodeInputWeightTarget, TensorWeightTarget[i]).
                                                             AddInput(NodeDropoutRate, TensorDropoutRate[i]).
                                                             AddInput(NodeLearningRate, TensorLearningRate[i]).
                                                             AddInput(NodeOrthogonalityRate, TensorOrthogonalityRate[i]).
                                                             Fetch(NodeOutputPredicted, NodeOutputLoss, NodeOutputLossKL, NodeBottleneck, NodeOpTrain),
                                                        nThreads);
            
            RunnerEncode = Helper.ArrayOfFunction(i => Session.GetRunner().
                                                         AddInput(NodeInputSource, TensorSource[i]).
                                                         AddInput(NodeInputWeightSource, TensorWeightSource[i]).
                                                         AddInput(NodeDropoutRate, TensorDropoutRate[i]).
                                                         Fetch(NodeBottleneck),
                                                    nThreads);

            RunnerRetrieveWeights0 = Session.GetRunner().Fetch(NodeWeights0);
            RunnerRetrieveWeights1 = Session.GetRunner().Fetch(NodeWeights1);

            if (ForTraining)
            {
                TensorWeights0 = TFTensor.FromBuffer(new TFShape(NWeights0, BoxDimensions.ElementsFFT() * 2),
                                                     new float[BoxDimensions.ElementsFFT() * 2 * NWeights0],
                                                     0,
                                                     (int)BoxDimensions.ElementsFFT() * 2 * NWeights0);

                RunnerAssignWeights0 = Session.GetRunner().AddInput(NodeWeights0Input, TensorWeights0).
                                                          Fetch(NodeWeights0Assign);
                RunnerAssignWeights1 = Session.GetRunner().AddInput(NodeWeights1Input, TensorWeights0).
                                                          Fetch(NodeWeights1Assign);
            }

            // Run prediction or training for one batch to claim all the memory needed
            float[] InitDecoded;
            float[] InitBottleneck;
            float[] InitLoss, InitLossKL;
            if (!ForTraining)
            {
                RandomNormal RandN = new RandomNormal(123);
                Predict(Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BottleneckWidth * BatchSize),
                        0,
                        out InitDecoded);
            }
            else
            {
                RandomNormal RandN = new RandomNormal();

                Encode(Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BatchSize * (int)BoxDimensions.ElementsFFT() * 2),
                       Helper.ArrayOfFunction(i => 1f, BatchSize * (int)BoxDimensions.ElementsFFT()),
                       0,
                       out InitBottleneck);

                Train(Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BatchSize * (int)BoxDimensions.ElementsFFT() * 2),
                      Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), BatchSize * (int)BoxDimensions.ElementsFFT() * 2),
                      Helper.ArrayOfFunction(i => 1f, BatchSize * (int)BoxDimensions.ElementsFFT()),
                      Helper.ArrayOfFunction(i => 1f, BatchSize * (int)BoxDimensions.ElementsFFT()),
                      0.5f,
                      1e-10f,
                      1e-5f,
                      0,
                      out InitDecoded,
                      out InitBottleneck,
                      out InitLoss,
                      out InitLossKL);
            }
        }

        public void Predict(float[] codes, int threadID, out float[] prediction)
        {
            if (ForTraining)
                throw new Exception("Network was loaded in training mode, but asked to predict.");

            Marshal.Copy(codes, 0, TensorCode[threadID].Data, BatchSize * BottleneckWidth);
            Marshal.Copy(new[] { 0f }, 0, TensorDropoutRate[threadID].Data, 1);

            var Output = RunnerPrediction[threadID].Run();

            Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);

            prediction = ResultPredicted[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
        }

        //public void Predict(IntPtr d_data, int threadID, out float[] prediction)
        //{
        //    //if (ForTraining)
        //    //    throw new Exception("Network was loaded in training mode, but asked to predict.");

        //    GPU.CopyDeviceToHostPinned(d_data, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.Elements());
        //    var Output = RunnerPrediction[threadID].Run();

        //    Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.Elements());

        //    prediction = ResultPredicted[threadID];

        //    foreach (var tensor in Output)
        //        tensor.Dispose();
        //}

        public void Train(float[] source,
                          float[] target,
                          float[] weightSource,
                          float[] weightTarget,
                          float dropoutRate,
                          float learningRate,
                          float orthogonalityRate,
                          int threadID,
                          out float[] prediction,
                          out float[] bottleneck,
                          out float[] loss,
                          out float[] lossKL)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            Marshal.Copy(source, 0, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            Marshal.Copy(target, 0, TensorTarget[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            Marshal.Copy(weightSource, 0, TensorWeightSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT());
            Marshal.Copy(weightTarget, 0, TensorWeightTarget[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT());
            Marshal.Copy(new[] { dropoutRate }, 0, TensorDropoutRate[threadID].Data, 1);
            Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);
            Marshal.Copy(new[] { orthogonalityRate }, 0, TensorOrthogonalityRate[threadID].Data, 1);

            var Output = RunnerTraining[threadID].Run();

            Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);

            Marshal.Copy(Output[1].Data, ResultLoss[threadID], 0, 1);
            Marshal.Copy(Output[2].Data, ResultLossKL[threadID], 0, 1);

            Marshal.Copy(Output[3].Data, ResultBottleneck[threadID], 0, BatchSize * BottleneckWidth);

            prediction = ResultPredicted[threadID];
            bottleneck = ResultBottleneck[threadID];
            loss = ResultLoss[threadID];
            lossKL = ResultLossKL[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
        }

        public void Train(IntPtr d_source,
                          IntPtr d_target,
                          IntPtr d_weightSource,
                          IntPtr d_weightTarget,
                          float dropoutRate,
                          float learningRate,
                          float orthogonalityRate,
                          int threadID,
                          out float[] prediction,
                          out float[] bottleneck,
                          out float[] loss,
                          out float[] lossKL)
        {
            if (!ForTraining)
                throw new Exception("Network was loaded in prediction mode, but asked to train.");

            GPU.CopyDeviceToHostPinned(d_source, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            GPU.CopyDeviceToHostPinned(d_target, TensorTarget[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            GPU.CopyDeviceToHostPinned(d_weightSource, TensorWeightSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT());
            GPU.CopyDeviceToHostPinned(d_weightTarget, TensorWeightTarget[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT());

            Marshal.Copy(new[] { dropoutRate }, 0, TensorDropoutRate[threadID].Data, 1);
            Marshal.Copy(new[] { learningRate }, 0, TensorLearningRate[threadID].Data, 1);
            Marshal.Copy(new[] { orthogonalityRate }, 0, TensorOrthogonalityRate[threadID].Data, 1);

            var Output = RunnerTraining[threadID].Run();

            Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);

            Marshal.Copy(Output[1].Data, ResultLoss[threadID], 0, 1);
            Marshal.Copy(Output[2].Data, ResultLossKL[threadID], 0, 1);

            Marshal.Copy(Output[3].Data, ResultBottleneck[threadID], 0, BatchSize * BottleneckWidth);

            prediction = ResultPredicted[threadID];
            bottleneck = ResultBottleneck[threadID];
            loss = ResultLoss[threadID];
            lossKL = ResultLossKL[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
        }

        public void Encode(float[] source,
                           float[] weight,
                           int threadID,
                           //out float[] prediction,
                           out float[] bottleneck)
        {
            //if (!ForTraining)
            //    throw new Exception("Network was loaded in prediction mode, but asked to train.");

            Marshal.Copy(source, 0, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            Marshal.Copy(weight, 0, TensorWeightSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT());
            Marshal.Copy(new[] { 0f }, 0, TensorDropoutRate[threadID].Data, 1);

            var Output = RunnerEncode[threadID].Run();

            //Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            Marshal.Copy(Output[0].Data, ResultBottleneck[threadID], 0, BatchSize * BottleneckWidth);

            //prediction = ResultPredicted[threadID];
            bottleneck = ResultBottleneck[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
        }

        public void Encode(IntPtr d_source,
                           IntPtr d_weight,
                           int threadID,
                           //out float[] prediction,
                           out float[] bottleneck)
        {
            //if (!ForTraining)
            //    throw new Exception("Network was loaded in prediction mode, but asked to train.");

            GPU.CopyDeviceToHostPinned(d_source, TensorSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            GPU.CopyDeviceToHostPinned(d_weight, TensorWeightSource[threadID].Data, BatchSize * (int)BoxDimensions.ElementsFFT());
            Marshal.Copy(new[] { 0f }, 0, TensorDropoutRate[threadID].Data, 1);

            var Output = RunnerEncode[threadID].Run();

            //Marshal.Copy(Output[0].Data, ResultPredicted[threadID], 0, BatchSize * (int)BoxDimensions.ElementsFFT() * 2);
            Marshal.Copy(Output[0].Data, ResultBottleneck[threadID], 0, BatchSize * BottleneckWidth);

            //prediction = ResultPredicted[threadID];
            bottleneck = ResultBottleneck[threadID];

            foreach (var tensor in Output)
                tensor.Dispose();
        }

        public float[] GetWeights(int layer)
        {
            if (layer == 0)
            {
                var Output = RunnerRetrieveWeights0.Run();

                Marshal.Copy(Output[0].Data, RetrievedWeights, 0, NWeights0 * (int)BoxDimensions.ElementsFFT() * 2);

                foreach (var tensor in Output)
                    tensor.Dispose();
            }
            else if (layer == 1)
            {
                var Output = RunnerRetrieveWeights1.Run();

                Marshal.Copy(Output[0].Data, RetrievedWeights, 0, NWeights0 * (int)BoxDimensions.ElementsFFT() * 2);

                foreach (var tensor in Output)
                    tensor.Dispose();
            }

            return RetrievedWeights;
        }

        public void AssignWeights(int layer, float[] data)
        {
            Marshal.Copy(data, 0, TensorWeights0.Data, NWeights0 * (int)BoxDimensions.ElementsFFT() * 2);

            if (layer == 0)
            {
                var Output = RunnerAssignWeights0.Run();
                foreach (var tensor in Output)
                    tensor.Dispose();
            }
            else if (layer==1)
            {
                var Output = RunnerAssignWeights1.Run();
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

        ~FlexNet3D()
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
                    foreach (var tensor in TensorWeightSource)
                        tensor.Dispose();

                    Session.DeleteSession();
                }
            }
        }
    }
}
