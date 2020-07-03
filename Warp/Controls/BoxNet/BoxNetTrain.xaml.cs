using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using LiveCharts;
using LiveCharts.Defaults;
using MahApps.Metro.Controls.Dialogs;
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for BoxNetTrain.xaml
    /// </summary>
    public partial class BoxNetTrain : UserControl
    {
        public string ModelName;
        public Options Options;
        public event Action Close;

        private string SuffixPositive, SuffixFalsePositive, SuffixUncertain;
        private string FolderPositive, FolderFalsePositive, FolderUncertain;

        private bool IsTrainingCanceled = false;

        public string NewName;

        private string NewExamplesPath;

        public BoxNetTrain(string modelName, Options options)
        {
            InitializeComponent();

            ModelName = modelName;
            Options = options;

            TextHeader.Text = "Retrain " + ModelName;
            TextNewName.Text = ModelName + "_2";

            int CorpusSize = 0;
            if (Directory.Exists(System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet2training")))
                foreach (var fileName in Directory.EnumerateFiles(System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet2training"), "*.tif"))
                {
                    MapHeader Header = MapHeader.ReadFromFile(fileName);
                    CorpusSize += Header.Dimensions.Z / 3;
                }

            CheckCorpus.Content = $" Also use {CorpusSize} micrographs in\n {System.IO.Path.Combine(Environment.CurrentDirectory, "boxnettraining")}";
            if (CorpusSize == 0)
            {
                CheckCorpus.IsChecked = false;
                CheckCorpus.IsEnabled = false;
            }
        }

        private async void ButtonRetrain2_OnClick(object sender, RoutedEventArgs e)
        {
            Options.MainWindow.MicrographDisplayControl.DropBoxNetworks();
            Options.MainWindow.MicrographDisplayControl.DropNoiseNetworks();

            Movie[] Movies = Options.MainWindow.FileDiscoverer.GetImmutableFiles();
            if (Movies.Length == 0)
            {
                Close?.Invoke();
                return;
            }

            TextNewName.Text = Helper.RemoveInvalidChars(TextNewName.Text);
            bool UseCorpus = (bool)CheckCorpus.IsChecked;
            bool TrainMasking = (bool)CheckTrainMask.IsChecked;
            float Diameter = (float)SliderDiameter.Value;

            string PotentialNewName = TextNewName.Text;

            PanelSettings.Visibility = Visibility.Collapsed;
            PanelTraining.Visibility = Visibility.Visible;

            await Task.Run(async () =>
            {
                GPU.SetDevice(0);

                #region Prepare new examples

                Dispatcher.Invoke(() => TextProgress.Text = "Preparing new examples...");

                Directory.CreateDirectory(Movies[0].DirectoryName + "boxnet2training/");
                NewExamplesPath = Movies[0].DirectoryName + "boxnet2training/" + PotentialNewName + ".tif";

                try
                {
                    PrepareData(NewExamplesPath);
                }
                catch (Exception exc)
                {
                    await Options.MainWindow.ShowMessageAsync("Oopsie", exc.ToString());
                    IsTrainingCanceled = true;

                    return;
                }

                #endregion

                #region Load background and new examples

                Dispatcher.Invoke(() => TextProgress.Text = "Loading examples...");

                int2 DimsLargest = new int2(1);

                List<float[]>[] AllMicrographs = { new List<float[]>(), new List<float[]>() };
                List<float[]>[] AllLabels = { new List<float[]>(), new List<float[]>() };
                List<float[]>[] AllUncertains = { new List<float[]>(), new List<float[]>() };
                List<int2>[] AllDims = { new List<int2>(), new List<int2>() };
                List<float3>[] AllLabelWeights = { new List<float3>(), new List<float3>() };

                string[][] AllPaths = UseCorpus
                                          ? new[]
                                          {
                                              new[] { NewExamplesPath },
                                              Directory.EnumerateFiles(System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet2training"), "*.tif").ToArray()
                                          }
                                          : new[] { new[] { NewExamplesPath } };

                long[] ClassHist = new long[3];

                for (int icorpus = 0; icorpus < AllPaths.Length; icorpus++)
                {
                    foreach (var examplePath in AllPaths[icorpus])
                    {
                        Image ExampleImage = Image.FromFile(examplePath);
                        int N = ExampleImage.Dims.Z / 3;

                        for (int n = 0; n < N; n++)
                        {
                            float[] Mic = ExampleImage.GetHost(Intent.Read)[n * 3 + 0];
                            MathHelper.FitAndSubtractGrid(Mic, new int2(ExampleImage.Dims), new int2(4));

                            AllMicrographs[icorpus].Add(Mic);
                            AllLabels[icorpus].Add(ExampleImage.GetHost(Intent.Read)[n * 3 + 1]);
                            AllUncertains[icorpus].Add(ExampleImage.GetHost(Intent.Read)[n * 3 + 2]);

                            AllDims[icorpus].Add(new int2(ExampleImage.Dims));

                            float[] Labels = ExampleImage.GetHost(Intent.Read)[n * 3 + 1];
                            float[] Uncertains = ExampleImage.GetHost(Intent.Read)[n * 3 + 2];
                            for (int i = 0; i < Labels.Length; i++)
                            {
                                int Label = (int)Labels[i];
                                if (!TrainMasking && Label == 2)
                                {
                                    Label = 0;
                                    Labels[i] = 0;
                                }
                                ClassHist[Label]++;
                            }
                        }

                        DimsLargest.X = Math.Max(DimsLargest.X, ExampleImage.Dims.X);
                        DimsLargest.Y = Math.Max(DimsLargest.Y, ExampleImage.Dims.Y);

                        ExampleImage.Dispose();
                    }
                }

                {
                    float[] LabelWeights = { 1f, 1f, 1f };
                    if (ClassHist[1] > 0)
                    {
                        LabelWeights[0] = (float)Math.Pow((float)ClassHist[1] / ClassHist[0], 1 / 3.0);
                        LabelWeights[2] = 1;//(float)Math.Sqrt((float)ClassHist[1] / ClassHist[2]);
                    }
                    else
                    {
                        LabelWeights[0] = (float)Math.Pow((float)ClassHist[2] / ClassHist[0], 1 / 3.0);
                    }

                    for (int icorpus = 0; icorpus < AllPaths.Length; icorpus++)
                        for (int i = 0; i < AllMicrographs[icorpus].Count; i++)
                            AllLabelWeights[icorpus].Add(new float3(LabelWeights[0], LabelWeights[1], LabelWeights[2]));
                }

                int NNewExamples = AllMicrographs[0].Count;
                int NOldExamples = UseCorpus ? AllMicrographs[1].Count : 0;

                #endregion

                #region Load models

                Dispatcher.Invoke(() => TextProgress.Text = "Loading old BoxNet model...");

                int NThreads = 2;

                Image.FreeDeviceAll();

                Dispatcher.Invoke(() =>
                {
                    Options.MainWindow.MicrographDisplayControl.FreeOnDevice();
                    Options.MainWindow.MicrographDisplayControl.DropBoxNetworks();
                    Options.MainWindow.MicrographDisplayControl.DropNoiseNetworks();
                });


                BoxNet2 NetworkTrain = null;
                try
                {
                    NetworkTrain = new BoxNet2(Options.MainWindow.LocatePickingModel(ModelName), GPU.GetDeviceWithMostMemory(), NThreads, 8, true);
                }
                catch   // It might be an old version of BoxNet that doesn't support batch size != 1
                {
                    NetworkTrain = new BoxNet2(Options.MainWindow.LocatePickingModel(ModelName), GPU.GetDeviceWithMostMemory(), NThreads, 1, true);
                }

                //BoxNet2 NetworkOld = new BoxNet2(Options.MainWindow.LocatePickingModel(ModelName), (GPU.GetDeviceCount() * 2 - 2) % GPU.GetDeviceCount(), NThreads, 1, false);

                #endregion

                #region Training

                Dispatcher.Invoke(() => TextProgress.Text = "Training...");

                int2 DimsAugmented = BoxNet2.BoxDimensionsTrain;
                int Border = (BoxNet2.BoxDimensionsTrain.X - BoxNet2.BoxDimensionsValidTrain.X) / 2;
                int BatchSize = NetworkTrain.BatchSize;
                int PlotEveryN = 10;
                int SmoothN = 30;

                List<ObservablePoint>[] AccuracyPoints = Helper.ArrayOfFunction(i => new List<ObservablePoint>(), 4);
                Queue<float>[] LastAccuracies = { new Queue<float>(SmoothN), new Queue<float>(SmoothN) };
                List<float>[] LastBaselines = { new List<float>(), new List<float>() };

                GPU.SetDevice(0);

                IntPtr d_MaskUncertain;
                {
                    float[] DataUncertain = new float[DimsAugmented.Elements()];
                    for (int y = 0; y < DimsAugmented.Y; y++)
                    {
                        for (int x = 0; x < DimsAugmented.X; x++)
                        {
                            if (x >= Border &&
                                y >= Border &&
                                x < DimsAugmented.X - Border &&
                                y < DimsAugmented.Y - Border)
                                DataUncertain[y * DimsAugmented.X + x] = 1;
                            else
                                DataUncertain[y * DimsAugmented.X + x] = 0.1f;
                        }
                    }

                    d_MaskUncertain = GPU.MallocDeviceFromHost(DataUncertain, DataUncertain.Length);
                }

                IntPtr[] d_OriData = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsLargest.Elements()), NetworkTrain.MaxThreads);
                IntPtr[] d_OriLabels = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsLargest.Elements()), NetworkTrain.MaxThreads);
                IntPtr[] d_OriUncertains = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsLargest.Elements()), NetworkTrain.MaxThreads);

                IntPtr[] d_AugmentedData = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsAugmented.Elements() * BatchSize), NetworkTrain.MaxThreads);
                IntPtr[] d_AugmentedLabels = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsAugmented.Elements() * BatchSize * 3), NetworkTrain.MaxThreads);
                IntPtr[] d_AugmentedWeights = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsAugmented.Elements() * BatchSize), NetworkTrain.MaxThreads);

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                Random[] RG = Helper.ArrayOfFunction(i => new Random(i), NetworkTrain.MaxThreads);
                RandomNormal[] RGN = Helper.ArrayOfFunction(i => new RandomNormal(i), NetworkTrain.MaxThreads);

                //float[][] h_AugmentedData = Helper.ArrayOfFunction(i => new float[DimsAugmented.Elements()], NetworkTrain.MaxThreads);
                //float[][] h_AugmentedLabels = Helper.ArrayOfFunction(i => new float[DimsAugmented.Elements()], NetworkTrain.MaxThreads);
                //float[][] h_AugmentedWeights = Helper.ArrayOfFunction(i => new float[DimsAugmented.Elements()], NetworkTrain.MaxThreads);
                //float[][] LabelsOneHot = Helper.ArrayOfFunction(i => new float[DimsAugmented.Elements() * 3], NetworkTrain.MaxThreads);

                int NIterations = NNewExamples * 100 * AllMicrographs.Length;

                int NDone = 0;
                Helper.ForCPU(0, NIterations, NetworkTrain.MaxThreads,

                              threadID => GPU.SetDevice(0),

                              (b, threadID) =>
                              {
                                  int icorpus;
                                  lock (Watch)
                                    icorpus = NDone % AllPaths.Length;

                                  float2[] PositionsGround;

                                  for (int ib = 0; ib < BatchSize; ib++)
                                  {
                                      int ExampleID = RG[threadID].Next(AllMicrographs[icorpus].Count);
                                      int2 Dims = AllDims[icorpus][ExampleID];

                                      float2[] Translations = Helper.ArrayOfFunction(x => new float2(RG[threadID].Next(Dims.X - Border * 2) + Border - DimsAugmented.X / 2,
                                                                                                     RG[threadID].Next(Dims.Y - Border * 2) + Border - DimsAugmented.Y / 2), 1);

                                      float[] Rotations = Helper.ArrayOfFunction(i => (float)(RG[threadID].NextDouble() * Math.PI * 2), 1);
                                      float3[] Scales = Helper.ArrayOfFunction(i => new float3(0.8f + (float)RG[threadID].NextDouble() * 0.4f,
                                                                                               0.8f + (float)RG[threadID].NextDouble() * 0.4f,
                                                                                               (float)(RG[threadID].NextDouble() * Math.PI * 2)), 1);
                                      float StdDev = (float)Math.Abs(RGN[threadID].NextSingle(0, 0.3f));

                                      float[] DataMicrograph = AllMicrographs[icorpus][ExampleID];
                                      float[] DataLabels = AllLabels[icorpus][ExampleID];
                                      float[] DataUncertains = AllUncertains[icorpus][ExampleID];

                                      GPU.CopyHostToDevice(DataMicrograph, d_OriData[threadID], Dims.Elements());
                                      GPU.CopyHostToDevice(DataLabels, d_OriLabels[threadID], Dims.Elements());
                                      GPU.CopyHostToDevice(DataUncertains, d_OriUncertains[threadID], Dims.Elements());

                                      //GPU.ValueFill(d_OriUncertains[threadID], Dims.Elements(), 1f);

                                      GPU.BoxNet2Augment(d_OriData[threadID],
                                                         d_OriLabels[threadID],
                                                         d_OriUncertains[threadID],
                                                         Dims,
                                                         new IntPtr((long)d_AugmentedData[threadID] + ib * DimsAugmented.Elements() * sizeof(float)),
                                                         new IntPtr((long)d_AugmentedLabels[threadID] + ib * DimsAugmented.Elements() * 3 * sizeof(float)),
                                                         new IntPtr((long)d_AugmentedWeights[threadID] + ib * DimsAugmented.Elements() * sizeof(float)),
                                                         DimsAugmented,
                                                         AllLabelWeights[icorpus][ExampleID],
                                                         Helper.ToInterleaved(Translations),
                                                         Rotations,
                                                         Helper.ToInterleaved(Scales),
                                                         StdDev,
                                                         RG[threadID].Next(99999),
                                                         (uint)1);
                                  }

                                  GPU.MultiplySlices(d_AugmentedWeights[threadID],
                                                     d_MaskUncertain,
                                                     d_AugmentedWeights[threadID],
                                                     DimsAugmented.Elements(),
                                                     (uint)BatchSize);

                                  float LearningRate = 0.00005f;

                                  long[][] ResultLabels = new long[2][];
                                  float[][] ResultProbabilities = new float[2][];

                                  float Loss = 0;

                                  lock (NetworkTrain)
                                      Loss = NetworkTrain.Train(d_AugmentedData[threadID],
                                                                d_AugmentedLabels[threadID],
                                                                d_AugmentedWeights[threadID],
                                                                LearningRate,
                                                                threadID,
                                                                out ResultLabels[0],
                                                                out ResultProbabilities[0]);
                                  //lock (NetworkOld)
                                  //    NetworkOld.Predict(d_AugmentedData[threadID],
                                  //                       threadID,
                                  //                       out ResultLabels[1],
                                  //                       out ResultProbabilities[1]);

                                  //float[] AccuracyParticles = new float[2];

                                  //for (int i = 0; i < 2; i++)
                                  //{
                                  //    for (int j = 0; j < ResultLabels[i].Length; j++)
                                  //    {
                                  //        long Label = ResultLabels[i][j];
                                  //        float Prob = ResultProbabilities[i][j * 3 + Label];
                                  //        if (Label == 1 && Prob < 0.4f)
                                  //            ResultLabels[i][j] = 0;
                                  //        else if (Label == 2 && Prob < 0.1f)
                                  //            ResultLabels[i][j] = 0;
                                  //    }

                                  //    float2[] PositionsPicked = GetCentroids(ResultLabels[i], DimsAugmented, Border);

                                  //    int FP = 0, FN = 0;

                                  //    foreach (var posGround in PositionsGround)
                                  //    {
                                  //        bool Found = false;
                                  //        foreach (var posPicked in PositionsPicked)
                                  //        {
                                  //            if ((posGround - posPicked).Length() < 5)
                                  //            {
                                  //                Found = true;
                                  //                break;
                                  //            }
                                  //        }
                                  //        if (!Found)
                                  //            FN++;
                                  //    }
                                  //    foreach (var posPicked in PositionsPicked)
                                  //    {
                                  //        bool Found = false;
                                  //        foreach (var posGround in PositionsGround)
                                  //        {
                                  //            if ((posGround - posPicked).Length() < 5)
                                  //            {
                                  //                Found = true;
                                  //                break;
                                  //            }
                                  //        }
                                  //        if (!Found)
                                  //            FP++;
                                  //    }
                                      
                                  //    AccuracyParticles[i] = (float)(PositionsPicked.Length - FP) / (PositionsGround.Length + 0 + FN);

                                  //    //if (float.IsNaN(AccuracyParticles[i]))
                                  //    //    throw new Exception();
                                  //}

                                  lock (Watch)
                                  {
                                      NDone++;

                                      //if (!float.IsNaN(AccuracyParticles[0]))
                                      {
                                          LastAccuracies[icorpus].Enqueue(Loss);
                                          if (LastAccuracies[icorpus].Count > SmoothN)
                                              LastAccuracies[icorpus].Dequeue();
                                      }
                                      //if (!float.IsNaN(AccuracyParticles[1]))
                                      //    LastBaselines[icorpus].Add(AccuracyParticles[1]);

                                      if (NDone % PlotEveryN == 0)
                                      {
                                          for (int iicorpus = 0; iicorpus < AllMicrographs.Length; iicorpus++)
                                          {
                                              AccuracyPoints[iicorpus * 2 + 0].Add(new ObservablePoint((float)NDone / NIterations * 100,
                                                                                                      MathHelper.Mean(LastAccuracies[iicorpus].Where(v => !float.IsNaN(v)))));

                                              //AccuracyPoints[iicorpus * 2 + 1].Clear();
                                              //AccuracyPoints[iicorpus * 2 + 1].Add(new ObservablePoint(0,
                                              //                                                        MathHelper.Mean(LastBaselines[iicorpus].Where(v => !float.IsNaN(v)))));
                                              //AccuracyPoints[iicorpus * 2 + 1].Add(new ObservablePoint((float)NDone / NIterations * 100,
                                              //                                                        MathHelper.Mean(LastBaselines[iicorpus].Where(v => !float.IsNaN(v)))));
                                          }

                                          long Elapsed = Watch.ElapsedMilliseconds;
                                          double Estimated = (double)Elapsed / NDone * NIterations;
                                          int Remaining = (int)(Estimated - Elapsed);
                                          TimeSpan SpanRemaining = new TimeSpan(0, 0, 0, 0, Remaining);

                                          Dispatcher.InvokeAsync(() =>
                                          {
                                              SeriesTrainAccuracy.Values = new ChartValues<ObservablePoint>(AccuracyPoints[0]);
                                              //SeriesTrainBaseline.Values = new ChartValues<ObservablePoint>(AccuracyPoints[1]);

                                              if (UseCorpus)
                                              {
                                                  SeriesBackgroundAccuracy.Values = new ChartValues<ObservablePoint>(AccuracyPoints[2]);
                                                  //SeriesBackgroundBaseline.Values = new ChartValues<ObservablePoint>(AccuracyPoints[3]);
                                              }

                                              TextProgress.Text = SpanRemaining.ToString((int)SpanRemaining.TotalHours > 0 ? @"hh\:mm\:ss" : @"mm\:ss");
                                          });
                                      }
                                  }
                              },

                              null);

                foreach (var ptr in d_OriData)
                    GPU.FreeDevice(ptr);
                foreach (var ptr in d_OriLabels)
                    GPU.FreeDevice(ptr);
                foreach (var ptr in d_OriUncertains)
                    GPU.FreeDevice(ptr);
                foreach (var ptr in d_AugmentedData)
                    GPU.FreeDevice(ptr);
                foreach (var ptr in d_AugmentedLabels)
                    GPU.FreeDevice(ptr);
                foreach (var ptr in d_AugmentedWeights)
                    GPU.FreeDevice(ptr);
                GPU.FreeDevice(d_MaskUncertain);

                #endregion

                if (!IsTrainingCanceled)
                {
                    Dispatcher.Invoke(() => TextProgress.Text = "Saving new BoxNet model...");

                    string BoxNetDir = System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet2models/");
                    Directory.CreateDirectory(BoxNetDir);

                    NetworkTrain.Export(BoxNetDir + PotentialNewName);
                }

                NetworkTrain.Dispose();
                //NetworkOld.Dispose();

                Image.FreeDeviceAll();
                TFHelper.TF_FreeAllMemory();
            });

            if (!IsTrainingCanceled)
                NewName = TextNewName.Text;

            TextProgress.Text = "Done.";

            if (IsTrainingCanceled)
            {
                Close?.Invoke();
            }
            else
            {
                ButtonCancelTraining.Content = "CLOSE";
                ButtonCancelTraining.Click -= ButtonCancelTraining_OnClick;
                ButtonCancelTraining.Click += async (a, b) =>
                {
                    Close?.Invoke();

                    if (MainWindow.GlobalOptions.ShowBoxNetReminder)
                    {
                        var DialogResult = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Sharing is caring 🙂",
                                                                                                               "BoxNet performs well because of the wealth of training data it\n" +
                                                                                                               "can use. However, it could do even better with the data you just\n" +
                                                                                                               "used for re-training! Would you like to open a website to guide\n" +
                                                                                                               "you through contributing your data to the central repository?\n\n" +
                                                                                                               $"Your training data have been saved to \n{NewExamplesPath.Replace('/', '\\')}.\n\n",
                                                                                                               MessageDialogStyle.AffirmativeAndNegative,
                                                                                                               new MetroDialogSettings()
                                                                                                               {
                                                                                                                   AffirmativeButtonText = "Yes",
                                                                                                                   NegativeButtonText = "No",
                                                                                                                   DialogMessageFontSize = 18,
                                                                                                                   DialogTitleFontSize = 28
                                                                                                               });
                        if (DialogResult == MessageDialogResult.Affirmative)
                        {
                            string argument = "/select, \"" + NewExamplesPath.Replace('/', '\\') + "\"";
                            Process.Start("explorer.exe", argument);

                            Process.Start("http://www.warpem.com/warp/?page_id=72");
                        }
                    }
                };
            }
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonSuffixPositive_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = Options.MainWindow.FileDiscoverer.GetImmutableFiles();

                bool FoundMatchingSuffix = false;
                string StarName = Helper.PathToName(OpenDialog.FileName);
                foreach (var item in Movies)
                {
                    if (StarName.Contains(item.RootName))
                    {
                        FoundMatchingSuffix = true;
                        SuffixPositive = StarName.Substring(item.RootName.Length);
                        FolderPositive = Helper.PathToFolder(OpenDialog.FileName);
                        ButtonSuffixPositiveText.Text = "*" + SuffixPositive;
                        ButtonSuffixPositiveText.ToolTip = FolderPositive + "*" + SuffixPositive;
                        break;
                    }
                }

                if (!FoundMatchingSuffix)
                {
                    await Options.MainWindow.ShowMessageAsync("Oopsie", "STAR file could not be matched to any of the movies to determine the suffix.");
                    return;
                }
            }
        }

        private async void ButtonSuffixFalsePositive_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = Options.MainWindow.FileDiscoverer.GetImmutableFiles();

                bool FoundMatchingSuffix = false;
                string StarName = Helper.PathToName(OpenDialog.FileName);
                foreach (var item in Movies)
                {
                    if (StarName.Contains(item.RootName))
                    {
                        FoundMatchingSuffix = true;
                        SuffixFalsePositive = StarName.Substring(item.RootName.Length);
                        FolderFalsePositive = Helper.PathToFolder(OpenDialog.FileName);
                        ButtonSuffixFalsePositiveText.Text = "*" + SuffixFalsePositive;
                        ButtonSuffixFalsePositiveText.ToolTip = FolderFalsePositive + "*" + SuffixFalsePositive;
                        break;
                    }
                }

                if (!FoundMatchingSuffix)
                {
                    await Options.MainWindow.ShowMessageAsync("Oopsie", "STAR file could not be matched to any of the movies to determine the suffix.");
                    return;
                }
            }
        }

        private async void ButtonSuffixUncertain_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                Movie[] Movies = Options.MainWindow.FileDiscoverer.GetImmutableFiles();

                bool FoundMatchingSuffix = false;
                string StarName = Helper.PathToName(OpenDialog.FileName);
                foreach (var item in Movies)
                {
                    if (StarName.Contains(item.RootName))
                    {
                        FoundMatchingSuffix = true;
                        SuffixUncertain = StarName.Substring(item.RootName.Length);
                        FolderUncertain = Helper.PathToFolder(OpenDialog.FileName);
                        ButtonSuffixUncertainText.Text = "*" + SuffixUncertain;
                        ButtonSuffixUncertain.ToolTip = FolderUncertain + "*" + SuffixUncertain;
                        break;
                    }
                }

                if (!FoundMatchingSuffix)
                {
                    await Options.MainWindow.ShowMessageAsync("Oopsie", "STAR file could not be matched to any of the movies to determine the suffix.");
                    return;
                }
            }
        }

        private void ButtonCancelTraining_OnClick(object sender, RoutedEventArgs e)
        {
            IsTrainingCanceled = true;
            ButtonCancelTraining.IsEnabled = false;
            ButtonCancelTraining.Content = "CANCELING...";
        }

        private void PrepareData(string savePath)
        {
            float Diameter = 1;
            Dispatcher.Invoke(() => Diameter = (float)SliderDiameter.Value);

            string ErrorString = "";

            //if (string.IsNullOrEmpty(NewName))
            //    ErrorString += "No name specified.\n";

            if (string.IsNullOrEmpty(SuffixPositive))
                ErrorString += "No positive examples selected.\n";

            if (!string.IsNullOrEmpty(ErrorString))
                throw new Exception(ErrorString);

            bool IsNegative = Options.Picking.DataStyle != "cryo";

            Movie[] Movies = Options.MainWindow.FileDiscoverer.GetImmutableFiles();
            List<Movie> ValidMovies = new List<Movie>();

            foreach (var movie in Movies)
            {
                if (movie.UnselectManual != null && (bool)movie.UnselectManual)
                    continue;

                if (!File.Exists(movie.AveragePath))
                    continue;

                bool HasExamples = File.Exists(FolderPositive + movie.RootName + SuffixPositive + ".star");
                if (!HasExamples)
                    continue;

                ValidMovies.Add(movie);
            }

            if (ValidMovies.Count == 0)
                throw new Exception("No movie averages could be found to create training examples. Please process the movies first to create the averages.");
            

            
            List<Image> AllAveragesBN = new List<Image>();
            List<Image> AllLabelsBN = new List<Image>();
            List<Image> AllCertainBN = new List<Image>();

            int MoviesDone = 0;
            foreach (var movie in ValidMovies)
            {
                MapHeader Header = MapHeader.ReadFromFile(movie.AveragePath);
                float PixelSize = Header.PixelSize.X;

                #region Load positions, and possibly move on to next movie

                List<float2> PosPositive = new List<float2>();
                List<float2> PosFalse = new List<float2>();
                List<float2> PosUncertain = new List<float2>();

                if (File.Exists(FolderPositive + movie.RootName + SuffixPositive + ".star"))
                    PosPositive.AddRange(Star.LoadFloat2(FolderPositive + movie.RootName + SuffixPositive + ".star",
                                                            "rlnCoordinateX",
                                                            "rlnCoordinateY").Select(v => v * PixelSize / BoxNet2.PixelSize));
                //if (PosPositive.Count == 0)
                //    continue;

                if (File.Exists(FolderFalsePositive + movie.RootName + SuffixFalsePositive + ".star"))
                    PosFalse.AddRange(Star.LoadFloat2(FolderFalsePositive + movie.RootName + SuffixFalsePositive + ".star",
                                                            "rlnCoordinateX",
                                                            "rlnCoordinateY").Select(v => v * PixelSize / BoxNet2.PixelSize));

                if (File.Exists(FolderUncertain + movie.RootName + SuffixUncertain + ".star"))
                    PosUncertain.AddRange(Star.LoadFloat2(FolderUncertain + movie.RootName + SuffixUncertain + ".star",
                                                            "rlnCoordinateX",
                                                            "rlnCoordinateY").Select(v => v * PixelSize / BoxNet2.PixelSize));

                #endregion

                Image Average = Image.FromFile(movie.AveragePath);
                int2 Dims = new int2(Average.Dims);

                Image Mask = null;
                if (File.Exists(movie.MaskPath))
                    Mask = Image.FromFile(movie.MaskPath);

                float RadiusParticle = Math.Max(1, Diameter / 2 / BoxNet2.PixelSize);
                float RadiusPeak = Math.Max(1.5f, Diameter / 2 / BoxNet2.PixelSize / 4);
                float RadiusFalse = Math.Max(1, Diameter / 2 / BoxNet2.PixelSize);
                float RadiusUncertain = Math.Max(1, Diameter / 2 / BoxNet2.PixelSize);

                #region Rescale everything and allocate memory

                int2 DimsBN = new int2(new float2(Dims) * PixelSize / BoxNet2.PixelSize + 0.5f) / 2 * 2;
                Image AverageBN = Average.AsScaled(DimsBN);
                Average.Dispose();

                if (IsNegative)
                    AverageBN.Multiply(-1f);

                GPU.Normalize(AverageBN.GetDevice(Intent.Read),
                                AverageBN.GetDevice(Intent.Write),
                                (uint)AverageBN.ElementsSliceReal,
                                1);

                Image MaskBN = null;
                if (Mask != null)
                {
                    MaskBN = Mask.AsScaled(DimsBN);
                    Mask.Dispose();
                }

                Image LabelsBN = new Image(new int3(DimsBN));
                Image CertainBN = new Image(new int3(DimsBN));
                CertainBN.Fill(1f);

                #endregion

                #region Paint all positive and uncertain peaks

                for (int i = 0; i < 3; i++)
                {
                    var positions = (new[] { PosPositive, PosFalse, PosUncertain })[i];
                    float R = (new[] { RadiusPeak, RadiusFalse, RadiusUncertain })[i];
                    float R2 = R * R;
                    float Label = (new[] { 1, 4, 0 })[i];
                    float[] ImageData = (new[] { LabelsBN.GetHost(Intent.ReadWrite)[0], CertainBN.GetHost(Intent.ReadWrite)[0], CertainBN.GetHost(Intent.ReadWrite)[0] })[i];

                    foreach (var pos in positions)
                    {
                        int2 Min = new int2(Math.Max(0, (int)(pos.X - R)), Math.Max(0, (int)(pos.Y - R)));
                        int2 Max = new int2(Math.Min(DimsBN.X - 1, (int)(pos.X + R)), Math.Min(DimsBN.Y - 1, (int)(pos.Y + R)));

                        for (int y = Min.Y; y <= Max.Y; y++)
                        {
                            float yy = y - pos.Y;
                            yy *= yy;
                            for (int x = Min.X; x <= Max.X; x++)
                            {
                                float xx = x - pos.X;
                                xx *= xx;

                                float r2 = xx + yy;
                                if (r2 <= R2)
                                    ImageData[y * DimsBN.X + x] = Label;
                            }
                        }
                    }
                }

                #endregion

                #region Add junk mask if there is one

                if (MaskBN != null)
                {
                    float[] LabelsBNData = LabelsBN.GetHost(Intent.ReadWrite)[0];
                    float[] MaskBNData = MaskBN.GetHost(Intent.Read)[0];
                    for (int i = 0; i < LabelsBNData.Length; i++)
                        if (MaskBNData[i] > 0.5f)
                            LabelsBNData[i] = 2;
                }

                #endregion

                #region Clean up

                MaskBN?.Dispose();

                AllAveragesBN.Add(AverageBN);
                AverageBN.FreeDevice();

                AllLabelsBN.Add(LabelsBN);
                LabelsBN.FreeDevice();

                AllCertainBN.Add(CertainBN);
                CertainBN.FreeDevice();

                #endregion
            }

            #region Figure out smallest dimensions that contain everything

            int2 DimsCommon = new int2(1);
            foreach (var image in AllAveragesBN)
            {
                DimsCommon.X = Math.Max(DimsCommon.X, image.Dims.X);
                DimsCommon.Y = Math.Max(DimsCommon.Y, image.Dims.Y);
            }

            #endregion

            #region Put everything in one stack and save

            Image Everything = new Image(new int3(DimsCommon.X, DimsCommon.Y, AllAveragesBN.Count * 3));
            float[][] EverythingData = Everything.GetHost(Intent.ReadWrite);

            for (int i = 0; i < AllAveragesBN.Count; i++)
            {
                if (AllAveragesBN[i].Dims == new int3(DimsCommon)) // No padding needed
                {
                    EverythingData[i * 3 + 0] = AllAveragesBN[i].GetHost(Intent.Read)[0];
                    EverythingData[i * 3 + 1] = AllLabelsBN[i].GetHost(Intent.Read)[0];
                    EverythingData[i * 3 + 2] = AllCertainBN[i].GetHost(Intent.Read)[0];
                }
                else // Padding needed
                {
                    {
                        Image Padded = AllAveragesBN[i].AsPadded(DimsCommon);
                        AllAveragesBN[i].Dispose();
                        EverythingData[i * 3 + 0] = Padded.GetHost(Intent.Read)[0];
                        Padded.FreeDevice();
                    }
                    {
                        Image Padded = AllLabelsBN[i].AsPadded(DimsCommon);
                        AllLabelsBN[i].Dispose();
                        EverythingData[i * 3 + 1] = Padded.GetHost(Intent.Read)[0];
                        Padded.FreeDevice();
                    }
                    {
                        Image Padded = AllCertainBN[i].AsPadded(DimsCommon);
                        AllCertainBN[i].Dispose();
                        EverythingData[i * 3 + 2] = Padded.GetHost(Intent.Read)[0];
                        Padded.FreeDevice();
                    }
                }
            }

            Everything.WriteTIFF(savePath, BoxNet2.PixelSize, typeof(float));

            #endregion
        }

        float2[] GetCentroids(long[] predictions, int2 dims, int border)
        {
            List<int2> Peaks = new List<int2>();

            List<List<int2>> Components = new List<List<int2>>();
            int[] PixelLabels = Helper.ArrayOfConstant(-1, predictions.Length);

            for (int y = 0; y < dims.Y; y++)
            {
                for (int x = 0; x < dims.X; x++)
                {
                    int2 peak = new int2(x, y);

                    if (predictions[dims.ElementFromPosition(peak)] != 1 || PixelLabels[dims.ElementFromPosition(peak)] >= 0)
                        continue;

                    List<int2> Component = new List<int2>() { peak };
                    int CN = Components.Count;

                    PixelLabels[dims.ElementFromPosition(peak)] = CN;
                    Queue<int2> Expansion = new Queue<int2>(100);
                    Expansion.Enqueue(peak);

                    while (Expansion.Count > 0)
                    {
                        int2 pos = Expansion.Dequeue();
                        int PosElement = dims.ElementFromPosition(pos);

                        if (pos.X > 0 && predictions[PosElement - 1] == 1 && PixelLabels[PosElement - 1] < 0)
                        {
                            PixelLabels[PosElement - 1] = CN;
                            Component.Add(pos + new int2(-1, 0));
                            Expansion.Enqueue(pos + new int2(-1, 0));
                        }
                        if (pos.X < dims.X - 1 && predictions[PosElement + 1] == 1 && PixelLabels[PosElement + 1] < 0)
                        {
                            PixelLabels[PosElement + 1] = CN;
                            Component.Add(pos + new int2(1, 0));
                            Expansion.Enqueue(pos + new int2(1, 0));
                        }

                        if (pos.Y > 0 && predictions[PosElement - dims.X] == 1 && PixelLabels[PosElement - dims.X] < 0)
                        {
                            PixelLabels[PosElement - dims.X] = CN;
                            Component.Add(pos + new int2(0, -1));
                            Expansion.Enqueue(pos + new int2(0, -1));
                        }
                        if (pos.Y < dims.Y - 1 && predictions[PosElement + dims.X] == 1 && PixelLabels[PosElement + dims.X] < 0)
                        {
                            PixelLabels[PosElement + dims.X] = CN;
                            Component.Add(pos + new int2(0, 1));
                            Expansion.Enqueue(pos + new int2(0, 1));
                        }
                    }

                    Components.Add(Component);
                }
            }

            return Components.Select(c => MathHelper.Mean(c.Select(v => new float2(v)))).Where(v => v.X >= border &&
                                                                                                    v.Y >= border &&
                                                                                                    v.X < dims.X - border &&
                                                                                                    v.Y < dims.Y - border).ToArray();
        }
    }
}
