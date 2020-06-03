using LiveCharts;
using LiveCharts.Defaults;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Warp;
using Warp.Tools;
using Path = System.IO.Path;

namespace Noise2Particle
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public decimal LearningRate
        {
            get { return (decimal)GetValue(LearningRateProperty); }
            set { SetValue(LearningRateProperty, value); }
        }

        public static readonly DependencyProperty LearningRateProperty = DependencyProperty.Register("LearningRate", typeof(decimal), typeof(MainWindow), new PropertyMetadata(0.0001M));

        public string ParticlesStarPath = "fordenoising_20180321.star";
        public Star TableParticleMetadata;


        private bool ShouldSaveModel = false;
        private bool ShouldSaveRecs = false;

        public float AngPixOri = 4.9875f;
        public float AngPix = 4.9875f;

        private string InputFolderPath = @"H:\sandra_denoising\";

        private int Dim = -1;
        private int DimOri = -1;

        private int BatchSize = 128;


        public int NParticles = 0;

        public int BatchLimit = 1024;

        public float[][] RawParticlesOdd = null;
        public float[][] RawParticlesEven = null;

        public float3[] RawAngles = null;
        public int[] RawSubsets = null;

        public MainWindow()
        {
            InitializeComponent();

            SliderLearningRate.DataContext = this;
        }

        private void ButtonStart_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonStart.IsEnabled = false;

            Task.Run(() =>
            {

                GPU.SetDevice(1);

                List<List<Image>> MapsSource = new List<List<Image>>();
                List<Image> MapsTarget = new List<Image>();


                #region Read STAR

                Console.WriteLine("Reading table...");

                TableParticleMetadata = new Star(InputFolderPath + ParticlesStarPath);
                TableParticleMetadata.RemoveRows(Helper.ArrayOfSequence(15000, TableParticleMetadata.RowCount, 1));
                
                float3[] ParticleAngles = TableParticleMetadata.GetRelionAngles().Select(a => a * Helper.ToRad).ToArray();
                float3[] ParticleShifts = TableParticleMetadata.GetRelionOffsets();

                CTF[] ParticleCTFParams = TableParticleMetadata.GetRelionCTF();
                {
                    float MeanNorm = MathHelper.Mean(ParticleCTFParams.Select(p => (float)p.Scale));
                    for (int p = 0; p < ParticleCTFParams.Length; p++)
                        ParticleCTFParams[p].Scale /= (decimal)MeanNorm;
                }

                RawSubsets = TableParticleMetadata.GetColumn("rlnRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

                string[] ParticleNames = TableParticleMetadata.GetColumn("rlnImageName");
                string[] UniqueMicrographs = Helper.GetUniqueElements(ParticleNames.Select(s => s.Substring(s.IndexOf('@') + 1))).ToArray();

                Console.WriteLine("Done.\n");

                #endregion

                #region Prepare data
                
                NParticles = TableParticleMetadata.RowCount;

                RawParticlesOdd = new float[NParticles][];
                RawParticlesEven = new float[NParticles][];
                RawAngles = new float3[NParticles];
                int NDone = 0;

                Helper.ForCPU(0, UniqueMicrographs.Length, 2, threadID => GPU.SetDevice(0),
                    (imic, threadID) =>
                    {
                        float[][][] RawParticles = { RawParticlesOdd, RawParticlesEven };
                        string[] Suffixes = { "odd", "even" };

                        for (int ioddity = 0; ioddity < 2; ioddity++)
                        {
                            int[] RowIndices = Helper.GetIndicesOf(ParticleNames, (s) => s.Substring(s.IndexOf('@') + 1) == UniqueMicrographs[imic]);
                            string StackFolder = Path.Combine(InputFolderPath, Helper.PathToFolder(ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1)), Suffixes[ioddity]);
                            string StackName = Helper.PathToNameWithExtension(ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1));
                            string StackPath = Path.Combine(StackFolder, StackName);

                            if (!File.Exists(StackPath))
                                throw new Exception($"No data found for {UniqueMicrographs[imic]}!");

                            Image OriginalStack = Image.FromFile(StackPath);

                            lock (TableParticleMetadata)
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

                            float[][] RelevantStackData = Helper.IndexedSubset(OriginalStack.GetHost(Intent.Read), SliceIndices);
                            Image RelevantStack = new Image(RelevantStackData, new int3(DimOri, DimOri, SliceIndices.Length));
                            OriginalStack.Dispose();

                            RelevantStack.ShiftSlices(MicShifts.Select(v => new float3(v.X, v.Y, 0)).ToArray());  // Shift and de-center

                            if (Dim != DimOri)
                            {
                                Image RelevantStackScaled = RelevantStack.AsScaled(new int2(Dim));
                                RelevantStack.Dispose();
                                RelevantStack = RelevantStackScaled;
                            }

                            RelevantStack.FreeDevice();

                            for (int p = 0; p < RowIndices.Length; p++)
                            {
                                RawParticles[ioddity][RowIndices[p]] = RelevantStack.GetHost(Intent.Read)[p];
                                RawAngles[RowIndices[p]] = ParticleAngles[RowIndices[p]];
                            }

                            RelevantStack.Dispose();
                        }

                        lock (TableParticleMetadata)
                        {
                            NDone++;
                            if (threadID == 0)
                                Debug.WriteLine(NDone + "/" + UniqueMicrographs.Length);
                        }
                    }, null);

                #endregion

                WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
                NoiseNet2D TrainModel = new NoiseNet2D(@"D:\Dev\warp2\Noise2Particle\noise2particle_model_export\Noise2Particle_80", new int2(Dim), 1, BatchSize, true, 0);
                WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

                GPU.SetDevice(1);

                Random Rand = new Random(123);
                
                Image ExtractedSource = new Image(new int3(Dim, Dim, BatchSize));
                Image ExtractedTarget = new Image(new int3(Dim, Dim, BatchSize));

                List<ObservablePoint> LossPoints = new List<ObservablePoint>();

                float2[] LearningRateStops =
                {
                    new float2(0, 1e-4f),
                    new float2(30000, 1e-5f),
                    new float2(60000, 1e-6f)
                };

                Func<float2[], int, float> GetInterpolatedRate = (stops, step) =>
                {
                    if (step <= stops[0].X)
                        return stops[0].Y;

                    int PrevStop = 0;
                    while (PrevStop < stops.Length && stops[PrevStop].X < step)
                        PrevStop++;
                    PrevStop--;

                    if (PrevStop >= stops.Length - 1)
                        return stops.Last().Y;

                    float[] LogY = stops.Select(v => (float)Math.Log(v.Y)).ToArray();

                    int NextStop = PrevStop + 1;
                    float Interp = (step - stops[PrevStop].X) / (stops[NextStop].X - stops[PrevStop].X);

                    return (float)Math.Exp(MathHelper.Lerp(LogY[PrevStop], LogY[NextStop], Interp));
                };

                int IterationsDone = 0;

                List<float> AllLosses = new List<float>();
                float[] PredictedData = null, Loss = null;
                float[] SourceData = null;
                float[] TargetData = null;
                float[] AverageData = null;

                while (true)
                {
                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NParticles, 1), BatchSize / 2, Rand.Next(9999));

                    for (int i = 0; i < BatchSize / 2; i++)
                    {
                        //bool Twist = Rand.Next(2) == 0;

                        //if (Twist)
                        {
                            ExtractedSource.GetHost(Intent.Write)[i * 2] = RawParticlesOdd[ShuffledMapIDs[i]];
                            ExtractedTarget.GetHost(Intent.Write)[i * 2] = RawParticlesEven[ShuffledMapIDs[i]];
                        }
                        //else
                        {
                            ExtractedSource.GetHost(Intent.Write)[i * 2 + 1] = RawParticlesEven[ShuffledMapIDs[i]];
                            ExtractedTarget.GetHost(Intent.Write)[i * 2 + 1] = RawParticlesOdd[ShuffledMapIDs[i]];
                        }
                    }

                    float CurrentLearningRate = 0;
                    //Dispatcher.Invoke(() => CurrentLearningRate = (float)LearningRate);
                    CurrentLearningRate = GetInterpolatedRate(LearningRateStops, IterationsDone);
                    
                    TrainModel.Train(ExtractedSource.GetDevice(Intent.Read),
                                    ExtractedTarget.GetDevice(Intent.Read),
                                    CurrentLearningRate,
                                    0,
                                    out PredictedData,
                                    out Loss);
                    AllLosses.Add(Loss[0]);
                    if (AllLosses.Count > 10)
                        AllLosses.RemoveAt(0);

                    SourceData = ExtractedSource.GetHost(Intent.Read)[0];
                    TargetData = ExtractedTarget.GetHost(Intent.Read)[0];

                    if (IterationsDone % 10 == 0)
                    {
                        //WriteToLog(MathHelper.Mean(AllLosses).ToString("F4"));
                        LossPoints.Add(new ObservablePoint(IterationsDone, MathHelper.Mean(AllLosses)));
                        if (LossPoints.Count > 1000)
                            LossPoints.RemoveAt(0);
                        Dispatcher.Invoke(() => SeriesLoss.Values = new ChartValues<ObservablePoint>(LossPoints));

                        // XY
                        {
                            float[] OneSliceXY = Helper.Subset(SourceData, 0, Dim * Dim);
                            float2 MeanStd = MathHelper.MeanAndStd(OneSliceXY);

                            byte[] BytesXY = new byte[OneSliceXY.Length];
                            for (int y = 0; y < Dim; y++)
                                for (int x = 0; x < Dim; x++)
                                {
                                    float Value = (OneSliceXY[y * Dim + x] - MeanStd.X) / MeanStd.Y;
                                    Value = (Value + 7f) / 14f;
                                    BytesXY[(Dim - 1 - y) * Dim + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                                }

                            ImageSource SliceXYImage = BitmapSource.Create(Dim, Dim, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, Dim);
                            SliceXYImage.Freeze();

                            Dispatcher.Invoke(() => ImageSource.Source = SliceXYImage);
                        }

                        // XZ
                        {
                            float[] OneSliceXZ = Helper.Subset(TargetData, 0, Dim * Dim);
                            float2 MeanStd = MathHelper.MeanAndStd(OneSliceXZ);

                            byte[] BytesXY = new byte[OneSliceXZ.Length];
                            for (int y = 0; y < Dim; y++)
                                for (int x = 0; x < Dim; x++)
                                {
                                    float Value = (OneSliceXZ[y * Dim + x] - MeanStd.X) / MeanStd.Y;
                                    Value = (Value + 7f) / 14f;
                                    BytesXY[(Dim - 1 - y) * Dim + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                                }

                            ImageSource SliceXZImage = BitmapSource.Create(Dim, Dim, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, Dim);
                            SliceXZImage.Freeze();

                            Dispatcher.Invoke(() => ImageTarget.Source = SliceXZImage);
                        }

                        // YZ
                        {
                            float[] OneSliceYZ = Helper.Subset(PredictedData, Dim * Dim, Dim * Dim * 2);
                            float2 MeanStd = MathHelper.MeanAndStd(OneSliceYZ);

                            byte[] BytesXY = new byte[OneSliceYZ.Length];
                            for (int y = 0; y < Dim; y++)
                                for (int x = 0; x < Dim; x++)
                                {
                                    float Value = (OneSliceYZ[y * Dim + x] - MeanStd.X) / MeanStd.Y;
                                    Value = (Value + 7f) / 14f;
                                    BytesXY[(Dim - 1 - y) * Dim + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                                }

                            ImageSource SliceYZImage = BitmapSource.Create(Dim, Dim, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, Dim);
                            SliceYZImage.Freeze();

                            Dispatcher.Invoke(() => ImageAverage.Source = SliceYZImage);
                        }

                        // YZ
                        {
                            float[] OneSliceYZ = Helper.Subset(PredictedData, 0, Dim * Dim);
                            float2 MeanStd = MathHelper.MeanAndStd(OneSliceYZ);

                            byte[] BytesXY = new byte[OneSliceYZ.Length];
                            for (int y = 0; y < Dim; y++)
                                for (int x = 0; x < Dim; x++)
                                {
                                    float Value = (OneSliceYZ[y * Dim + x] - MeanStd.X) / MeanStd.Y;
                                    Value = (Value + 7f) / 14f;
                                    BytesXY[(Dim - 1 - y) * Dim + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                                }

                            ImageSource SliceYZImage = BitmapSource.Create(Dim, Dim, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, Dim);
                            SliceYZImage.Freeze();

                            Dispatcher.Invoke(() => ImagePrediction.Source = SliceYZImage);
                        }

                        if (ShouldSaveModel)
                        {
                            ShouldSaveModel = false;

                            TrainModel.Export(InputFolderPath + @"noise2particle_trained\Noise2Particle_80_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                            Thread.Sleep(10000);

                            Dispatcher.Invoke(() => ButtonSave.IsEnabled = true);
                        }
                    }

                    IterationsDone++;
                    //Dispatcher.Invoke(() => TextCoverage.Text = $"{IterationsDone} iterations done");
                }
            });
        }

        private void ButtonSave_OnClick(object sender, RoutedEventArgs e)
        {
            ShouldSaveModel = true;
            ButtonSave.IsEnabled = false;
        }

        private void WriteToLog(string line)
        {
            Dispatcher.Invoke(() =>
            {
                TextOutput.Text += line + "\n";
                TextOutput.ScrollToLine(TextOutput.LineCount - 1);
            });
        }

        private void ButtonTest_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonTest.IsEnabled = false;

            Task.Run(() =>
            {
                WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
                NoiseNet2D TrainModel = new NoiseNet2D(@"H:\sandra_denoising\noise2particle_trained\Noise2Particle_80_20190715_190302", new int2(80), 1, BatchSize, false, 0);
                WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

                GPU.SetDevice(1);

                #region Read STAR

                Console.WriteLine("Reading table...");

                TableParticleMetadata = new Star(InputFolderPath + ParticlesStarPath);
                //TableParticleMetadata.RemoveRows(Helper.ArrayOfSequence(5000, TableParticleMetadata.RowCount, 1));

                float3[] ParticleAngles = TableParticleMetadata.GetRelionAngles().Select(a => a * Helper.ToRad).ToArray();
                float3[] ParticleShifts = TableParticleMetadata.GetRelionOffsets();

                CTF[] ParticleCTFParams = TableParticleMetadata.GetRelionCTF();
                {
                    float MeanNorm = MathHelper.Mean(ParticleCTFParams.Select(p => (float)p.Scale));
                    for (int p = 0; p < ParticleCTFParams.Length; p++)
                        ParticleCTFParams[p].Scale /= (decimal)MeanNorm;
                }

                RawSubsets = TableParticleMetadata.GetColumn("rlnRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

                string[] ParticleNames = TableParticleMetadata.GetColumn("rlnImageName");
                string[] UniqueMicrographs = Helper.GetUniqueElements(ParticleNames.Select(s => s.Substring(s.IndexOf('@') + 1))).ToArray();

                Console.WriteLine("Done.\n");

                #endregion

                #region Prepare data

                NParticles = TableParticleMetadata.RowCount;

                RawParticlesOdd = new float[NParticles][];
                RawParticlesEven = new float[NParticles][];
                RawAngles = new float3[NParticles];
                int NDone = 0;

                Image ImagesForDenoising = new Image(new int3(80, 80, BatchSize));

                Helper.ForCPU(0, UniqueMicrographs.Length, 1, threadID => GPU.SetDevice(0),
                    (imic, threadID) =>
                    {
                        float[][][] RawParticles = { RawParticlesOdd, RawParticlesEven };
                        string[] Suffixes = { "odd", "even" };

                        ImagesForDenoising.Fill(0);
                        int NSlices = 0;

                        for (int ioddity = 0; ioddity < 2; ioddity++)
                        {
                            int[] RowIndices = Helper.GetIndicesOf(ParticleNames, (s) => s.Substring(s.IndexOf('@') + 1) == UniqueMicrographs[imic]);
                            string StackFolder = Path.Combine(InputFolderPath, Helper.PathToFolder(ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1)), Suffixes[ioddity]);
                            string StackName = Helper.PathToNameWithExtension(ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1));
                            string StackPath = Path.Combine(StackFolder, StackName);

                            if (!File.Exists(StackPath))
                                throw new Exception($"No data found for {UniqueMicrographs[imic]}!");

                            Image OriginalStack = Image.FromFile(StackPath);

                            lock (TableParticleMetadata)
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

                            float[][] RelevantStackData = Helper.IndexedSubset(OriginalStack.GetHost(Intent.Read), SliceIndices);
                            Image RelevantStack = new Image(RelevantStackData, new int3(DimOri, DimOri, SliceIndices.Length));
                            OriginalStack.Dispose();

                            RelevantStack.ShiftSlices(MicShifts.Select(v => new float3(v.X, v.Y, 0)).ToArray());  // Shift and de-center

                            if (Dim != DimOri)
                            {
                                Image RelevantStackScaled = RelevantStack.AsScaled(new int2(Dim));
                                RelevantStack.Dispose();
                                RelevantStack = RelevantStackScaled;
                            }

                            for (int b = 0; b < BatchSize; b++)
                            {
                                GPU.AddToSlices(ImagesForDenoising.GetDeviceSlice(b, Intent.Read),
                                                RelevantStack.GetDeviceSlice(b % RelevantStack.Dims.Z, Intent.Read),
                                                ImagesForDenoising.GetDeviceSlice(b, Intent.Write),
                                                Dim * Dim,
                                                1);
                            }

                            NSlices = RelevantStack.Dims.Z;

                            RelevantStack.Dispose();
                        }

                        {
                            ImagesForDenoising.Multiply(0.5f);

                            float[] Denoised = null;
                            TrainModel.Predict(ImagesForDenoising.GetDevice(Intent.Read), 0, out Denoised);

                            Image DenoisedImage = new Image(Denoised.Take(Dim * Dim * NSlices).ToArray(), new int3(Dim, Dim, NSlices));

                            int[] RowIndices = Helper.GetIndicesOf(ParticleNames, (s) => s.Substring(s.IndexOf('@') + 1) == UniqueMicrographs[imic]);
                            string StackFolder = Path.Combine(InputFolderPath, Helper.PathToFolder(ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1)), "denoised");
                            string StackName = Helper.PathToNameWithExtension(ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1));
                            string StackPath = Path.Combine(StackFolder, StackName);

                            DenoisedImage.WriteMRC(StackPath, AngPix, true);
                            DenoisedImage.Dispose();
                        }
                        WriteToLog(++NDone + "/" + UniqueMicrographs.Length);
                    }, null);

                #endregion
            });
        }
    }
}
