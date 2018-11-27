using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
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
using System.Windows.Threading;
using Accord.Math.Optimization;
using LiveCharts;
using LiveCharts.Defaults;
using MahApps.Metro;
using MahApps.Metro.Controls;
using MahApps.Metro.Controls.Dialogs;
using Warp.Controls;
using Warp.Controls.TaskDialogs.Tomo;
using Warp.Controls.TaskDialogs.TwoD;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using Menu = System.Windows.Forms.Menu;

namespace Warp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MahApps.Metro.Controls.MetroWindow
    {
        private const string DefaultAnalyticsName = "analytics.settings";
        public static Analytics Analytics = new Analytics();

        #region MAIN WINDOW

        private CheckBox[] CheckboxesGPUStats;
        private int[] BaselinesGPUStats;
        private DispatcherTimer TimerGPUStats;
        private DispatcherTimer TimerCheckUpdates;

        public MainWindow()
        {
            #region Make sure everything is OK with GPUs
            try
            {
                Options.Runtime.DeviceCount = GPU.GetDeviceCount();
                if (Options.Runtime.DeviceCount <= 0)
                    throw new Exception();
            }
            catch (Exception exc)
            {
                MessageBox.Show("No CUDA devices found, or couldn't load GPUAcceleration.dll due to missing dependencies, shutting down.\n\n" +
                                "First things to check:\n" +
                                "-At least one GPU with Maxwell (GeForce 9xx, Quadro Mxxxx, Tesla Mxx) or later architecture available?\n" +
                                "-Latest GPU driver installed?\n" +
                                "-VC++ 2017 redistributable installed?\n" +
                                "-Any bundled libraries missing? (reinstall Warp to be sure)\n" +
                                "\n" +
                                "If none of this works, please report the issue in https://groups.google.com/forum/#!forum/warp-em");
                Close();
            }

            GPU.SetDevice(0);
            #endregion

            DataContext = Options;

            InitializeComponent();

            #region Options events

            Options.PropertyChanged += Options_PropertyChanged;
            Options.Import.PropertyChanged += OptionsImport_PropertyChanged;
            Options.CTF.PropertyChanged += OptionsCTF_PropertyChanged;
            Options.Movement.PropertyChanged += OptionsMovement_PropertyChanged;
            Options.Grids.PropertyChanged += OptionsGrids_PropertyChanged;
            Options.Tomo.PropertyChanged += OptionsTomo_PropertyChanged;
            Options.Picking.PropertyChanged += OptionsPicking_PropertyChanged;
            Options.Export.PropertyChanged += OptionsExport_PropertyChanged;
            Options.Tasks.PropertyChanged += OptionsTasks_PropertyChanged;
            Options.Filter.PropertyChanged += OptionsFilter_PropertyChanged;
            Options.Advanced.PropertyChanged += OptionsAdvanced_PropertyChanged;
            Options.Runtime.PropertyChanged += OptionsRuntime_PropertyChanged;

            #endregion

            Closing += MainWindow_Closing;

            #region GPU statistics

            CheckboxesGPUStats = Helper.ArrayOfFunction(i =>
                                                        {
                                                            CheckBox NewCheckBox = new CheckBox
                                                            {
                                                                Foreground = Brushes.White,
                                                                Margin = new Thickness(10, 0, 10, 0),
                                                                IsChecked = true,
                                                                Opacity = 0.5,
                                                                Focusable = false
                                                            };
                                                            NewCheckBox.MouseEnter += (a, b) => NewCheckBox.Opacity = 1;
                                                            NewCheckBox.MouseLeave += (a, b) => NewCheckBox.Opacity = 0.5;

                                                            return NewCheckBox;
                                                        },
                                                        GPU.GetDeviceCount());
            foreach (var checkBox in CheckboxesGPUStats)
                PanelGPUStats.Children.Add(checkBox);
            BaselinesGPUStats = Helper.ArrayOfFunction(i => (int)GPU.GetFreeMemory(i), GPU.GetDeviceCount());

            TimerGPUStats = new DispatcherTimer(new TimeSpan(0, 0, 0, 0, 200), DispatcherPriority.Background, (a, b) =>
            {
                for (int i = 0; i < CheckboxesGPUStats.Length; i++)
                {
                    int CurrentMemory = (int)GPU.GetFreeMemory(i);
                    CheckboxesGPUStats[i].Content = $"GPU {i}: {CurrentMemory} MB";

                    //float Full = 1 - (float)CurrentMemory / BaselinesGPUStats[i];
                    //Color ColorFull = Color.FromRgb((byte)MathHelper.Lerp(Colors.White.R, Colors.DeepPink.R, Full),
                    //                                (byte)MathHelper.Lerp(Colors.White.G, Colors.DeepPink.G, Full),
                    //                                (byte)MathHelper.Lerp(Colors.White.B, Colors.DeepPink.B, Full));
                    //SolidColorBrush BrushFull = new SolidColorBrush(ColorFull);
                    //BrushFull.Freeze();
                    //CheckboxesGPUStats[i].Foreground = BrushFull;
                }
            }, Dispatcher);

            #endregion

            #region Control set definitions

            DisableWhenPreprocessing = new List<UIElement>
            {
                GridOptionsIO,
                GridOptionsIOTomo,
                GridOptionsPreprocessing,
                GridOptionsCTF,
                GridOptionsMovement,
                GridOptionsGrids,
                GridOptionsPicking,
                GridOptionsPostprocessing,
                ButtonOptionsSave,
                ButtonOptionsLoad,
                ButtonOptionsAdopt,
                ButtonProcessOneItemCTF,
                SwitchProcessCTF,
                SwitchProcessMovement,
                SwitchProcessPicking,
                PanelOverviewTasks2D,
                PanelOverviewTasks3D
            };
            DisableWhenPreprocessing.AddRange(CheckboxesGPUStats);

            HideWhen2D = new List<UIElement>
            {
                CheckCTFSimultaneous,
                GridOptionsIOTomo,
                PanelOverviewTasks3D
            };
            foreach (var element in HideWhen2D)
                element.Visibility = Visibility.Collapsed;

            HideWhenTomo = new List<UIElement>
            {
                ButtonTasksAdjustDefocus,
                ButtonTasksExportParticles,
                //CheckCTFDoIce,
                PanelProcessMovement,
                GridOptionsPicking,
                GridMovement,
                GridOptionsMovement,
                LabelModelsHeader,
                GridOptionsGrids,
                GridOptionsPicking,
                SwitchProcessPicking,
                LabelOutputHeader,
                GridOptionsPostprocessing,
                GridOptionsIO2D,
                PanelOverviewTasks2D
            };

            HideWhenNoActiveItem = new List<UIElement>
            {
                ButtonProcessOneItemCTF
            };

            #endregion

            #region File discoverer

            FileDiscoverer = new FileDiscoverer();
            FileDiscoverer.FilesChanged += FileDiscoverer_FilesChanged;
            FileDiscoverer.IncubationStarted += FileDiscoverer_IncubationStarted;
            FileDiscoverer.IncubationEnded += FileDiscoverer_IncubationEnded;

            #endregion

            ProcessingStatusBar.ActiveItemStatusChanged += UpdateStatsStatus;

            // Load settings from previous session
            if (File.Exists(DefaultOptionsName))
                Options.Load(DefaultOptionsName);

            OptionsAutoSave = true;
            OptionsLookForFolderOptions = true;

            Options.MainWindow = this;

            if (File.Exists(DefaultAnalyticsName))
                Analytics.Load(DefaultAnalyticsName);

            UpdateStatsAll();

            #region Perform version check online

            if (Analytics.PromptShown)
                Dispatcher.InvokeAsync(async () =>
                {
                    try
                    {
                        Version CurrentVersion = Assembly.GetExecutingAssembly().GetName().Version;
                        Version LatestVersion = Analytics.GetLatestVersion();

                        if (CurrentVersion < LatestVersion)
                        {
                            var MessageResult = await this.ShowMessageAsync("How the time flies!",
                                                                            $"It's been a while since you updated Warp. You're running version {CurrentVersion}, but version {LatestVersion} is even better! Would you like to go to the download page?",
                                                                            MessageDialogStyle.AffirmativeAndNegative,
                                                                            new MetroDialogSettings
                                                                            {
                                                                                AffirmativeButtonText = "Yes",
                                                                                NegativeButtonText = "No"
                                                                            });

                            if (MessageResult == MessageDialogResult.Affirmative)
                            {
                                Process.Start("http://www.warpem.com/warp/?page_id=65");
                            }

                            ButtonUpdateAvailable.Visibility = Visibility.Visible;
                        }
                    }
                    catch
                    {
                    }
                }, DispatcherPriority.ApplicationIdle);

            TimerCheckUpdates = new DispatcherTimer();
            TimerCheckUpdates.Interval = new TimeSpan(0, 10, 0);
            TimerCheckUpdates.Tick += (sender, e) =>
            {
                Dispatcher.InvokeAsync(() =>
                {
                    try
                    {
                        Version CurrentVersion = Assembly.GetExecutingAssembly().GetName().Version;
                        Version LatestVersion = Analytics.GetLatestVersion();

                        if (CurrentVersion < LatestVersion)
                            ButtonUpdateAvailable.Visibility = Visibility.Visible;
                    }
                    catch
                    {
                    }
                });
            };
            TimerCheckUpdates.Start();

            #endregion

            #region Show prompt on first run

            if (Analytics.ShowTiffReminder)
                Dispatcher.InvokeAsync(async () =>
                {
                    var DialogResult = await this.ShowMessageAsync("Careful there!",
                                                                   "As of v1.0.6, Warp handles TIFF files differently. Find out more at http://www.warpem.com/warp/?page_id=361.\n" +
                                                                   "Go there now?", MessageDialogStyle.AffirmativeAndNegative);
                    if (DialogResult == MessageDialogResult.Affirmative)
                    {
                        Process.Start("http://www.warpem.com/warp/?page_id=361");
                    }
                    else
                    {
                        Analytics.ShowTiffReminder = false;
                        Analytics.Save(DefaultAnalyticsName);
                    }
                }, DispatcherPriority.ApplicationIdle);

            if (!Analytics.PromptShown)
                Dispatcher.InvokeAsync(async () =>
                {
                    CustomDialog Dialog = new CustomDialog();
                    Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                    FirstRunPrompt DialogContent = new FirstRunPrompt();
                    DialogContent.Close += () =>
                    {
                        this.HideMetroDialogAsync(Dialog);
                        Analytics.PromptShown = true;
                        Analytics.AllowCollection = (bool)DialogContent.CheckAgree.IsChecked;

                        Analytics.Save(DefaultAnalyticsName);

                        Analytics.LogEnvironment();
                    };
                    Dialog.Content = DialogContent;

                    this.ShowMetroDialogAsync(Dialog);
                }, DispatcherPriority.ApplicationIdle);

            #endregion

            // Report hardware environment
            Analytics.LogEnvironment();

            #region Set up log message windows

            Logger.SetBufferCount(GPU.GetDeviceCount());
            Logger.MessageLogged += Logger_MessageLogged;

            GridMessageLogs.Children.Clear();
            GridMessageLogs.ColumnDefinitions.Clear();
            LogMessagePanels = new StackPanel[GPU.GetDeviceCount()];

            for (int c = 0; c < GPU.GetDeviceCount(); c++)
            {
                GridMessageLogs.ColumnDefinitions.Add(new ColumnDefinition());

                TextBlock TextLogTitle = new TextBlock
                {
                    Text = "GPU " + c,
                    FontSize = 20
                };
                GridMessageLogs.Children.Add(TextLogTitle);
                Grid.SetColumn(TextLogTitle, c);

                ScrollViewer ViewerLog = new ScrollViewer
                {
                    HorizontalScrollBarVisibility = ScrollBarVisibility.Hidden,
                    VerticalScrollBarVisibility = ScrollBarVisibility.Visible,
                    HorizontalAlignment = HorizontalAlignment.Stretch,
                    VerticalAlignment = VerticalAlignment.Stretch
                };
                GridMessageLogs.Children.Add(ViewerLog);
                Grid.SetRow(ViewerLog, 1);
                Grid.SetColumn(ViewerLog, c);

                StackPanel PanelLog = new StackPanel()
                {
                    Orientation = Orientation.Vertical,
                    HorizontalAlignment = HorizontalAlignment.Stretch,
                    VerticalAlignment = VerticalAlignment.Top
                };
                ViewerLog.Content = PanelLog;
                LogMessagePanels[c] = PanelLog;
            }

            Logger.Write(new LogMessage("Test.", "System"));
            Logger.Write(new LogMessage("Nothing really important.", "General"));
            Logger.Write(new LogMessage("Lorem ipsum.", "System"));
            //for (int i = 0; i < 200; i++)
            //{
            //    Logger.Write(new LogMessage(Guid.NewGuid().ToString(), "System " + i));
            //}

            #endregion

            #region Create mockup

            {
                //Options.WriteToXML(null);

                float2[] SplinePoints = { new float2(0f, 0f), new float2(1f / 3f, 1f)};//, new float2(2f / 3f, 0f)};//, new float2(1f, 1f) };
                Cubic1D ReferenceSpline = new Cubic1D(SplinePoints);
                Cubic1DShort ShortSpline = Cubic1DShort.GetInterpolator(SplinePoints);
                for (float i = -1f; i < 2f; i += 0.01f)
                {
                    float Reference = ReferenceSpline.Interp(i);
                    float Test = ShortSpline.Interp(i);
                    if (Math.Abs(Reference - Test) > 1e-6f)
                        throw new Exception();
                }

                Random Rnd = new Random(123);
                int3 GridDim = new int3(1, 1, 1);
                float[] GridValues = new float[GridDim.Elements()];
                for (int i = 0; i < GridValues.Length; i++)
                    GridValues[i] = (float)Rnd.NextDouble();
                CubicGrid CGrid = new CubicGrid(GridDim, GridValues);
                float[] Managed = CGrid.GetInterpolated(new int3(16, 16, 16), new float3(0, 0, 0));
                float[] Native = CGrid.GetInterpolatedNative(new int3(16, 16, 16), new float3(0, 0, 0));
                for (int i = 0; i < Managed.Length; i++)
                    if (Math.Abs(Managed[i] - Native[i]) > 1e-6f)
                        throw new Exception();

                Matrix3 A = new Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9);
                Matrix3 B = new Matrix3(11, 12, 13, 14, 15, 16, 17, 18, 19);
                Matrix3 C = A * B;

                // Euler matrix
                {
                    Matrix3 E = Matrix3.Euler(0 * Helper.ToRad, 20 * Helper.ToRad, 0 * Helper.ToRad);
                    float3 EE = Matrix3.EulerFromMatrix(E.Transposed()) * Helper.ToDeg;

                    float3 Transformed = E * new float3(1, 0, 0);
                    Transformed.Y = 0;
                }

                //{
                //    Image VolIntensities = Image.FromFile("E:\\Dropbox\\GRC2017\\models\\80S.mrc");
                //    RendererMock1.Supersampling = 2;
                //    RendererMock1.Camera.SurfaceThreshold = 0.025M;
                //    RendererMock1.Camera.IntensityRangeMin = 0;
                //    RendererMock1.Camera.IntensityRangeMax = 1;
                //    RendererMock1.Volume = VolIntensities;
                //}

                //{
                //    Image VolIntensities = Image.FromFile("E:\\Dropbox\\GRC2017\\models\\chloro70S.mrc");
                //    RendererMock2.Supersampling = 2;
                //    RendererMock2.Camera.SurfaceThreshold = 0.025M;
                //    RendererMock2.Camera.IntensityRangeMin = 0;
                //    RendererMock2.Camera.IntensityRangeMax = 1;
                //    RendererMock2.Volume = VolIntensities;
                //}

                //{
                //    Image VolIntensities = Image.FromFile("E:\\Dropbox\\GRC2017\\models\\26S.mrc");
                //    RendererMock3.Supersampling = 2;
                //    RendererMock3.Camera.SurfaceThreshold = 0.025M;
                //    RendererMock3.Camera.IntensityRangeMin = 0;
                //    RendererMock3.Camera.IntensityRangeMax = 1;
                //    RendererMock3.Volume = VolIntensities;
                //}

                //float3[] HealpixAngles = Helper.GetHealpixAngles(3, "D4");

                // Deconvolve reconstructions using a separate CTF
                //{
                //    for (int i = 1; i <= 24; i++)
                //    {
                //        Image Map = StageDataLoad.LoadMap($"F:\\stefanribo\\vlion\\warped_{i}.mrc", new int2(1, 1), 0, typeof(float));
                //        Image MapFT = Map.AsFFT(true);
                //        Map.Dispose();

                //        Image CTF = StageDataLoad.LoadMap($"F:\\stefanribo\\vlion\\warped_ctf_{i}.mrc", new int2(1, 1), 0, typeof(float));
                //        foreach (var slice in CTF.GetHost(Intent.ReadWrite))
                //            for (int s = 0; s < slice.Length; s++)
                //                slice[s] = Math.Max(1e-3f, slice[s]);

                //        MapFT.Divide(CTF);
                //        Map = MapFT.AsIFFT(true);
                //        MapFT.Dispose();

                //        Map.WriteMRC($"F:\\stefanribo\\vlion\\warped_deconv_{i}.mrc");
                //        Map.Dispose();
                //    }
                //}

                //{
                //    Image SumFT = new Image(new int3(64, 64, 64), true, true);
                //    Image SumWeights = new Image(new int3(64, 64, 64), true);

                //    Star TableIn = new Star("F:\\badaben\\ppca_nomem\\nomembrane_membrane_class.star");
                //    string[] ColumnClass = TableIn.GetColumn("rlnClassNumber");

                //    int read = 0;
                //    foreach (var tomoPath in Directory.EnumerateFiles("F:\\badaben\\ppca_nomem\\particles", "*_*.mrc"))
                //    {
                //        string CN = ColumnClass[read];

                //        if (read++ % 10 != 0 || CN != "1")
                //            continue;

                //        FileInfo Info = new FileInfo(tomoPath);

                //        Image Tomo = StageDataLoad.LoadMap(tomoPath, new int2(1, 1), 0, typeof(float));
                //        Image TomoFT = Tomo.AsFFT(true);
                //        Tomo.Dispose();

                //        Image TomoWeights = StageDataLoad.LoadMap("F:\\badaben\\ppca_nomem\\particlectf\\" + Info.Name, new int2(1, 1), 0, typeof(float));

                //        TomoFT.Multiply(TomoWeights);
                //        TomoWeights.Multiply(TomoWeights);

                //        SumFT.Add(TomoFT);
                //        SumWeights.Add(TomoWeights);

                //        TomoFT.Dispose();
                //        TomoWeights.Dispose();

                //        Debug.WriteLine(read);
                //    }

                //    foreach (var slice in SumWeights.GetHost(Intent.ReadWrite))
                //        for (int i = 0; i < slice.Length; i++)
                //            slice[i] = Math.Max(1e-3f, slice[i]);

                //    SumFT.Divide(SumWeights);
                //    Image Sum = SumFT.AsIFFT(true);
                //    Sum.WriteMRC("F:\\badaben\\ppca_nomem\\subtomomean.mrc");

                //    SumFT.Dispose();
                //    SumWeights.Dispose();
                //    Sum.Dispose();
                //}

                //{
                //    Image Subtrahend = StageDataLoad.LoadMap("E:\\martinsried\\stefan\\membranebound\\vlion\\relion_subtrahend.mrc", new int2(1, 1), 0, typeof(float));
                //    Image SubtrahendFT = Subtrahend.AsFFT(true);

                //    int read = 0;
                //    foreach (var tomoPath in Directory.EnumerateFiles("E:\\martinsried\\stefan\\membranebound\\oridata\\particles", "tomo*.mrc"))
                //    {
                //        FileInfo Info = new FileInfo(tomoPath);

                //        Image Tomo = StageDataLoad.LoadMap(tomoPath, new int2(1, 1), 0, typeof(float));
                //        Image TomoFT = Tomo.AsFFT(true);
                //        Tomo.Dispose();

                //        Image TomoWeights = StageDataLoad.LoadMap("E:\\martinsried\\stefan\\membranebound\\oridata\\particlectf\\" + Info.Name, new int2(1, 1), 0, typeof(float));

                //        Image SubtrahendFTMult = new Image(SubtrahendFT.GetDevice(Intent.Read), SubtrahendFT.Dims, true, true);
                //        SubtrahendFTMult.Multiply(TomoWeights);

                //        TomoFT.Subtract(SubtrahendFTMult);
                //        Tomo = TomoFT.AsIFFT(true);

                //        Tomo.WriteMRC("D:\\stefanribo\\particles\\" + Info.Name);

                //        Tomo.Dispose();
                //        TomoFT.Dispose();
                //        SubtrahendFTMult.Dispose();
                //        TomoWeights.Dispose();

                //        Debug.WriteLine(read++);
                //    }
                //}

                //{
                //    Image SubtRef1 = StageDataLoad.LoadMap("E:\\martinsried\\stefan\\membranebound\\vlion\\warp_subtrahend.mrc", new int2(1, 1), 0, typeof(float));
                //    Projector Subt = new Projector(SubtRef1, 2);
                //    SubtRef1.Dispose();

                //    Image ProjFT = Subt.Project(new int2(220, 220), new[] { new float3(0, 0, 0) }, 110);
                //    Image Proj = ProjFT.AsIFFT();
                //    Proj.RemapFromFT();

                //    Proj.WriteMRC("d_testproj.mrc");
                //}

                // Projector
                /*{
                    Image MapForProjector = StageDataLoad.LoadMap("E:\\youwei\\run36_half1_class001_unfil.mrc", new int2(1, 1), 0, typeof (float));
                    Projector Proj = new Projector(MapForProjector, 2);
                    Image Projected = Proj.Project(new int2(240, 240), new[] { new float3(0, 0, 0) }, 120);
                    Projected = Projected.AsIFFT();
                    Projected.RemapFromFT();
                    Projected.WriteMRC("d_projected.mrc");
                }*/

                // Backprojector
                /*{
                    Image Dot = new Image(new int3(32, 32, 360));
                    for (int a = 0; a < 360; a++)
                        Dot.GetHost(Intent.Write)[a][0] = 1;
                    Dot = Dot.AsFFT();
                    Dot.AsAmplitudes().WriteMRC("d_dot.mrc");

                    Image DotWeights = new Image(new int3(32, 32, 360), true);
                    for (int a = 0; a < 360; a++)
                        for (int i = 0; i < DotWeights.ElementsSliceReal; i++)
                            DotWeights.GetHost(Intent.Write)[a][i] = 1;

                    float3[] Angles = new float3[360];
                    for (int a = 0; a < 360; a++)
                        Angles[a] = new float3(0, a * Helper.ToRad * 0.05f, 0);

                    Projector Proj = new Projector(new int3(32, 32, 32), 2);
                    Proj.BackProject(Dot, DotWeights, Angles);

                    Proj.Weights.WriteMRC("d_weights.mrc");
                    //Image Re = Proj.Data.AsImaginary();
                    //Re.WriteMRC("d_projdata.mrc");

                    Image Rec = Proj.Reconstruct(true);
                    Rec.WriteMRC("d_rec.mrc");
                }*/

                //Star Models = new Star("D:\\rado27\\Refine3D\\run1_ct5_it005_half1_model.star", "data_model_group_2");
                //Debug.WriteLine(Models.GetRow(0)[0]);

                /*Image Volume = StageDataLoad.LoadMap("F:\\carragher20s\\ref256.mrc", new int2(1, 1), 0, typeof (float));
                Image VolumePadded = Volume.AsPadded(new int3(512, 512, 512));
                VolumePadded.WriteMRC("d_padded.mrc");
                Volume.Dispose();
                VolumePadded.RemapToFT(true);
                Image VolumeFT = VolumePadded.AsFFT(true);
                VolumePadded.Dispose();

                Image VolumeProjFT = VolumeFT.AsProjections(new[] { new float3(Helper.ToRad * 0, Helper.ToRad * 0, Helper.ToRad * 0) }, new int2(256, 256), 2f);
                Image VolumeProj = VolumeProjFT.AsIFFT();
                VolumeProjFT.Dispose();
                VolumeProj.RemapFromFT();
                VolumeProj.WriteMRC("d_proj.mrc");
                VolumeProj.Dispose();*/

                /*Options.Movies.Add(new Movie(@"D:\Dev\warp\May19_21.44.54.mrc"));
                Options.Movies.Add(new Movie(@"D:\Dev\warp\May19_21.49.06.mrc"));
                Options.Movies.Add(new Movie(@"D:\Dev\warp\May19_21.50.48.mrc"));
                Options.Movies.Add(new Movie(@"D:\Dev\warp\May19_21.52.16.mrc"));
                Options.Movies.Add(new Movie(@"D:\Dev\warp\May19_21.53.43.mrc"));

                CTFDisplay.PS2D = new BitmapImage();*/

                /*float2[] SimCoords = new float2[512 * 512];
                for (int y = 0; y < 512; y++)
                    for (int x = 0; x < 512; x++)
                    {
                        int xcoord = x - 512, ycoord = y - 512;
                        SimCoords[y * 512 + x] = new float2((float) Math.Sqrt(xcoord * xcoord + ycoord * ycoord),
                            (float) Math.Atan2(ycoord, xcoord));
                    }
                float[] Sim2D = new CTF {Defocus = -2M}.Get2D(SimCoords, 512, true);
                byte[] Sim2DBytes = new byte[Sim2D.Length];
                for (int i = 0; i < 512 * 512; i++)
                    Sim2DBytes[i] = (byte) (Sim2D[i] * 255f);
                BitmapSource Sim2DSource = BitmapSource.Create(512, 512, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, Sim2DBytes, 512);
                CTFDisplay.Simulated2D = Sim2DSource;*/

                /*float2[] PointsPS1D = new float2[512];
                for (int i = 0; i < PointsPS1D.Length; i++)
                    PointsPS1D[i] = new float2(i, (float) Math.Exp(-i / 300f));
                CTFDisplay.PS1D = PointsPS1D;

                float[] SimCTF = new CTF { Defocus = -2M }.Get1D(512, true);
                float2[] PointsSim1D = new float2[SimCTF.Length];
                for (int i = 0; i < SimCTF.Length; i++)
                    PointsSim1D[i] = new float2(i, SimCTF[i] * (float)Math.Exp(-i / 100f) + (float)Math.Exp(-i / 300f));
                CTFDisplay.Simulated1D = PointsSim1D;*/

                /*CubicGrid Grid = new CubicGrid(new int3(5, 5, 5), 0, 0, Dimension.X);
                Grid.Values[2, 2, 2] = 1f;
                float[] Data = new float[11 * 11 * 11];
                int i = 0;
                for (float z = 0f; z < 1.05f; z += 0.1f)
                    for (float y = 0f; y < 1.05f; y += 0.1f)
                        for (float x = 0f; x < 1.05f; x += 0.1f)
                            Data[i++] = Grid.GetInterpolated(new float3(x, y, z));
                Image DataImage = new Image(Data, new int3(11, 11, 11));
                DataImage.WriteMRC("bla.mrc");

                Image GPUImage = new Image(DataImage.GetDevice(Intent.Read), new int3(11, 11, 11));
                GPUImage.WriteMRC("gpu.mrc");*/

                /*CubicGrid WiggleGrid = new CubicGrid(new int3(2, 2, 1));
                float[][] WiggleWeights = WiggleGrid.GetWiggleWeights(new int3(3, 3, 1));*/
            }

            #endregion
        }

        private void MainWindow_Closing(object sender, CancelEventArgs e)
        {
            try
            {
                SaveDefaultSettings();
                FileDiscoverer.Shutdown();
            }
            catch (Exception)
            {
                // ignored
            }
        }
        
        private void ButtonUpdateAvailable_OnClick(object sender, RoutedEventArgs e)
        {
            Process.Start("http://www.warpem.com/warp/?page_id=65");
        }

        private void SwitchDayNight_OnClick(object sender, RoutedEventArgs e)
        {
            if (ThemeManager.DetectAppStyle().Item1.Name == "BaseLight")
            {
                ThemeManager.ChangeAppStyle(Application.Current,
                                            ThemeManager.GetAccent("Blue"),
                                            ThemeManager.GetAppTheme("BaseDark"));

                this.GlowBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#304160"));
                this.WindowTitleBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#304160"));

                SwitchDayNight.Content = "🦇";
            }
            else
            {
                ThemeManager.ChangeAppStyle(Application.Current,
                                            ThemeManager.GetAccent("Blue"),
                                            ThemeManager.GetAppTheme("BaseLight"));

                this.GlowBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#41b1e1"));
                this.WindowTitleBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#41b1e1"));

                SwitchDayNight.Content = "🔆";
            }

            GridCTF.UpdateLines();
            GridMovement.UpdateLines();
        }

        #region Hot keys

        public ActionCommand HotKeyLeft
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (TabProcessingCTF.IsSelected || TabProcessingMovement.IsSelected || TabProcessingCTFAndMovement.IsSelected)
                        ProcessingStatusBar.MoveToOtherItem(-1);
                });
            }
        }

        public ActionCommand HotKeyRight
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (TabProcessingCTF.IsSelected || TabProcessingMovement.IsSelected || TabProcessingCTFAndMovement.IsSelected)
                        ProcessingStatusBar.MoveToOtherItem(1);
                });
            }
        }

        public ActionCommand HotKeyW
        {
            get
            {
                return new ActionCommand(() =>
                {
                    TabProcessingOverview.IsSelected = true;
                });
            }
        }

        public ActionCommand HotKeyE
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (IsPreprocessingCollapsed)
                        TabProcessingCTFAndMovement.IsSelected = true;
                    else
                        TabProcessingCTF.IsSelected = true;
                });
            }
        }

        public ActionCommand HotKeyR
        {
            get 
            {
                return new ActionCommand(() =>
                {
                    if (IsPreprocessingCollapsed)
                        TabProcessingCTFAndMovement.IsSelected = true;
                    else
                        TabProcessingMovement.IsSelected = true;
                });
            }
        }

        public ActionCommand HotKeyF
        {
            get
            {
                return new ActionCommand(() =>
                {
                    if (TabProcessingCTF.IsSelected || TabProcessingMovement.IsSelected || TabProcessingCTFAndMovement.IsSelected)
                        if (Options.Runtime.DisplayedMovie != null)
                        {
                            if (Options.Runtime.DisplayedMovie.UnselectManual == null || !(bool)Options.Runtime.DisplayedMovie.UnselectManual)
                                Options.Runtime.DisplayedMovie.UnselectManual = true;
                            else
                                Options.Runtime.DisplayedMovie.UnselectManual = false;
                            //ProcessingStatusBar.UpdateElements();
                        }
                });
            }
        }

        #endregion

        #endregion

        #region Options

        #region Helper variables

        const string DefaultOptionsName = "previous.settings";

        public static Options Options = new Options();
        static bool OptionsAutoSave = false;
        static bool OptionsLookForFolderOptions = false;

        #endregion

        private async void Options_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "Import.Folder")
            {
                if (!IOHelper.CheckFolderPermission(Options.Import.Folder))
                {
                    Options.Import.Folder = "";
                    return;
                }
                ButtonInputPathText.Text = Options.Import.Folder == "" ? "Select folder..." : Options.Import.Folder;

                if (OptionsLookForFolderOptions)
                {
                    OptionsLookForFolderOptions = false;

                    if (File.Exists(Options.Import.Folder + DefaultOptionsName))
                    {
                        var MessageResult = await this.ShowMessageAsync("Options File Found in Folder",
                                                                        "A file with options from a previous Warp session was found in this folder. Load it?",
                                                                        MessageDialogStyle.AffirmativeAndNegative,
                                                                        new MetroDialogSettings
                                                                        {
                                                                            AffirmativeButtonText = "Yes",
                                                                            NegativeButtonText = "No"
                                                                        });

                        if (MessageResult == MessageDialogResult.Affirmative)
                        {
                            string SelectedFolder = Options.Import.Folder;

                            Options.Load(Options.Import.Folder + DefaultOptionsName);

                            Options.Import.Folder = SelectedFolder;
                        }
                    }

                    OptionsLookForFolderOptions = true;
                }

                AdjustInput();
                TomoAdjustInterface();
            }
            else if (e.PropertyName == "Import.Extension")
            {
                AdjustInput();
                TomoAdjustInterface();
            }
            else if (e.PropertyName == "Import.GainPath")
            {
                if (!File.Exists(Options.Import.GainPath))
                {
                    Options.Import.GainPath = "";
                    return;
                }

                Options.Runtime.GainReferenceHash = MathHelper.GetSHA1(Options.Import.GainPath, 1 << 20);
                ButtonGainPathText.Text = Options.Import.GainPath == "" ? "Select gain reference..." : Options.Import.GainPath;
                ButtonGainPathText.ToolTip = Options.Import.GainPath == "" ? null : Options.Import.GainPath;
            }
            else if (e.PropertyName == "CTF.Window")
            {
                CTFDisplayControl.Width = CTFDisplayControl.Height = Math.Min(1024, Options.CTF.Window);
            }
            else if (e.PropertyName == "CTF.DoPhase")
            {
                UpdateFilterRanges();
            }
            else if (e.PropertyName == "PixelSizeX")
            {
                UpdateFilterRanges();

                if (OptionsAutoSave)
                {
                    Options.Tasks.TomoFullReconstructPixel = Math.Max(Options.PixelSizeMean, Options.Tasks.TomoFullReconstructPixel);
                    Options.Tasks.TomoSubReconstructPixel = Math.Max(Options.PixelSizeMean, Options.Tasks.TomoSubReconstructPixel);
                }
            }
            else if (e.PropertyName == "PixelSizeY")
            {
                UpdateFilterRanges();

                if (OptionsAutoSave)
                {
                    Options.Tasks.TomoFullReconstructPixel = Math.Max(Options.PixelSizeMean, Options.Tasks.TomoFullReconstructPixel);
                    Options.Tasks.TomoSubReconstructPixel = Math.Max(Options.PixelSizeMean, Options.Tasks.TomoSubReconstructPixel);
                }
            }
            else if (e.PropertyName == "PixelSizeAngle")
            {
                TransformPixelAngle.Angle = -(double)Options.PixelSizeAngle;
            }
            else if (e.PropertyName == "ProcessCTF")
            {
                if (!Options.ProcessCTF)
                {
                    GridCTF.Opacity = 0.5;
                    PanelGridCTFParams.Opacity = 0.5;
                }
                else
                {
                    GridCTF.Opacity = 1;
                    PanelGridCTFParams.Opacity = 1;
                }

                UpdateFilterRanges();
                SaveDefaultSettings();
            }
            else if (e.PropertyName == "ProcessMovement")
            {
                if (!Options.ProcessMovement)
                {
                    GridMovement.Opacity = 0.5;
                    PanelGridMovementParams.Opacity = 0.5;
                }
                else
                {
                    GridMovement.Opacity = 1;
                    PanelGridMovementParams.Opacity = 1;
                }

                UpdateFilterRanges();
                SaveDefaultSettings();
            }
            else if (e.PropertyName == "Runtime.DisplayedMovie")
            {
                foreach (var element in HideWhenNoActiveItem)
                    element.Visibility = Options.Runtime.DisplayedMovie == null ? Visibility.Collapsed : Visibility.Visible;
            }
            else if (e.PropertyName == "Picking.ModelPath")
            {
                if (string.IsNullOrEmpty(LocatePickingModel(Options.Picking.ModelPath)))
                {
                    Options.Picking.ModelPath = "";
                    //return;
                }
                ButtonPickingModelNameText.Text = Options.Picking.ModelPath == "" ? "Select BoxNet model..." : Options.Picking.ModelPath;
                MicrographDisplayControl.UpdateBoxNetName(Options.Picking.ModelPath);
            }

            if (OptionsAutoSave && !e.PropertyName.StartsWith("Tasks"))
            {
                Dispatcher.Invoke(() =>
                {
                    ProcessingStatusBar.UpdateElements();
                    UpdateStatsStatus();
                    UpdateButtonOptionsAdopt();
                });
            }
        }

        private void OptionsImport_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsCTF_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsMovement_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsGrids_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsTomo_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsPicking_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsExport_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsTasks_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsFilter_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            UpdateFilterRanges();
            UpdateFilterResult();
            UpdateStatsStatus();
            SaveDefaultSettings();
        }

        private void OptionsAdvanced_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            SaveDefaultSettings();
        }

        private void OptionsRuntime_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "DisplayedMovie")
            {
                UpdateButtonOptionsAdopt();
            }
        }

        private void SaveDefaultSettings()
        {
            if (OptionsAutoSave)
            {
                try
                {
                    Options.Save(DefaultOptionsName);
                    Analytics.Save(DefaultAnalyticsName);
                } catch { }

                if (Options.Import.Folder != "")
                    try
                    {
                        Options.Save(Options.Import.Folder + DefaultOptionsName);
                    } catch { }
            }
        }

        #endregion

        #region File Discoverer

        public readonly FileDiscoverer FileDiscoverer;

        private void FileDiscoverer_FilesChanged()
        {
            Movie[] ImmutableItems = null;
            Helper.Time("FileDiscoverer.GetImmutableFiles", () => ImmutableItems = FileDiscoverer.GetImmutableFiles());

            Dispatcher.InvokeAsync(() =>
            {
                ProcessingStatusBar.Items = new ObservableCollection<Movie>(ImmutableItems);
                if (Options.Runtime.DisplayedMovie == null && ImmutableItems.Length > 0)
                    Options.Runtime.DisplayedMovie = ImmutableItems[0];
            });

            Helper.Time("FileDiscoverer.UpdateStatsAll", () => UpdateStatsAll());
        }

        private void FileDiscoverer_IncubationStarted()
        {
            
        }

        private void FileDiscoverer_IncubationEnded()
        {
            
        }

        #endregion

        #region TAB: RAW DATA

        #region Helper variables

        public static bool IsPreprocessing = false;
        static Task PreprocessingTask = null;

        bool IsPreprocessingCollapsed = false;
        int PreprocessingWidth = 450;

        readonly List<UIElement> DisableWhenPreprocessing;
        readonly List<UIElement> HideWhen2D, HideWhenTomo;
        readonly List<UIElement> HideWhenNoActiveItem;

        #endregion

        #region Left menu panel

        private void ButtonInputPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.Folder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (!IOHelper.CheckFolderPermission(Dialog.SelectedPath))
                {
                    MessageBox.Show("Don't have permission to access the selected folder.");
                    return;
                }

                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                OptionsAutoSave = false;
                Options.Import.Folder = Dialog.SelectedPath;
                OptionsAutoSave = true;
            }
        }

        private void ButtonInputExtension_OnClick(object sender, RoutedEventArgs e)
        {
            PopupInputExtension.IsOpen = true;
        }

        private void ButtonGainPath_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "Image Files|*.dm4;*.mrc;*.em",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Options.Import.GainPath = Dialog.FileName;
                Options.Import.CorrectGain = true;
            }
        }

        private void ButtonOptionsSave_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.SaveFileDialog Dialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "Setting Files|*.settings"
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();
            if (Result == System.Windows.Forms.DialogResult.OK)
            {
                Options.Save(Dialog.FileName);
            }
        }

        private void ButtonOptionsLoad_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "Setting Files|*.settings",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();
            if (Result == System.Windows.Forms.DialogResult.OK)
            {
                Options.Load(Dialog.FileName);
            }
        }

        #region Options adoption

        private void ButtonOptionsAdopt_OnClick(object sender, RoutedEventArgs e)
        {
            if (Options.Runtime.DisplayedMovie != null)
            {
                ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();

                ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(Options.Runtime.DisplayedMovie, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options);

                if (Status == ProcessingStatus.Outdated)
                {
                    ButtonOptionsAdopt.Visibility = Visibility.Visible;
                    OptionsAutoSave = false;

                    if (Options.Runtime.DisplayedMovie.OptionsCTF != null)
                        Options.Adopt(Options.Runtime.DisplayedMovie.OptionsCTF);
                    if (Options.Runtime.DisplayedMovie.OptionsMovement != null)
                        Options.Adopt(Options.Runtime.DisplayedMovie.OptionsMovement);
                    if (Options.Runtime.DisplayedMovie.OptionsMovieExport != null)
                        Options.Adopt(Options.Runtime.DisplayedMovie.OptionsMovieExport);

                    OptionsAutoSave = true;
                    SaveDefaultSettings();
                    ProcessingStatusBar.UpdateElements();
                    UpdateStatsStatus();

                    bool IsConflicted = Options.Runtime.DisplayedMovie.AreOptionsConflicted();
                    if (IsConflicted)
                        this.ShowMessageAsync("Don't panic, but...", "... the input options for individual processing steps have conflicting parameters. Please reprocess the data with uniform input options.");
                }
            }

            ButtonOptionsAdopt.Visibility = Visibility.Hidden;
        }

        private void UpdateButtonOptionsAdopt()
        {
            ButtonOptionsAdopt.Visibility = Visibility.Hidden;

            if (Options.Runtime.DisplayedMovie != null)
            {
                ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();

                ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(Options.Runtime.DisplayedMovie, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options);

                if (Status == ProcessingStatus.Outdated)
                    ButtonOptionsAdopt.Visibility = Visibility.Visible;
            }
        }

        #endregion

        #region Picking
        
        private void ButtonPickingModelName_OnClick(object sender, RoutedEventArgs e)
        {
            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            BoxNetSelect DialogContent = new BoxNetSelect(Options.Picking.ModelPath, Options);
            DialogContent.Close += () =>
            {
                Options.Picking.ModelPath = DialogContent.ModelName;
                this.HideMetroDialogAsync(Dialog);
            };
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        public string LocatePickingModel(string name)
        {
            if (string.IsNullOrEmpty(name))
                return null;

            if (Directory.Exists(name))
            {
                return name;
            }
            else if (Directory.Exists(System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet2models/" + name)))
            {
                return System.IO.Path.Combine(Environment.CurrentDirectory, "boxnet2models/" + name);
            }

            return null;
        }

        #endregion

        #endregion

        private async void ButtonStartProcessing_OnClick(object sender, RoutedEventArgs e)
        {
            if (!IsPreprocessing)
            {
                foreach (var item in DisableWhenPreprocessing)
                    item.IsEnabled = false;
                MicrographDisplayControl.SetProcessingMode(true);

                ButtonStartProcessing.Content = "STOP PROCESSING";
                ButtonStartProcessing.Foreground = Brushes.Red;
                IsPreprocessing = true;

                PreprocessingTask = Task.Run(async () =>
                {
                    List<int> UsedDevices = GetDeviceList();

                    #region Check if options are compatible

                    {
                        string ErrorMessage = "";
                    }

                    #endregion

                    #region Load gain reference if needed

                    Image[] ImageGain = new Image[GPU.GetDeviceCount()];
                    if (!string.IsNullOrEmpty(Options.Import.GainPath) && Options.Import.CorrectGain && File.Exists(Options.Import.GainPath))
                        foreach (int d in UsedDevices)
                            try
                            {
                                GPU.SetDevice(d);
                                //ImageGain[d] = Image.FromFilePatient(50, 500,
                                //                                     Options.Import.GainPath,
                                //                                     new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                //                                     (int)Options.Import.HeaderlessOffset,
                                //                                     ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));
                                ImageGain[d] = LoadAndPrepareGainReference();
                            }
                            catch (Exception exc)
                            {
                                foreach (int dd in UsedDevices)
                                    ImageGain[dd]?.Dispose();

                                await Dispatcher.InvokeAsync(async () =>
                                {
                                    await this.ShowMessageAsync("Oopsie",
                                                                "Something went wrong when trying to load the gain reference.\n\n" +
                                                                "The exception raised is:\n" + exc);

                                    ButtonStartProcessing_OnClick(sender, e);
                                });

                                return;
                            }

                    #endregion

                    #region Load BoxNet model if needed

                    BoxNet2[] BoxNetworks = new BoxNet2[GPU.GetDeviceCount()];
                    if (Options.ProcessPicking)
                    {
                        ProgressDialogController ProgressDialog = null;

                        try
                        {
                            await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Loading {Options.Picking.ModelPath} model...", ""));
                            ProgressDialog.SetIndeterminate();

                            if (string.IsNullOrEmpty(Options.Picking.ModelPath) || LocatePickingModel(Options.Picking.ModelPath) == null)
                                throw new Exception("No BoxNet model selected. Please use the options panel to select a model.");

                            MicrographDisplayControl.DropBoxNetworks();
                            foreach (var d in UsedDevices)
                                BoxNetworks[d] = new BoxNet2(LocatePickingModel(Options.Picking.ModelPath), d, 2, 1, false);
                        }
                        catch (Exception exc)
                        {
                            await Dispatcher.Invoke(async () =>
                            {
                                await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                    "There was an error loading the specified BoxNet model for picking.\n\n" +
                                                                                                    "The exception raised is:\n" + exc);

                                ButtonStartProcessing_OnClick(sender, e);
                            });

                            foreach (int dd in UsedDevices)
                                ImageGain[dd]?.Dispose();

                            await ProgressDialog.CloseAsync();

                            return;
                        }

                        await ProgressDialog.CloseAsync();
                    }

                    #endregion

                    /*#region Wait until all discovered files have been loaded

                    if (FileDiscoverer.IsIncubating())
                    {
                        ProgressDialogController ProgressDialog = null;
                        await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Waiting for all discovered items to be loaded...", ""));
                        ProgressDialog.SetIndeterminate();

                        while (FileDiscoverer.IsIncubating())
                            Thread.Sleep(50);

                        await ProgressDialog.CloseAsync();
                    }

                    #endregion*/

                    #region Load or create STAR table for BoxNet output, if needed

                    string BoxNetSuffix = Helper.PathToNameWithExtension(Options.Picking.ModelPath);

                    Star TableBoxNetAll = null;
                    string PathBoxNetAll = Options.Import.Folder + "allparticles_" + BoxNetSuffix + ".star";
                    string PathBoxNetAllSubset = Options.Import.Folder + "allparticles_last" + Options.Picking.RunningWindowLength + "_" + BoxNetSuffix + ".star";
                    string PathBoxNetFiltered = Options.Import.Folder + "goodparticles_" + BoxNetSuffix + ".star";
                    string PathBoxNetFilteredSubset = Options.Import.Folder + "goodparticles_last" + Options.Picking.RunningWindowLength + "_" + BoxNetSuffix + ".star";
                    object TableBoxNetAllWriteLock = new object();
                    int TableBoxNetConcurrent = 0;

                    // Switch filter suffix to the one used in current processing
                    //if (Options.ProcessPicking)
                    //    Dispatcher.Invoke(() => Options.Filter.ParticlesSuffix = "_" + BoxNetSuffix);

                    Dictionary<Movie, List<List<string>>> AllMovieParticleRows = new Dictionary<Movie, List<List<string>>>();

                    if (Options.ProcessPicking && Options.Picking.DoExport && !string.IsNullOrEmpty(Options.Picking.ModelPath))
                    {
                        Movie[] TempMovies = FileDiscoverer.GetImmutableFiles();

                        if (File.Exists(PathBoxNetAll))
                        {
                            ProgressDialogController ProgressDialog = null;
                            await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Loading particle metadata from previous run...", ""));
                            ProgressDialog.SetIndeterminate();

                            TableBoxNetAll = new Star(PathBoxNetAll);

                            Dictionary<string, Movie> NameMapping = new Dictionary<string, Movie>();
                            string[] ColumnMicName = TableBoxNetAll.GetColumn("rlnMicrographName");
                            for (int r = 0; r < ColumnMicName.Length; r++)
                            {
                                if (!NameMapping.ContainsKey(ColumnMicName[r]))
                                {
                                    var Movie = TempMovies.Where(m => ColumnMicName[r].Contains(m.Name));
                                    if (Movie.Count() != 1)
                                        continue;

                                    NameMapping.Add(ColumnMicName[r], Movie.First());
                                    AllMovieParticleRows.Add(Movie.First(), new List<List<string>>());
                                }

                                AllMovieParticleRows[NameMapping[ColumnMicName[r]]].Add(TableBoxNetAll.GetRow(r));
                            }

                            await ProgressDialog.CloseAsync();
                        }
                        else
                        {
                            TableBoxNetAll = new Star(new string[] { });
                        }

                        #region Make sure all columns are there

                        if (!TableBoxNetAll.HasColumn("rlnCoordinateX"))
                            TableBoxNetAll.AddColumn("rlnCoordinateX", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnCoordinateY"))
                            TableBoxNetAll.AddColumn("rlnCoordinateY", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnMagnification"))
                            TableBoxNetAll.AddColumn("rlnMagnification", "10000.0");
                        else
                            TableBoxNetAll.SetColumn("rlnMagnification", Helper.ArrayOfConstant("10000.0", TableBoxNetAll.RowCount));

                        if (!TableBoxNetAll.HasColumn("rlnDetectorPixelSize"))
                            TableBoxNetAll.AddColumn("rlnDetectorPixelSize", Options.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture));
                        else
                            TableBoxNetAll.SetColumn("rlnDetectorPixelSize", Helper.ArrayOfConstant(Options.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture), TableBoxNetAll.RowCount));

                        if (!TableBoxNetAll.HasColumn("rlnVoltage"))
                            TableBoxNetAll.AddColumn("rlnVoltage", "300.0");

                        if (!TableBoxNetAll.HasColumn("rlnSphericalAberration"))
                            TableBoxNetAll.AddColumn("rlnSphericalAberration", "2.7");

                        if (!TableBoxNetAll.HasColumn("rlnAmplitudeContrast"))
                            TableBoxNetAll.AddColumn("rlnAmplitudeContrast", "0.07");

                        if (!TableBoxNetAll.HasColumn("rlnPhaseShift"))
                            TableBoxNetAll.AddColumn("rlnPhaseShift", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnDefocusU"))
                            TableBoxNetAll.AddColumn("rlnDefocusU", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnDefocusV"))
                            TableBoxNetAll.AddColumn("rlnDefocusV", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnDefocusAngle"))
                            TableBoxNetAll.AddColumn("rlnDefocusAngle", "0.0");

                        if (!TableBoxNetAll.HasColumn("rlnCtfMaxResolution"))
                            TableBoxNetAll.AddColumn("rlnCtfMaxResolution", "999.0");

                        if (!TableBoxNetAll.HasColumn("rlnImageName"))
                            TableBoxNetAll.AddColumn("rlnImageName", "None");

                        if (!TableBoxNetAll.HasColumn("rlnMicrographName"))
                            TableBoxNetAll.AddColumn("rlnMicrographName", "None");

                        #endregion

                        #region Repair

                        var RepairMovies = TempMovies.Where(m => !AllMovieParticleRows.ContainsKey(m) && m.OptionsBoxNet != null && File.Exists(m.MatchingDir + m.RootName + "_" + BoxNetSuffix + ".star")).ToList();
                        if (RepairMovies.Count() > 0)
                        {
                            ProgressDialogController ProgressDialog = null;
                            await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Repairing particle metadata...", ""));

                            int NRepaired = 0;
                            foreach (var item in RepairMovies)
                            {
                                float2[] Positions = Star.LoadFloat2(item.MatchingDir + item.RootName + "_" + BoxNetSuffix + ".star",
                                                                     "rlnCoordinateX",
                                                                     "rlnCoordinateY");

                                float[] Defoci = new float[Positions.Length];
                                if (item.GridCTF != null)
                                    Defoci = item.GridCTF.GetInterpolated(Positions.Select(v => new float3(v.X / (item.OptionsBoxNet.Dimensions.X / (float)item.OptionsBoxNet.BinnedPixelSizeMean),
                                                                                                           v.Y / (item.OptionsBoxNet.Dimensions.Y / (float)item.OptionsBoxNet.BinnedPixelSizeMean),
                                                                                                           0.5f)).ToArray());
                                float Astigmatism = (float)item.CTF.DefocusDelta / 2;
                                float PhaseShift = item.GridCTFPhase.GetInterpolated(new float3(0.5f)) * 180;

                                List<List<string>> NewRows = new List<List<string>>();
                                for (int r = 0; r < Positions.Length; r++)
                                {
                                    string[] Row = Helper.ArrayOfConstant("0", TableBoxNetAll.ColumnCount);

                                    Row[TableBoxNetAll.GetColumnID("rlnMagnification")] = "10000.0";
                                    Row[TableBoxNetAll.GetColumnID("rlnDetectorPixelSize")] = item.OptionsBoxNet.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture);

                                    Row[TableBoxNetAll.GetColumnID("rlnDefocusU")] = ((Defoci[r] + Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnDefocusV")] = ((Defoci[r] - Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnDefocusAngle")] = item.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnVoltage")] = item.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnSphericalAberration")] = item.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnAmplitudeContrast")] = item.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnPhaseShift")] = PhaseShift.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnCtfMaxResolution")] = item.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnCoordinateX")] = Positions[r].X.ToString("F2", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnCoordinateY")] = Positions[r].Y.ToString("F2", CultureInfo.InvariantCulture);
                                    Row[TableBoxNetAll.GetColumnID("rlnImageName")] = (r + 1).ToString("D7") + "@particles/" + item.RootName + "_" + BoxNetSuffix + ".mrcs";
                                    Row[TableBoxNetAll.GetColumnID("rlnMicrographName")] = item.Name;

                                    NewRows.Add(Row.ToList());
                                }

                                AllMovieParticleRows.Add(item, NewRows);

                                NRepaired++;
                                Dispatcher.Invoke(() => ProgressDialog.SetProgress((float)NRepaired / RepairMovies.Count));
                            }

                            await ProgressDialog.CloseAsync();
                        }

                        #endregion
                    }

                    #endregion

                    bool CheckedGainDims = ImageGain[0] == null;

                    while (true)
                    {
                        if (!IsPreprocessing)
                            break;

                        #region Figure out what needs preprocessing

                        Movie[] ImmutableItems = FileDiscoverer.GetImmutableFiles();
                        List<Movie> NeedProcessing = new List<Movie>();

                        ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
                        ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
                        ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();
                        ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();

                        bool DoCTF = Options.ProcessCTF;
                        bool DoMovement = Options.ProcessMovement;
                        bool DoPicking = Options.ProcessPicking;

                        foreach (var item in ImmutableItems)
                        {
                            ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(item, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options, false);

                            if (Status == ProcessingStatus.Outdated || Status == ProcessingStatus.Unprocessed)
                                NeedProcessing.Add(item);
                        }

                        #endregion

                        if (NeedProcessing.Count == 0)
                        {
                            Task.Delay(20);
                            continue;
                        }

                        #region Make sure gain dims match those of first image to be processed

                        if (!CheckedGainDims)
                        {
                            string ItemPath;

                            if (NeedProcessing[0].GetType() == typeof(Movie))
                                ItemPath = NeedProcessing[0].Path;
                            else
                                ItemPath = ((TiltSeries)NeedProcessing[0]).TiltMoviePaths[0];

                            MapHeader Header = MapHeader.ReadFromFilePatient(50, 500,
                                                                             ItemPath,
                                                                             new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                                             Options.Import.HeaderlessOffset,
                                                                             ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));

                            if (Header.Dimensions.X != ImageGain[0].Dims.X || Header.Dimensions.Y != ImageGain[0].Dims.Y)
                            {
                                await Dispatcher.InvokeAsync(async () =>
                                {
                                    await this.ShowMessageAsync("Oopsie", "Image dimensions do not match those of the gain reference. Maybe it needs to be rotated or transposed?");

                                    ButtonStartProcessing_OnClick(sender, e);
                                });

                                break;
                            }

                            CheckedGainDims = true;
                        }

                        #endregion

                        Dispatcher.Invoke(() =>
                        {
                            ProcessingStatusBar.ShowProgressBar();
                            StatsProgressIndicator.Visibility = Visibility.Visible;
                        });

                        int NSimultaneous = 1;
                        int NPreread = NSimultaneous * 1;

                        Dictionary<int, SemaphoreSlim> SemaphoresPreread = new Dictionary<int, SemaphoreSlim>();
                        Dictionary<int, SemaphoreSlim> SemaphoresProcessing = new Dictionary<int, SemaphoreSlim>();
                        foreach (var device in UsedDevices)
                        {
                            SemaphoresPreread.Add(device, new SemaphoreSlim(1));
                            SemaphoresProcessing.Add(device, new SemaphoreSlim(NSimultaneous));
                        }

                        #region Perform preprocessing on all available GPUs

                        Helper.ForEachGPU(NeedProcessing, (item, gpuID) =>
                        {
                            if (!IsPreprocessing)
                                return true;    // This cancels the iterator

                            Image OriginalStack = null;
                            bool AcquiredProcessingSemaphore = false;

                            try
                            {
                                var TimerOverall = BenchmarkAllProcessing.Start();

                                bool IsTomo = item.GetType() == typeof(TiltSeries);

                                ProcessingOptionsMovieCTF CurrentOptionsCTF = Options.GetProcessingMovieCTF();
                                ProcessingOptionsMovieMovement CurrentOptionsMovement = Options.GetProcessingMovieMovement();
                                ProcessingOptionsBoxNet CurrentOptionsBoxNet = Options.GetProcessingBoxNet();
                                ProcessingOptionsMovieExport CurrentOptionsExport = Options.GetProcessingMovieExport();

                                bool DoExport = OptionsExport.DoAverage || OptionsExport.DoStack || OptionsExport.DoDeconv || (DoPicking && !File.Exists(item.AveragePath));

                                bool NeedsNewCTF = CurrentOptionsCTF != item.OptionsCTF && DoCTF;
                                bool NeedsNewMotion = CurrentOptionsMovement != item.OptionsMovement && DoMovement;
                                bool NeedsNewPicking = DoPicking &&
                                                       (CurrentOptionsBoxNet != item.OptionsBoxNet ||
                                                        NeedsNewMotion);
                                bool NeedsNewExport = DoExport &&
                                                      (NeedsNewMotion ||
                                                       CurrentOptionsExport != item.OptionsMovieExport ||
                                                       (CurrentOptionsExport.DoDeconv && NeedsNewCTF));

                                bool NeedsMoreDenoisingExamples = !Directory.Exists(item.DenoiseTrainingDirOdd) || 
                                                                 Directory.EnumerateFiles(item.DenoiseTrainingDirOdd, "*.mrc").Count() < 128;   // Having more than 128 examples is a waste of space
                                bool DoesDenoisingExampleExist = File.Exists(item.DenoiseTrainingOddPath);
                                bool NeedsDenoisingExample = NeedsMoreDenoisingExamples || (DoesDenoisingExampleExist && (NeedsNewCTF || NeedsNewExport));
                                CurrentOptionsExport.DoDenoise = NeedsDenoisingExample;

                                MapHeader OriginalHeader = null;
                                decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                                bool NeedStack = NeedsNewCTF ||
                                                 NeedsNewMotion ||
                                                 NeedsNewExport ||
                                                 (NeedsNewPicking && CurrentOptionsBoxNet.ExportParticles);

                                if (!IsTomo)
                                {
                                    SemaphoresPreread[gpuID].Wait();

                                    Debug.WriteLine(GPU.GetDevice() + " loading...");
                                    var TimerRead = BenchmarkRead.Start();

                                    LoadAndPrepareHeaderAndMap(item.Path, ImageGain[gpuID], ScaleFactor, out OriginalHeader, out OriginalStack, NeedStack);

                                    BenchmarkRead.Finish(TimerRead);
                                    Debug.WriteLine(GPU.GetDevice() + " loaded.");
                                    SemaphoresPreread[gpuID].Release();
                                }

                                // Store original dimensions in Angstrom
                                if (!IsTomo)
                                {
                                    CurrentOptionsCTF.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.PixelSizeMean);
                                    CurrentOptionsMovement.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.PixelSizeMean);
                                    CurrentOptionsBoxNet.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.PixelSizeMean);
                                    CurrentOptionsExport.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.PixelSizeMean);
                                }
                                else
                                {
                                    ((TiltSeries)item).LoadMovieSizes(CurrentOptionsCTF);

                                    float3 StackDims = new float3(((TiltSeries)item).ImageDimensionsPhysical[0], ((TiltSeries)item).NTilts);
                                    CurrentOptionsCTF.Dimensions = StackDims;
                                    CurrentOptionsMovement.Dimensions = StackDims;
                                    CurrentOptionsExport.Dimensions = StackDims;
                                }
                                
                                SemaphoresProcessing[gpuID].Wait();
                                AcquiredProcessingSemaphore = true;
                                Debug.WriteLine(GPU.GetDevice() + " processing...");

                                if (!IsPreprocessing)
                                {
                                    OriginalStack?.Dispose();
                                    SemaphoresProcessing[gpuID].Release();
                                    return true;
                                } // These checks are needed to abort the processing faster

                                if (DoCTF && NeedsNewCTF)
                                {
                                    var TimerCTF = BenchmarkCTF.Start();

                                    if (item.GetType() == typeof(Movie))
                                        item.ProcessCTF(OriginalStack, CurrentOptionsCTF);
                                    else
                                        ((TiltSeries)item).ProcessCTFSimultaneous(CurrentOptionsCTF);

                                    BenchmarkCTF.Finish(TimerCTF);
                                    Analytics.LogProcessingCTF(CurrentOptionsCTF, item.CTF, (float)item.CTFResolutionEstimate);
                                }
                                if (!IsPreprocessing)
                                {
                                    OriginalStack?.Dispose();
                                    SemaphoresProcessing[gpuID].Release();
                                    return true;
                                }

                                if (DoMovement && NeedsNewMotion && item.GetType() != typeof(TiltSeries))
                                {
                                    var TimerMotion = BenchmarkMotion.Start();

                                    item.ProcessShift(OriginalStack, CurrentOptionsMovement);

                                    BenchmarkMotion.Finish(TimerMotion);
                                    Analytics.LogProcessingMovement(CurrentOptionsMovement, (float)item.MeanFrameMovement);
                                }
                                if (!IsPreprocessing)
                                {
                                    OriginalStack?.Dispose();
                                    SemaphoresProcessing[gpuID].Release();
                                    return true;
                                }

                                if (DoExport && NeedsNewExport)
                                {
                                    var TimerOutput = BenchmarkOutput.Start();

                                    item.ExportMovie(OriginalStack, CurrentOptionsExport);

                                    BenchmarkOutput.Finish(TimerOutput);
                                }

                                // In case no average was written out, force its creation to enable micrograph display

                                if (!File.Exists(item.AveragePath) && item.GetType() == typeof(Movie))
                                {
                                    if (!Directory.Exists(item.AverageDir))
                                        Directory.CreateDirectory(item.AverageDir);
                                    
                                    var TimerOutput = BenchmarkOutput.Start();

                                    Image StackAverage = OriginalStack.AsReducedAlongZ();
                                    OriginalStack.FreeDevice();
                                    //Task.Run(() =>
                                    {
                                        StackAverage.WriteMRC(item.AveragePath, (float)Options.BinnedPixelSizeMean, true);
                                        StackAverage.Dispose();

                                        if (!DoExport) // No settings stored yet
                                        {
                                            item.OptionsMovieExport = CurrentOptionsExport;
                                            item.SaveMeta();
                                        }
                                    }//);
                                    
                                    BenchmarkOutput.Finish(TimerOutput);
                                }

                                if (!File.Exists(item.ThumbnailsPath))
                                    item.CreateThumbnail(384, 2.5f);

                                if (DoPicking && NeedsNewPicking)
                                {
                                    var TimerPicking = BenchmarkPicking.Start();

                                    Image AverageForPicking = Image.FromFilePatient(50, 500, item.AveragePath);
                                    item.MatchBoxNet2(new[] { BoxNetworks[gpuID] }, AverageForPicking, CurrentOptionsBoxNet, null);

                                    Analytics.LogProcessingBoxNet(CurrentOptionsBoxNet, item.GetParticleCount("_" + BoxNetSuffix));

                                    #region Export particles if needed

                                    if (CurrentOptionsBoxNet.ExportParticles)
                                    {
                                        float2[] Positions = Star.LoadFloat2(item.MatchingDir + item.RootName + "_" + BoxNetSuffix + ".star",
                                                                             "rlnCoordinateX",
                                                                             "rlnCoordinateY").Select(v => v * AverageForPicking.PixelSize).ToArray();

                                        ProcessingOptionsParticlesExport ParticleOptions = new ProcessingOptionsParticlesExport
                                        {
                                            Suffix = "_" + BoxNetSuffix,

                                            BoxSize = CurrentOptionsBoxNet.ExportBoxSize,
                                            Diameter = (int)CurrentOptionsBoxNet.ExpectedDiameter,
                                            Invert = CurrentOptionsBoxNet.ExportInvert,
                                            Normalize = CurrentOptionsBoxNet.ExportNormalize,
                                            CorrectAnisotropy = true,

                                            PixelSizeX = CurrentOptionsBoxNet.PixelSizeX,
                                            PixelSizeY = CurrentOptionsBoxNet.PixelSizeY,
                                            PixelSizeAngle = CurrentOptionsBoxNet.PixelSizeAngle,
                                            Dimensions = CurrentOptionsBoxNet.Dimensions,

                                            BinTimes = CurrentOptionsBoxNet.BinTimes,
                                            GainPath = CurrentOptionsBoxNet.GainPath,
                                            DosePerAngstromFrame = Options.Import.DosePerAngstromFrame,

                                            DoAverage = true,
                                            DoStack = false,
                                            StackGroupSize = 1,
                                            SkipFirstN = Options.Export.SkipFirstN,
                                            SkipLastN = Options.Export.SkipLastN,

                                            Voltage = Options.CTF.Voltage
                                        };

                                        item.ExportParticles(OriginalStack, Positions, ParticleOptions);

                                        OriginalStack?.Dispose();
                                        Debug.WriteLine(GPU.GetDevice() + " processed.");
                                        SemaphoresProcessing[gpuID].Release();

                                        float[] Defoci = new float[Positions.Length];
                                        if (item.GridCTF != null)
                                            Defoci = item.GridCTF.GetInterpolated(Positions.Select(v => new float3(v.X / CurrentOptionsBoxNet.Dimensions.X,
                                                                                                                   v.Y / CurrentOptionsBoxNet.Dimensions.Y,
                                                                                                                   0.5f)).ToArray());
                                        float Astigmatism = (float)item.CTF.DefocusDelta / 2;
                                        float PhaseShift = item.GridCTFPhase.GetInterpolated(new float3(0.5f)) * 180;

                                        List<List<string>> NewRows = new List<List<string>>();
                                        for (int r = 0; r < Positions.Length; r++)
                                        {
                                            string[] Row = Helper.ArrayOfConstant("0", TableBoxNetAll.ColumnCount);

                                            Row[TableBoxNetAll.GetColumnID("rlnMagnification")] = "10000.0";
                                            Row[TableBoxNetAll.GetColumnID("rlnDetectorPixelSize")] = Options.BinnedPixelSizeMean.ToString("F5", CultureInfo.InvariantCulture);

                                            Row[TableBoxNetAll.GetColumnID("rlnDefocusU")] = ((Defoci[r] + Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnDefocusV")] = ((Defoci[r] - Astigmatism) * 1e4f).ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnDefocusAngle")] = item.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnVoltage")] = item.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnSphericalAberration")] = item.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnAmplitudeContrast")] = item.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnPhaseShift")] = PhaseShift.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnCtfMaxResolution")] = item.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnCoordinateX")] = (Positions[r].X / (float)CurrentOptionsBoxNet.BinnedPixelSizeMean).ToString("F2", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnCoordinateY")] = (Positions[r].Y / (float)CurrentOptionsBoxNet.BinnedPixelSizeMean).ToString("F2", CultureInfo.InvariantCulture);
                                            Row[TableBoxNetAll.GetColumnID("rlnImageName")] = (r + 1).ToString("D7") + "@particles/" + item.RootName + "_" + BoxNetSuffix + ".mrcs";
                                            Row[TableBoxNetAll.GetColumnID("rlnMicrographName")] = item.Name;

                                            NewRows.Add(Row.ToList());
                                        }

                                        List<List<string>> RowsAll = new List<List<string>>();
                                        List<List<string>> RowsGood = new List<List<string>>();

                                        lock (AllMovieParticleRows)
                                        {
                                            if (!AllMovieParticleRows.ContainsKey(item))
                                                AllMovieParticleRows.Add(item, NewRows);
                                            else
                                                AllMovieParticleRows[item] = NewRows;

                                            foreach (var pair in AllMovieParticleRows)
                                            {
                                                RowsAll.AddRange(pair.Value);
                                                if (!(pair.Key.UnselectFilter || (pair.Key.UnselectManual != null && pair.Key.UnselectManual.Value)))
                                                    RowsGood.AddRange(pair.Value);
                                            }
                                        }

                                        if (TableBoxNetConcurrent == 0)
                                        {
                                            lock (TableBoxNetAllWriteLock)
                                                TableBoxNetConcurrent++;

                                            Task.Run(() =>
                                            {
                                                Star TempTableAll = new Star(TableBoxNetAll.GetColumnNames());
                                                TempTableAll.AddRow(RowsAll);

                                                bool SuccessAll = false;
                                                while (!SuccessAll)
                                                {
                                                    try
                                                    {
                                                        TempTableAll.Save(PathBoxNetAll + "_" + item.RootName);
                                                        lock (TableBoxNetAllWriteLock)
                                                        {
                                                            if (File.Exists(PathBoxNetAll))
                                                                File.Delete(PathBoxNetAll);
                                                            File.Move(PathBoxNetAll + "_" + item.RootName, PathBoxNetAll);

                                                            if (Options.Picking.DoRunningWindow && TempTableAll.RowCount > 0)
                                                            {
                                                                TempTableAll.CreateSubset(Helper.ArrayOfSequence(Math.Max(0, TempTableAll.RowCount - Options.Picking.RunningWindowLength), 
                                                                                                                 TempTableAll.RowCount - 1, 
                                                                                                                 1)).Save(PathBoxNetAllSubset);
                                                            }
                                                        }
                                                        SuccessAll = true;
                                                    }
                                                    catch { }
                                                }

                                                Star TempTableGood = new Star(TableBoxNetAll.GetColumnNames());
                                                TempTableGood.AddRow(RowsGood);

                                                bool SuccessGood = false;
                                                while (!SuccessGood)
                                                {
                                                    try
                                                    {
                                                        TempTableGood.Save(PathBoxNetFiltered + "_" + item.RootName);
                                                        lock (TableBoxNetAllWriteLock)
                                                        {
                                                            if (File.Exists(PathBoxNetFiltered))
                                                                File.Delete(PathBoxNetFiltered);
                                                            File.Move(PathBoxNetFiltered + "_" + item.RootName, PathBoxNetFiltered);

                                                            if (Options.Picking.DoRunningWindow && TempTableGood.RowCount > 0)
                                                            {
                                                                TempTableGood.CreateSubset(Helper.ArrayOfSequence(Math.Max(0, TempTableGood.RowCount - Options.Picking.RunningWindowLength),
                                                                                                                  TempTableGood.RowCount - 1,
                                                                                                                  1)).Save(PathBoxNetFilteredSubset);
                                                            }
                                                        }
                                                        SuccessGood = true;
                                                    }
                                                    catch { }
                                                }

                                                lock (TableBoxNetAllWriteLock)
                                                    TableBoxNetConcurrent--;
                                            });
                                        }
                                    }
                                    else
                                    {
                                        OriginalStack?.Dispose();
                                        Debug.WriteLine(GPU.GetDevice() + " processed.");
                                        SemaphoresProcessing[gpuID].Release();
                                    }

                                    #endregion

                                    AverageForPicking.Dispose();

                                    BenchmarkPicking.Finish(TimerPicking);
                                }
                                else
                                {
                                    OriginalStack?.Dispose();
                                    Debug.WriteLine(GPU.GetDevice() + " processed.");
                                    SemaphoresProcessing[gpuID].Release();
                                }


                                Dispatcher.Invoke(() =>
                                {
                                    if (Options.Runtime.DisplayedMovie == item)
                                        UpdateButtonOptionsAdopt();

                                    ProcessingStatusBar.ApplyFilter();
                                    ProcessingStatusBar.UpdateElements();
                                });

                                BenchmarkAllProcessing.Finish(TimerOverall);

                                UpdateStatsAll();

                                return false; // No need to cancel GPU ForEach iterator
                            }
                            catch (Exception exc)
                            {
                                OriginalStack?.Dispose();

                                item.UnselectManual = true;
                                UpdateStatsAll();

                                Dispatcher.Invoke(() =>
                                {
                                    ProcessingStatusBar.ApplyFilter();
                                    ProcessingStatusBar.UpdateElements();
                                });

                                if (AcquiredProcessingSemaphore)
                                    SemaphoresProcessing[gpuID].Release();

                                return false;
                            }
                        }, Math.Max(NPreread, NSimultaneous), UsedDevices);


                        Dispatcher.Invoke(() =>
                        {
                            UpdateStatsAll();
                            ProcessingStatusBar.UpdateElements();
                        });

                        #endregion

                        Dispatcher.Invoke(() =>
                        {
                            ProcessingStatusBar.HideProgressBar();
                            StatsProgressIndicator.Visibility = Visibility.Hidden;
                        });
                    }

                    foreach (int d in UsedDevices)
                    {
                        ImageGain[d]?.Dispose();
                        BoxNetworks[d]?.Dispose();
                    }

                    #region Make sure all particle tables are written out in their most recent form

                    if (Options.ProcessPicking && Options.Picking.DoExport && !string.IsNullOrEmpty(Options.Picking.ModelPath))
                    {
                        ProgressDialogController ProgressDialog = null;
                        await Dispatcher.Invoke(async () => ProgressDialog = await this.ShowProgressAsync($"Waiting for the last particle files to be written out...", ""));
                        ProgressDialog.SetIndeterminate();

                        List<List<string>> RowsAll = new List<List<string>>();
                        List<List<string>> RowsGood = new List<List<string>>();

                        lock (AllMovieParticleRows)
                        {
                            foreach (var pair in AllMovieParticleRows)
                            {
                                RowsAll.AddRange(pair.Value);
                                if (!(pair.Key.UnselectFilter || (pair.Key.UnselectManual != null && pair.Key.UnselectManual.Value)))
                                    RowsGood.AddRange(pair.Value);
                            }
                        }

                        while (TableBoxNetConcurrent > 0)
                            Thread.Sleep(50);
                        
                        Star TempTableAll = new Star(TableBoxNetAll.GetColumnNames());
                        TempTableAll.AddRow(RowsAll);

                        bool SuccessAll = false;
                        while (!SuccessAll)
                        {
                            try
                            {
                                TempTableAll.Save(PathBoxNetAll + "_temp");
                                lock (TableBoxNetAllWriteLock)
                                {
                                    if (File.Exists(PathBoxNetAll))
                                        File.Delete(PathBoxNetAll);
                                    File.Move(PathBoxNetAll + "_temp", PathBoxNetAll);
                                }
                                SuccessAll = true;
                            }
                            catch { }
                        }

                        Star TempTableGood = new Star(TableBoxNetAll.GetColumnNames());
                        TempTableGood.AddRow(RowsGood);

                        bool SuccessGood = false;
                        while (!SuccessGood)
                        {
                            try
                            {
                                TempTableGood.Save(PathBoxNetFiltered + "_temp");
                                lock (TableBoxNetAllWriteLock)
                                {
                                    if (File.Exists(PathBoxNetFiltered))
                                        File.Delete(PathBoxNetFiltered);
                                    File.Move(PathBoxNetFiltered + "_temp", PathBoxNetFiltered);
                                }
                                SuccessGood = true;
                            }
                            catch { }
                        }

                        await ProgressDialog.CloseAsync();
                    }

                    #endregion
                });
            }
            else
            {
                ButtonStartProcessing.IsEnabled = false;
                ButtonStartProcessing.Content = "STOPPING...";

                IsPreprocessing = false;
                if (PreprocessingTask != null)
                {
                    await PreprocessingTask;
                    PreprocessingTask = null;
                }

                foreach (var item in DisableWhenPreprocessing)
                    item.IsEnabled = true;
                MicrographDisplayControl.SetProcessingMode(false);
                
                #region Timers

                BenchmarkAllProcessing.Clear();
                BenchmarkRead.Clear();
                BenchmarkCTF.Clear();
                BenchmarkMotion.Clear();
                BenchmarkPicking.Clear();
                BenchmarkOutput.Clear();

                #endregion

                UpdateStatsAll();

                ButtonStartProcessing.IsEnabled = true;
                ButtonStartProcessing.Content = "START PROCESSING";
                ButtonStartProcessing.Foreground = new LinearGradientBrush(Colors.DeepSkyBlue, Colors.DeepPink, 0);
            }
        }

        private void ButtonCollapseMainPreprocessing_OnClick(object sender, RoutedEventArgs e)
        {
            if (!IsPreprocessingCollapsed)
            {
                PreprocessingWidth = (int)ColumnMainPreprocessing.Width.Value;

                ButtonCollapseMainPreprocessing.Content = "▶";
                ButtonCollapseMainPreprocessing.ToolTip = "Expand";
                GridMainPreprocessing.Visibility = Visibility.Collapsed;
                ColumnMainPreprocessing.Width = new GridLength(0);

                bool NeedsTabSwitch = TabsProcessingView.SelectedItem == TabProcessingCTF || TabsProcessingView.SelectedItem == TabProcessingMovement;

                // Reorder controls
                GridCTFDisplay.Children.Clear();
                GridMicrographDisplay.Children.Clear();

                GridMergedCTFAndMovement.Children.Add(CTFDisplayControl);
                GridMergedCTFAndMovement.Children.Add(ButtonProcessOneItemCTF);
                GridMergedCTFAndMovement.Children.Add(MicrographDisplayControl);
                Grid.SetColumn(MicrographDisplayControl, 2);

                // Hide and show tabs
                TabProcessingCTFAndMovement.Visibility = Visibility.Visible;
                TabProcessingCTF.Visibility = Visibility.Collapsed;
                TabProcessingMovement.Visibility = Visibility.Collapsed;

                if (NeedsTabSwitch)
                    TabsProcessingView.SelectedItem = TabProcessingCTFAndMovement;
            }
            else
            {
                ButtonCollapseMainPreprocessing.Content = "◀";
                ButtonCollapseMainPreprocessing.ToolTip = "Collapse";
                GridMainPreprocessing.Visibility = Visibility.Visible;
                ColumnMainPreprocessing.Width = new GridLength(PreprocessingWidth);

                bool NeedsTabSwitch = TabsProcessingView.SelectedItem == TabProcessingCTFAndMovement;

                // Reorder controls
                GridMergedCTFAndMovement.Children.Clear();

                GridCTFDisplay.Children.Add(CTFDisplayControl);
                GridCTFDisplay.Children.Add(ButtonProcessOneItemCTF);
                GridMicrographDisplay.Children.Add(MicrographDisplayControl);
                Grid.SetColumn(MicrographDisplayControl, 0);

                // Hide and show tabs
                TabProcessingCTFAndMovement.Visibility = Visibility.Collapsed;
                TabProcessingCTF.Visibility = Visibility.Visible;
                TabProcessingMovement.Visibility = Visibility.Visible;

                if (NeedsTabSwitch)
                    TabsProcessingView.SelectedItem = TabProcessingCTF;
            }

            IsPreprocessingCollapsed = !IsPreprocessingCollapsed;
        }

        #region L2 TAB: OVERVIEW

        #region Statistics and filters

        #region Variables

        private int StatsAstigmatismZoom = 1;

        private BenchmarkTimer BenchmarkRead = new BenchmarkTimer("File read");
        private BenchmarkTimer BenchmarkCTF = new BenchmarkTimer("CTF");
        private BenchmarkTimer BenchmarkMotion = new BenchmarkTimer("Motion");
        private BenchmarkTimer BenchmarkPicking = new BenchmarkTimer("Picking");
        private BenchmarkTimer BenchmarkOutput = new BenchmarkTimer("Output");

        private BenchmarkTimer BenchmarkAllProcessing = new BenchmarkTimer("All processing");

        #endregion

        public void UpdateStatsAll()
        {
            UpdateFilterRanges();
            UpdateFilterResult();
            UpdateStatsAstigmatismPlot();
            UpdateStatsStatus();
            UpdateFilterSuffixMenu();
            UpdateBenchmarkTimes();
        }

        private void UpdateStatsStatus()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();

            bool HaveCTF = Options.ProcessCTF || Items.Any(v => v.OptionsCTF != null && v.CTF != null);
            bool HavePhase = Options.CTF.DoPhase || Items.Any(v => v.OptionsCTF != null && v.OptionsCTF.DoPhase);
            bool HaveMovement = Options.ProcessMovement || Items.Any(v => v.OptionsMovement != null);
            bool HaveParticles = Items.Any(m => m.HasParticleSuffix(Options.Filter.ParticlesSuffix));

            ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
            ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
            ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();
            ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();

            int[] ColorIDs = new int[Items.Length];
            int NProcessed = 0, NOutdated = 0, NUnprocessed = 0, NFilteredOut = 0, NUnselected = 0;
            for (int i = 0; i < Items.Length; i++)
            {
                ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(Items[i], OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options);
                int ID = 0;
                switch (Status)
                {
                    case ProcessingStatus.Processed:
                        ID = 0;
                        NProcessed++;
                        break;
                    case ProcessingStatus.Outdated:
                        ID = 1;
                        NOutdated++;
                        break;
                    case ProcessingStatus.Unprocessed:
                        ID = 2;
                        NUnprocessed++;
                        break;
                    case ProcessingStatus.FilteredOut:
                        ID = 3;
                        NFilteredOut++;
                        break;
                    case ProcessingStatus.LeaveOut:
                        ID = 4;
                        NUnselected++;
                        break;
                }
                ColorIDs[i] = ID;
            }

            Dispatcher.InvokeAsync(() =>
            {
                StatsSeriesStatusProcessed.Visibility = NProcessed == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusOutdated.Visibility = NOutdated == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusUnprocessed.Visibility = NUnprocessed == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusUnfiltered.Visibility = NFilteredOut == 0 ? Visibility.Collapsed : Visibility.Visible;
                StatsSeriesStatusUnselected.Visibility = NUnselected == 0 ? Visibility.Collapsed : Visibility.Visible;

                StatsSeriesStatusProcessed.Values = new ChartValues<ObservableValue> { new ObservableValue(NProcessed) };
                StatsSeriesStatusOutdated.Values = new ChartValues<ObservableValue> { new ObservableValue(NOutdated) };
                StatsSeriesStatusUnprocessed.Values = new ChartValues<ObservableValue> { new ObservableValue(NUnprocessed) };
                StatsSeriesStatusUnfiltered.Values = new ChartValues<ObservableValue> { new ObservableValue(NFilteredOut) };
                StatsSeriesStatusUnselected.Values = new ChartValues<ObservableValue> { new ObservableValue(NUnselected) };
            });

            if (HaveCTF)
            {
                #region Defocus

                double[] DefocusValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].OptionsCTF != null && Items[i].CTF != null)
                        DefocusValues[i] = (double)Items[i].CTF.Defocus;
                    else
                        DefocusValues[i] = double.NaN;

                SingleAxisPoint[] DefocusPlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    DefocusPlotValues[i] = new SingleAxisPoint(DefocusValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsDefocus.Points = new ObservableCollection<SingleAxisPoint>(DefocusPlotValues));

                #endregion

                #region Phase

                if (HavePhase)
                {
                    double[] PhaseValues = new double[Items.Length];
                    for (int i = 0; i < Items.Length; i++)
                        if (Items[i].OptionsCTF != null && Items[i].CTF != null)
                            PhaseValues[i] = (double)Items[i].CTF.PhaseShift;
                        else
                            PhaseValues[i] = double.NaN;

                    SingleAxisPoint[] PhasePlotValues = new SingleAxisPoint[Items.Length];
                    for (int i = 0; i < Items.Length; i++)
                        PhasePlotValues[i] = new SingleAxisPoint(PhaseValues[i], ColorIDs[i], Items[i]);

                    Dispatcher.InvokeAsync(() => PlotStatsPhase.Points = new ObservableCollection<SingleAxisPoint>(PhasePlotValues));
                }
                else
                    Dispatcher.InvokeAsync(() => PlotStatsPhase.Points = null);

                #endregion

                #region Resolution

                double[] ResolutionValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].CTFResolutionEstimate > 0)
                        ResolutionValues[i] = (double)Items[i].CTFResolutionEstimate;
                    else
                        ResolutionValues[i] = double.NaN;

                SingleAxisPoint[] ResolutionPlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    ResolutionPlotValues[i] = new SingleAxisPoint(ResolutionValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsResolution.Points = new ObservableCollection<SingleAxisPoint>(ResolutionPlotValues));

                #endregion
            }
            else
            {
                Dispatcher.InvokeAsync(() =>
                {
                    //StatsSeriesAstigmatism0.Values = new ChartValues<ObservablePoint>();
                    PlotStatsDefocus.Points = null;
                    PlotStatsPhase.Points = null;
                    PlotStatsResolution.Points = null;
                });
            }

            if (HaveMovement)
            {
                double[] MovementValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].MeanFrameMovement > 0)
                        MovementValues[i] = (double)Items[i].MeanFrameMovement;
                    else
                        MovementValues[i] = double.NaN;

                SingleAxisPoint[] MovementPlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    MovementPlotValues[i] = new SingleAxisPoint(MovementValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsMotion.Points = new ObservableCollection<SingleAxisPoint>(MovementPlotValues));
            }
            else
            {
                Dispatcher.InvokeAsync(() => PlotStatsMotion.Points = null);
            }

            if (HaveParticles)
            {
                int CountSum = 0, CountFilteredSum = 0;
                double[] ParticleValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                {
                    int Count = Items[i].GetParticleCount(Options.Filter.ParticlesSuffix);
                    if (Count >= 0)
                    {
                        ParticleValues[i] = Count;
                        CountSum += Count;

                        if (!(Items[i].UnselectFilter || (Items[i].UnselectManual != null && Items[i].UnselectManual.Value)))
                            CountFilteredSum += Count;
                    }
                    else
                        ParticleValues[i] = double.NaN;
                }

                SingleAxisPoint[] ParticlePlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    ParticlePlotValues[i] = new SingleAxisPoint(ParticleValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() =>
                {
                    PlotStatsParticles.Points = new ObservableCollection<SingleAxisPoint>(ParticlePlotValues);
                    TextStatsParticlesOverall.Value = CountSum.ToString();
                    TextStatsParticlesFiltered.Value = CountFilteredSum.ToString();
                });
            }
            else
            {
                Dispatcher.InvokeAsync(() => PlotStatsParticles.Points = null);
            }

            {
                double[] MaskPercentageValues = new double[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    if (Items[i].MaskPercentage >= 0)
                        MaskPercentageValues[i] = (double)Items[i].MaskPercentage;
                    else
                        MaskPercentageValues[i] = double.NaN;

                SingleAxisPoint[] MaskPercentagePlotValues = new SingleAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                    MaskPercentagePlotValues[i] = new SingleAxisPoint(MaskPercentageValues[i], ColorIDs[i], Items[i]);

                Dispatcher.InvokeAsync(() => PlotStatsMaskPercentage.Points = new ObservableCollection<SingleAxisPoint>(MaskPercentagePlotValues));
            }
        }

        private void UpdateStatsAstigmatismPlot()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();

            bool HaveCTF = Options.ProcessCTF || Items.Any(v => v.OptionsCTF != null && v.CTF != null);

            if (HaveCTF)
            {
                #region Astigmatism

                DualAxisPoint[] AstigmatismPoints = new DualAxisPoint[Items.Length];
                for (int i = 0; i < Items.Length; i++)
                {
                    Movie item = Items[i];
                    DualAxisPoint P = new DualAxisPoint();
                    P.Context = item;
                    P.ColorID = i * 4 / Items.Length;
                    if (item.OptionsCTF != null && item.CTF != null)
                    {
                        P.X = Math.Round(Math.Cos((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta, 4);
                        P.Y = Math.Round(Math.Sin((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta, 4);
                        P.Label = item.CTF.DefocusDelta.ToString("F4");
                    }
                    else
                        P.Label = "";

                    AstigmatismPoints[i] = P;
                }

                Dispatcher.InvokeAsync(() =>
                {
                    PlotStatsAstigmatism.Points = new ObservableCollection<DualAxisPoint>(AstigmatismPoints);
                });

                #endregion
            }
            else
            {
                Dispatcher.InvokeAsync(() =>
                {
                    PlotStatsAstigmatism.Points = new ObservableCollection<DualAxisPoint>();
                });
            }
        }

        private void UpdateFilterRanges()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();
            Movie[] ItemsWithCTF = Items.Where(v => v.OptionsCTF != null && v.CTF != null).ToArray();
            Movie[] ItemsWithMovement = Items.Where(v => v.OptionsMovement != null).ToArray();

            #region Astigmatism (includes adjusting the plot elements)

            float2 AstigmatismMean = new float2();
            float AstigmatismStd = 0.1f;
            float AstigmatismMax = 0.4f;

            // Get all items with valid CTF information
            List<float2> AstigmatismPoints = new List<float2>(ItemsWithCTF.Length);
            foreach (var item in ItemsWithCTF)
                AstigmatismPoints.Add(new float2((float)Math.Cos((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta,
                                                    (float)Math.Sin((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta));

            // Calculate mean and stddev of all points in Cartesian coords
            if (AstigmatismPoints.Count > 0)
            {
                AstigmatismMean = new float2();
                AstigmatismMax = 0;
                foreach (var point in AstigmatismPoints)
                {
                    AstigmatismMean += point;
                    AstigmatismMax = Math.Max(AstigmatismMax, point.LengthSq());
                }
                AstigmatismMax = (float)Math.Sqrt(AstigmatismMax);
                AstigmatismMean /= AstigmatismPoints.Count;

                AstigmatismStd = 0;
                foreach (var point in AstigmatismPoints)
                    AstigmatismStd += (point - AstigmatismMean).LengthSq();
                AstigmatismStd = (float)Math.Max(1e-4, Math.Sqrt(AstigmatismStd / AstigmatismPoints.Count));
            }

            AstigmatismMax = Math.Max(1e-4f, (float)Math.Ceiling(AstigmatismMax * 20) / 20);

            // Set the labels for outer and inner circle
            Dispatcher.InvokeAsync(() =>
            {
                StatsAstigmatismLabelOuter.Value = (AstigmatismMax / StatsAstigmatismZoom).ToString("F3", CultureInfo.InvariantCulture);
                StatsAstigmatismLabelInner.Value = (AstigmatismMax / StatsAstigmatismZoom / 2).ToString("F3", CultureInfo.InvariantCulture);

                // Adjust plot axes
                
                PlotStatsAstigmatism.AxisMax = AstigmatismMax / StatsAstigmatismZoom;

                // Scale and position the valid range ellipse
                StatsAstigmatismEllipseSigma.Width = AstigmatismStd * StatsAstigmatismZoom * (float)Options.Filter.AstigmatismMax / AstigmatismMax * 256;
                StatsAstigmatismEllipseSigma.Height = AstigmatismStd * StatsAstigmatismZoom * (float)Options.Filter.AstigmatismMax / AstigmatismMax * 256;
                Canvas.SetLeft(StatsAstigmatismEllipseSigma, AstigmatismMean.X / AstigmatismMax * 128 * StatsAstigmatismZoom + 128 - StatsAstigmatismEllipseSigma.Width / 2);
                Canvas.SetTop(StatsAstigmatismEllipseSigma, AstigmatismMean.Y / AstigmatismMax * 128 * StatsAstigmatismZoom + 128 - StatsAstigmatismEllipseSigma.Height / 2);
            });

            lock (Options)
            {
                Options.AstigmatismMean = AstigmatismMean;
                Options.AstigmatismStd = AstigmatismStd;
            }

            #endregion

            bool HaveCTF = Options.ProcessCTF || ItemsWithCTF.Length > 0;
            bool HavePhase = Options.CTF.DoPhase || ItemsWithCTF.Any(v => v.OptionsCTF.DoPhase);
            bool HaveMovement = Options.ProcessMovement || ItemsWithMovement.Length > 0;
            bool HaveParticles = Items.Any(m => m.HasAnyParticleSuffixes());

            Dispatcher.InvokeAsync(() =>
            {
                PanelStatsAstigmatism.Visibility = HaveCTF ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsDefocus.Visibility = HaveCTF ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsPhase.Visibility = HaveCTF && HavePhase ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsResolution.Visibility = HaveCTF ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsMotion.Visibility = HaveMovement ? Visibility.Visible : Visibility.Collapsed;
                PanelStatsParticles.Visibility = HaveParticles ? Visibility.Visible : Visibility.Collapsed;
            });
        }

        private void UpdateFilterResult()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();

            float2 AstigmatismMean;
            float AstigmatismStd;
            lock (Options)
            {
                AstigmatismMean = Options.AstigmatismMean;
                AstigmatismStd = Options.AstigmatismStd;
            }

            foreach (var item in Items)
            {
                bool FilterStatus = true;

                if (item.OptionsCTF != null)
                {
                    FilterStatus &= item.CTF.Defocus >= Options.Filter.DefocusMin && item.CTF.Defocus <= Options.Filter.DefocusMax;
                    float AstigmatismDeviation = (new float2((float)Math.Cos((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta,
                                                             (float)Math.Sin((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta) - AstigmatismMean).Length() / AstigmatismStd;
                    FilterStatus &= AstigmatismDeviation <= (float)Options.Filter.AstigmatismMax;

                    FilterStatus &= item.CTFResolutionEstimate <= Options.Filter.ResolutionMax;

                    if (Options.CTF.DoPhase)
                        FilterStatus &= item.CTF.PhaseShift >= Options.Filter.PhaseMin && item.CTF.PhaseShift <= Options.Filter.PhaseMax;
                }

                if (item.OptionsMovement != null)
                {
                    FilterStatus &= item.MeanFrameMovement <= Options.Filter.MotionMax;
                }

                if (item.HasAnyParticleSuffixes())
                {
                    int Count = item.GetParticleCount(Options.Filter.ParticlesSuffix);
                    if (Count >= 0)
                        FilterStatus &= Count >= Options.Filter.ParticlesMin;
                }

                FilterStatus &= item.MaskPercentage <= Options.Filter.MaskPercentage;

                item.UnselectFilter = !FilterStatus;
            }

            // Calculate average CTF
            Task.Run(() =>
            {
                try
                {
                    CTF[] AllCTFs = Items.Where(m => m.OptionsCTF != null && !m.UnselectFilter).Select(m => m.CTF.GetCopy()).ToArray();
                    decimal PixelSize = Options.BinnedPixelSizeMean;

                    Dispatcher.Invoke(() => StatsDefocusAverageCTFFrequencyLabel.Text = $"1/{PixelSize:F1} Å");

                    float[] AverageCTFValues = new float[192];
                    foreach (var ctf in AllCTFs)
                    {
                        ctf.PixelSize = PixelSize;
                        float[] Simulated = ctf.Get1D(AverageCTFValues.Length, true);

                        for (int i = 0; i < Simulated.Length; i++)
                            AverageCTFValues[i] += Simulated[i];
                    }

                    if (AllCTFs.Length > 1)
                        for (int i = 0; i < AverageCTFValues.Length; i++)
                            AverageCTFValues[i] /= AllCTFs.Length;

                    float MinAverage = MathHelper.Min(AverageCTFValues);

                    Dispatcher.Invoke(() =>
                    {
                        IEnumerable<Point> TrackPoints = AverageCTFValues.Select((v, i) => new Point(i, 24 - 1 - (24 * v)));

                        System.Windows.Shapes.Path TrackPath = new System.Windows.Shapes.Path()
                        {
                            Stroke = StatsDefocusAverageCTFFrequencyLabel.Foreground,
                            StrokeThickness = 1,
                            StrokeLineJoin = PenLineJoin.Bevel,
                            IsHitTestVisible = false
                        };
                        PolyLineSegment PlotSegment = new PolyLineSegment(TrackPoints, true);
                        PathFigure PlotFigure = new PathFigure
                        {
                            Segments = new PathSegmentCollection { PlotSegment },
                            StartPoint = TrackPoints.First()
                        };
                        TrackPath.Data = new PathGeometry { Figures = new PathFigureCollection { PlotFigure } };

                        StatsDefocusAverageCTFCanvas.Children.Clear();
                        StatsDefocusAverageCTFCanvas.Children.Add(TrackPath);
                        Canvas.SetBottom(TrackPath, 24 * MinAverage);
                    });
                }
                catch { }
            });
        }

        public void UpdateFilterSuffixMenu()
        {
            Movie[] Items = FileDiscoverer.GetImmutableFiles();
            List<string> Suffixes = new List<string>();

            foreach (var movie in Items)
                foreach (var suffix in movie.GetParticlesSuffixes())
                    if (!Suffixes.Contains(suffix))
                        Suffixes.Add(suffix);

            Suffixes.Sort();
            Dispatcher.InvokeAsync(() =>
            {
                MenuParticlesSuffix.Items.Clear();
                foreach (var suffix in Suffixes)
                    MenuParticlesSuffix.Items.Add(suffix);

                if ((string.IsNullOrEmpty(Options.Filter.ParticlesSuffix) || !Suffixes.Contains(Options.Filter.ParticlesSuffix))
                    && Suffixes.Count > 0)
                    Options.Filter.ParticlesSuffix = Suffixes[0];
            });
        }

        public void UpdateBenchmarkTimes()
        {
            Dispatcher.Invoke(() =>
            {
                StatsBenchmarkOverall.Text = "";

                if (BenchmarkAllProcessing.NItems < 5)
                    return;

                int NMeasurements = Math.Min(BenchmarkAllProcessing.NItems, 100);

                StatsBenchmarkOverall.Text = ((int)Math.Round(BenchmarkAllProcessing.GetPerSecondConcurrent(NMeasurements) * 3600)) + " / h";

                StatsBenchmarkInput.Text = BenchmarkRead.NItems > 0 ? (BenchmarkRead.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkCTF.Text = BenchmarkCTF.NItems > 0 ? (BenchmarkCTF.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkMotion.Text = BenchmarkMotion.NItems > 0 ? (BenchmarkMotion.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkPicking.Text = BenchmarkPicking.NItems > 0 ? (BenchmarkPicking.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
                StatsBenchmarkOutput.Text = BenchmarkOutput.NItems > 0 ? (BenchmarkOutput.GetAverageMilliseconds(NMeasurements) / 1000).ToString("F1") + " s" : "";
            });
        }

        #region GUI events

        private void PlotStatsAstigmatism_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            Options.Runtime.DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsDefocus_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;
            
            Options.Runtime.DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsPhase_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            Options.Runtime.DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsResolution_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            Options.Runtime.DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingCTF.IsSelected = true;
            });
        }

        private void PlotStatsMotion_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            Options.Runtime.DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingMovement.IsSelected = true;
            });
        }

        private void PlotStatsParticles_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            Options.Runtime.DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingMovement.IsSelected = true;
            });
        }

        private void PlotStatsMaskPercentage_OnPointClicked(Movie obj)
        {
            if (obj == null)
                return;

            Options.Runtime.DisplayedMovie = obj;
            Dispatcher.InvokeAsync(() =>
            {
                if (IsPreprocessingCollapsed)
                    TabProcessingCTFAndMovement.IsSelected = true;
                else
                    TabProcessingMovement.IsSelected = true;
            });
        }

        private void StatsAstigmatismBackground_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            StatsAstigmatismZoom = Math.Max(1, StatsAstigmatismZoom + Math.Sign(e.Delta));
            UpdateFilterRanges();
        }

        private void MenuParticlesSuffix_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateFilterSuffixMenu();
        }
        
        private void StatsBenchmarkOverall_OnMouseEnter(object sender, MouseEventArgs e)
        {
            StatsBenchmarkDetails.Visibility = Visibility.Visible;
        }

        private void StatsBenchmarkDetails_OnMouseLeave(object sender, MouseEventArgs e)
        {
            StatsBenchmarkDetails.Visibility = Visibility.Hidden;
        }

        #endregion

        #endregion

        #region Task button events (micrograph, particle export, adjustment etc.)

        #region 2D

        private void ButtonTasksExportMicrographs_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.SaveFileDialog FileDialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Movie[] Movies = FileDiscoverer.GetImmutableFiles();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                Dialog2DList DialogContent = new Dialog2DList(Movies, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksAdjustDefocus_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
                {
                    Filter = "STAR Files|*.star"
                };
                System.Windows.Forms.DialogResult ResultSave = SaveDialog.ShowDialog();

                if (ResultSave.ToString() == "OK")
                {
                    Movie[] Movies = FileDiscoverer.GetImmutableFiles();

                    CustomDialog Dialog = new CustomDialog();
                    Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                    Dialog2DDefocusUpdate DialogContent = new Dialog2DDefocusUpdate(Movies, OpenDialog.FileName, SaveDialog.FileName, Options);
                    DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                    Dialog.Content = DialogContent;

                    this.ShowMetroDialogAsync(Dialog);
                }
            }
        }

        private void ButtonTasksExportParticles_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
                {
                    Filter = "STAR Files|*.star"
                };
                System.Windows.Forms.DialogResult ResultSave = SaveDialog.ShowDialog();

                if (ResultSave.ToString() == "OK")
                {
                    Movie[] Movies = FileDiscoverer.GetImmutableFiles();

                    CustomDialog Dialog = new CustomDialog();
                    Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                    Dialog2DParticleExport DialogContent = new Dialog2DParticleExport(Movies, OpenDialog.FileName, SaveDialog.FileName, Options);
                    DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                    Dialog.Content = DialogContent;

                    this.ShowMetroDialogAsync(Dialog);
                }
            }
        }

        private void ButtonTasksImportParticles_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog FileDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Movie[] Movies = FileDiscoverer.GetImmutableFiles();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                Dialog2DParticleImport DialogContent = new Dialog2DParticleImport(Movies, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksMatch_OnClick(object sender, RoutedEventArgs e)
        {
            if (Options.Import.ExtensionTomoSTAR)   // This is not for tomo
                return;
            System.Windows.Forms.OpenFileDialog FileDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC Volumes|*.mrc",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Movie[] ImmutableItems = FileDiscoverer.GetImmutableFiles();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                Dialog2DMatch DialogContent = new Dialog2DMatch(ImmutableItems, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksExportBoxNet_OnClick(object sender, RoutedEventArgs e)
        {
            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            BoxNetExport DialogContent = new BoxNetExport(Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        #endregion

        #region Tomo
               
        private void ButtonTasksImportImod_Click(object sender, RoutedEventArgs e)
        {
            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogTomoImportImod DialogContent = new DialogTomoImportImod(Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        private void ButtonTasksExportTomograms_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.SaveFileDialog FileDialog = new System.Windows.Forms.SaveFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                TiltSeries[] Series = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                DialogTomoList DialogContent = new DialogTomoList(Series, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        private void ButtonTasksReconstructTomograms_OnClick(object sender, RoutedEventArgs e)
        {
            if (!Options.Import.ExtensionTomoSTAR)
                return;

            TiltSeries[] ImmutableItems = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogTomoReconstruction DialogContent = new DialogTomoReconstruction(ImmutableItems, Options);
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
        }

        private void ButtonTasksReconstructSubtomograms_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "STAR Files|*.star"
            };
            System.Windows.Forms.DialogResult ResultOpen = OpenDialog.ShowDialog();

            if (ResultOpen.ToString() == "OK")
            {
                System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
                {
                    Filter = "STAR Files|*.star"
                };
                System.Windows.Forms.DialogResult ResultSave = SaveDialog.ShowDialog();

                if (ResultSave.ToString() == "OK")
                {
                    TiltSeries[] Series = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

                    CustomDialog Dialog = new CustomDialog();
                    Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                    DialogTomoParticleExport DialogContent = new DialogTomoParticleExport(Series, OpenDialog.FileName, SaveDialog.FileName, Options);
                    DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                    Dialog.Content = DialogContent;

                    this.ShowMetroDialogAsync(Dialog);
                }
            }
        }

        private void ButtonTasksMatchTomograms_OnClick(object sender, RoutedEventArgs e)
        {
            if (!Options.Import.ExtensionTomoSTAR)
                return;
            System.Windows.Forms.OpenFileDialog FileDialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "MRC Volumes|*.mrc",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = FileDialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                TiltSeries[] ImmutableItems = FileDiscoverer.GetImmutableFiles().Cast<TiltSeries>().ToArray();

                CustomDialog Dialog = new CustomDialog();
                Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                DialogTomoMatch DialogContent = new DialogTomoMatch(ImmutableItems, FileDialog.FileName, Options);
                DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);
                Dialog.Content = DialogContent;

                this.ShowMetroDialogAsync(Dialog);
            }
        }

        #endregion

        #endregion

        private void TomoAdjustInterface()
        {
            if (Options.Import.ExtensionTomoSTAR)            // Tomo interface
            {
                foreach (var element in HideWhen2D)
                    element.Visibility = Visibility.Visible;
                foreach (var element in HideWhenTomo)
                    element.Visibility = Visibility.Collapsed;

                Options.ProcessMovement = false;
                Options.Export.DoAverage = false;
                Options.Export.DoDeconvolve = false;
                Options.Export.DoStack = false;
            }
            else                                        // SPA interface
            {
                foreach (var element in HideWhen2D)
                    element.Visibility = Visibility.Collapsed;
                foreach (var element in HideWhenTomo)
                    element.Visibility = Visibility.Visible;
            }
        }

        #endregion

        #region L2 TAB: CTF

        private async void ButtonProcessOneItemCTF_OnClick(object sender, RoutedEventArgs e)
        {
            if (Options.Runtime.DisplayedMovie == null)
                return;

            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            Movie Item = Options.Runtime.DisplayedMovie;

            var Dialog = await this.ShowProgressAsync("Please wait...", $"Processing CTF for {Item.Name}...");
            Dialog.SetIndeterminate();

            await Task.Run(async () =>
            {
                Image ImageGain = null;
                Image OriginalStack = null;

                try
                {
                    #region Get gain ref if needed

                    if (!string.IsNullOrEmpty(Options.Import.GainPath) && Options.Import.CorrectGain && File.Exists(Options.Import.GainPath))
                        ImageGain = LoadAndPrepareGainReference();

                    #endregion

                    bool IsTomo = Item.GetType() == typeof(TiltSeries);

                    #region Load movie

                    MapHeader OriginalHeader = null;
                    decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.Import.BinTimes);

                    if (!IsTomo)
                        LoadAndPrepareHeaderAndMap(Item.Path, ImageGain, ScaleFactor, out OriginalHeader, out OriginalStack);

                    #endregion

                    Watch.Stop();
                    Debug.WriteLine(Watch.ElapsedMilliseconds / 1e3);

                    ProcessingOptionsMovieCTF CurrentOptionsCTF = Options.GetProcessingMovieCTF();

                    // Store original dimensions in Angstrom
                    if (!IsTomo)
                    {
                        CurrentOptionsCTF.Dimensions = OriginalHeader.Dimensions.MultXY((float)Options.PixelSizeMean);
                    }
                    else
                    {
                        ((TiltSeries)Item).LoadMovieSizes(CurrentOptionsCTF);

                        float3 StackDims = new float3(((TiltSeries)Item).ImageDimensionsPhysical[0], ((TiltSeries)Item).NTilts);
                        CurrentOptionsCTF.Dimensions = StackDims;
                    }

                    if (Item.GetType() == typeof(Movie))
                        Item.ProcessCTF(OriginalStack, CurrentOptionsCTF);
                    else
                        ((TiltSeries)Item).ProcessCTFSimultaneous(CurrentOptionsCTF);

                    Dispatcher.Invoke(() =>
                    {
                        UpdateButtonOptionsAdopt();

                        ProcessingStatusBar.UpdateElements();
                    });

                    UpdateStatsAll();

                    OriginalStack?.Dispose();
                    ImageGain?.Dispose();

                    await Dialog.CloseAsync();
                }
                catch (Exception exc)
                {
                    ImageGain?.Dispose();
                    OriginalStack?.Dispose();

                    await Dispatcher.Invoke(async () =>
                    {
                        if (Dialog.IsOpen)
                            await Dialog.CloseAsync();

                        Item.UnselectManual = true;

                        await this.ShowMessageAsync("Oopsie", "An error occurred while fitting the CTF. Likely reasons include:\n\n" +
                                                              "– Insufficient read/write permissions in this folder.\n" +
                                                              "– Too low defocus to yield more than one CTF peak in the processing range.\n" +
                                                              "– Mismatch in gain reference and image dimensions.\n\n" +
                                                              "The exception raised is:\n" + exc.ToString());
                    });
                }
            });
        }

        #endregion

        private void TabsProcessingView_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ProcessingStatusBar == null)
                return;

            if (TabsProcessingView.SelectedItem == TabProcessingOverview)
                ProcessingStatusBar.Visibility = Visibility.Collapsed;
            else
                ProcessingStatusBar.Visibility = Visibility.Visible;
        }

        #endregion

        #region Logging

        StackPanel[] LogMessagePanels;

        private void ButtonOpenLog_OnClick(object sender, RoutedEventArgs e)
        {
            //if (GridMessageLogs.Visibility == Visibility.Collapsed)
            //{
            //    GridMessageLogs.Visibility = Visibility.Visible;
            //    ButtonOpenLog.Content = "CLOSE LOG";
            //}
            //else
            //{
            //    GridMessageLogs.Visibility = Visibility.Collapsed;
            //    ButtonOpenLog.Content = "OPEN LOG";
            //}
        }

        private void UpdateLog(List<LogMessage> log, int logID)
        {
            LogMessagePanels[logID].Children.Clear();

            Dictionary<string, List<LogMessage>> Groups = new Dictionary<string, List<LogMessage>>();
            foreach (var message in log)
            {
                if (!Groups.ContainsKey(message.GroupTitle))
                    Groups.Add(message.GroupTitle, new List<LogMessage>());
                Groups[message.GroupTitle].Add(message);
            }

            foreach (var group in Groups)
                group.Value.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));

            List<List<LogMessage>> GroupsList = new List<List<LogMessage>>(Groups.Values);
            GroupsList.Sort((a, b) => a[0].Timestamp.CompareTo(b[0].Timestamp));

            foreach (var group in GroupsList)
            {
                LogMessageGroup GroupDisplay = new LogMessageGroup();
                GroupDisplay.Margin = new Thickness(0, 4, 0, 0);
                GroupDisplay.GroupName = group[0].GroupTitle;
                GroupDisplay.Messages = group;
                LogMessagePanels[logID].Children.Add(GroupDisplay);
            }

            ((ScrollViewer)LogMessagePanels[logID].Parent).ScrollToEnd();
        }

        private void Logger_MessageLogged(LogMessage message, int bufferID, List<LogMessage> buffer)
        {
            UpdateLog(buffer, bufferID);
        }

        #endregion

        #region Helper methods

        void AdjustInput()
        {
            FileDiscoverer.ChangePath(Options.Import.Folder, Options.Import.Extension);
        }

        public static Image LoadAndPrepareGainReference()
        {
            Image Gain = Image.FromFilePatient(50, 500,
                                               Options.Import.GainPath,
                                               new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                               (int)Options.Import.HeaderlessOffset,
                                               ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));

            if (Options.Import.GainFlipX)
                Gain = Gain.AsFlippedX();
            if (Options.Import.GainFlipY)
                Gain = Gain.AsFlippedY();
            if (Options.Import.GainTranspose)
                Gain = Gain.AsTransposed();

            return Gain;
        }

        public static void LoadAndPrepareHeaderAndMap(string path, Image imageGain, decimal scaleFactor, out MapHeader header, out Image stack, bool needStack = true, int maxThreads = 8)
        {
            header = MapHeader.ReadFromFilePatient(50, 500,
                                                   path,
                                                   new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                   Options.Import.HeaderlessOffset,
                                                   ImageFormatsHelper.StringToType(Options.Import.HeaderlessType));

            if (imageGain != null)
                if (header.Dimensions.X != imageGain.Dims.X || header.Dimensions.Y != imageGain.Dims.Y)
                    throw new Exception("Gain reference dimensions do not match image.");

            bool NeedIOLock = header.GetType() != typeof(HeaderTiff);   // No need to create additional IO competition without compression
            bool IsTiff = header.GetType() == typeof(HeaderTiff);
            object IOLock = new object();

            int NThreads = Math.Min(IsTiff ? 8 : 2, maxThreads);

            int CurrentDevice = GPU.GetDevice();

            if (needStack)
            {
                byte[] TiffBytes = null;
                if (IsTiff)
                {
                    MemoryStream Stream = new MemoryStream();
                    using (Stream BigBufferStream = IOHelper.OpenWithBigBuffer(path))
                        BigBufferStream.CopyTo(Stream);
                    TiffBytes = Stream.GetBuffer();
                }

                if (scaleFactor == 1M)
                {
                    stack = new Image(header.Dimensions);
                    float[][] OriginalStackData = stack.GetHost(Intent.Write);

                    Helper.ForCPU(0, header.Dimensions.Z, NThreads, threadID => GPU.SetDevice(CurrentDevice), (z, threadID) =>
                    {
                        Image Layer = null;
                        MemoryStream TiffStream = TiffBytes != null ? new MemoryStream(TiffBytes) : null;

                        if (NeedIOLock)
                        {
                            lock (IOLock)
                                Layer = Image.FromFilePatient(50, 500,
                                                              path,
                                                              new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                              (int)Options.Import.HeaderlessOffset,
                                                              ImageFormatsHelper.StringToType(Options.Import.HeaderlessType),
                                                              z,
                                                              TiffStream);
                        }
                        else
                        {
                            Layer = Image.FromFilePatient(50, 500,
                                                          path,
                                                          new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                          (int)Options.Import.HeaderlessOffset,
                                                          ImageFormatsHelper.StringToType(Options.Import.HeaderlessType),
                                                          z,
                                                          TiffStream);
                        }

                        lock (OriginalStackData)
                        {
                            if (imageGain != null)
                                Layer.MultiplySlices(imageGain);
                            Layer.Xray(20f);

                            OriginalStackData[z] = Layer.GetHost(Intent.Read)[0];
                            Layer.Dispose();
                        }

                    }, null);
                }
                else
                {
                    int3 ScaledDims = new int3((int)Math.Round(header.Dimensions.X * scaleFactor) / 2 * 2,
                                               (int)Math.Round(header.Dimensions.Y * scaleFactor) / 2 * 2,
                                               header.Dimensions.Z);

                    stack = new Image(ScaledDims);
                    float[][] OriginalStackData = stack.GetHost(Intent.Write);

                    int PlanForw = GPU.CreateFFTPlan(header.Dimensions.Slice(), 1);
                    int PlanBack = GPU.CreateIFFTPlan(ScaledDims.Slice(), 1);

                    Helper.ForCPU(0, ScaledDims.Z, NThreads, threadID => GPU.SetDevice(CurrentDevice), (z, threadID) =>
                                  {
                                      Image Layer = null;
                                      MemoryStream TiffStream = TiffBytes != null ? new MemoryStream(TiffBytes) : null;

                                      if (NeedIOLock)
                                      {
                                          lock (IOLock)
                                              Layer = Image.FromFilePatient(50, 500,
                                                                            path,
                                                                            new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                                            (int)Options.Import.HeaderlessOffset,
                                                                            ImageFormatsHelper.StringToType(Options.Import.HeaderlessType),
                                                                            z,
                                                                            TiffStream);
                                      }
                                      else
                                      {
                                          Layer = Image.FromFilePatient(50, 500,
                                                                        path,
                                                                        new int2(Options.Import.HeaderlessWidth, Options.Import.HeaderlessHeight),
                                                                        (int)Options.Import.HeaderlessOffset,
                                                                        ImageFormatsHelper.StringToType(Options.Import.HeaderlessType),
                                                                        z,
                                                                        TiffStream);
                                      }

                                      Image ScaledLayer = null;
                                      lock (OriginalStackData)
                                      {
                                          if (imageGain != null)
                                              Layer.MultiplySlices(imageGain);
                                          Layer.Xray(20f);

                                          ScaledLayer = Layer.AsScaled(new int2(ScaledDims), PlanForw, PlanBack);
                                          Layer.Dispose();
                                      }

                                      OriginalStackData[z] = ScaledLayer.GetHost(Intent.Read)[0];
                                      ScaledLayer.Dispose();

                                  }, null);

                    GPU.DestroyFFTPlan(PlanForw);
                    GPU.DestroyFFTPlan(PlanBack);
                }
            }
            else
            {
                stack = null;
            }
        }

        public List<int> GetDeviceList()
        {
            List<int> Devices = new List<int>();
            Dispatcher.Invoke(() =>
            {
                for (int i = 0; i < CheckboxesGPUStats.Length; i++)
                    if ((bool)CheckboxesGPUStats[i].IsChecked)
                        Devices.Add(i);
            });

            return Devices;
        }

        #endregion

        #region Experimental

        private void ButtonExportParticles_OnClick(object sender, RoutedEventArgs e)
        {
            //System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            //{
            //    Filter = "STAR Files|*.star",
            //    Multiselect = false
            //};
            //System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();
            //if (Result == System.Windows.Forms.DialogResult.OK)
            //{
            //    System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
            //    {
            //        Filter = "STAR Files|*.star"
            //    };
            //    System.Windows.Forms.DialogResult SaveResult = SaveDialog.ShowDialog();
            //    if (SaveResult == System.Windows.Forms.DialogResult.OK)
            //    {
            //        Thread ProcessThread = new Thread(() =>
            //        {
            //            Star TableIn = new Star(Dialog.FileName);
            //            //if (TableIn.GetColumn("rlnCtfImage") == null)
            //            //    TableIn.AddColumn("rlnCtfImage");

            //            string[] ColumnNames = TableIn.GetColumn("rlnMicrographName");

            //            string[] Excluded = Options.Movies.Where(m => m.Status == ProcessingStatus.Skip).Select(m => m.RootName).ToArray();
            //            List<int> ForDelete = new List<int>();
            //            for (int r = 0; r < TableIn.RowCount; r++)
            //                for (int ex = 0; ex < Excluded.Length; ex++)
            //                    if (ColumnNames[r].Contains(Excluded[ex]))
            //                        ForDelete.Add(r);
            //            TableIn.RemoveRows(ForDelete.ToArray());

            //            ColumnNames = TableIn.GetColumn("rlnMicrographName");
            //            string[] ColumnCoordsX = TableIn.GetColumn("rlnCoordinateX");
            //            string[] ColumnCoordsY = TableIn.GetColumn("rlnCoordinateY");

            //            Star TableOut = new Star(TableIn.GetColumnNames());

            //            int MaxDevices = 999;
            //            int UsedDevices = Math.Min(MaxDevices, GPU.GetDeviceCount());

            //            Queue<DeviceToken> Devices = new Queue<DeviceToken>();
            //            for (int d = 0; d < UsedDevices; d++)
            //                Devices.Enqueue(new DeviceToken(d));
            //            for (int d = 0; d < UsedDevices; d++)
            //                Devices.Enqueue(new DeviceToken(d));
            //            int NTokens = Devices.Count;

            //            DeviceToken[] IOSync = new DeviceToken[UsedDevices];
            //            for (int d = 0; d < UsedDevices; d++)
            //                IOSync[d] = new DeviceToken(d);
            //            for (int d = 0; d < UsedDevices; d++)
            //                IOSync[d] = new DeviceToken(d);
            //            for (int d = 0; d < UsedDevices; d++)
            //                IOSync[d] = new DeviceToken(d);

            //            Image[] ImageGain = new Image[UsedDevices];
            //            if (!string.IsNullOrEmpty(Options.GainPath) && Options.CorrectGain && File.Exists(Options.GainPath))
            //                for (int d = 0; d < UsedDevices; d++)
            //                    try
            //                    {
            //                        GPU.SetDevice(d);
            //                        ImageGain[d] = StageDataLoad.LoadMap(Options.GainPath,
            //                                                             new int2(MainWindow.Options.InputDatWidth, MainWindow.Options.InputDatHeight),
            //                                                             MainWindow.Options.InputDatOffset,
            //                                                             ImageFormatsHelper.StringToType(MainWindow.Options.InputDatType));
            //                    }
            //                    catch
            //                    {
            //                        return;
            //                    }

            //            Dictionary<int, Projector>[] DeviceReferences = new Dictionary<int, Projector>[GPU.GetDeviceCount()];
            //            Dictionary<int, Projector>[] DeviceReconstructions = new Dictionary<int, Projector>[GPU.GetDeviceCount()];
            //            Dictionary<int, Projector>[] DeviceCTFReconstructions = new Dictionary<int, Projector>[GPU.GetDeviceCount()];
            //            for (int d = 0; d < GPU.GetDeviceCount(); d++)
            //            {
            //                GPU.SetDevice(d);

            //                Dictionary<int, Projector> References = new Dictionary<int, Projector>();
            //                {
            //                    //Image Ref1 = StageDataLoad.LoadMap("F:\\26s\\vlion\\warp_ref1.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref2 = StageDataLoad.LoadMap("F:\\26s\\vlion\\warp_ref2.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref3 = StageDataLoad.LoadMap("F:\\26s\\vlion\\warp_ref3.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref4 = StageDataLoad.LoadMap("F:\\26s\\vlion\\warp_ref4.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref5 = StageDataLoad.LoadMap("F:\\26s\\vlion\\warp_ref5.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref6 = StageDataLoad.LoadMap("F:\\26s\\vlion\\warp_ref6.mrc", new int2(1, 1), 0, typeof(float));

            //                    Image Ref1 = StageDataLoad.LoadMap("G:\\lucas_warp\\warp_ref1.mrc", new int2(1, 1), 0, typeof(float));
            //                    Image Ref2 = StageDataLoad.LoadMap("G:\\lucas_warp\\warp_ref2.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref3 = StageDataLoad.LoadMap("F:\\badaben\\vlion4\\warp_ref3.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref4 = StageDataLoad.LoadMap("F:\\badaben\\vlion123\\warp_ref4.mrc", new int2(1, 1), 0, typeof(float));

            //                    References.Add(1, new Projector(Ref1, 2));
            //                    References.Add(2, new Projector(Ref2, 2));
            //                    //References.Add(2, new Projector(Ref2, 2));
            //                    //References.Add(3, new Projector(Ref3, 2));
            //                    //References.Add(4, new Projector(Ref4, 2));
            //                    //References.Add(5, new Projector(Ref5, 2));
            //                    //References.Add(6, new Projector(Ref6, 2));

            //                    Ref1.Dispose();
            //                    Ref2.Dispose();
            //                    //Ref3.Dispose();
            //                    //Ref4.Dispose();
            //                    //Ref5.Dispose();
            //                    //Ref6.Dispose();

            //                    //Image Ref1 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref1.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref2 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref2.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref3 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref3.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref4 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref4.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref5 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref5.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref6 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref6.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref7 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref7.mrc", new int2(1, 1), 0, typeof(float));
            //                    //Image Ref8 = StageDataLoad.LoadMap("D:\\nucleosome\\vlion\\warp_ref8.mrc", new int2(1, 1), 0, typeof(float));

            //                    //References.Add(1, new Projector(Ref1, 2));
            //                    //References.Add(2, new Projector(Ref2, 2));
            //                    //References.Add(3, new Projector(Ref3, 2));
            //                    //References.Add(4, new Projector(Ref4, 2));
            //                    //References.Add(5, new Projector(Ref5, 2));
            //                    //References.Add(6, new Projector(Ref6, 2));
            //                    //References.Add(7, new Projector(Ref7, 2));
            //                    //References.Add(8, new Projector(Ref8, 2));

            //                    //Ref1.Dispose();
            //                    //Ref2.Dispose();
            //                    //Ref3.Dispose();
            //                    //Ref4.Dispose();
            //                    //Ref5.Dispose();
            //                    //Ref6.Dispose();
            //                    //Ref7.Dispose();
            //                    //Ref8.Dispose();
            //                }
            //                DeviceReferences[d] = References;

            //                Dictionary<int, Projector> Reconstructions = new Dictionary<int, Projector>();
            //                foreach (var reference in References)
            //                {
            //                    Reconstructions.Add(reference.Key, new Projector(reference.Value.Dims, reference.Value.Oversampling));
            //                    Reconstructions[reference.Key].FreeDevice();
            //                }
            //                DeviceReconstructions[d] = Reconstructions;

            //                Dictionary<int, Projector> CTFReconstructions = new Dictionary<int, Projector>();
            //                foreach (var reference in References)
            //                {
            //                    CTFReconstructions.Add(reference.Key, new Projector(reference.Value.Dims, reference.Value.Oversampling));
            //                    CTFReconstructions[reference.Key].FreeDevice();
            //                }
            //                DeviceCTFReconstructions[d] = CTFReconstructions;
            //            }

            //            int NTilts = (int)MathHelper.Max(Options.Movies.Select(m => (float)((TiltSeries)m).NTilts));
            //            Dictionary<int, Projector[]> PerAngleReconstructions = new Dictionary<int, Projector[]>();
            //            Dictionary<int, Projector[]> PerAngleWeightReconstructions = new Dictionary<int, Projector[]>();
            //            //{
            //            //    int[] ColumnSubset = TableIn.GetColumn("rlnRandomSubset").Select(s => int.Parse(s)).ToArray();
            //            //    List<int> SubsetIDs = new List<int>();
            //            //    foreach (var subset in ColumnSubset)
            //            //        if (!SubsetIDs.Contains(subset))
            //            //            SubsetIDs.Add(subset);
            //            //    SubsetIDs.Sort();
            //            //    SubsetIDs.Remove(1);
            //            //    SubsetIDs.Remove(2);

            //            //    int Size = Options.ExportParticleSize;

            //            //    //foreach (var subsetID in SubsetIDs)
            //            //    //{
            //            //    //    PerAngleReconstructions.Add(subsetID, new Projector[NTilts]);
            //            //    //    PerAngleWeightReconstructions.Add(subsetID, new Projector[NTilts]);

            //            //    //    for (int t = 0; t < NTilts; t++)
            //            //    //    {
            //            //    //        PerAngleReconstructions[subsetID][t] = new Projector(new int3(Size, Size, Size), 2);
            //            //    //        PerAngleReconstructions[subsetID][t].FreeDevice();
            //            //    //        PerAngleWeightReconstructions[subsetID][t] = new Projector(new int3(Size, Size, Size), 2);
            //            //    //        PerAngleWeightReconstructions[subsetID][t].FreeDevice();
            //            //    //    }
            //            //    //}
            //            //}

            //            TableOut = new Star(new string[0]);

            //            foreach (var movie in Options.Movies)
            //                if (movie.DoProcess)
            //                {
            //                    //if (((TiltSeries)movie).GlobalBfactor < -200)
            //                    //    continue;

            //                    while (Devices.Count <= 0)
            //                        Thread.Sleep(20);

            //                    DeviceToken CurrentDevice;
            //                    lock (Devices)
            //                        CurrentDevice = Devices.Dequeue();

            //                    Thread DeviceThread = new Thread(() =>
            //                    {
            //                        GPU.SetDevice(CurrentDevice.ID);

            //                        MapHeader OriginalHeader = null;
            //                        Image OriginalStack = null;
            //                        decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.PostBinTimes);

            //                        lock (IOSync[CurrentDevice.ID])
            //                            LoadAndPrepareHeaderAndMap(movie.Path, ImageGain[CurrentDevice.ID], ScaleFactor, out OriginalHeader, out OriginalStack);

            //                        if (movie.GetType() == typeof (Movie))
            //                        {
            //                            movie.UpdateStarDefocus(TableIn, ColumnNames, ColumnCoordsX, ColumnCoordsY);
            //                            //movie.ExportParticles(TableIn, TableOut, OriginalHeader, OriginalStack, Options.ExportParticleSize, Options.ExportParticleRadius, ScaleFactor);
            //                        }
            //                        else if (movie.GetType() == typeof (TiltSeries))
            //                        {
            //                            //((TiltSeries)movie).ExportSubtomos(TableIn, OriginalStack, Options.ExportParticleSize, new int3(928, 928, 300));
            //                            //((TiltSeries)movie).ExportSubtomos(TableIn, OriginalStack, Options.ExportParticleSize, new int3(3712, 3712, 1400), 4);
            //                            //((TiltSeries)movie).Export2DParticles(TableIn, TableOut, OriginalStack, Options.ExportParticleSize, new int3(3712, 3712, 1400), 8);

            //                            //((TiltSeries)movie).Reconstruct(OriginalStack, 128, 3.42f * 4 * 2, new int3(3712, 3712, 1400));

            //                            /*for (int refinement = 0; refinement < 20; refinement++)
            //                            {*/

            //                            //OriginalStack.FreeDevice();
            //                            ((TiltSeries)movie).PerformOptimizationStep(TableIn,
            //                                                                        OriginalStack,
            //                                                                        Options.ExportParticleSize,
            //                                                                        new int3(4096, 4096, 1200),
            //                                                                        DeviceReferences[CurrentDevice.ID],
            //                                                                        18f,
            //                                                                        300f,
            //                                                                        30,
            //                                                                        true,
            //                                                                        true,
            //                                                                        false,
            //                                                                        true,
            //                                                                        DeviceReconstructions[0],
            //                                                                        DeviceCTFReconstructions[0]);

            //                            //((TiltSeries)movie).RealspaceRefineGlobal(TableIn, OriginalStack, Options.ExportParticleSize, new int3(3838, 3710, 1200), References, 30, 2, "D4", Reconstructions);


            //                            //Image Simulated = ((TiltSeries)movie).SimulateTiltSeries(TableIn, OriginalStack.Dims, Options.ExportParticleSize, new int3(3712, 3712, 1200), References, 15);
            //                            //Simulated.WriteMRC("d_simulatedseries.mrc");

            //                            //((TiltSeries)movie).AlignTiltMovies(TableIn, OriginalStack.Dims, Options.ExportParticleSize, new int3(3712, 3712, 1200), References, 100);


            //                            /*TableIn.Save(SaveDialog.FileName + $".it{refinement:D2}.star");
            //                            }*/

            //                            //((TiltSeries)movie).ExportSubtomos(TableIn, OriginalStack, 192, new int3(3712, 3712, 1400), 3);

            //                            //OriginalStack.FreeDevice();
            //                            //Image Reference = StageDataLoad.LoadMap("F:\\badaben\\ref_from_refinement.mrc", new int2(1, 1), 0, typeof(float));
            //                            //((TiltSeries)movie).Correlate(OriginalStack, Reference, 128, 3.42f * 4 * 2, 400, new int3(3712, 3712, 1400), 5000, 2, "C1");
            //                            //Reference.Dispose();

            //                            //GPU.SetDevice(0);
            //                            //((TiltSeries)movie).MakePerTomogramReconstructions(TableIn, OriginalStack, Options.ExportParticleSize, new int3(3712, 3712, 1400));
            //                            //((TiltSeries)movie).AddToPerAngleReconstructions(TableIn, OriginalStack, Options.ExportParticleSize, new int3(3710, 3710, 1400), PerAngleReconstructions, PerAngleWeightReconstructions);
            //                        }

            //                        OriginalStack?.Dispose();
            //                        //Debug.WriteLine(movie.Path);
            //                        //TableIn.Save(SaveDialog.FileName);

            //                        lock (Devices)
            //                            Devices.Enqueue(CurrentDevice);

            //                        Debug.WriteLine("Done: " + movie.RootName);
            //                    });
            //                    DeviceThread.Name = movie.RootName + " thread";
            //                    DeviceThread.Start();
            //                }

            //            while (Devices.Count != NTokens)
            //                Thread.Sleep(20);

            //            for (int d = 0; d < UsedDevices; d++)
            //            {
            //                ImageGain[d]?.Dispose();
            //            }

            //            for (int d = 0; d < DeviceReferences.Length; d++)
            //            {
            //                GPU.SetDevice(d);
            //                foreach (var reconstruction in DeviceReconstructions[d])
            //                {
            //                    if (d == 0)
            //                    {
            //                        Image ReconstructedMap = reconstruction.Value.Reconstruct(false);
            //                        //ReconstructedMap.WriteMRC($"F:\\chloroplastribo\\vlion12\\warped_{reconstruction.Key}_nodeconv.mrc");

            //                        Image ReconstructedCTF = DeviceCTFReconstructions[d][reconstruction.Key].Reconstruct(true);

            //                        Image ReconstructedMapFT = ReconstructedMap.AsFFT(true);
            //                        ReconstructedMap.Dispose();

            //                        int Dim = ReconstructedMap.Dims.Y;
            //                        int DimFT = Dim / 2 + 1;
            //                        int R2 = Dim / 2 - 2;
            //                        R2 *= R2;
            //                        foreach (var slice in ReconstructedCTF.GetHost(Intent.ReadWrite))
            //                        {
            //                            for (int y = 0; y < Dim; y++)
            //                            {
            //                                int yy = y < Dim / 2 + 1 ? y : y - Dim;
            //                                yy *= yy;

            //                                for (int x = 0; x < DimFT; x++)
            //                                {
            //                                    int xx = x * x;

            //                                    slice[y * DimFT + x] = xx + yy < R2 ? Math.Max(1e-2f, slice[y * DimFT + x]) : 1f;
            //                                }
            //                            }
            //                        }

            //                        ReconstructedMapFT.Divide(ReconstructedCTF);
            //                        ReconstructedMap = ReconstructedMapFT.AsIFFT(true);
            //                        ReconstructedMapFT.Dispose();

            //                        //GPU.SphereMask(ReconstructedMap.GetDevice(Intent.Read),
            //                        //               ReconstructedMap.GetDevice(Intent.Write),
            //                        //               ReconstructedMap.Dims,
            //                        //               (float)(ReconstructedMap.Dims.X / 2 - 8),
            //                        //               8,
            //                        //               1);

            //                        ReconstructedMap.WriteMRC($"G:\\lucas_warp\\warped_{reconstruction.Key}.mrc");
            //                        ReconstructedMap.Dispose();
            //                    }

            //                    reconstruction.Value.Dispose();
            //                    DeviceReferences[d][reconstruction.Key].Dispose();
            //                    DeviceCTFReconstructions[d][reconstruction.Key].Dispose();
            //                }
            //            }

            //            //string WeightOptimizationDir = ((TiltSeries)Options.Movies[0]).WeightOptimizationDir;
            //            //foreach (var subset in PerAngleReconstructions)
            //            //{
            //            //    for (int t = 0; t < NTilts; t++)
            //            //    {
            //            //        Image Reconstruction = PerAngleReconstructions[subset.Key][t].Reconstruct(false);
            //            //        PerAngleReconstructions[subset.Key][t].Dispose();
            //            //        Reconstruction.WriteMRC(WeightOptimizationDir + $"subset{subset.Key}_tilt{t.ToString("D3")}.mrc");
            //            //        Reconstruction.Dispose();

            //            //        foreach (var slice in PerAngleWeightReconstructions[subset.Key][t].Weights.GetHost(Intent.ReadWrite))
            //            //            for (int i = 0; i < slice.Length; i++)
            //            //                slice[i] = Math.Min(1, slice[i]);
            //            //        Image WeightReconstruction = PerAngleWeightReconstructions[subset.Key][t].Reconstruct(true);
            //            //        PerAngleWeightReconstructions[subset.Key][t].Dispose();
            //            //        WeightReconstruction.WriteMRC(WeightOptimizationDir + $"subset{subset.Key}_tilt{t.ToString("D3")}.weight.mrc");
            //            //        WeightReconstruction.Dispose();
            //            //    }
            //            //}

            //            TableIn.Save(SaveDialog.FileName);
            //            //TableOut.Save(SaveDialog.FileName);
            //        });
            //        ProcessThread.Start();
            //    }
            //}
        }

        private void ButtonPolishParticles_OnClick(object sender, RoutedEventArgs e)
        {
            //System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            //{
            //    Filter = "STAR Files|*.star",
            //    Multiselect = false
            //};
            //System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();
            //if (Result == System.Windows.Forms.DialogResult.OK)
            //{
            //    System.Windows.Forms.SaveFileDialog SaveDialog = new System.Windows.Forms.SaveFileDialog
            //    {
            //        Filter = "STAR Files|*.star"
            //    };
            //    System.Windows.Forms.DialogResult SaveResult = SaveDialog.ShowDialog();
            //    if (SaveResult == System.Windows.Forms.DialogResult.OK)
            //    {
            //        Thread ProcessThread = new Thread(() =>
            //        {
            //            Star TableIn = new Star(Dialog.FileName);
            //            if (TableIn.GetColumn("rlnCtfImage") == null)
            //                TableIn.AddColumn("rlnCtfImage");

            //            string[] ColumnNames = TableIn.GetColumn("rlnMicrographName");
            //            string[] ColumnCoordsX = TableIn.GetColumn("rlnCoordinateX");
            //            string[] ColumnCoordsY = TableIn.GetColumn("rlnCoordinateY");

            //            Star TableOut = new Star(TableIn.GetColumnNames());

            //            Image ImageGain = null;
            //            if (!string.IsNullOrEmpty(Options.GainPath) && Options.CorrectGain)
            //                try
            //                {
            //                    ImageGain = StageDataLoad.LoadMap(Options.GainPath,
            //                                                      new int2(MainWindow.Options.InputDatWidth, MainWindow.Options.InputDatHeight),
            //                                                      MainWindow.Options.InputDatOffset,
            //                                                      ImageFormatsHelper.StringToType(MainWindow.Options.InputDatType));
            //                }
            //                catch
            //                {
            //                    return;
            //                }

            //            foreach (var movie in Options.Movies)
            //                if (movie.DoProcess)
            //                {
            //                    MapHeader OriginalHeader = null;
            //                    Image OriginalStack = null;
            //                    decimal ScaleFactor = 1M / (decimal)Math.Pow(2, (double)Options.PostBinTimes);

            //                    LoadAndPrepareHeaderAndMap(movie.Path, ImageGain, ScaleFactor, out OriginalHeader, out OriginalStack);

            //                    //OriginalStack.WriteMRC("d_stack.mrc");
            //                    movie.UpdateStarDefocus(TableIn, ColumnNames, ColumnCoordsX, ColumnCoordsY);
            //                    movie.ExportParticlesMovie(TableIn, TableOut, OriginalHeader, OriginalStack, Options.ExportParticleSize, Options.ExportParticleRadius, ScaleFactor);

            //                    OriginalStack?.Dispose();
            //                    //Debug.WriteLine(movie.Path);
            //                    TableOut.Save(SaveDialog.FileName);
            //                }

            //            TableOut.Save(SaveDialog.FileName);

            //            ImageGain?.Dispose();
            //        });
            //        ProcessThread.Start();
            //    }
            //}
        }

        private void OptimizePerTomoWeights()
        {
            //if (!Options.Movies.Any(m => m.GetType() == typeof (TiltSeries)))
            //    return;

            //Image Mask1 = StageDataLoad.LoadMap("F:\\chloroplastribo\\vlion12\\mask_C2_post.mrc", new int2(1, 1), 0, typeof (float));
            ////Image Mask2 = StageDataLoad.LoadMap("F:\\badaben\\vlion\\mask_C3_post.mrc", new int2(1, 1), 0, typeof(float));
            ////Image Mask3 = StageDataLoad.LoadMap("F:\\badaben\\vlion\\mask_C4_post.mrc", new int2(1, 1), 0, typeof(float));
            //List<Image> SubsetMasks = new List<Image> { Mask1, Mask1 };

            //int3 Dims = Mask1.Dims;
            //List<WeightOptContainer> Reconstructions = new List<WeightOptContainer>();
            //Dictionary<TiltSeries, int> SeriesIndices = new Dictionary<TiltSeries, int>();

            //foreach (Movie movie in Options.Movies)
            //{
            //    if (!movie.DoProcess)
            //        continue;

            //    TiltSeries Series = (TiltSeries)movie;
            //    string[] FileNames = Directory.EnumerateFiles(Series.WeightOptimizationDir, Series.RootName + "_subset*.mrc").Where(v => v.Contains("subset3") || v.Contains("subset4")).Select(v => new FileInfo(v).Name).ToArray();
            //    if (FileNames.Length == 0)
            //        continue;

            //    string[] MapNames = FileNames.Where(v => !v.Contains(".weight.mrc")).ToArray();
            //    string[] WeightNames = FileNames.Where(v => v.Contains(".weight.mrc")).ToArray();
            //    if (MapNames.Length != WeightNames.Length)
            //        throw new Exception("Number of reconstructions and weights does not match!");

            //    string[] MapSuffixes = MapNames.Select(v => v.Substring(Series.RootName.Length)).ToArray();
            //    int[] MapSubsets = MapSuffixes.Select(v =>
            //    {
            //        string S = v.Substring(v.IndexOf("subset") + "subset".Length);
            //        return int.Parse(S.Substring(0, S.IndexOf(".mrc"))) - 3;
            //    }).ToArray();

            //    SeriesIndices.Add(Series, SeriesIndices.Count);

            //    for (int i = 0; i < MapNames.Length; i++)
            //    {
            //        Image Map = StageDataLoad.LoadMap(Series.WeightOptimizationDir + MapNames[i], new int2(1, 1), 0, typeof (float));
            //        Image MapFT = Map.AsFFT(true);
            //        float[] MapData = MapFT.GetHostContinuousCopy();
            //        Map.Dispose();
            //        MapFT.Dispose();

            //        Image Weights = StageDataLoad.LoadMap(Series.WeightOptimizationDir + WeightNames[i], new int2(1, 1), 0, typeof (float));
            //        float[] WeightsData = Weights.GetHostContinuousCopy();
            //        Weights.Dispose();

            //        Reconstructions.Add(new WeightOptContainer(SeriesIndices[Series], MapSubsets[i], MapData, WeightsData, 0, 0));
            //    }

            //    //break;
            //}

            //float[][] PackedRecFT = new float[SeriesIndices.Count][];
            //float[][] PackedRecWeights = new float[SeriesIndices.Count][];
            //foreach (var s in SeriesIndices)
            //{
            //    WeightOptContainer[] SeriesRecs = Reconstructions.Where(r => r.SeriesID == s.Value).ToArray();
            //    PackedRecFT[s.Value] = new float[SeriesRecs.Length * SeriesRecs[0].DataFT.Length];
            //    PackedRecWeights[s.Value] = new float[SeriesRecs.Length * SeriesRecs[0].DataWeights.Length];
                
            //    for (int n = 0; n < SeriesRecs.Length; n++)
            //    {
            //        Array.Copy(SeriesRecs[n].DataFT, 0, PackedRecFT[s.Value], n * SeriesRecs[0].DataFT.Length, SeriesRecs[0].DataFT.Length);
            //        Array.Copy(SeriesRecs[n].DataWeights, 0, PackedRecWeights[s.Value], n * SeriesRecs[0].DataWeights.Length, SeriesRecs[0].DataWeights.Length);
            //    }
            //}

            //float PixelSize = (float)Options.Movies[0].CTF.PixelSize;
            //float FreqMin = 1f / (12.5f / PixelSize), FreqMin2 = FreqMin * FreqMin;
            //float FreqMax = 1f / (10.7f / PixelSize), FreqMax2 = FreqMax * FreqMax;

            //int ShellMin = (int)(Dims.X * FreqMin);
            //int ShellMax = (int)(Dims.X * FreqMax);
            //int NShells = ShellMax - ShellMin;

            //float[] R2 = new float[(Dims.X / 2 + 1) * Dims.Y * Dims.Z];
            //int[] ShellIndices = new int[R2.Length];

            //for (int z = 0; z < Dims.Z; z++)
            //{
            //    int zz = z < Dims.Z / 2 + 1 ? z : z - Dims.Z;
            //    zz *= zz;
            //    for (int y = 0; y < Dims.Y; y++)
            //    {
            //        int yy = y < Dims.Y / 2 + 1 ? y : y - Dims.Y;
            //        yy *= yy;
            //        for (int x = 0; x < Dims.X / 2 + 1; x++)
            //        {
            //            int xx = x;
            //            xx *= x;

            //            float r = (float)Math.Sqrt(zz + yy + xx) / Dims.X / PixelSize;
            //            R2[(z * Dims.Y + y) * (Dims.X / 2 + 1) + x] = r * r;
            //            int ir = (int)Math.Round(Math.Sqrt(zz + yy + xx));
            //            ShellIndices[(z * Dims.Y + y) * (Dims.X / 2 + 1) + x] = ir < Dims.X / 2 ? ir : -1;
            //        }
            //    }
            //}

            //float[] SeriesWeights = new float[SeriesIndices.Count];
            //float[] SeriesBfacs = new float[SeriesIndices.Count];

            //Func<double[], float[]> WeightedFSC = input =>
            //{
            //    // Set parameters from input vector
            //    //{
            //        int Skip = 0;
            //        SeriesWeights = input.Take(SeriesWeights.Length).Select(v => (float)v / 100f).ToArray();
            //        Skip += SeriesWeights.Length;
            //        SeriesBfacs = input.Skip(Skip).Take(SeriesBfacs.Length).Select(v => (float)v * 10f).ToArray();
            //    //}

            //    // Initialize sum vectors
            //    float[] FSC = new float[Dims.X / 2];

            //    float[] MapSum1 = new float[Dims.ElementsFFT() * 2], MapSum2 = new float[Dims.ElementsFFT() * 2];
            //    float[] WeightSum1 = new float[Dims.ElementsFFT()], WeightSum2 = new float[Dims.ElementsFFT()];

            //    int ElementsFT = (int)Dims.ElementsFFT();

            //    foreach (var s in SeriesIndices)
            //    {
            //        WeightOptContainer[] SeriesRecs = Reconstructions.Where(r => r.SeriesID == s.Value).ToArray();

            //        float[] PrecalcWeights = new float[SeriesRecs.Length];
            //        float[] PrecalcBfacs = new float[SeriesRecs.Length];
            //        int[] PrecalcSubsets = new int[SeriesRecs.Length];

            //        for (int n = 0; n < SeriesRecs.Length; n++)
            //        {
            //            WeightOptContainer reconstruction = SeriesRecs[n];
            //            // Weight is Weight(Series) * exp(Bfac(Series) / 4 * r^2)

            //            float SeriesWeight = (float)Math.Exp(SeriesWeights[reconstruction.SeriesID]);
            //            float SeriesBfac = SeriesBfacs[reconstruction.SeriesID];

            //            PrecalcWeights[n] = SeriesWeight;
            //            PrecalcBfacs[n] = SeriesBfac * 0.25f;
            //            PrecalcSubsets[n] = reconstruction.Subset;
            //        }

            //        CPU.OptimizeWeights(SeriesRecs.Length,
            //                            PackedRecFT[s.Value],
            //                            PackedRecWeights[s.Value],
            //                            R2,
            //                            ElementsFT,
            //                            PrecalcSubsets,
            //                            PrecalcBfacs,
            //                            PrecalcWeights,
            //                            MapSum1,
            //                            MapSum2,
            //                            WeightSum1,
            //                            WeightSum2);
            //    }

            //    for (int i = 0; i < ElementsFT; i++)
            //    {
            //        float Weight = Math.Max(1e-3f, WeightSum1[i]);
            //        MapSum1[i * 2] /= Weight;
            //        MapSum1[i * 2 + 1] /= Weight;

            //        Weight = Math.Max(1e-3f, WeightSum2[i]);
            //        MapSum2[i * 2] /= Weight;
            //        MapSum2[i * 2 + 1] /= Weight;
            //    }
                
            //    Image Map1FT = new Image(MapSum1, Dims, true, true);
            //    Image Map1 = Map1FT.AsIFFT(true);
            //    Map1.Multiply(SubsetMasks[0]);
            //    Image MaskedFT1 = Map1.AsFFT(true);
            //    float[] MaskedFT1Data = MaskedFT1.GetHostContinuousCopy();

            //    Map1FT.Dispose();
            //    Map1.Dispose();
            //    MaskedFT1.Dispose();

            //    Image Map2FT = new Image(MapSum2, Dims, true, true);
            //    Image Map2 = Map2FT.AsIFFT(true);
            //    Map2.Multiply(SubsetMasks[1]);
            //    Image MaskedFT2 = Map2.AsFFT(true);
            //    float[] MaskedFT2Data = MaskedFT2.GetHostContinuousCopy();

            //    Map2FT.Dispose();
            //    Map2.Dispose();
            //    MaskedFT2.Dispose();

            //    float[] Nums = new float[Dims.X / 2];
            //    float[] Denoms1 = new float[Dims.X / 2];
            //    float[] Denoms2 = new float[Dims.X / 2];
            //    for (int i = 0; i < ElementsFT; i++)
            //    {
            //        int Shell = ShellIndices[i];
            //        if (Shell < 0)
            //            continue;

            //        Nums[Shell] += MaskedFT1Data[i * 2] * MaskedFT2Data[i * 2] + MaskedFT1Data[i * 2 + 1] * MaskedFT2Data[i * 2 + 1];
            //        Denoms1[Shell] += MaskedFT1Data[i * 2] * MaskedFT1Data[i * 2] + MaskedFT1Data[i * 2 + 1] * MaskedFT1Data[i * 2 + 1];
            //        Denoms2[Shell] += MaskedFT2Data[i * 2] * MaskedFT2Data[i * 2] + MaskedFT2Data[i * 2 + 1] * MaskedFT2Data[i * 2 + 1];
            //    }

            //    for (int i = 0; i < Dims.X / 2; i++)
            //        FSC[i] = Nums[i] / (float)Math.Sqrt(Denoms1[i] * Denoms2[i]);

            //    return FSC;
            //};

            //Func<double[], double> EvalForGrad = input =>
            //{
            //    return WeightedFSC(input).Skip(ShellMin).Take(NShells).Sum() * Reconstructions.Count;
            //};

            //Func<double[], double> Eval = input =>
            //{
            //    double Score = EvalForGrad(input);
            //    Debug.WriteLine(Score);

            //    return Score;
            //};

            //int Iterations = 0;

            //Func<double[], double[]> Grad = input =>
            //{
            //    double[] Result = new double[input.Length];
            //    double Step = 4;

            //    if (Iterations++ > 15)
            //        return Result;

            //    //Parallel.For(0, input.Length, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i =>
            //    for (int i = 0; i < input.Length; i++)
            //    {
            //        double[] InputCopy = input.ToList().ToArray();
            //        double Original = InputCopy[i];
            //        InputCopy[i] = Original + Step;
            //        double ResultPlus = EvalForGrad(InputCopy);
            //        InputCopy[i] = Original - Step;
            //        double ResultMinus = EvalForGrad(InputCopy);
            //        InputCopy[i] = Original;

            //        Result[i] = (ResultPlus - ResultMinus) / (Step * 2);
            //    }//);

            //    return Result;
            //};

            //List<double> StartParamsList = new List<double>();
            //StartParamsList.AddRange(SeriesWeights.Select(v => (double)v));
            //StartParamsList.AddRange(SeriesBfacs.Select(v => (double)v));

            //double[] StartParams = StartParamsList.ToArray();

            //BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            //Optimizer.Epsilon = 3e-7;
            //Optimizer.Maximize(StartParams);
            
            //EvalForGrad(StartParams);
            
            //foreach (var s in SeriesIndices)
            //{
            //    s.Key.GlobalWeight = (float)Math.Exp(SeriesWeights[s.Value] - MathHelper.Max(SeriesWeights));   // Minus, because exponential
            //    s.Key.GlobalBfactor = SeriesBfacs[s.Value] - MathHelper.Max(SeriesBfacs);

            //    s.Key.SaveMeta();
            //}
        }

        private void OptimizePerTiltWeights()
        {
            //if (!Options.Movies.Any(m => m.GetType() == typeof(TiltSeries)))
            //    return;

            //Image Mask = StageDataLoad.LoadMap("F:\\stefanribo\\vlion\\mask_warped2_OST_post.mrc", new int2(1, 1), 0, typeof(float));
            //List<Image> SubsetMasks = new List<Image> { Mask, Mask };

            //int3 Dims = Mask.Dims;
            //float AngleMin = float.MaxValue, AngleMax = float.MinValue;
            //float DoseMax = float.MinValue;
            //List<WeightOptContainer> Reconstructions = new List<WeightOptContainer>();
            //Dictionary<TiltSeries, int> SeriesIndices = new Dictionary<TiltSeries, int>();

            //int NTilts = 0;
            
            //{
            //    TiltSeries Series = (TiltSeries)Options.Movies[0];
            //    string[] FileNames = Directory.EnumerateFiles(Series.WeightOptimizationDir, "subset*.mrc").Where(p => p.Contains("subset3") || p.Contains("subset4")).Select(v => new FileInfo(v).Name).ToArray();

            //    string[] MapNames = FileNames.Where(v => !v.Contains(".weight.mrc")).ToArray();
            //    string[] WeightNames = FileNames.Where(v => v.Contains(".weight.mrc")).ToArray();
            //    if (MapNames.Length != WeightNames.Length)
            //        throw new Exception("Number of reconstructions and weights does not match!");

            //    string[] MapSuffixes = MapNames;
            //    int[] MapSubsets = MapSuffixes.Select(v =>
            //    {
            //        string S = v.Substring(v.IndexOf("subset") + "subset".Length);
            //        return int.Parse(S.Substring(0, S.IndexOf("_"))) - 3;
            //    }).ToArray();
            //    int[] MapTilts = MapSuffixes.Select(v =>
            //    {
            //        string S = v.Substring(v.IndexOf("tilt") + "tilt".Length);
            //        return int.Parse(S.Substring(0, S.IndexOf(".mrc")));
            //    }).ToArray();

            //    SeriesIndices.Add(Series, SeriesIndices.Count);

            //    float[] MapAngles = MapTilts.Select(t => Series.AnglesCorrect[t]).ToArray();
            //    float[] MapDoses = MapTilts.Select(t => Series.Dose[t]).ToArray();

            //    for (int i = 0; i < MapNames.Length; i++)
            //    {
            //        Image Map = StageDataLoad.LoadMap(Series.WeightOptimizationDir + MapNames[i], new int2(1, 1), 0, typeof(float));
            //        Image MapFT = Map.AsFFT(true);
            //        float[] MapData = MapFT.GetHostContinuousCopy();
            //        Map.Dispose();
            //        MapFT.Dispose();

            //        Image Weights = StageDataLoad.LoadMap(Series.WeightOptimizationDir + WeightNames[i], new int2(1, 1), 0, typeof(float));
            //        float[] WeightsData = Weights.GetHostContinuousCopy();
            //        Weights.Dispose();

            //        Reconstructions.Add(new WeightOptContainer(SeriesIndices[Series], MapSubsets[i], MapData, WeightsData, MapAngles[i], MapDoses[i]));
            //    }

            //    AngleMin = Math.Min(MathHelper.Min(MapAngles), AngleMin);
            //    AngleMax = Math.Max(MathHelper.Max(MapAngles), AngleMax);
            //    DoseMax = Math.Max(MathHelper.Max(MapDoses), DoseMax);

            //    NTilts = Series.NTilts;

            //    //break;
            //}

            //float[][] PackedRecFT = new float[SeriesIndices.Count][];
            //float[][] PackedRecWeights = new float[SeriesIndices.Count][];
            //foreach (var s in SeriesIndices)
            //{
            //    WeightOptContainer[] SeriesRecs = Reconstructions.Where(r => r.SeriesID == s.Value).ToArray();
            //    PackedRecFT[s.Value] = new float[SeriesRecs.Length * SeriesRecs[0].DataFT.Length];
            //    PackedRecWeights[s.Value] = new float[SeriesRecs.Length * SeriesRecs[0].DataWeights.Length];

            //    for (int n = 0; n < SeriesRecs.Length; n++)
            //    {
            //        Array.Copy(SeriesRecs[n].DataFT, 0, PackedRecFT[s.Value], n * SeriesRecs[0].DataFT.Length, SeriesRecs[0].DataFT.Length);
            //        Array.Copy(SeriesRecs[n].DataWeights, 0, PackedRecWeights[s.Value], n * SeriesRecs[0].DataWeights.Length, SeriesRecs[0].DataWeights.Length);
            //    }
            //}

            //float PixelSize = (float)Options.Movies[0].CTF.PixelSize;
            //float FreqMin = 1f / (10f / PixelSize), FreqMin2 = FreqMin * FreqMin;
            //float FreqMax = 1f / (8.5f / PixelSize), FreqMax2 = FreqMax * FreqMax;

            //int ShellMin = (int)(Dims.X * FreqMin);
            //int ShellMax = (int)(Dims.X * FreqMax);
            //int NShells = ShellMax - ShellMin;

            //float[] R2 = new float[(Dims.X / 2 + 1) * Dims.Y * Dims.Z];
            //int[] ShellIndices = new int[R2.Length];

            //for (int z = 0; z < Dims.Z; z++)
            //{
            //    int zz = z < Dims.Z / 2 + 1 ? z : z - Dims.Z;
            //    zz *= zz;
            //    for (int y = 0; y < Dims.Y; y++)
            //    {
            //        int yy = y < Dims.Y / 2 + 1 ? y : y - Dims.Y;
            //        yy *= yy;
            //        for (int x = 0; x < Dims.X / 2 + 1; x++)
            //        {
            //            int xx = x;
            //            xx *= x;

            //            float r = (float)Math.Sqrt(zz + yy + xx) / Dims.X / PixelSize;
            //            R2[(z * Dims.Y + y) * (Dims.X / 2 + 1) + x] = r * r;
            //            int ir = (int)Math.Round(Math.Sqrt(zz + yy + xx));
            //            ShellIndices[(z * Dims.Y + y) * (Dims.X / 2 + 1) + x] = ir < Dims.X / 2 ? ir : -1;
            //        }
            //    }
            //}

            //float[] SeriesWeights = new float[SeriesIndices.Count];
            //float[] SeriesBfacs = new float[SeriesIndices.Count];
            //float[] InitGridAngle = new float[NTilts], InitGridDose = new float[NTilts];
            //for (int i = 0; i < InitGridAngle.Length; i++)
            //{
            //    InitGridAngle[i] = (float)Math.Cos((i / (float)(InitGridAngle.Length - 1) * (AngleMax - AngleMin) + AngleMin) * Helper.ToRad) * 100f;
            //    InitGridDose[i] = -8 * i / (float)(InitGridAngle.Length - 1) * DoseMax / 10f;
            //}
            //CubicGrid GridAngle = new CubicGrid(new int3(NTilts, 1, 1), InitGridAngle);
            //CubicGrid GridDose = new CubicGrid(new int3(NTilts, 1, 1), InitGridDose);

            //Func<double[], float[]> WeightedFSC = input =>
            //{
            //    // Set parameters from input vector
            //    {
            //        int Skip = 0;
            //        GridAngle = new CubicGrid(GridAngle.Dimensions, input.Skip(Skip).Take((int)GridAngle.Dimensions.Elements()).Select(v => (float)v / 100f).ToArray());
            //        Skip += (int)GridAngle.Dimensions.Elements();
            //        GridDose = new CubicGrid(GridDose.Dimensions, input.Skip(Skip).Take((int)GridDose.Dimensions.Elements()).Select(v => (float)v * 10f).ToArray());
            //    }

            //    // Initialize sum vectors
            //    float[] FSC = new float[Dims.X / 2];

            //    float[] MapSum1 = new float[Dims.ElementsFFT() * 2], MapSum2 = new float[Dims.ElementsFFT() * 2];
            //    float[] WeightSum1 = new float[Dims.ElementsFFT()], WeightSum2 = new float[Dims.ElementsFFT()];

            //    int ElementsFT = (int)Dims.ElementsFFT();

            //    foreach (var s in SeriesIndices)
            //    {
            //        WeightOptContainer[] SeriesRecs = Reconstructions.Where(r => r.SeriesID == s.Value).ToArray();

            //        float[] PrecalcWeights = new float[SeriesRecs.Length];
            //        float[] PrecalcBfacs = new float[SeriesRecs.Length];
            //        int[] PrecalcSubsets = new int[SeriesRecs.Length];

            //        for (int n = 0; n < SeriesRecs.Length; n++)
            //        {
            //            WeightOptContainer reconstruction = SeriesRecs[n];
            //            // Weight is Weight(Series) * Weight(Angle) * exp((Bfac(Series) + Bfac(Dose)) / 4 * r^2)                        
                        
            //            float AngleWeight = GridAngle.GetInterpolated(new float3((reconstruction.Angle - AngleMin) / (AngleMax - AngleMin), 0.5f, 0.5f));
            //            float DoseBfac = GridDose.GetInterpolated(new float3(reconstruction.Dose / DoseMax, 0.5f, 0.5f));

            //            PrecalcWeights[n] = AngleWeight;
            //            PrecalcBfacs[n] = DoseBfac * 0.25f;
            //            PrecalcSubsets[n] = reconstruction.Subset;
            //        }

            //        CPU.OptimizeWeights(SeriesRecs.Length,
            //                            PackedRecFT[s.Value],
            //                            PackedRecWeights[s.Value],
            //                            R2,
            //                            ElementsFT,
            //                            PrecalcSubsets,
            //                            PrecalcBfacs,
            //                            PrecalcWeights,
            //                            MapSum1,
            //                            MapSum2,
            //                            WeightSum1,
            //                            WeightSum2);
            //    }

            //    for (int i = 0; i < ElementsFT; i++)
            //    {
            //        float Weight = Math.Max(1e-3f, WeightSum1[i]);
            //        MapSum1[i * 2] /= Weight;
            //        MapSum1[i * 2 + 1] /= Weight;

            //        Weight = Math.Max(1e-3f, WeightSum2[i]);
            //        MapSum2[i * 2] /= Weight;
            //        MapSum2[i * 2 + 1] /= Weight;
            //    }

            //    lock (GridAngle)
            //    {
            //        Image Map1FT = new Image(MapSum1, Dims, true, true);
            //        Image Map1 = Map1FT.AsIFFT(true);
            //        Map1.Multiply(SubsetMasks[0]);
            //        Image MaskedFT1 = Map1.AsFFT(true);
            //        float[] MaskedFT1Data = MaskedFT1.GetHostContinuousCopy();

            //        Map1FT.Dispose();
            //        Map1.Dispose();
            //        MaskedFT1.Dispose();

            //        Image Map2FT = new Image(MapSum2, Dims, true, true);
            //        Image Map2 = Map2FT.AsIFFT(true);
            //        Map2.Multiply(SubsetMasks[1]);
            //        Image MaskedFT2 = Map2.AsFFT(true);
            //        float[] MaskedFT2Data = MaskedFT2.GetHostContinuousCopy();

            //        Map2FT.Dispose();
            //        Map2.Dispose();
            //        MaskedFT2.Dispose();

            //        float[] Nums = new float[Dims.X / 2];
            //        float[] Denoms1 = new float[Dims.X / 2];
            //        float[] Denoms2 = new float[Dims.X / 2];
            //        for (int i = 0; i < ElementsFT; i++)
            //        {
            //            int Shell = ShellIndices[i];
            //            if (Shell < 0)
            //                continue;

            //            Nums[Shell] += MaskedFT1Data[i * 2] * MaskedFT2Data[i * 2] + MaskedFT1Data[i * 2 + 1] * MaskedFT2Data[i * 2 + 1];
            //            Denoms1[Shell] += MaskedFT1Data[i * 2] * MaskedFT1Data[i * 2] + MaskedFT1Data[i * 2 + 1] * MaskedFT1Data[i * 2 + 1];
            //            Denoms2[Shell] += MaskedFT2Data[i * 2] * MaskedFT2Data[i * 2] + MaskedFT2Data[i * 2 + 1] * MaskedFT2Data[i * 2 + 1];
            //        }

            //        for (int i = 0; i < Dims.X / 2; i++)
            //            FSC[i] = Nums[i] / (float)Math.Sqrt(Denoms1[i] * Denoms2[i]);
            //    }

            //    return FSC;
            //};

            //Func<double[], double> EvalForGrad = input =>
            //{
            //    return WeightedFSC(input).Skip(ShellMin).Take(NShells).Sum() * Reconstructions.Count;
            //};

            //Func<double[], double> Eval = input =>
            //{
            //    double Score = EvalForGrad(input);
            //    Debug.WriteLine(Score);

            //    return Score;
            //};

            //int Iterations = 0;

            //Func<double[], double[]> Grad = input =>
            //{
            //    double[] Result = new double[input.Length];
            //    double Step = 1;

            //    if (Iterations++ > 15)
            //        return Result;

            //    //Parallel.For(0, input.Length, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i =>
            //    for (int i = 0; i < input.Length; i++)
            //    {
            //        double[] InputCopy = input.ToList().ToArray();
            //        double Original = InputCopy[i];
            //        InputCopy[i] = Original + Step;
            //        double ResultPlus = EvalForGrad(InputCopy);
            //        InputCopy[i] = Original - Step;
            //        double ResultMinus = EvalForGrad(InputCopy);
            //        InputCopy[i] = Original;

            //        Result[i] = (ResultPlus - ResultMinus) / (Step * 2);
            //    }//);

            //    return Result;
            //};

            //List<double> StartParamsList = new List<double>();
            //StartParamsList.AddRange(GridAngle.FlatValues.Select(v => (double)v));
            //StartParamsList.AddRange(GridDose.FlatValues.Select(v => (double)v));

            //double[] StartParams = StartParamsList.ToArray();

            //BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
            //Optimizer.Epsilon = 3e-7;
            //Optimizer.Maximize(StartParams);

            //EvalForGrad(StartParams);

            //float MaxAngleWeight = MathHelper.Max(GridAngle.FlatValues);
            //GridAngle = new CubicGrid(GridAngle.Dimensions, GridAngle.FlatValues.Select(v => v / MaxAngleWeight).ToArray());

            //float MaxDoseBfac = MathHelper.Max(GridDose.FlatValues);
            //GridDose = new CubicGrid(GridDose.Dimensions, GridDose.FlatValues.Select(v => v - MaxDoseBfac).ToArray());

            //foreach (var s in Options.Movies)
            //{
            //    TiltSeries Series = (TiltSeries)s;
                
            //    List<float> AngleWeights = new List<float>();
            //    List<float> DoseBfacs = new List<float>();
            //    for (int i = 0; i < Series.Angles.Length; i++)
            //    {
            //        float AngleWeight = GridAngle.GetInterpolated(new float3(Math.Min(1, (Series.AnglesCorrect[i] - AngleMin) / (AngleMax - AngleMin)), 0.5f, 0.5f));
            //        float DoseBfac = GridDose.GetInterpolated(new float3(Math.Min(1, Series.Dose[i] / DoseMax), 0.5f, 0.5f));

            //        AngleWeights.Add(AngleWeight);
            //        DoseBfacs.Add(DoseBfac);
            //    }

            //    Series.GridAngleWeights = new CubicGrid(new int3(1, 1, AngleWeights.Count), AngleWeights.ToArray());
            //    Series.GridDoseBfacs = new CubicGrid(new int3(1, 1, DoseBfacs.Count), DoseBfacs.ToArray());

            //    Series.SaveMeta();
            //}
        }

        private void CreateRandomSubsetReconstructions(Star tableIn, int size, int nRec, int particlesPerRec)
        {
            List<string> ParticleNames = new List<string>();
            Dictionary<string, List<int>> ParticleRows = new Dictionary<string, List<int>>();
            Dictionary<string, float3[]> ParticleAngles = new Dictionary<string, float3[]>();
            Dictionary<string, Image> ParticleImages = new Dictionary<string, Image>();
            Dictionary<string, Image> ParticleCTFs = new Dictionary<string, Image>();

            string[] ColumnAngleRot = tableIn.GetColumn("rlnAngleRot");
            string[] ColumnAngleTilt = tableIn.GetColumn("rlnAngleTilt");
            string[] ColumnAnglePsi = tableIn.GetColumn("rlnAnglePsi");
            string[] ColumnImageName = tableIn.GetColumn("rlnImageName");
            string[] ColumnClassNumber = tableIn.GetColumn("rlnClassNumber");

            for (int r = 0; r < tableIn.RowCount; r++)
            {
                if (ColumnClassNumber[r] == "2")
                    continue;

                string ParticleName = ColumnImageName[r];
                ParticleName = ParticleName.Substring(ParticleName.LastIndexOf("/") + 1);
                ParticleName = ParticleName.Substring(0, ParticleName.Length - 1);
                if (!ParticleRows.ContainsKey(ParticleName))
                {
                    ParticleNames.Add(ParticleName);
                    ParticleRows.Add(ParticleName, new List<int>());
                }

                ParticleRows[ParticleName].Add(r);
            }

            foreach (var particleName in ParticleNames)
            {
                {
                    float3[] Angles = new float3[ParticleRows[particleName].Count];
                    int a = 0;
                    foreach (var r in ParticleRows[particleName])
                    {
                        float3 Angle = new float3(float.Parse(ColumnAngleRot[r]),
                                                  float.Parse(ColumnAngleTilt[r]),
                                                  float.Parse(ColumnAnglePsi[r]));
                        Angles[a++] = Angle * Helper.ToRad;
                    }
                    ParticleAngles.Add(particleName, Angles);
                }

                {
                    Image ImageCTF = Image.FromFile("F:\\badaben\\ppca\\particlectf2d\\" + particleName, new int2(1, 1), 0, typeof (float));
                    ImageCTF.IsFT = true;
                    ImageCTF.Dims.X = ImageCTF.Dims.Y;
                    Image ImageParticle = Image.FromFile("F:\\badaben\\ppca\\particles2d\\" + particleName, new int2(1, 1), 0, typeof(float));
                    ImageParticle.RemapToFT();
                    Image ImageParticleFT = ImageParticle.AsFFT();
                    ImageParticle.Dispose();

                    ImageParticleFT.Multiply(ImageCTF);
                    ImageCTF.Multiply(ImageCTF);

                    ParticleImages.Add(particleName, ImageParticleFT);
                    ParticleCTFs.Add(particleName, ImageCTF);

                    ImageCTF.FreeDevice();
                    ImageParticleFT.FreeDevice();
                }
            }

            Random Rnd = new Random(123);
            int PlanForw, PlanBack, PlanForwCTF;
            Projector.GetPlans(new int3(size, size, size), 2, out PlanForw, out PlanBack, out PlanForwCTF);

            for (int n = 0; n < nRec; n++)
            {
                Projector Proj = new Projector(new int3(size, size, size), 2);

                List<string> ChooseFrom = new List<string>(ParticleNames);
                for (int p = 0; p < particlesPerRec; p++)
                {
                    string ParticleName = ChooseFrom[Rnd.Next(ChooseFrom.Count)];
                    ChooseFrom.Remove(ParticleName);

                    Image Images = ParticleImages[ParticleName];
                    Image CTFs = ParticleCTFs[ParticleName];

                    Proj.BackProject(Images, CTFs, ParticleAngles[ParticleName]);

                    Images.FreeDevice();
                    CTFs.FreeDevice();
                }

                Image Rec = Proj.Reconstruct(false, PlanForw, PlanBack, PlanForwCTF);
                Proj.Dispose();

                Rec.WriteMRC("F:\\badaben\\ppca_nomem\\randomsubsets\\" + n.ToString("D6") + ".mrc");
                Rec.Dispose();
            }
        }

        #endregion
    }

    class WeightOptContainer
    {
        public int SeriesID;
        public int Subset;
        public float[] DataFT;
        public float[] DataWeights;
        public float Angle;
        public float Dose;

        public WeightOptContainer(int seriesID, int subset, float[] dataFT, float[] dataWeights, float angle, float dose)
        {
            SeriesID = seriesID;
            Subset = subset;
            DataFT = dataFT;
            DataWeights = dataWeights;
            Angle = angle;
            Dose = dose;
        }
    }

    public class ActionCommand : ICommand
    {
        private readonly Action _action;

        public ActionCommand(Action action)
        {
            _action = action;
        }

        public void Execute(object parameter)
        {
            _action();
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public event EventHandler CanExecuteChanged;
    }
}
