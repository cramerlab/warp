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
using M.Controls.Sociology.Dialogs;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using Warp;
using Menu = System.Windows.Forms.Menu;
using M.Controls.Sociology;
using Image = Warp.Image;
using M.Controls;

namespace M
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MahApps.Metro.Controls.MetroWindow
    {
        private const string DefaultAnalyticsName = "analytics.settings";

        #region MAIN WINDOW

        private CheckBox[] CheckboxesGPUStats;
        private int[] BaselinesGPUStats;
        private DispatcherTimer TimerGPUStats;
        private DispatcherTimer TimerCheckUpdates;

        readonly List<UIElement> DisableWhenPreprocessing;

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
            };
            DisableWhenPreprocessing.AddRange(CheckboxesGPUStats);

            #endregion

            // Load settings from previous session
            if (File.Exists(DefaultOptionsName))
                Options.Load(DefaultOptionsName);

            OptionsAutoSave = true;

            Options.MainWindow = this;

            #region TEMP

            PanelPopulationLanding.Visibility = Visibility.Visible;
            PanelPopulationMain.Visibility = Visibility.Hidden;

            #endregion
        }

        private void MainWindow_Closing(object sender, CancelEventArgs e)
        {
            try
            {
                SaveDefaultSettings();
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

                this.GlowBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#8ecc1a"));
                this.WindowTitleBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#8ecc1a"));

                SwitchDayNight.Content = "🔆";
            }
        }

        #region Hot keys

        #endregion

        #endregion

        #region Options

        #region Helper variables

        const string DefaultOptionsName = "m.settings";

        public static Options Options = new Options();
        static bool OptionsAutoSave = false;

        #endregion

        private async void Options_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {

        }

        private void OptionsRuntime_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {

        }

        private void SaveDefaultSettings()
        {
            if (OptionsAutoSave)
            {
                try
                {
                    Options.Save(DefaultOptionsName);
                }
                catch { }
            }
        }

        #endregion

        #region TAB: POPULATION

        private Population _ActivePopulation = null;
        public Population ActivePopulation
        {
            get
            {
                return _ActivePopulation;
            }
            set
            {
                if (value != _ActivePopulation)
                {
                    Population OldValue = _ActivePopulation;
                    Population NewValue = value;

                    _ActivePopulation = value;

                    if (OldValue != null)
                    {
                        NewValue.Sources.CollectionChanged += PopulationSources_CollectionChanged;
                        NewValue.Species.CollectionChanged += PopulationSpecies_CollectionChanged;
                    }

                    if (NewValue != null)
                    {
                        NewValue.Sources.CollectionChanged += PopulationSources_CollectionChanged;
                        NewValue.Species.CollectionChanged += PopulationSpecies_CollectionChanged;
                    }

                    GridPopulation.DataContext = NewValue;
                    PopulationUpdateSourceStats();
                    UpdateSpeciesDisplay();
                }
            }
        }

        #region Statistics display

        private void PopulationUpdateSourceStats()
        {
            if (ActivePopulation == null)
                return;

            int NSources = ActivePopulation.Sources.Count;
            int N2D = 0;
            int NTilt = 0;

            foreach (var source in ActivePopulation.Sources)
                foreach (var file in source.Files)
                    if (file.Value.LastIndexOf(".tomostar") > -1)
                        NTilt++;
                    else
                        N2D++;

            Dispatcher.Invoke(() =>
            {
                ButtonPopulationEditSources.Content = NSources > 0 ? $"{NSources} data source{(NSources != 1 ? "s" : "")}" : "Manage data sources";
                TextPopulationSourcesStats.Text = $" ({NTilt} tilt series, {N2D} micrograph{(N2D != 1 ? "s" : "")})";
            });
        }

        #endregion

        #region Population event handling

        private void PopulationSources_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            PopulationUpdateSourceStats();
        }

        private void PopulationSpecies_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            UpdateSpeciesDisplay();
        }

        #endregion

        #region Species display

        public void UpdateSpeciesDisplay()
        {
            PanelAllSpecies.Children.Clear();

            if (ActivePopulation != null)
            {
                foreach (var species in ActivePopulation.Species)
                {
                    PanelAllSpecies.Children.Add(new SpeciesView() { Species = species });
                }

                ButtonPopulationRefine.Visibility = ActivePopulation.Species.Count == 0 ? Visibility.Hidden : Visibility.Visible;
            }

            PanelAllSpecies.Children.Add(ButtonPopulationAddSpecies);
        }

        public void ClearSpeciesDisplay()
        {
            foreach (var item in PanelAllSpecies.Children)
                if (item.GetType() == typeof(SpeciesView))
                    ((SpeciesView)item).Species = null;

            PanelAllSpecies.Children.Clear();
        }

        #endregion

        #region Button event handling

        private void ButtonPopulationCreateNew_OnClick(object sender, RoutedEventArgs e)
        {
            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogCreatePopulation DialogContent = new DialogCreatePopulation();
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);

            DialogContent.Create += () =>
            {
                string PopulationName = Helper.RemoveInvalidChars(DialogContent.TextPopulationName.Text);
                string PopulationPath = (string)DialogContent.ButtonChangeFolder.Content + PopulationName + ".population";

                Population NewPopulation = new Population(PopulationPath);
                NewPopulation.Name = DialogContent.TextPopulationName.Text;
                NewPopulation.Save();

                ActivePopulation = NewPopulation;

                PanelPopulationLanding.Visibility = Visibility.Hidden;
                PanelPopulationMain.Visibility = Visibility.Visible;

                this.HideMetroDialogAsync(Dialog);
            };

            Dialog.Content = DialogContent;

            this.ShowMetroDialogAsync(Dialog);
            Dispatcher.InvokeAsync(() => DialogContent.TextPopulationName.TriggerEdit(), DispatcherPriority.ApplicationIdle);
        }

        private void ButtonPopulationLoad_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog
            {
                Filter = "Population Files|*.population",
                Multiselect = false
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Population LoadedPopulation = new Population(Dialog.FileName);
                ActivePopulation = LoadedPopulation;

                PanelPopulationLanding.Visibility = Visibility.Hidden;
                PanelPopulationMain.Visibility = Visibility.Visible;
            }
        }

        private void ButtonPopulationEditSources_OnClick(object sender, RoutedEventArgs e)
        {
            if (ActivePopulation == null)
                return;

            CustomDialog Dialog = new CustomDialog();
            Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogDataSources DialogContent = new DialogDataSources(ActivePopulation);
            DialogContent.MaxHeight = this.ActualHeight - 200;
            DialogContent.Close += () => this.HideMetroDialogAsync(Dialog);

            Dialog.Content = DialogContent;
            this.ShowMetroDialogAsync(Dialog);
        }

        private async void ButtonPopulationAddSpecies_OnClick(object sender, RoutedEventArgs e)
        {
            if (ActivePopulation == null)
                return;

            CustomDialog AddDialog = new CustomDialog();
            AddDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogAddSpecies AddDialogContent = new DialogAddSpecies();
            AddDialogContent.Close += async () => await this.HideMetroDialogAsync(AddDialog);

            AddDialogContent.Add += async () =>
            {
                await this.HideMetroDialogAsync(AddDialog);

                if ((bool)AddDialogContent.RadioLocal.IsChecked)
                {
                    await this.ShowMessageAsync("Oopsie", "This feature is not implemented yet.");
                }
                else if ((bool)AddDialogContent.RadioRemote.IsChecked)
                {
                    await this.ShowMessageAsync("Oopsie", "This feature is not implemented yet.");
                }
                else if ((bool)AddDialogContent.RadioScratch.IsChecked)
                {
                    CustomDialog NewDialog = new CustomDialog();
                    NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                    DialogCreateNewSpecies NewDialogContent = new DialogCreateNewSpecies(ActivePopulation);
                    NewDialogContent.Close += async () => await this.HideMetroDialogAsync(NewDialog);

                    NewDialogContent.Finish += async () =>
                    {
                        await this.HideMetroDialogAsync(NewDialog);

                        string DefaultProgressMessage = "Please wait while various metrics are calculated for the new map. This will take at least a few minutes.\n";
                        var NewSpeciesProgress = await this.ShowProgressAsync("Creating new species...",
                                                                              DefaultProgressMessage);
                        NewSpeciesProgress.SetIndeterminate();

                        Species NewSpecies = new Species(NewDialogContent.Halfmap1Final, NewDialogContent.Halfmap2Final, NewDialogContent.MaskFinal)
                        {
                            Name = NewDialogContent.SpeciesName,
                            PixelSize = NewDialogContent.HalfmapPixelSize,

                            Symmetry = NewDialogContent.SpeciesSymmetry,
                            DiameterAngstrom = NewDialogContent.SpeciesDiameter,
                            MolecularWeightkDa = NewDialogContent.SpeciesWeight,

                            TemporalResolutionMovement = NewDialogContent.TemporalResMov,
                            TemporalResolutionRotation = NewDialogContent.TemporalResMov,

                            Particles = NewDialogContent.ParticlesFinal
                        };

                        await Task.Run(() =>
                        {
                            NewSpecies.Path = ActivePopulation.SpeciesDir + NewSpecies.GUID.ToString().Substring(0, 8) + "\\" + NewSpecies.NameSafe + ".species";
                            Directory.CreateDirectory(NewSpecies.FolderPath);

                            NewSpecies.CalculateResolutionAndFilter(-1, (message) => Dispatcher.Invoke(() => NewSpeciesProgress.SetMessage(DefaultProgressMessage + "\n" + message)));

                            Dispatcher.Invoke(() => NewSpeciesProgress.SetMessage(DefaultProgressMessage + "\n" + "Calculating particle statistics"));
                            NewSpecies.CalculateParticleStats();


                            Dispatcher.Invoke(() => NewSpeciesProgress.SetMessage(DefaultProgressMessage + "\n" + "Committing results"));
                            NewSpecies.Commit();
                            NewSpecies.Save();
                        });

                        ActivePopulation.Species.Add(NewSpecies);

                        await NewSpeciesProgress.CloseAsync();
                    };

                    NewDialog.Content = NewDialogContent;
                    await this.ShowMetroDialogAsync(NewDialog);
                }
            };

            AddDialog.Content = AddDialogContent;
            await this.ShowMetroDialogAsync(AddDialog);
        }

        private async void ButtonPopulationRefine_Click(object sender, RoutedEventArgs e)
        {
            if (ActivePopulation == null)
                return;

            //if (ActivePopulation.Species.Count > 1)
            //{
            //    await this.ShowMessageAsync("Oopsie", "Multiple species aren't supported yet. Oh, the irony!");
            //    return;
            //}

            if (ActivePopulation.Species.Count == 0)
            {
                await this.ShowMessageAsync("Oopsie", "There are no species in this population. Please create one first.");
                return;
            }

            CustomDialog SettingsDialog = new CustomDialog();
            SettingsDialog.HorizontalContentAlignment = HorizontalAlignment.Center;
            SettingsDialog.DataContext = ActivePopulation.LastRefinementOptions;

            DialogRefinementSettings SettingsDialogContent = new DialogRefinementSettings();
            SettingsDialogContent.DataContext = ActivePopulation.LastRefinementOptions;
            SettingsDialogContent.Close += async () => await this.HideMetroDialogAsync(SettingsDialog);
            SettingsDialogContent.StartRefinement += async () =>
            {
                await this.HideMetroDialogAsync(SettingsDialog);

                ClearSpeciesDisplay();  // To avoid UI updates from refinement thread

                ProcessingOptionsMPARefine Options = ActivePopulation.LastRefinementOptions;

                Options.BFactorWeightingThreshold = 0.25M;

                Options.DoAstigmatismDelta = Options.DoDefocus;
                Options.DoAstigmatismAngle = Options.DoDefocus;

                Dispatcher.Invoke(() => { SettingsDialogContent.DataContext = null; });

                PerformRefinementIteration(Options);

                UpdateSpeciesDisplay();
            };

            SettingsDialog.Content = SettingsDialogContent;
            await this.ShowMetroDialogAsync(SettingsDialog);
        }

        private async void PerformRefinementIteration(ProcessingOptionsMPARefine options)
        {
            bool DoMultiProcess = true;

            #region Create worker processes

            int NDevices = GPU.GetDeviceCount();
            List<int> UsedDevices = Options.MainWindow.GetDeviceList();

            WorkerWrapper[] Workers = new WorkerWrapper[GPU.GetDeviceCount()];
            string[] WorkerFolders = new string[Workers.Length];
            string[] WorkerLogs = new string[Workers.Length];

            foreach (var gpuID in UsedDevices)
            {
                WorkerFolders[gpuID] = System.IO.Path.Combine(ActivePopulation.FolderPath, "refinement_temp", $"worker{gpuID}");
                Directory.CreateDirectory(WorkerFolders[gpuID]);
            }

            if (DoMultiProcess)
                foreach (var gpuID in UsedDevices)
                {
                    Workers[gpuID] = new WorkerWrapper(gpuID);
                    Workers[gpuID].SetHeaderlessParams(new int2(2),
                                                       0,
                                                       "float");

                    WorkerLogs[gpuID] = System.IO.Path.Combine(ActivePopulation.FolderPath, "refinement_temp", $"worker{gpuID}", "run.out");
                }

            #endregion

            var Progress = await this.ShowProgressAsync("Preparing for refinement – this will take a few minutes per species", "");
            Progress.SetIndeterminate();

            int ItemsCompleted = 0;
            int ItemsToDo = ActivePopulation.Sources.Select(s => s.Files.Count).Sum();

            long PinnedMemoryLimit = 1 << 30;

            string[] CurrentlyRefinedItems = new string[GPU.GetDeviceCount()];

            System.Timers.Timer StatusUpdater = null;
            if (DoMultiProcess)
            {
                StatusUpdater = new System.Timers.Timer(1001);
                StatusUpdater.Elapsed += (s, e) =>
                {
                    lock (CurrentlyRefinedItems)
                    {
                        StringBuilder StatusMessage = new StringBuilder();

                        foreach (var gpuID in UsedDevices)
                        {
                            if (CurrentlyRefinedItems[gpuID] == null)
                                continue;

                            try
                            {
                                string ItemMessage = File.ReadLines(WorkerLogs[gpuID]).Last();
                                StatusMessage.Append(CurrentlyRefinedItems[gpuID] + ": " + ItemMessage + "\n");
                            }
                            catch { }
                        }

                        Dispatcher.Invoke(() => Progress.SetMessage(StatusMessage.ToString()));
                    }
                };
            }

            try
            {
                await Task.Run(() =>
                {
                    Dispatcher.InvokeAsync(() => Progress.SetMessage($"Figuring out memory capacity..."));
                    WorkerWrapper[] MemoryTesters = Helper.ArrayOfFunction(i => new WorkerWrapper(0), 4);
                    try
                    {
                        int Tested = 0;
                        while (true)
                        {
                            long ChunkSize = (long)1 << 30; // 1 GB
                            long IncreaseBy = (long)1 << 31; // 2 GB
                            MemoryTesters[Tested % MemoryTesters.Length].TryAllocatePinnedMemory(Helper.ArrayOfConstant(ChunkSize, (int)(IncreaseBy / ChunkSize)));

                            PinnedMemoryLimit += IncreaseBy;
                            Tested++;
                        }
                    }
                    catch
                    {
                        PinnedMemoryLimit = PinnedMemoryLimit * 15 / 100; // Take 15% of that limit because Windows is weird
                    }
                    foreach (var item in MemoryTesters)
                        item.Dispose();

                    if (DoMultiProcess)
                    {
                        Dispatcher.InvokeAsync(() => Progress.SetMessage($"Preparing refinement requisites..."));

                        Helper.ForEachGPUOnce(gpuID =>
                        {
                            Workers[gpuID].MPAPreparePopulation(ActivePopulation.Path);
                        }, UsedDevices);

                        foreach (var species in ActivePopulation.Species)
                            species.PrepareRefinementRequisites(true, 0);
                    }
                    else
                    {
                        foreach (var species in ActivePopulation.Species)
                        {
                            Dispatcher.InvokeAsync(() => Progress.SetMessage($"Preprocessing {species.Name}..."));

                            species.PrepareRefinementRequisites();
                        }
                    }

                    GPU.CheckGPUExceptions();

                    Dispatcher.InvokeAsync(() => Progress.SetTitle("Performing refinement"));

                    Image.PrintObjectIDs();
                    if (true)
                        foreach (var source in ActivePopulation.Sources)
                        {
                            //break;
                            Dispatcher.InvokeAsync(() => Progress.SetMessage($"Loading gain reference for {source.Name}..."));

                            Image[] GainRefs = new Image[GPU.GetDeviceCount()];

                            try
                            {
                                if (DoMultiProcess)
                                {
                                    Helper.ForEachGPUOnce(gpuID =>
                                    {
                                        Workers[gpuID].LoadGainRef(source.GainPath,
                                                                   source.GainFlipX,
                                                                   source.GainFlipY,
                                                                   source.GainTranspose,
                                                                   source.DefectsPath);
                                    }, UsedDevices);
                                }
                                else
                                {
                                    Image GainRef = source.LoadAndPrepareGainReference();
                                    if (GainRef != null)
                                        GainRefs = Helper.ArrayOfFunction(i => GainRef.GetCopy(), GPU.GetDeviceCount());
                                }
                            }
                            catch
                            {
                                throw new Exception($"Could not load gain reference for {source.Name}.");
                            }

                            if (DoMultiProcess)
                                StatusUpdater.Start();

                            #region Load all items and determine pinned memory footprint for each of them

                            List<Movie> AllItems = source.Files.Select(pair => source.IsTiltSeries ? new TiltSeries(source.FolderPath + pair.Value) :
                                                                                                     new Movie(source.FolderPath + pair.Value)).ToList();

                            Dictionary<Movie, long> ItemFootprints = new Dictionary<Movie, long>();
                            foreach (var item in AllItems)
                                ItemFootprints.Add(item, item.MultiParticleRefinementCalculateHostMemory(options, ActivePopulation.Species.ToArray(), source));

                            AllItems.Sort((a, b) => ItemFootprints[a].CompareTo(ItemFootprints[b]));

                            #endregion

                            long OverallFootprint = 0;

                            Queue<DeviceToken> Devices = new Queue<DeviceToken>();
                            for (int d = UsedDevices.Count - 1; d >= 0; d--)
                                Devices.Enqueue(new DeviceToken(UsedDevices[d]));

                            int NTokens = Devices.Count;
                            bool IsCanceled = false;

                            // A modified version of Helper.ForEachGPU()
                            int NDone = 0;
                            while (AllItems.Count > 0)
                            {
                                if (IsCanceled)
                                    break;

                                //if (NDone++ < 200)
                                //{
                                //    AllItems.RemoveAt(AllItems.Count - 1);
                                //    continue;
                                //}

                                //if (NDone++ > 20)
                                //    break;

                                while (Devices.Count <= 0)
                                    Thread.Sleep(5);

                                DeviceToken CurrentDevice = null;
                                Movie CurrentItem = null;

                                while (CurrentItem == null)
                                {
                                    int ItemID = AllItems.Count - 1;
                                    lock (Devices)  // Don't want OverallFootprint to change while checking
                                        while (ItemID >= 0 && OverallFootprint + ItemFootprints[AllItems[ItemID]] > PinnedMemoryLimit)
                                            ItemID--;

                                    // No suitable item found and there is hope more memory will become available later
                                    if (ItemID < 0 && OverallFootprint > 0)
                                    {
                                        Thread.Sleep(5);
                                        continue;
                                    }

                                    // Either item can fit, or there is no hope for more memory later, so try anyway
                                    if (ItemID < 0 && OverallFootprint == 0)
                                        ItemID = AllItems.Count - 1;
                                    ItemID = Math.Max(0, ItemID);
                                    CurrentItem = AllItems[ItemID];
                                    AllItems.Remove(CurrentItem);

                                    lock (Devices)
                                    {
                                        CurrentDevice = Devices.Dequeue();
                                        OverallFootprint += ItemFootprints[CurrentItem];
                                    }

                                    break;
                                }

                                Thread DeviceThread = new Thread(() =>
                                {
                                    int GPUID = CurrentDevice.ID;
                                    GPU.SetDevice(GPUID);

                                    if (DoMultiProcess)
                                    {
                                        lock (CurrentlyRefinedItems)
                                            CurrentlyRefinedItems[GPUID] = CurrentItem.Name;

                                        Workers[GPUID].MPARefine(CurrentItem.Path,
                                                                 WorkerFolders[GPUID],
                                                                 WorkerLogs[GPUID],
                                                                 options,
                                                                 source);

                                        lock (CurrentlyRefinedItems)
                                            CurrentlyRefinedItems[GPUID] = null;
                                    }
                                    else
                                    {
                                        Dispatcher.InvokeAsync(() => Progress.SetTitle($"Refining {CurrentItem.Name}..."));

                                        CurrentItem.PerformMultiParticleRefinement(WorkerFolders[GPUID], options, ActivePopulation.Species.ToArray(), source, GainRefs[GPUID], null, (s) =>
                                        {
                                            Dispatcher.InvokeAsync(() =>
                                            {
                                                Progress.SetMessage(s);
                                            });
                                        });

                                        CurrentItem.SaveMeta();

                                        GPU.CheckGPUExceptions();
                                    }

                                    Dispatcher.Invoke(() =>
                                    {
                                        ItemsCompleted++;

                                        Progress.Maximum = ItemsToDo;
                                        Progress.SetProgress(ItemsCompleted);
                                    });

                                    lock (Devices)
                                    {
                                        Devices.Enqueue(CurrentDevice);
                                        OverallFootprint -= ItemFootprints[CurrentItem];
                                    }
                                })
                                { Name = $"ForEachGPU Device {CurrentDevice.ID}" };

                                DeviceThread.Start();
                            }

                            while (Devices.Count != NTokens)
                                Thread.Sleep(5);

                            if (DoMultiProcess)
                                StatusUpdater.Stop();

                            source.Commit();
                        }

                    Image.PrintObjectIDs();

                    Dispatcher.InvokeAsync(() => Progress.SetTitle("Finishing refinement"));

                    if (DoMultiProcess)
                    {
                        Dispatcher.InvokeAsync(() => Progress.SetMessage("Saving intermediate results"));

                        Helper.ForEachGPUOnce(gpuID =>
                        {
                            Workers[gpuID].MPASaveProgress(WorkerFolders[gpuID]);
                            Workers[gpuID].Dispose();
                        }, UsedDevices);

                        Dispatcher.InvokeAsync(() => Progress.SetMessage("Gathering intermediate results"));

                        ActivePopulation.GatherRefinementProgress(UsedDevices.Select(gpuID => WorkerFolders[gpuID]).ToArray());

                        foreach (var folder in WorkerFolders)
                            try
                            {
                                Directory.Delete(folder, true);
                            }
                            catch { }
                    }

                    foreach (var species in ActivePopulation.Species)
                    {
                        Dispatcher.InvokeAsync(() => Progress.SetMessage($"Reconstructing and filtering {species.Name}..."));

                        species.FinishRefinement();
                        species.Commit();
                    }

                    ActivePopulation.Save();
                });
            }
            catch (Exception exc)
            {
                await Progress.CloseAsync();
                await this.ShowMessageAsync("Oopsie", "Something went wrong during refinement. Sorry! Here are the details:\n\n" +
                                                        exc.ToString());
            }

            await Progress.CloseAsync();
        }

        #endregion

        #endregion

        #region Helper methods

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
