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
                } catch { }
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

            if (ActivePopulation.Species.Count >= 1)
            {
                await this.ShowMessageAsync("Oopsie", "Multiple species aren't supported yet. Oh, the irony!");
                return;
            }

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

                        var NewSpeciesProgress = await this.ShowProgressAsync("Creating new species...",
                                                                              "Please wait while various metrics are calculated for the new map. This will take at least a few minutes.");
                        NewSpeciesProgress.SetIndeterminate();

                        Species NewSpecies = new Species(NewDialogContent.Halfmap1Final, NewDialogContent.Halfmap2Final, NewDialogContent.MaskFinal)
                        {
                            Name = NewDialogContent.SpeciesName,
                            PixelSize = NewDialogContent.HalfmapPixelSize,

                            Symmetry = NewDialogContent.SpeciesSymmetry,
                            DiameterAngstrom = NewDialogContent.SpeciesDiameter,
                            MolecularWeightkDa = NewDialogContent.SpeciesWeight,

                            TemporalResolutionMovement = NewDialogContent.TemporalResMov,
                            TemporalResolutionRotation = NewDialogContent.TemporalResRot,

                            Particles = NewDialogContent.ParticlesFinal
                        };

                        await Task.Run(() =>
                        {
                            NewSpecies.CalculateResolutionAndFilter();
                            NewSpecies.CalculateParticleStats();

                            NewSpecies.Path = ActivePopulation.SpeciesDir + NewSpecies.GUID + "\\" + NewSpecies.NameSafe + ".species";
                            Directory.CreateDirectory(NewSpecies.FolderPath);

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

            if (ActivePopulation.Species.Count > 1)
            {
                await this.ShowMessageAsync("Oopsie", "Multiple species aren't supported yet. Oh, the irony!");
                return;
            }

            if (ActivePopulation.Species.Count == 0)
            {
                await this.ShowMessageAsync("Oopsie", "There are no species in this population. Please create one first.");
                return;
            }

            if (ActivePopulation.Sources.Count(s => s.Files.Any(f => f.Value.LastIndexOf(".tomostar") > -1)) == 0)
            {
                await this.ShowMessageAsync("Oopsie", "Only refinement of tilt series is supported at the moment!");
                return;
            }

            var Progress = await this.ShowProgressAsync("Preparing for refinement...", "");
            Progress.SetIndeterminate();

            ClearSpeciesDisplay();  // To avoid UI updates from refinement thread

            try
            {
                await Task.Run(() =>
                {
                    foreach (var species in ActivePopulation.Species)
                    {
                        Dispatcher.InvokeAsync(() => Progress.SetMessage($"Preprocessing {species.Name}..."));

                        species.PrepareRefinementRequisites();
                    }

                    foreach (var source in ActivePopulation.Sources)
                    {
                        foreach (var pair in source.Files)
                        {
                            if (pair.Value.LastIndexOf(".tomostar") < 0)
                                continue;

                            TiltSeries Series = new TiltSeries(source.FolderPath + pair.Value);

                            Dispatcher.InvokeAsync(() => Progress.SetTitle($"Refining {Series.Name}..."));

                            Series.PerformMultiParticleRefinement(ActivePopulation.Species.ToArray(), source, (s) =>
                                {
                                    Dispatcher.InvokeAsync(() =>
                                    {
                                        Progress.SetMessage(s);
                                    });
                                });

                            Series.SaveMeta();
                        }

                        source.Commit();
                    }

                    Dispatcher.InvokeAsync(() => Progress.SetTitle("Finishing refinement..."));

                    foreach (var species in ActivePopulation.Species)
                    {
                        Dispatcher.InvokeAsync(() => Progress.SetMessage($"Reconstructing and filtering {species.Name}..."));

                        species.FinishRefinement();
                        //species.Commit();
                    }
                });
            }
            catch (Exception exc)
            {
                await Progress.CloseAsync();
                await this.ShowMessageAsync("Oopsie", "Something went wrong during refinement. Sorry! Here are the details:\n\n" +
                                                        exc.ToString());
            }

            await Progress.CloseAsync();

            UpdateSpeciesDisplay();
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
