using System;
using System.Collections.Generic;
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
using Warp.Sociology;
using Warp.Tools;

namespace Warp.Controls.Sociology.Dialogs
{
    /// <summary>
    /// Interaction logic for DialogDataSources.xaml
    /// </summary>
    public partial class DialogDataSources : UserControl
    {
        private Population Population;

        public event Action Close;

        public DialogDataSources(Population population)
        {
            InitializeComponent();

            Population = population;
            UpdateGridItems();
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private void UpdateGridItems()
        {
            GridSources.Children.Clear();
            GridSources.RowDefinitions.Clear();

            double TableFontSize = 14;

            if (Population.Sources.Count > 0)
            {
                int NLocal = Population.Sources.Sum(s => s.IsRemote ? 0 : 1);
                int NRemote = Population.Sources.Count - NLocal;
                bool AnyTomo = Population.Sources.Any(s => s.IsTiltSeries);

                Style ButtonStyle = (Style)FindResource("ChromelessButtonStyle");

                #region Header with column names

                GridSources.RowDefinitions.Add(new RowDefinition { Height = new GridLength(30) });

                {
                    TextBlock Text = new TextBlock
                    {
                        Text = "Name",
                        FontSize = TableFontSize,
                        FontWeight = FontWeights.Medium,
                        Margin = new Thickness(0, 0, 16, 0)
                    };
                    Grid.SetColumn(Text, 1);
                    GridSources.Children.Add(Text);
                }
                {
                    TextBlock Text = new TextBlock
                    {
                        Text = "Path",
                        FontSize = TableFontSize,
                        FontWeight = FontWeights.Medium,
                        Margin = new Thickness(0, 0, 16, 0)
                    };
                    Grid.SetColumn(Text, 2);
                    GridSources.Children.Add(Text);
                }
                {
                    TextBlock Text = new TextBlock
                    {
                        Text = "Files",
                        FontSize = TableFontSize,
                        FontWeight = FontWeights.Medium,
                        Margin = new Thickness(0, 0, 16, 0)
                    };
                    Grid.SetColumn(Text, 3);
                    GridSources.Children.Add(Text);
                }
                {
                    TextBlock Text = new TextBlock
                    {
                        Text = "Pixel size",
                        FontSize = TableFontSize,
                        FontWeight = FontWeights.Medium,
                        Margin = new Thickness(0, 0, 16, 0)
                    };
                    Grid.SetColumn(Text, 4);
                    GridSources.Children.Add(Text);
                }
                if (AnyTomo)
                {
                    TextBlock Text = new TextBlock
                    {
                        Text = "Volume dimensions",
                        FontSize = TableFontSize,
                        FontWeight = FontWeights.Medium,
                        Margin = new Thickness(0, 0, 16, 0)
                    };
                    Grid.SetColumn(Text, 5);
                    GridSources.Children.Add(Text);
                }
                if (NLocal > 0)
                {
                    TextBlock Text = new TextBlock
                    {
                        Text = "Freeze",
                        FontSize = TableFontSize,
                        FontWeight = FontWeights.Medium,
                        Margin = new Thickness(0, 0, 16, 0),
                        ToolTip = "Don't change frame/tilt series alignment."
                    };
                    Grid.SetColumn(Text, 6);
                    GridSources.Children.Add(Text);
                }

                #endregion

                #region Source rows

                for (int s = 0; s < Population.Sources.Count; s++)
                {
                    GridSources.RowDefinitions.Add(new RowDefinition { Height = new GridLength(30) });

                    DataSource Source = Population.Sources[s];
                    
                    {
                        TextBlock Text = new TextBlock
                        {
                            Text = Source.IsRemote ? "☁" : "📀",
                            FontSize = TableFontSize + 2,
                            VerticalAlignment = VerticalAlignment.Center,
                            Width = 32
                        };
                        Grid.SetColumn(Text, 0);
                        Grid.SetRow(Text, s + 1);
                        GridSources.Children.Add(Text);
                    }
                    {
                        TextBlock Text = new TextBlock
                        {
                            Text = Source.Name,
                            FontSize = TableFontSize,
                            VerticalAlignment = VerticalAlignment.Center,
                            Margin = new Thickness(0, 0, 16, 0)
                        };
                        Grid.SetColumn(Text, 1);
                        Grid.SetRow(Text, s + 1);
                        GridSources.Children.Add(Text);
                    }
                    {
                        string TrimmedPath = Helper.ShortenString(Source.Path, 50);

                        TextBlock Text = new TextBlock
                        {
                            Text = TrimmedPath,
                            FontSize = TableFontSize,
                            HorizontalAlignment = HorizontalAlignment.Left,
                            VerticalAlignment = VerticalAlignment.Center,
                            Margin = new Thickness(0, 0, 16, 0)
                        };
                        Grid.SetColumn(Text, 2);
                        Grid.SetRow(Text, s + 1);
                        GridSources.Children.Add(Text);
                    }
                    {
                        TextBlock Text = new TextBlock
                        {
                            Text = Source.Files.Count.ToString(),
                            FontSize = TableFontSize,
                            Margin = new Thickness(0, 0, 16, 0),
                            HorizontalAlignment = HorizontalAlignment.Right,
                            VerticalAlignment = VerticalAlignment.Center
                        };
                        Grid.SetColumn(Text, 3);
                        Grid.SetRow(Text, s + 1);
                        GridSources.Children.Add(Text);
                    }
                    {
                        TextBlock Text = new TextBlock
                        {
                            Text = $"{Source.PixelSizeX}/{Source.PixelSizeY} Å, {Source.PixelSizeAngle} °",
                            FontSize = TableFontSize,
                            VerticalAlignment = VerticalAlignment.Center,
                            Margin = new Thickness(0, 0, 16, 0)
                        };
                        Grid.SetColumn(Text, 4);
                        Grid.SetRow(Text, s + 1);
                        GridSources.Children.Add(Text);
                    }
                    if (Source.IsTiltSeries)
                    {
                        TextBlock Text = new TextBlock
                        {
                            Text = $"{Source.DimensionsX} x {Source.DimensionsY} x {Source.DimensionsZ} Å",
                            FontSize = TableFontSize,
                            VerticalAlignment = VerticalAlignment.Center,
                            Margin = new Thickness(0, 0, 16, 0)
                        };
                        Grid.SetColumn(Text, 5);
                        Grid.SetRow(Text, s + 1);
                        GridSources.Children.Add(Text);
                    }
                    if (NLocal > 0)
                    {
                        CheckBox Box = new CheckBox
                        {
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center,
                            Margin = new Thickness(0, 0, 8, 0),
                            ToolTip = "Don't change frame/tilt series alignment."
                        };
                        if (Source.IsRemote)
                        {
                            Box.IsChecked = true;
                            Box.IsEnabled = false;
                            Box.Opacity = 0.4;
                        }
                        Grid.SetColumn(Box, 6);
                        Grid.SetRow(Box, s + 1);
                        GridSources.Children.Add(Box);
                    }
                    if (Source.IsRemote)
                    {
                        Button Button = new Button
                        {
                            Content = "DOWNLOAD",
                            FontSize = TableFontSize,
                            FontWeight = FontWeights.Medium,
                            Foreground = Brushes.CornflowerBlue,
                            Style = ButtonStyle,
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center,
                            Margin = new Thickness(10, 0, 10, 0)
                        };
                        Grid.SetColumn(Button, 7);
                        Grid.SetRow(Button, s + 1);
                        GridSources.Children.Add(Button);

                        Button.Click += async (sender, args) =>
                        {
                            await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                "Not implemented yet.",
                                                                                                MessageDialogStyle.Affirmative);
                        };
                    }
                    {
                        Button Button = new Button
                        {
                            Content = "REMOVE",
                            FontSize = TableFontSize,
                            FontWeight = FontWeights.Medium,
                            Foreground = Brushes.Red,
                            Style = ButtonStyle,
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center,
                            Margin = new Thickness(10, 0, 0, 0)
                        };
                        Grid.SetColumn(Button, 8);
                        Grid.SetRow(Button, s + 1);
                        GridSources.Children.Add(Button);

                        Button.Click += async (sender, args) =>
                        {
                            var Result = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Would you like to remove this data source?",
                                                                                                             "This will not physically delete any files.",
                                                                                                             MessageDialogStyle.AffirmativeAndNegative, new MetroDialogSettings()
                                                                                                             {
                                                                                                                 AffirmativeButtonText = "Yes",
                                                                                                                 NegativeButtonText = "No"
                                                                                                             });
                            if (Result == MessageDialogResult.Affirmative)
                            {
                                if (Population.Sources.Contains(Source))
                                {
                                    Population.Sources.Remove(Source);
                                    UpdateGridItems();
                                }
                            }
                        };
                    }
                }

                #endregion
            }
        }

        private async void ButtonAddLocal_OnClick(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog OpenDialog = new System.Windows.Forms.OpenFileDialog();
            OpenDialog.Filter = "Warp Folder Settings|*.settings|Warp Data Source|*.source";
            System.Windows.Forms.DialogResult OpenResult = OpenDialog.ShowDialog();
            
            if (OpenResult.ToString() == "OK")
            {
                FileInfo Info = new FileInfo(OpenDialog.FileName);

                #region Check if user wants to use an existing source file instead

                //if (Info.Extension.ToLower() == ".settings" && 
                //    File.Exists(OpenDialog.FileName.Substring(0, OpenDialog.FileName.LastIndexOf(".")) + ".source"))
                //{
                //    var Result = await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Similar source file found",
                //                                                                                     $"Would you like to use {Helper.PathToName(Info.Name) + ".source"} instead?",
                //                                                                                     MessageDialogStyle.AffirmativeAndNegativeAndSingleAuxiliary,
                //                                                                                     new MetroDialogSettings
                //                                                                                     {
                //                                                                                         AffirmativeButtonText = "Use existing source",
                //                                                                                         NegativeButtonText = "Replace with new source from settings",
                //                                                                                         FirstAuxiliaryButtonText = "Cancel"
                //                                                                                     });

                //    if (Result == MessageDialogResult.FirstAuxiliary)   // Cancel
                //    {
                //        return;
                //    }
                //    else if (Result == MessageDialogResult.Affirmative) // Use existing .source
                //    {
                //        OpenDialog.FileName = OpenDialog.FileName.Substring(0, OpenDialog.FileName.LastIndexOf(".")) + ".source";
                //        Info = new FileInfo(OpenDialog.FileName);
                //    }
                //}

                #endregion

                if (Info.Extension.ToLower() == ".settings")
                {
                    #region Load preprocessing options

                    Options Options = new Options();
                    try
                    {
                        Options.Load(OpenDialog.FileName);
                    }
                    catch
                    {
                        await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                            $"An error was encountered when reading {Info.Name}.",
                                                                                            MessageDialogStyle.Affirmative);
                        return;
                    }

                    #endregion

                    #region Load items with metadata

                    List<Movie> Items = new List<Movie>();

                    string FileExtension = Options.Import.Extension;
                    var AvailableFiles = Directory.EnumerateFiles(Info.DirectoryName, FileExtension);

                    {
                        var ProgressDialog = await ((MainWindow)Application.Current.MainWindow).ShowProgressAsync("Loading metadata...", "");
                        ProgressDialog.Maximum = AvailableFiles.Count();

                        await Task.Run(() =>
                        {
                            int Done = 0;
                            foreach (var file in AvailableFiles)
                            {
                                string XmlPath = file.Substring(0, file.LastIndexOf(".")) + ".xml";
                                if (File.Exists(XmlPath))
                                    Items.Add(FileExtension == "*.tomostar" ? new TiltSeries(file) : new Movie(file));

                                ProgressDialog.SetProgress(++Done);
                            }
                        });
                        await ProgressDialog.CloseAsync();
                    }

                    if (Items.Count == 0)
                    {
                        await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                            $"No micrographs or tilt series found to match these settings.",
                                                                                            MessageDialogStyle.Affirmative);
                        return;
                    }

                    #endregion

                    #region Figure out filtering status

                    #region Astigmatism statistics

                    Movie[] ItemsWithCTF = Items.Where(v => v.OptionsCTF != null && v.CTF != null).ToArray();
                    List<float2> AstigmatismPoints = new List<float2>(ItemsWithCTF.Length);
                    foreach (var item in ItemsWithCTF)
                        AstigmatismPoints.Add(new float2((float)Math.Cos((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta,
                                                         (float)Math.Sin((float)item.CTF.DefocusAngle * 2 * Helper.ToRad) * (float)item.CTF.DefocusDelta));
                    
                    float2 AstigmatismMean = new float2();
                    float AstigmatismStd = 0.1f;

                    if (AstigmatismPoints.Count > 0)
                    {
                        AstigmatismMean = new float2();
                        foreach (var point in AstigmatismPoints)
                            AstigmatismMean += point;
                        AstigmatismMean /= AstigmatismPoints.Count;

                        AstigmatismStd = 0;
                        foreach (var point in AstigmatismPoints)
                            AstigmatismStd += (point - AstigmatismMean).LengthSq();
                        AstigmatismStd = (float)Math.Max(1e-4, Math.Sqrt(AstigmatismStd / AstigmatismPoints.Count));
                    }

                    #endregion

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

                            FilterStatus &= item.CTF.PhaseShift >= Options.Filter.PhaseMin && item.CTF.PhaseShift <= Options.Filter.PhaseMax;
                        }

                        if (item.OptionsMovement != null)
                        {
                            FilterStatus &= item.MeanFrameMovement <= Options.Filter.MotionMax;
                        }

                        item.UnselectFilter = !FilterStatus;
                    }

                    ProcessingOptionsMovieCTF OptionsCTF = Options.GetProcessingMovieCTF();
                    ProcessingOptionsMovieMovement OptionsMovement = Options.GetProcessingMovieMovement();
                    ProcessingOptionsBoxNet OptionsBoxNet = Options.GetProcessingBoxNet();
                    ProcessingOptionsMovieExport OptionsExport = Options.GetProcessingMovieExport();
                    
                    List<Movie> ItemsProcessed = new List<Movie>();
                    List<Movie> ItemsFilteredOut = new List<Movie>();
                    List<Movie> ItemsUnselected = new List<Movie>();

                    foreach (Movie item in Items)
                    {
                        ProcessingStatus Status = StatusBar.GetMovieProcessingStatus(item, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, Options);

                        if (Status == ProcessingStatus.Processed || (Status == ProcessingStatus.Outdated && !item.UnselectFilter))
                            ItemsProcessed.Add(item);
                        else if (Status == ProcessingStatus.FilteredOut || (Status == ProcessingStatus.Outdated && item.UnselectFilter))
                            ItemsFilteredOut.Add(item);
                        else if (Status == ProcessingStatus.LeaveOut)
                            ItemsUnselected.Add(item);
                    }

                    #endregion

                    #region Show dialog

                    CustomDialog Dialog = new CustomDialog();
                    Dialog.HorizontalContentAlignment = HorizontalAlignment.Center;

                    DialogCreateSourceFromSettings DialogContent = new DialogCreateSourceFromSettings();
                    DialogContent.TextTitle.Text = $"Create data source from\n{Info.Name}";
                    DialogContent.StatsSeriesStatusProcessed.Values = new ChartValues<ObservableValue> { new ObservableValue(ItemsProcessed.Count) };
                    DialogContent.StatsSeriesStatusUnfiltered.Values = new ChartValues<ObservableValue> { new ObservableValue(ItemsFilteredOut.Count) };
                    DialogContent.StatsSeriesStatusUnselected.Values = new ChartValues<ObservableValue> { new ObservableValue(ItemsUnselected.Count) };

                    DialogContent.Close += () =>
                    {
                        ((MainWindow)Application.Current.MainWindow).HideMetroDialogAsync(Dialog);
                    };

                    DialogContent.Create += async () =>
                    {
                        #region Create source metadata and check if one with the same path already exists

                        DataSource NewSource = new DataSource
                        {
                            PixelSizeX = Options.PixelSizeX,
                            PixelSizeY = Options.PixelSizeY,
                            PixelSizeAngle = Options.PixelSizeAngle,

                            DimensionsX = Options.Tomo.DimensionsX,
                            DimensionsY = Options.Tomo.DimensionsY,
                            DimensionsZ = Options.Tomo.DimensionsZ,

                            Name = DialogContent.TextSourceName.Text,
                            Path = Info.DirectoryName + "\\" + Helper.RemoveInvalidChars(DialogContent.TextSourceName.Text) + ".source"
                        };

                        if (Population.Sources.Any(s => s.Path == NewSource.Path))
                        {
                            await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                $"{Helper.PathToNameWithExtension(NewSource.Path)} already exists in this population. Please choose a different name.",
                                                                                                MessageDialogStyle.Affirmative);
                            return;
                        }

                        await ((MainWindow)Application.Current.MainWindow).HideMetroDialogAsync(Dialog);

                        #endregion

                        #region Add all items and their data hashes

                        List<Movie> AllItems = new List<Movie>(ItemsProcessed);
                        if ((bool)DialogContent.CheckFilter.IsChecked)
                            AllItems.AddRange(ItemsFilteredOut);
                        if ((bool)DialogContent.CheckManual.IsChecked)
                            AllItems.AddRange(ItemsUnselected);

                        if (AllItems.Count == 0)
                        {
                            await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                $"No micrographs or tilt series found to match these settings.",
                                                                                                MessageDialogStyle.Affirmative);
                            return;
                        }

                        {
                            var ProgressDialog = await ((MainWindow)Application.Current.MainWindow).ShowProgressAsync("Calculating data hashes...", "");
                            ProgressDialog.Maximum = AllItems.Count;

                            await Task.Run(() =>
                            {
                                int Done = 0;
                                foreach (var item in AllItems)
                                {
                                    NewSource.Files.Add(item.GetDataHash(), item.Name);

                                    ProgressDialog.SetProgress(++Done);
                                }
                            });
                            await ProgressDialog.CloseAsync();
                        }

                        #endregion

                        #region Check for overlapping hashes

                        string[] Overlapping = Helper.Combine(Population.Sources.Select(s => s.Files.Where(f => NewSource.Files.ContainsKey(f.Key)).Select(f => f.Value).ToArray()));
                        if (Overlapping.Length > 0)
                        {
                            string Offenders = "";
                            for (int o = 0; o < Math.Min(5, Overlapping.Length); o++)
                                Offenders += "\n" + Overlapping[o];
                            if (Overlapping.Length > 5)
                                Offenders += $"\n... and {Overlapping.Length - 5} more.";

                            await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                "The new source contains files that are already used in this population:" + Offenders,
                                                                                                MessageDialogStyle.Affirmative);
                            return;
                        }

                        #endregion

                        {
                            var ProgressDialog = await ((MainWindow)Application.Current.MainWindow).ShowProgressAsync("Committing initial version...", "");
                            ProgressDialog.Maximum = AllItems.Count;

                            await Task.Run(() =>
                            {
                                NewSource.Commit();
                            });
                            await ProgressDialog.CloseAsync();
                        }

                        Population.Sources.Add(NewSource);

                        UpdateGridItems();
                    };

                    Dialog.Content = DialogContent;
                    await ((MainWindow)Application.Current.MainWindow).ShowMetroDialogAsync(Dialog, new MetroDialogSettings() { });

                    #endregion
                }
                else
                {
                    try
                    {
                        if (Population.Sources.Any(s => s.Path == OpenDialog.FileName))
                        {
                            await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                                "This data source is already part of this population.",
                                                                                                MessageDialogStyle.Affirmative);
                            return;
                        }

                        DataSource NewSource = DataSource.FromFile(OpenDialog.FileName);
                        Population.Sources.Add(NewSource);

                        UpdateGridItems();
                    }
                    catch
                    {
                        await ((MainWindow)Application.Current.MainWindow).ShowMessageAsync("Oopsie",
                                                                                            $"An error was encountered when reading {Info.Name}.",
                                                                                            MessageDialogStyle.Affirmative);
                        return;
                    }
                }
            }
        }

        private void ButtonAddRemote_OnClick(object sender, RoutedEventArgs e)
        {
            DataSource NewSource = new DataSource();
            NewSource.Path = "https://www.multiparticle.com/data/test.source";
            NewSource.PixelSizeX = NewSource.PixelSizeY = 1.2543M;

            Population.Sources.Add(NewSource);
            UpdateGridItems();
        }
    }
}
