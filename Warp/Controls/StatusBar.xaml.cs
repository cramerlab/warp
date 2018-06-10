using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Channels;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for StatusBar.xaml
    /// </summary>
    public partial class StatusBar : UserControl
    {
        public ObservableCollection<Movie> Items
        {
            get { return (ObservableCollection<Movie>)GetValue(ItemsProperty); }
            set { SetValue(ItemsProperty, value); }
        }

        public static readonly DependencyProperty ItemsProperty =
            DependencyProperty.Register("Items", typeof (ObservableCollection<Movie>), typeof (StatusBar), 
                new PropertyMetadata(new ObservableCollection<Movie>(), (sender, e) => ((StatusBar)sender).ItemsChanged(sender, e)));

        public Movie ActiveItem
        {
            get { return (Movie)GetValue(ActiveItemProperty); }
            set { SetValue(ActiveItemProperty, value); }
        }
        public static readonly DependencyProperty ActiveItemProperty = DependencyProperty.Register("ActiveItem", typeof(Movie), typeof(StatusBar), 
            new PropertyMetadata(null, (sender, e) => ((StatusBar)sender).ActiveItemChanged(sender, e)));

        private Movie HighlightItem;

        public event Action ActiveItemStatusChanged;

        private static SolidColorBrush BrushProcessed = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#cc77cd77"));
        private static SolidColorBrush BrushOutdated = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ccffc456"));
        private static SolidColorBrush BrushFilteredOut = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#cc7edfff"));
        private static SolidColorBrush BrushUnprocessed = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ccff7d7d"));
        private static SolidColorBrush BrushDeselected = new SolidColorBrush(Colors.LightGray);

        private static SolidColorBrush BrushProcessedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ff75f575"));
        private static SolidColorBrush BrushOutdatedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ffffc456"));
        private static SolidColorBrush BrushFilteredOutOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ff7edfff"));
        private static SolidColorBrush BrushUnprocessedOpaque = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#ffff7d7d"));

        public StatusBar()
        {
            SizeChanged += (sender, e) => UpdateElements();
            IsVisibleChanged += (sender, e) => { if (IsVisible) UpdateElements(); };
            InitializeComponent();
            
            UpdateElements();
        }

        private void ItemsChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ObservableCollection<Movie> OldValue = e.OldValue as ObservableCollection<Movie>;
            if (OldValue != null)
            {
                OldValue.CollectionChanged -= Movies_CollectionChanged;
                foreach (var item in OldValue)
                    item.ProcessingChanged -= Item_ProcessingChanged;
            }
            ObservableCollection<Movie> NewValue = e.NewValue as ObservableCollection<Movie>;
            if (NewValue != null)
            {
                NewValue.CollectionChanged += Movies_CollectionChanged;
                foreach (var item in NewValue)
                    item.ProcessingChanged += Item_ProcessingChanged;

                if (!NewValue.Contains(ActiveItem)) // Switched to a different folder
                    ActiveItem = null;
            }

            UpdateElements();
        }

        private void ActiveItemChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            UpdateTracker();
        }

        private void Movies_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (e.OldItems != null)
                foreach (var item in e.OldItems.Cast<Movie>())
                    item.ProcessingChanged -= Item_ProcessingChanged;

            if (e.NewItems != null)
                foreach (var item in e.NewItems.Cast<Movie>())
                    item.ProcessingChanged += Item_ProcessingChanged;

            UpdateElements();
        }

        private void Item_ProcessingChanged(object sender, EventArgs e)
        {
            DispatchUpdateElements();
        }

        private void DispatchUpdateElements()
        {
            Dispatcher.Invoke(() => UpdateElements());
        }

        public void UpdateElements()
        {
            PanelSegments.Children.Clear();

            ProcessingOptionsMovieCTF OptionsCTF = MainWindow.Options.GetProcessingMovieCTF();
            ProcessingOptionsMovieMovement OptionsMovement = MainWindow.Options.GetProcessingMovieMovement();
            ProcessingOptionsBoxNet OptionsBoxNet = MainWindow.Options.GetProcessingBoxNet();
            ProcessingOptionsMovieExport OptionsExport = MainWindow.Options.GetProcessingMovieExport();

            List<Tuple<Movie, ProcessingStatus>> WithStatus = new List<Tuple<Movie, ProcessingStatus>>(Items.Count);
            foreach (var item in Items)
                WithStatus.Add(new Tuple<Movie, ProcessingStatus>(item, GetMovieProcessingStatus(item, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, MainWindow.Options)));

            // Update number of processed movies.
            int NProcessed = WithStatus.Sum(m => m.Item2 == ProcessingStatus.Processed || m.Item2 == ProcessingStatus.FilteredOut ? 1 : 0);
            int NProcessable = WithStatus.Sum(m => m.Item2 != ProcessingStatus.LeaveOut ? 1 : 0);
            TextNumberProcessed.Text = $"Processed {NProcessed}/{NProcessable}.";
            
            if (Items.Count == 0) // Hide unnecessary elements.
            {
                HideElements();
            }
            else
            {
                ShowElements();

                // Create colored navigation bar.
                double StepLength = ActualWidth / Items.Count;
                ProcessingStatus CurrentStatus = WithStatus[0].Item2;
                int CurrentSteps = 0;
                double OverallOffset = 0;
                foreach (var movie in WithStatus)
                {
                    if (movie.Item2 != CurrentStatus)
                    {
                        Rectangle Segment = new Rectangle
                        {
                            Width = CurrentSteps * StepLength + 0.3,
                            Height = 12,
                            Fill = StatusToBrush(CurrentStatus),
                            SnapsToDevicePixels = false
                        };
                        PanelSegments.Children.Add(Segment);
                        Canvas.SetLeft(Segment, OverallOffset);

                        CurrentStatus = movie.Item2;
                        OverallOffset += CurrentSteps * StepLength;
                        CurrentSteps = 0;
                    }

                    CurrentSteps++;
                }
                if (CurrentSteps > 0)
                {
                    Rectangle Segment = new Rectangle
                    {
                        Width = CurrentSteps * StepLength,
                        Height = 12,
                        Fill = StatusToBrush(CurrentStatus),
                        SnapsToDevicePixels = false
                    };
                    PanelSegments.Children.Add(Segment);
                    Canvas.SetLeft(Segment, OverallOffset);
                }

                UpdateTracker();
            }
        }

        private void UpdateTracker()
        {
            if (ActiveItem != null && Items.Count > 0)
            {
                PathPosition.Visibility = Visibility.Visible;
                CheckCurrentName.Visibility = Visibility.Visible;
                TextCurrentName.Visibility = Visibility.Visible;

                double StepLength = ActualWidth / Items.Count;

                int PositionIndex = Items.IndexOf(ActiveItem);
                if (PositionIndex < 0)  // Can't find it in Items, hide everything
                {
                    PathPosition.Visibility = Visibility.Hidden;
                    CheckCurrentName.Visibility = Visibility.Hidden;
                    TextCurrentName.Visibility = Visibility.Hidden;
                    return;
                }

                double IdealOffset = (PositionIndex + 0.5) * StepLength;
                PathPosition.Margin =
                    new Thickness(Math.Max(0, Math.Min(IdealOffset - PathPosition.ActualWidth / 2, ActualWidth - PathPosition.ActualWidth)),
                                  0, 0, 0);

                string TextString = $"{ActiveItem.RootName} ({PositionIndex + 1}/{Items.Count})";
                Size TextSize = MeasureString(TextString);
                CheckCurrentName.Margin =
                    new Thickness(Math.Max(0, Math.Min(IdealOffset - TextSize.Width / 2, ActualWidth - TextSize.Width) - 24),
                                  0, 0, 0);
                TextCurrentName.Text = TextString;

                ProcessingOptionsMovieCTF OptionsCTF = MainWindow.Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = MainWindow.Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = MainWindow.Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = MainWindow.Options.GetProcessingMovieExport();

                PenCurrentName.Brush = StatusToBrush(GetMovieProcessingStatus(ActiveItem, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, MainWindow.Options));
                //PenCurrentName.Brush.Opacity = 0.3;
            }
            else
            {
                PathPosition.Visibility = Visibility.Hidden;
                CheckCurrentName.Visibility = Visibility.Hidden;
                TextCurrentName.Visibility = Visibility.Hidden;
            }
        }

        private void UpdateHighlightTracker()
        {
            if (HighlightItem != null && Items.Count > 0)
            {
                PathPosition.Opacity = 0.3;
                CheckCurrentName.Opacity = 0.5;
                BlurCurrentName.Radius = 6;

                PathHighlightPosition.Visibility = Visibility.Visible;
                CheckHighlightCurrentName.Visibility = Visibility.Visible;

                double StepLength = ActualWidth / Items.Count;

                int PositionIndex = Items.IndexOf(HighlightItem);
                if (PositionIndex < 0)  // Can't find it in Items, hide everything
                {
                    PathHighlightPosition.Visibility = Visibility.Hidden;
                    CheckHighlightCurrentName.Visibility = Visibility.Hidden;
                    return;
                }

                double IdealOffset = (PositionIndex + 0.5) * StepLength;
                PathHighlightPosition.Margin = new Thickness(Math.Max(0, Math.Min(IdealOffset - PathPosition.ActualWidth / 2, ActualWidth - PathPosition.ActualWidth)),
                                                             0, 0, 0);

                string TextString = $"{HighlightItem.RootName} ({PositionIndex + 1}/{Items.Count})";
                Size TextSize = MeasureString(TextString);
                CheckHighlightCurrentName.Margin = new Thickness(Math.Max(0, Math.Min(IdealOffset - TextSize.Width / 2, ActualWidth - TextSize.Width) - 24),
                                                        0, 0, 0);
                TextHighlightCurrentName.Text = TextString;
                CheckHighlightCurrentName.IsChecked = HighlightItem.UnselectManual != null ? !(bool)HighlightItem.UnselectManual : (bool?)null;

                ProcessingOptionsMovieCTF OptionsCTF = MainWindow.Options.GetProcessingMovieCTF();
                ProcessingOptionsMovieMovement OptionsMovement = MainWindow.Options.GetProcessingMovieMovement();
                ProcessingOptionsBoxNet OptionsBoxNet = MainWindow.Options.GetProcessingBoxNet();
                ProcessingOptionsMovieExport OptionsExport = MainWindow.Options.GetProcessingMovieExport();

                PenHighlightCurrentName.Brush = StatusToBrush(GetMovieProcessingStatus(HighlightItem, OptionsCTF, OptionsMovement, OptionsBoxNet, OptionsExport, MainWindow.Options));
                
                PanelSegmentHighlight.Children.Clear();
                Rectangle Segment = new Rectangle
                {
                    Width = StepLength,
                    Height = 12,
                    Fill = new SolidColorBrush(Colors.Transparent),
                    Stroke = PathHighlightPosition.Fill,
                    StrokeThickness = 1,
                    SnapsToDevicePixels = false
                };
                PanelSegmentHighlight.Children.Add(Segment);
                Canvas.SetLeft(Segment, PositionIndex * StepLength);

                if (File.Exists(HighlightItem.ThumbnailsPath))
                {
                    ImageSource Image = new BitmapImage(new Uri(HighlightItem.ThumbnailsPath));
                    ImageThumbnail.Source = Image;

                    PopupThumbnail.HorizontalOffset = Mouse.GetPosition(PanelSegments).X - Image.Width / 2;
                    ImageThumbnail.Visibility = Visibility.Visible;
                }
                else
                {
                    PopupThumbnail.HorizontalOffset = Mouse.GetPosition(PanelSegments).X;
                    ImageThumbnail.Visibility = Visibility.Collapsed;
                }

                #region Figure out processing status

                bool DoCTF = MainWindow.Options.ProcessCTF;
                bool DoMovement = MainWindow.Options.ProcessMovement;
                bool DoPicking = MainWindow.Options.ProcessPicking;
                bool DoExport = OptionsExport.DoAverage || OptionsExport.DoStack || OptionsExport.DoDeconv || DoPicking;

                bool NeedsNewCTF = OptionsCTF != HighlightItem.OptionsCTF;
                bool NeedsNewMotion = OptionsMovement != HighlightItem.OptionsMovement;
                bool NeedsNewPicking = (OptionsBoxNet != HighlightItem.OptionsBoxNet ||
                                        NeedsNewMotion);
                bool NeedsNewExport = (DoMovement && NeedsNewMotion) ||
                                       OptionsExport != HighlightItem.OptionsMovieExport ||
                                       (OptionsExport.DoDeconv && NeedsNewCTF && DoCTF);

                if (HighlightItem.OptionsCTF != null)
                    IndicatorCTF.Foreground = NeedsNewCTF ? (DoCTF ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoCTF)
                    IndicatorCTF.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorCTF.Foreground = BrushDeselected;

                if (HighlightItem.OptionsMovement != null)
                    IndicatorMotion.Foreground = NeedsNewMotion ? (DoMovement ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoMovement)
                    IndicatorMotion.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorMotion.Foreground = BrushDeselected;

                if (HighlightItem.OptionsBoxNet != null)
                    IndicatorPicking.Foreground = NeedsNewPicking ? (DoPicking ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoPicking)
                    IndicatorPicking.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorPicking.Foreground = BrushDeselected;

                if (HighlightItem.OptionsMovieExport != null)
                    IndicatorExport.Foreground = NeedsNewExport ? (DoExport ? BrushOutdatedOpaque : BrushDeselected) : BrushProcessedOpaque;
                else if (DoExport)
                    IndicatorExport.Foreground = BrushUnprocessedOpaque;
                else
                    IndicatorExport.Foreground = BrushDeselected;

                #endregion

                PopupThumbnail.PlacementTarget = PanelSegments;
                PopupThumbnail.VerticalOffset = -12;
                PopupThumbnail.IsOpen = true;
            }
            else
            {
                PathPosition.Opacity = 1;
                CheckCurrentName.Opacity = 1;
                BlurCurrentName.Radius = 0;

                PathHighlightPosition.Visibility = Visibility.Hidden;
                CheckHighlightCurrentName.Visibility = Visibility.Hidden;

                PanelSegmentHighlight.Children.Clear();

                PopupThumbnail.IsOpen = false;
            }
        }

        private void ShowElements()
        {
            PanelSegments.Visibility = Visibility.Visible;
            PathPosition.Visibility = Visibility.Visible;
            TextCurrentName.Visibility = Visibility.Visible;
            CheckCurrentName.Visibility = Visibility.Visible;
        }

        private void HideElements()
        {
            PanelSegments.Visibility = Visibility.Hidden;
            PathPosition.Visibility = Visibility.Hidden;
            TextCurrentName.Visibility = Visibility.Hidden;
            CheckCurrentName.Visibility = Visibility.Hidden;
        }

        private static Brush StatusToBrush(ProcessingStatus status)
        {
            if (status == ProcessingStatus.Processed)
                return BrushProcessed.CloneCurrentValue();
            else if (status == ProcessingStatus.Outdated)
                return BrushOutdated.CloneCurrentValue();
            else if (status == ProcessingStatus.Unprocessed)
                return BrushUnprocessed.CloneCurrentValue();
            else if (status == ProcessingStatus.FilteredOut)
                return BrushFilteredOut.CloneCurrentValue();
            else
                return BrushDeselected.CloneCurrentValue();
        }

        private void MainGrid_OnPreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            MoveToOtherItem(-Math.Sign(e.Delta));
        }

        public void MoveToOtherItem(int delta)
        {
            if (Items.Count == 0)
                return;

            if (ActiveItem == null || !Items.Contains(ActiveItem))
            {
                ActiveItem = Items[0];
                return;
            }
            else
            {
                int NewIndex = Items.IndexOf(ActiveItem) + delta;
                ActiveItem = Items[Math.Max(0, Math.Min(NewIndex, Items.Count - 1))];
            }
        }

        private void MainGrid_OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (Items.Count == 0)
                return;

            float StepLength = (float)ActualWidth / Items.Count;
            int NewIndex = (int) ((float) e.GetPosition(this).X / StepLength);
            NewIndex = Math.Max(0, Math.Min(NewIndex, Items.Count - 1));

            ActiveItem = Items[NewIndex];
        }

        private void PanelSegments_OnMouseMove(object sender, MouseEventArgs e)
        {
            //if (PanelSegments.ToolTip == null || PanelSegments.ToolTip.GetType() != typeof (ToolTip))
            //    PanelSegments.ToolTip = new ToolTip();

            if (Items == null || Items.Count == 0)
            {
                //((ToolTip)PanelSegments.ToolTip).Content = "";
                //((ToolTip)PanelSegments.ToolTip).IsOpen = false;
                HighlightItem = null;
                UpdateHighlightTracker();
            }
            else
            {
                //ProcessingOptionsMovieCTF OptionsCTF = MainWindow.Options.GetProcessingMovieCTF();
                //ProcessingOptionsMovieMovement OptionsMovement = MainWindow.Options.GetProcessingMovieMovement();
                //ProcessingOptionsBoxNet OptionsBoxNet = MainWindow.Options.GetProcessingBoxNet();
                //ProcessingOptionsMovieExport OptionsExport = MainWindow.Options.GetProcessingMovieExport();
                
                double StepLength = PanelSegments.ActualWidth / Items.Count;
                int ID = (int)(e.GetPosition(PanelSegments).X / StepLength);
                HighlightItem = Items[Math.Max(0, Math.Min(Items.Count - 1, ID))];
                UpdateHighlightTracker();

                //((ToolTip)PanelSegments.ToolTip).Content = Items[Math.Max(0, Math.Min(Items.Count - 1, ID))].RootName;
                //((ToolTip)PanelSegments.ToolTip).Foreground = StatusToBrush(GetMovieProcessingStatus(Items[Math.Max(0, Math.Min(Items.Count - 1, ID))],
                //                                                                                     OptionsCTF,
                //                                                                                     OptionsMovement,
                //                                                                                     OptionsBoxNet,
                //                                                                                     OptionsExport,
                //                                                                                     MainWindow.Options));
                //((ToolTip)PanelSegments.ToolTip).FontWeight = FontWeights.Bold;
                //((ToolTip)PanelSegments.ToolTip).IsOpen = false;
                //((ToolTip)PanelSegments.ToolTip).IsOpen = true;
            }
        }

        private void PanelSegments_OnMouseLeave(object sender, MouseEventArgs e)
        {
            //ToolTip Tip = PanelSegments.ToolTip as ToolTip;
            //if (Tip != null)
            //    Tip.IsOpen = false;
            HighlightItem = null;
            UpdateHighlightTracker();
        }

        private Size MeasureString(string candidate)
        {
            FormattedText Test = new FormattedText(candidate,
                                                   CultureInfo.CurrentUICulture,
                                                   FlowDirection.LeftToRight,
                                                   new Typeface(TextCurrentName.FontFamily, TextCurrentName.FontStyle, TextCurrentName.FontWeight, TextCurrentName.FontStretch),
                                                   TextCurrentName.FontSize,
                                                   Brushes.Black);

            return new Size(Test.Width, Test.Height);
        }

        public void ShowProgressBar()
        {
            ProgressDiscovery.Visibility = Visibility.Visible;
        }

        public void HideProgressBar()
        {
            ProgressDiscovery.Visibility = Visibility.Hidden;
        }

        public static ProcessingStatus GetMovieProcessingStatus(Movie movie, ProcessingOptionsMovieCTF optionsCTF, ProcessingOptionsMovieMovement optionsMovement, ProcessingOptionsBoxNet optionsBoxNet, ProcessingOptionsMovieExport optionsExport, Options options, bool considerFilter = true)
        {
            bool DoCTF = options.ProcessCTF;
            bool DoMovement = options.ProcessMovement;
            bool DoBoxNet = options.ProcessPicking;
            bool DoExport = optionsExport.DoAverage || optionsExport.DoStack || optionsExport.DoDeconv;
            
            ProcessingStatus Status = ProcessingStatus.Processed;

            if (movie.UnselectManual != null && (bool)movie.UnselectManual)
            {
                Status = ProcessingStatus.LeaveOut;
            }
            else if (movie.OptionsCTF == null && movie.OptionsMovement == null && movie.OptionsMovieExport == null)
            {
                Status = ProcessingStatus.Unprocessed;
            }
            else
            {
                if (DoCTF && (movie.OptionsCTF == null || movie.OptionsCTF != optionsCTF))
                    Status = ProcessingStatus.Outdated;
                else if (DoMovement && (movie.OptionsMovement == null || movie.OptionsMovement != optionsMovement))
                    Status = ProcessingStatus.Outdated;
                else if (DoBoxNet && (movie.OptionsBoxNet == null || movie.OptionsBoxNet != optionsBoxNet))
                    Status = ProcessingStatus.Outdated;
                else if (DoExport && (movie.OptionsMovieExport == null || movie.OptionsMovieExport != optionsExport))
                    Status = ProcessingStatus.Outdated;
            }

            if (Status == ProcessingStatus.Processed && movie.UnselectFilter && movie.UnselectManual == null && considerFilter)
                Status = ProcessingStatus.FilteredOut;

            return Status;
        }

        private void CheckCurrentName_OnClick(object sender, RoutedEventArgs e)
        {
            ActiveItemStatusChanged?.Invoke();
        }
    }

    public enum ProcessingStatus
    {
        Processed = 1,
        Outdated = 2,
        Unprocessed = 3,
        FilteredOut = 4,
        LeaveOut = 5
    }

    [ValueConversion(typeof(bool), typeof(bool))]
    public class InverseBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            //if (targetType != typeof(bool))
            //    throw new InvalidOperationException("The target must be a boolean");

            return !(bool?)value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return !(bool?)value;
        }
    }
}
