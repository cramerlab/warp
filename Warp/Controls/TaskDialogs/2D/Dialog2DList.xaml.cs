using System;
using System.Collections.Generic;
using System.Globalization;
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
using Warp.Tools;

namespace Warp.Controls
{
    public partial class Dialog2DList : UserControl
    {
        public Movie[] Movies;
        public string ExportPath;
        public Options Options;
        public event Action Close;

        public Dialog2DList(Movie[] movies, string exportPath, Options options)
        {
            InitializeComponent();

            Movies = movies;
            Options = options;
            ExportPath = exportPath;

            DataContext = Options;
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonWrite_OnClick(object sender, RoutedEventArgs e)
        {
            bool Relative = (bool)CheckRelative.IsChecked;
            bool Filter = (bool)CheckFilter.IsChecked;
            bool Manual = (bool)CheckManual.IsChecked;

            ProgressWrite.Visibility = Visibility.Visible;
            PanelButtons.Visibility = Visibility.Hidden;

            await Task.Run(() =>
            {
                List<Movie> ValidMovies = Movies.Where(v =>
                {
                    if (!Filter && v.UnselectFilter && v.UnselectManual == null)
                        return false;
                    if (!Manual && v.UnselectManual != null && (bool)v.UnselectManual)
                        return false;
                    return true;
                }).ToList();

                string PathPrefix = "";
                if (ValidMovies.Count > 0 && Relative)
                {
                    Uri UriStar = new Uri(ExportPath);
                    PathPrefix = UriStar.MakeRelativeUri(new Uri(ValidMovies[0].AveragePath)).ToString();
                    
                    PathPrefix = PathPrefix.Substring(0, PathPrefix.IndexOf(Helper.PathToNameWithExtension(PathPrefix)));
                }

                Dispatcher.Invoke(() => ProgressWrite.Maximum = ValidMovies.Count);

                bool IncludeCTF = ValidMovies.Any(v => v.OptionsCTF != null);

                Star TableOut = new Star(new[] { "rlnMicrographName" });
                if (IncludeCTF)
                {
                    TableOut.AddColumn("rlnMagnification");
                    TableOut.AddColumn("rlnDetectorPixelSize");
                    TableOut.AddColumn("rlnVoltage");
                    TableOut.AddColumn("rlnSphericalAberration");
                    TableOut.AddColumn("rlnAmplitudeContrast");
                    TableOut.AddColumn("rlnPhaseShift");
                    TableOut.AddColumn("rlnDefocusU");
                    TableOut.AddColumn("rlnDefocusV");
                    TableOut.AddColumn("rlnDefocusAngle");
                    TableOut.AddColumn("rlnCtfMaxResolution");
                }

                int r = 0;
                foreach (var movie in ValidMovies)
                {
                    List<string> Row = new List<string>() { PathPrefix + movie.RootName + ".mrc" };

                    if (movie.OptionsCTF != null)
                        Row.AddRange(new[]
                        {
                            (1e4M / movie.OptionsCTF.BinnedPixelSizeMean).ToString("F1", CultureInfo.InvariantCulture),
                            "1.0",
                            movie.CTF.Voltage.ToString("F1", CultureInfo.InvariantCulture),
                            movie.CTF.Cs.ToString("F4", CultureInfo.InvariantCulture),
                            movie.CTF.Amplitude.ToString("F3", CultureInfo.InvariantCulture),
                            (movie.CTF.PhaseShift * 180M).ToString("F1", CultureInfo.InvariantCulture),
                            ((movie.CTF.Defocus + movie.CTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                            ((movie.CTF.Defocus - movie.CTF.DefocusDelta / 2) * 1e4M).ToString("F1", CultureInfo.InvariantCulture),
                            movie.CTF.DefocusAngle.ToString("F1", CultureInfo.InvariantCulture),
                            movie.CTFResolutionEstimate.ToString("F1", CultureInfo.InvariantCulture)
                        });
                    else
                        Row.AddRange(Helper.ArrayOfFunction(i => "0.0", 10));

                    TableOut.AddRow(Row);

                    Dispatcher.Invoke(() => ProgressWrite.Value = ++r);
                }

                TableOut.Save(ExportPath);
            });

            Close?.Invoke();
        }
    }
}
