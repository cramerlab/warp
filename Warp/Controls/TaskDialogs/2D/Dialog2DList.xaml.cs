using System;
using System.Collections.Generic;
using System.Globalization;
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
using Warp.Headers;
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

                string PathPrefix = "", PathPrefixAverage = "", GainPath = "";
                if (ValidMovies.Count > 0 && Relative)
                {
                    Uri UriStar = new Uri(ExportPath);

                    PathPrefix = Helper.PathToFolder(UriStar.MakeRelativeUri(new Uri(ValidMovies[0].Path)).ToString());
                    if (PathPrefix == ValidMovies[0].Name)
                        PathPrefix = "";
                    PathPrefixAverage = Helper.PathToFolder(UriStar.MakeRelativeUri(new Uri(ValidMovies[0].AveragePath)).ToString());

                    if (!string.IsNullOrEmpty(Options.Import.GainPath) && Options.Import.CorrectGain)
                    {
                        GainPath = UriStar.MakeRelativeUri(new Uri(Options.Import.GainPath)).ToString();
                    }
                }

                Dispatcher.Invoke(() => ProgressWrite.Maximum = ValidMovies.Count);

                bool IncludeCTF = ValidMovies.Any(v => v.OptionsCTF != null);
                bool IncludePolishing = ValidMovies.Any(v => v.OptionsMovement != null) && Options.Tasks.MicListMakePolishing;

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
                if (IncludePolishing)
                    TableOut.AddColumn("rlnMicrographMetadata");

                int r = 0;
                foreach (var movie in ValidMovies)
                {
                    List<string> Row = new List<string>() { PathPrefixAverage + movie.RootName + ".mrc" };

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

                    if (IncludePolishing)
                        Row.Add(PathPrefix + "motion/" + movie.RootName + ".star");

                    TableOut.AddRow(Row);

                    if (IncludePolishing)
                    {
                        MapHeader Header = MapHeader.ReadFromFile(movie.Path);

                        List<string> HeaderNames = new List<string>()
                        {
                            "rlnImageSizeX",
                            "rlnImageSizeY",
                            "rlnImageSizeZ",
                            "rlnMicrographMovieName",
                            "rlnMicrographBinning",
                            "rlnMicrographOriginalPixelSize",
                            "rlnMicrographDoseRate",
                            "rlnMicrographPreExposure",
                            "rlnVoltage",
                            "rlnMicrographStartFrame",
                            "rlnMotionModelVersion"
                        };

                        List<string> HeaderValues = new List<string>()
                        {
                            Header.Dimensions.X.ToString(),
                            Header.Dimensions.Y.ToString(),
                            Header.Dimensions.Z.ToString(),
                            PathPrefix + movie.Name,
                            Math.Pow(2.0, (double)Options.Import.BinTimes).ToString("F5"),
                            Options.PixelSizeMean.ToString("F5"),
                            Options.Import.DosePerAngstromFrame.ToString("F5"),
                            "0",
                            Options.CTF.Voltage.ToString(),
                            "1",
                            "0"
                        };

                        if (!string.IsNullOrEmpty(GainPath))
                        {
                            HeaderNames.Add("rlnMicrographGainName");
                            HeaderValues.Add(GainPath);
                        }

                        StarParameters ParamsTable = new StarParameters(HeaderNames.ToArray(), HeaderValues.ToArray());

                        float2[] MotionTrack = movie.GetMotionTrack(new float2(0.5f, 0.5f), 1).Select(v => v / (float)Options.PixelSizeMean).ToArray();
                        Star TrackTable = new Star(new[]
                        {
                            Helper.ArrayOfSequence(1, MotionTrack.Length + 1, 1).Select(v => v.ToString()).ToArray(),
                            MotionTrack.Select(v => (-v.X).ToString("F5")).ToArray(),
                            MotionTrack.Select(v => (-v.Y).ToString("F5")).ToArray()
                        },
                        new[]
                        {
                            "rlnMicrographFrameNumber",
                            "rlnMicrographShiftX",
                            "rlnMicrographShiftY"
                        });

                        Directory.CreateDirectory(movie.DirectoryName + "motion");
                        Star.SaveMultitable(movie.DirectoryName + "motion/" + movie.RootName + ".star", new Dictionary<string, Star>()
                        {
                            { "general", ParamsTable },
                            { "global_shift", TrackTable },
                        });
                    }

                    Dispatcher.Invoke(() => ProgressWrite.Value = ++r);
                }

                TableOut.Save(ExportPath);
            });

            Close?.Invoke();
        }
    }
}
