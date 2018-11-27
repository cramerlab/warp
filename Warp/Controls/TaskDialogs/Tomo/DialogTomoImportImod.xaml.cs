using MahApps.Metro.Controls.Dialogs;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Warp.Headers;
using Warp.Tools;

namespace Warp.Controls.TaskDialogs.Tomo
{
    /// <summary>
    /// Interaction logic for DialogTomoImportImod.xaml
    /// </summary>
    public partial class DialogTomoImportImod : UserControl
    {
        public string PathMdoc
        {
            get { return (string)GetValue(PathMdocProperty); }
            set { SetValue(PathMdocProperty, value); }
        }
        public static readonly DependencyProperty PathMdocProperty = DependencyProperty.Register("PathMdoc", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public string PathImod
        {
            get { return (string)GetValue(PathImodProperty); }
            set { SetValue(PathImodProperty, value); }
        }
        public static readonly DependencyProperty PathImodProperty = DependencyProperty.Register("PathImod", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public string PathMovie
        {
            get { return (string)GetValue(PathMovieProperty); }
            set { SetValue(PathMovieProperty, value); }
        }
        public static readonly DependencyProperty PathMovieProperty = DependencyProperty.Register("PathMovie", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));
        
        public string Suffixes
        {
            get { return (string)GetValue(SuffixesProperty); }
            set { SetValue(SuffixesProperty, value); }
        }
        public static readonly DependencyProperty SuffixesProperty = DependencyProperty.Register("Suffixes", typeof(string), typeof(DialogTomoImportImod), new PropertyMetadata("", (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));
               
        public bool InvertTilts
        {
            get { return (bool)GetValue(InvertTiltsProperty); }
            set { SetValue(InvertTiltsProperty, value); }
        }
        public static readonly DependencyProperty InvertTiltsProperty = DependencyProperty.Register("InvertTilts", typeof(bool), typeof(DialogTomoImportImod), new PropertyMetadata(false, (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public decimal PixelSize
        {
            get { return (decimal)GetValue(PixelSizeProperty); }
            set { SetValue(PixelSizeProperty, value); }
        }
        public static readonly DependencyProperty PixelSizeProperty = DependencyProperty.Register("PixelSize", typeof(decimal), typeof(DialogTomoImportImod), new PropertyMetadata(1M, (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public decimal Dose
        {
            get { return (decimal)GetValue(DoseProperty); }
            set { SetValue(DoseProperty, value); }
        }
        public static readonly DependencyProperty DoseProperty = DependencyProperty.Register("Dose", typeof(decimal), typeof(DialogTomoImportImod), new PropertyMetadata(0M, (sender, args) => ((DialogTomoImportImod)sender).Reevaluate()));

        public ObservableCollection<ParsedEntry> ParsedEntries
        {
            get { return (ObservableCollection<ParsedEntry>)GetValue(ParsedEntriesProperty); }
            set { SetValue(ParsedEntriesProperty, value); }
        }
        public static readonly DependencyProperty ParsedEntriesProperty = DependencyProperty.Register("ParsedEntries", typeof(ObservableCollection<ParsedEntry>), typeof(DialogTomoImportImod), new PropertyMetadata(null));
        

        public Options Options;
        public event Action Close;

        public DialogTomoImportImod(Options options)
        {
            InitializeComponent();

            Options = options;

            ParsedEntries = new ObservableCollection<ParsedEntry>();

            DataContext = this;
        }

        private void Reevaluate()
        {
            ObservableCollection<ParsedEntry> OldEntries = ParsedEntries;
            ParsedEntries = new ObservableCollection<ParsedEntry>();
            
            try
            {
                List<string> SuffixList = Suffixes.Replace(" ", "").Split(new[] { ',', ';', '|' }, StringSplitOptions.RemoveEmptyEntries).ToList();
                SuffixList.Add("");
                SuffixList = SuffixList.Select(s => s + ".mdoc").ToList();

                if (string.IsNullOrEmpty(PathMdoc))
                    throw new Exception("No folder specified for .mdoc files!");

                Dictionary<string, List<string>> MdocTuples = new Dictionary<string, List<string>>();
                foreach (var filepath in Directory.EnumerateFiles(PathMdoc, "*.mdoc"))
                {
                    FileInfo Info = new FileInfo(filepath);
                    string FileName = Info.Name;

                    string Root = FileName;
                    foreach (var suffix in SuffixList)
                        if (FileName.Contains(suffix))
                        {
                            Root = FileName.Substring(0, FileName.IndexOf(suffix));
                            break;
                        }

                    //Root = Helper.PathToName(Root);

                    if (!MdocTuples.ContainsKey(Root))
                        MdocTuples.Add(Root, new List<string>());

                    MdocTuples[Root].Add(FileName);
                }

                if (MdocTuples.Count == 0)
                    throw new Exception($"No .mdoc files found in {PathMdoc}.");

                foreach (var mdocNames in MdocTuples)
                {
                    float AxisAngle = 0;
                    List<MdocEntry> Entries = new List<MdocEntry>();
                    bool FoundTime = false;

                    foreach (var mdocName in mdocNames.Value)
                    {
                        using (TextReader Reader = new StreamReader(File.OpenRead(Path.Combine(PathMdoc, mdocName))))
                        {
                            string Line;
                            while ((Line = Reader.ReadLine()) != null)
                            {
                                if (Line.Contains("Tilt axis angle = "))
                                {
                                    string Suffix = Line.Substring(Line.IndexOf("Tilt axis angle = ") + "Tilt axis angle = ".Length);
                                    Suffix = Suffix.Substring(0, Suffix.IndexOf(","));

                                    AxisAngle = float.Parse(Suffix, CultureInfo.InvariantCulture);
                                    continue;
                                }

                                if (Line.Length < 7 || Line.Substring(0, 7) != "[ZValue")
                                    continue;

                                MdocEntry NewEntry = new MdocEntry();

                                {
                                    string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                                    if (Parts[0] == "[ZValue")
                                        NewEntry.ZValue = int.Parse(Parts[1].Replace("]", ""));
                                }

                                while ((Line = Reader.ReadLine()) != null)
                                {
                                    string[] Parts = Line.Split(new[] { " = " }, StringSplitOptions.RemoveEmptyEntries);
                                    if (Parts.Length < 2)
                                        break;

                                    if (Parts[0] == "TiltAngle")
                                        NewEntry.TiltAngle = (float)Math.Round(float.Parse(Parts[1], CultureInfo.InvariantCulture), 1);
                                    else if (Parts[0] == "ExposureDose")
                                        NewEntry.Dose = float.Parse(Parts[1], CultureInfo.InvariantCulture);
                                    else if (Parts[0] == "SubFramePath")
                                        NewEntry.Name = Parts[1].Substring(Parts[1].LastIndexOf("\\") + 1);
                                    else if (Parts[0] == "DateTime")
                                    {
                                        NewEntry.Time = DateTime.ParseExact(Parts[1], "dd-MMM-yy  HH:mm:ss", CultureInfo.InvariantCulture);
                                        FoundTime = true;
                                    }
                                }

                                if (mdocNames.Value.Count == 1)
                                    Entries.RemoveAll(v => v.ZValue == NewEntry.ZValue);

                                Entries.Add(NewEntry);
                            }
                        }
                    }

                    //if (Dose > 0) // Try to get time stamps from file names
                    //    try
                    //    {
                    //        foreach (var entry in Entries)
                    //            entry.Time = DateTime.ParseExact(entry.Name.Substring(0, entry.Name.IndexOf(".mrc")), "MMMdd_HH.mm.ss", CultureInfo.InvariantCulture);
                    //    }
                    //    catch
                    //    {
                    //        throw new Exception("No time stamps found, need them for accumulated dose. Set default dose to 0 to ignore.");
                    //    }

                    List<MdocEntry> SortedTime = new List<MdocEntry>(Entries);
                    SortedTime.Sort((a, b) => a.Time.CompareTo(b.Time));

                    // Do running dose
                    float Accumulated = 0;
                    foreach (var entry in SortedTime)
                    {
                        Accumulated += entry.Dose;
                        entry.Dose = Accumulated;
                    }

                    // In case mdoc doesn't tell anything about the dose, use default value
                    if (Dose > 0)
                        for (int i = 0; i < SortedTime.Count; i++)
                        {
                            SortedTime[i].Dose = (i + 0.5f) * (float)Dose;
                            Accumulated += (float)Dose;
                        }

                    // Sort entires by angle and time (accumulated dose)
                    List<MdocEntry> SortedAngle = new List<MdocEntry>(Entries);
                    SortedAngle.Sort((a, b) => a.TiltAngle.CompareTo(b.TiltAngle));
                    // Sometimes, there will be 2 0-tilts at the beginning of plus and minus series. 
                    // Sort them according to dose, considering in which order plus and minus were acquired
                    float DoseMinus = SortedAngle.Take(SortedAngle.Count / 2).Select(v => v.Dose).Sum();
                    float DosePlus = SortedAngle.Skip(SortedAngle.Count / 2).Take(SortedAngle.Count / 2).Select(v => v.Dose).Sum();
                    int OrderCorrection = DoseMinus < DosePlus ? 1 : -1;
                    SortedAngle.Sort((a, b) => a.TiltAngle.CompareTo(b.TiltAngle) != 0 ? a.TiltAngle.CompareTo(b.TiltAngle) : a.Dose.CompareTo(b.Dose) * OrderCorrection);

                    //foreach (var entry in Entries)
                    //{
                    //    if (File.Exists(TiltMoviesFolder + entry.Name.Substring(0, entry.Name.LastIndexOf(".")) + ".xml"))
                    //    {

                    //    }
                    //}

                    //string PrexfPath = DataRoot + ImodFolder + mdocNames.Key + "\\" + mdocNames.Key + ".prexg";
                    //if (File.Exists(PrexfPath))
                    //{
                    //    using (TextReader Reader = new StreamReader(File.OpenRead(PrexfPath)))
                    //    {
                    //        string Line;
                    //        for (int i = 0; i < SortedAngle.Count; i++)
                    //        {
                    //            Line = Reader.ReadLine();
                    //            string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    //            float2 VecX = new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture),
                    //                                     float.Parse(Parts[2], CultureInfo.InvariantCulture));
                    //            float2 VecY = new float2(float.Parse(Parts[1], CultureInfo.InvariantCulture),
                    //                                     float.Parse(Parts[3], CultureInfo.InvariantCulture));

                    //            SortedAngle[i].Shift += VecX * float.Parse(Parts[4], CultureInfo.InvariantCulture) + VecY * float.Parse(Parts[5], CultureInfo.InvariantCulture);
                    //        }
                    //    }
                    //}

                    if (string.IsNullOrEmpty(PathMovie))
                        throw new Exception("No folder specified for the original movies.");

                    SortedAngle.RemoveAll(v => IsMovieExcluded(Path.Combine(PathMovie, v.Name)));

                    if (SortedAngle.Count == 0)
                        throw new Exception($"No movies found for {mdocNames.Key},\nor none have been processed in Warp yet.");

                    string XfPath = Path.Combine(PathImod, mdocNames.Key, mdocNames.Key + ".xf");
                    bool HasAlignmentData = false;
                    if (File.Exists(XfPath))
                    {
                        HasAlignmentData = true;
                        using (TextReader Reader = new StreamReader(File.OpenRead(XfPath)))
                        {
                            string Line;
                            for (int i = 0; i < SortedAngle.Count; i++)
                            {
                                Line = Reader.ReadLine();
                                string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                                float2 VecX = new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture),
                                                         float.Parse(Parts[2], CultureInfo.InvariantCulture));
                                float2 VecY = new float2(float.Parse(Parts[1], CultureInfo.InvariantCulture),
                                                         float.Parse(Parts[3], CultureInfo.InvariantCulture));

                                Matrix3 Rotation = new Matrix3(VecX.X, VecX.Y, 0, VecY.X, VecY.Y, 0, 0, 0, 1);
                                float3 Euler = Matrix3.EulerFromMatrix(Rotation);

                                SortedAngle[i].AxisAngle = Euler.Z * Helper.ToDeg;

                                //SortedAngle[i].Shift += VecX * float.Parse(Parts[4], CultureInfo.InvariantCulture) + VecY * float.Parse(Parts[5], CultureInfo.InvariantCulture);
                                float3 Shift = new float3(-float.Parse(Parts[4], CultureInfo.InvariantCulture), -float.Parse(Parts[5], CultureInfo.InvariantCulture), 0);
                                Shift = Rotation.Transposed() * Shift;

                                SortedAngle[i].Shift += new float2(Shift);
                            }
                        }
                    }

                    string SolutionPath = Path.Combine(PathImod, mdocNames.Key, "taSolution.log");
                    if (File.Exists(SolutionPath))
                    {
                        using (TextReader Reader = new StreamReader(File.OpenRead(SolutionPath)))
                        {
                            string Line;
                            while (true)
                            {
                                Line = Reader.ReadLine();
                                if (Line.ToLower().Contains("view") && Line.ToLower().Contains("rotation") && Line.ToLower().Contains("mag"))
                                    break;
                            }

                            for (int i = 0; i < SortedAngle.Count; i++)
                            {
                                Line = Reader.ReadLine();
                                string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                                //SortedAngle[i].AxisAngle = float.Parse(Parts[1], CultureInfo.InvariantCulture);
                                SortedAngle[i].TiltAngle += float.Parse(Parts[3], CultureInfo.InvariantCulture);
                            }
                        }
                    }
                    else
                    {
                        foreach (var mdocEntry in SortedAngle)
                            mdocEntry.AxisAngle = AxisAngle;
                    }

                    //if (CreateStacks)
                    //{
                    //    Console.Write("Reading images");
                    //    foreach (var mdocEntry in Entries)
                    //    {
                    //        MapHeader Header = MapHeader.ReadFromFile(DataRoot + TiltAveragesFolder + mdocEntry.Name, new int2(1, 1), 0, typeof (float));
                    //        float[][] Data;

                    //        Data = IOHelper.ReadMapFloat(DataRoot + TiltAveragesFolder + mdocEntry.Name, new int2(1, 1), 0, typeof (float));

                    //        mdocEntry.Micrograph = new Image(Data,
                    //                                         new int3(Header.Dimensions.X, Header.Dimensions.Y, Header.Dimensions.Z),
                    //                                         false,
                    //                                         false);
                    //        Console.Write(".");
                    //    }
                    //    Console.WriteLine("");
                    //}

                    Star Table = new Star(new[]
                    {
                        "wrpMovieName",
                        "wrpAngleTilt",
                        "wrpAxisAngle",
                        "wrpAxisOffsetX",
                        "wrpAxisOffsetY",
                        "wrpDose"
                    });

                    //Image Stack = CreateStacks ? new Image(new int3(Entries[0].Micrograph.Dims.X, Entries[0].Micrograph.Dims.Y, Entries.Count)) : null;
                    //float[][] StackData = Stack?.GetHost(Intent.Write);
                                       
                    for (int i = 0; i < SortedAngle.Count; i++)
                    {
                        //if (CreateStacks)
                        //    StackData[i] = SortedAngle[i].Micrograph.GetHost(Intent.Read)[0];

                        string PathToMovie = Path.Combine(PathMovie, SortedAngle[i].Name);

                        Uri UriMovie = new Uri(Options.Import.Folder);
                        string MovieRelativePath = UriMovie.MakeRelativeUri(new Uri(PathToMovie)).ToString();

                        Table.AddRow(new List<string>()
                        {
                            MovieRelativePath,
                            (SortedAngle[i].TiltAngle * (InvertTilts ? -1 : 1)).ToString(CultureInfo.InvariantCulture),
                            SortedAngle[i].AxisAngle.ToString(CultureInfo.InvariantCulture),
                            (SortedAngle[i].Shift.X * (float)PixelSize).ToString(CultureInfo.InvariantCulture),
                            (SortedAngle[i].Shift.Y * (float)PixelSize).ToString(CultureInfo.InvariantCulture),
                            SortedAngle[i].Dose.ToString(CultureInfo.InvariantCulture)
                        });
                    }

                    //Table.Save(Path.Combine(Options.Import.Folder, mdocNames.Key + ".tomostar"));
                    ParsedEntries.Add(new ParsedEntry()
                    {
                        DoImport = OldEntries.Any(v => v.Name == mdocNames.Key) ? OldEntries.First(v => v.Name == mdocNames.Key).DoImport : true,
                        Name = mdocNames.Key,
                        NTilts = Table.RowCount,
                        Dose = (int)Math.Round(SortedAngle.Select(v => v.Dose).Max()),
                        Aligned = HasAlignmentData,
                        Table = Table
                    });

                    // Write out tilt series into IMOD folder + its individual subfolder
                    //if (CreateStacks)
                    //{
                    //    Console.WriteLine("Writing tilt series.");
                    //    Directory.CreateDirectory(DataRoot + ImodFolder + mdocNames.Key);
                    //    Stack.WriteMRC(DataRoot + ImodFolder + mdocNames.Key + "\\" + mdocNames.Key + ".st");
                    //}
                }
            }
            catch (Exception exc)
            {
                TextErrors.Content = exc.Message;
                TextErrors.Visibility = Visibility.Visible;
                ListParsedEntries.Visibility = Visibility.Collapsed;
                PanelImportResults.Visibility = Visibility.Visible;

                ButtonWrite.IsEnabled = false;
                ButtonCreateStacks.IsEnabled = false;

                return;
            }

            ButtonWrite.IsEnabled = true;
            ButtonCreateStacks.IsEnabled = true;

            PanelImportResults.Visibility = Visibility.Visible;
            ListParsedEntries.Visibility = Visibility.Visible;
            TextErrors.Visibility = Visibility.Collapsed;
        }

        private void ButtonMdocPath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.Folder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                PathMdoc = Dialog.SelectedPath;
                ButtonMdocPathText.Text = PathMdoc;
            }
        }

        private void ButtonImodPath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.Folder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                PathImod = Dialog.SelectedPath;
                ButtonImodPathText.Text = PathImod;
            }
        }

        private void ButtonMoviePath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                SelectedPath = Options.Import.Folder
            };
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                    Dialog.SelectedPath += '\\';

                PathMovie = Dialog.SelectedPath;
                ButtonMoviePathText.Text = PathMovie;
            }
        }

        private void ButtonWrite_Click(object sender, RoutedEventArgs e)
        {
            foreach (var entry in ParsedEntries.Where(v => v.DoImport))
            {
                string TablePath = Path.Combine(Options.Import.Folder, entry.Name + ".tomostar");
                entry.Table.Save(TablePath);
            }

            Close?.Invoke();
        }

        private void ButtonCancel_OnClick(object sender, RoutedEventArgs e)
        {
            Close?.Invoke();
        }

        private async void ButtonCreateStacks_Click(object sender, RoutedEventArgs e)
        {
            var ProgressDialog = await Options.MainWindow.ShowProgressAsync("Writing data...", "");

            ObservableCollection<ParsedEntry> _ParsedEntries = ParsedEntries;

            await Task.Run(() =>
            {
                int idone = 0;
                foreach (var entry in _ParsedEntries)
                {
                    Dispatcher.Invoke(() => ProgressDialog.SetTitle($"Writing stack for {entry.Name}..."));

                    Directory.CreateDirectory(Path.Combine(Options.Import.Folder, "imod", entry.Name));

                    Movie[] Movies = Helper.ArrayOfFunction(i => new Movie(Path.Combine(Options.Import.Folder, entry.Table.GetRowValue(i, "wrpMovieName"))), entry.Table.RowCount);

                    MapHeader Header = MapHeader.ReadFromFile(Movies[0].AveragePath);

                    Image Stack = new Image(new int3(Header.Dimensions.X, Header.Dimensions.Y, Movies.Length));
                    float[][] StackData = Stack.GetHost(Intent.Write);

                    for (int i = 0; i < Movies.Length; i++)
                    {
                        Image Layer = Image.FromFile(Movies[i].AveragePath);
                        StackData[i] = Layer.GetHost(Intent.Read)[0];
                    }

                    Stack.WriteMRC(Path.Combine(Options.Import.Folder, "imod", entry.Name, entry.Name + ".st"));

                    Dispatcher.Invoke(() => ProgressDialog.SetProgress(++idone / (double)_ParsedEntries.Count));
                }
            });

            await ProgressDialog.CloseAsync();

            Close?.Invoke();
        }

        Dictionary<string, bool> MovieExcludedStatus = new Dictionary<string, bool>();

        private bool IsMovieExcluded(string path)
        {
            if (!MovieExcludedStatus.ContainsKey(path))
            {
                bool Status = false;

                string PathXml = path.Substring(0, path.LastIndexOf('.')) + ".xml";
                if (!File.Exists(PathXml) || !File.Exists(path))
                    Status = true;

                Movie M = new Movie(path);
                if (M.UnselectManual != null && (bool)M.UnselectManual)
                    Status = true;

                if (!File.Exists(M.AveragePath))
                    Status = true;

                MovieExcludedStatus.Add(path, Status);
            }

            return MovieExcludedStatus[path];
        }
    }

    public class MdocEntry
    {
        public int ZValue;
        public string Name;
        public float TiltAngle;
        public float AxisAngle;
        public float Dose;
        public DateTime Time;
        public Image Micrograph;
        public float2 Shift;
        public float MaxTranslation;
    }

    public class ParsedEntry : WarpBase
    {
        private bool _DoImport = true;
        public bool DoImport
        {
            get { return _DoImport; }
            set { if (value != _DoImport) { _DoImport = value; OnPropertyChanged(); } }
        }

        private string _Name = "";
        public string Name
        {
            get { return _Name; }
            set { if (value != _Name) { _Name = value; OnPropertyChanged(); } }
        }

        private int _NTilts = 0;
        public int NTilts
        {
            get { return _NTilts; }
            set { if (value != _NTilts) { _NTilts = value; OnPropertyChanged(); } }
        }

        private int _Dose = 0;
        public int Dose
        {
            get { return _Dose; }
            set { if (value != _Dose) { _Dose = value; OnPropertyChanged(); } }
        }

        private bool _Aligned = false;
        public bool Aligned
        {
            get { return _Aligned; }
            set { if (value != _Aligned) { _Aligned = value; OnPropertyChanged(); } }
        }

        public Star Table;
    }
}
