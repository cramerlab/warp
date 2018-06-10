using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Accord.Math.Optimization;
using Warp.Headers;
using Warp.Tools;
using System.Threading;
using Accord;
using TensorFlow;

namespace Warp
{
    public class Movie : WarpBase
    {
        #region Paths and names

        private string _Path = "";
        public string Path
        {
            get { return _Path; }
            set
            {
                if (value != _Path)
                {
                    _Path = value;
                    OnPropertyChanged();
                }
            }
        }

        public string Name => Helper.PathToNameWithExtension(Path);

        public string RootName => Helper.PathToName(Path);

        public string DirectoryName
        {
            get
            {
                if (Path.Length == 0)
                    return "";

                string Dir = "";
                if (Path.Contains("\\") || Path.Contains("/"))
                    Dir = Path.Substring(0, Math.Max(Path.LastIndexOf("\\"), Path.LastIndexOf("/")));

                if (Dir[Dir.Length - 1] != '/')
                    Dir += "/";

                return Dir;
            }
        }

        public string PowerSpectrumDir => DirectoryName + "powerspectrum/";
        public string AverageDir => DirectoryName + "average/";
        public string DeconvolvedDir => DirectoryName + "deconv/";
        public string ShiftedStackDir => DirectoryName + "stack/";
        public string MaskDir => DirectoryName + "mask/";
        public string ParticlesDir => DirectoryName + "particles/";
        public string ParticleTiltsDir => DirectoryName + "particletilts/";
        public string ParticleCTFDir => DirectoryName + "particlectf/";
        public string ParticleTiltsCTFDir => DirectoryName + "particletiltsctf/";
        public string ParticleMoviesDir => DirectoryName + "particlemovies/";
        public string ParticleMoviesCTFDir => DirectoryName + "particlemoviesctf/";
        public string MatchingDir => DirectoryName + "matching/";
        public string ThumbnailsDir => DirectoryName + "thumbnails/";

        public string PowerSpectrumPath => PowerSpectrumDir + RootName + ".mrc";
        public string AveragePath => AverageDir + RootName + ".mrc";
        public string DeconvolvedPath => DeconvolvedDir + RootName + ".mrc";
        public string ShiftedStackPath => ShiftedStackDir + RootName + "_movie.mrcs";
        public string MaskPath => MaskDir + RootName + ".tif";
        public string ParticlesPath => ParticlesDir + RootName + "_particles.mrcs";
        public string ParticleCTFPath => ParticleCTFDir + RootName + "_particlesctf.mrcs";
        public string ParticleMoviesPath => ParticleMoviesDir + RootName + "_particlemovies.mrcs";
        public string ParticleMoviesCTFPath => ParticleMoviesCTFDir + RootName + "_particlemoviesctf.mrcs";
        public string ThumbnailsPath => ThumbnailsDir + RootName + ".png";

        public string XMLName => RootName + ".xml";
        public string XMLPath => DirectoryName + XMLName;

        #endregion

        #region Selection

        protected Nullable<bool> _UnselectManual = null;
        public Nullable<bool> UnselectManual
        {
            get { return _UnselectManual; }
            set
            {
                if (value != _UnselectManual)
                {
                    _UnselectManual = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    SaveMeta();
                }
            }
        }

        protected bool _UnselectFilter = false;
        public bool UnselectFilter
        {
            get { return _UnselectFilter; }
            set
            {
                if (value != _UnselectFilter)
                {
                    _UnselectFilter = value;
                    OnPropertyChanged();
                    //SaveMeta();
                }
            }
        }

        #endregion

        #region Power spectrum and CTF

        private CTF _CTF = new CTF();
        public CTF CTF
        {
            get { return _CTF; }
            set
            {
                if (value != _CTF)
                {
                    _CTF = value;
                    OnPropertyChanged();
                }
            }
        }

        private float2[] _PS1D;
        public float2[] PS1D
        {
            get { return _PS1D; }
            set
            {
                if (value != _PS1D)
                {
                    _PS1D = value;
                    OnPropertyChanged();
                }
            }
        }

        private float2[] _Simulated1D;
        public float2[] Simulated1D
        {
            get { return _Simulated1D ?? (_Simulated1D = GetSimulated1D()); }
            set
            {
                if (value != _Simulated1D)
                {
                    _Simulated1D = value;
                    OnPropertyChanged();
                }
            }
        }

        protected float2[] GetSimulated1D()
        {
            if (PS1D == null || SimulatedScale == null)
                return null;
            
            float[] SimulatedCTF = CTF.Get1DWithIce(PS1D.Length, true);

            float2[] Result = new float2[PS1D.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = new float2(PS1D[i].X, SimulatedCTF[i] * SimulatedScale.Interp(PS1D[i].X));

            return Result;
        }

        private Cubic1D _SimulatedBackground;
        public Cubic1D SimulatedBackground
        {
            get { return _SimulatedBackground; }
            set
            {
                if (value != _SimulatedBackground)
                {
                    _SimulatedBackground = value;
                    OnPropertyChanged();
                }
            }
        }

        private Cubic1D _SimulatedScale = new Cubic1D(new[] { new float2(0, 1), new float2(1, 1) });
        public Cubic1D SimulatedScale
        {
            get { return _SimulatedScale; }
            set
            {
                if (value != _SimulatedScale)
                {
                    _SimulatedScale = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _CTFResolutionEstimate = 0;
        public decimal CTFResolutionEstimate
        {
            get { return _CTFResolutionEstimate; }
            set { if (value != _CTFResolutionEstimate) { _CTFResolutionEstimate = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Movement

        private decimal _MeanFrameMovement = 0;
        public decimal MeanFrameMovement
        {
            get { return _MeanFrameMovement; }
            set { if (value != _MeanFrameMovement) { _MeanFrameMovement = value; OnPropertyChanged(); } }
        }

        public bool HasLocalMovement => (GridLocalX != null && GridLocalX.FlatValues.Length > 1) || (PyramidShiftX != null && PyramidShiftX.Count > 1);
        public bool HasGlobalMovement => (GridMovementX != null && GridMovementX.FlatValues.Length > 1) || (PyramidShiftX != null && PyramidShiftX.Count > 0);

        #endregion

        #region Grids

        private CubicGrid _GridCTF = new CubicGrid(new int3(1));
        public CubicGrid GridCTF
        {
            get { return _GridCTF; }
            set
            {
                if (value != _GridCTF)
                {
                    _GridCTF = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridCTFPhase = new CubicGrid(new int3(1));
        public CubicGrid GridCTFPhase
        {
            get { return _GridCTFPhase; }
            set
            {
                if (value != _GridCTFPhase)
                {
                    _GridCTFPhase = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridMovementX = new CubicGrid(new int3(1));
        public CubicGrid GridMovementX
        {
            get { return _GridMovementX; }
            set
            {
                if (value != _GridMovementX)
                {
                    _GridMovementX = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridMovementY = new CubicGrid(new int3(1));
        public CubicGrid GridMovementY
        {
            get { return _GridMovementY; }
            set
            {
                if (value != _GridMovementY)
                {
                    _GridMovementY = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridLocalX = new CubicGrid(new int3(1));
        public CubicGrid GridLocalX
        {
            get { return _GridLocalX; }
            set
            {
                if (value != _GridLocalX)
                {
                    _GridLocalX = value;
                    OnPropertyChanged();
                }
            }
        }

        private CubicGrid _GridLocalY = new CubicGrid(new int3(1));
        public CubicGrid GridLocalY
        {
            get { return _GridLocalY; }
            set
            {
                if (value != _GridLocalY)
                {
                    _GridLocalY = value;
                    OnPropertyChanged();
                }
            }
        }

        private List<CubicGrid> _PyramidShiftX = new List<CubicGrid>();
        public List<CubicGrid> PyramidShiftX
        {
            get { return _PyramidShiftX; }
            set
            {
                if (value != _PyramidShiftX)
                {
                    _PyramidShiftX = value;
                    OnPropertyChanged();
                }
            }
        }

        private List<CubicGrid> _PyramidShiftY = new List<CubicGrid>();
        public List<CubicGrid> PyramidShiftY
        {
            get { return _PyramidShiftY; }
            set
            {
                if (value != _PyramidShiftY)
                {
                    _PyramidShiftY = value;
                    OnPropertyChanged();
                }
            }
        }

        #endregion

        #region Processing options

        private bool _IsProcessing = false;
        public bool IsProcessing
        {
            get { return _IsProcessing; }
            set
            {
                if (value != _IsProcessing)
                {
                    _IsProcessing = value;
                    if (value)
                        OnProcessingStarted();
                    else
                        OnProcessingFinished();
                    OnPropertyChanged();
                }
            }
        }

        private ProcessingOptionsMovieCTF _OptionsCTF = null;
        public ProcessingOptionsMovieCTF OptionsCTF
        {
            get { return _OptionsCTF; }
            set
            {
                if (value != _OptionsCTF)
                {
                    _OptionsCTF = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    OnCTF1DChanged();
                    OnCTF2DChanged();
                    OnPS2DChanged();
                }
            }
        }

        private ProcessingOptionsMovieMovement _OptionsMovement = null;
        public ProcessingOptionsMovieMovement OptionsMovement
        {
            get { return _OptionsMovement; }
            set
            {
                if (value != _OptionsMovement)
                {
                    _OptionsMovement = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    OnMovementChanged();
                }
            }
        }

        private ProcessingOptionsMovieExport _OptionsMovieExport = null;
        public ProcessingOptionsMovieExport OptionsMovieExport
        {
            get { return _OptionsMovieExport; }
            set
            {
                if (value != _OptionsMovieExport)
                {
                    _OptionsMovieExport = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                    OnAverageChanged();
                }
            }
        }

        private ProcessingOptionsParticlesExport _OptionsParticlesExport = null;
        public ProcessingOptionsParticlesExport OptionsParticlesExport
        {
            get { return _OptionsParticlesExport; }
            set { if (value != _OptionsParticlesExport) { _OptionsParticlesExport = value; OnPropertyChanged(); } }
        }

        private ProcessingOptionsBoxNet _OptionsBoxNet = null;
        public ProcessingOptionsBoxNet OptionsBoxNet
        {
            get { return _OptionsBoxNet; }
            set
            {
                if (value != _OptionsBoxNet)
                {
                    _OptionsBoxNet = value;
                    OnPropertyChanged();
                    OnProcessingChanged();
                }
            }
        }

        public bool AreOptionsConflicted()
        {
            bool Result = false;

            if (OptionsCTF != null && OptionsMovement != null)
                Result |= OptionsCTF != OptionsMovement;
            if (OptionsCTF != null && OptionsMovieExport != null)
                Result |= OptionsCTF != OptionsMovieExport;
            if (OptionsMovement != null && OptionsMovieExport != null)
                Result |= OptionsMovement != OptionsMovieExport;

            return Result;
        }

        #endregion

        #region Picking and particles

        public readonly Dictionary<string, decimal> PickingThresholds = new Dictionary<string, decimal>();
        private readonly Dictionary<string, int> ParticleCounts = new Dictionary<string, int>();

        public int GetParticleCount(string suffix)
        {
            if (string.IsNullOrEmpty(suffix))
                return -1;

            lock (ParticleCounts)
            {
                if (!ParticleCounts.ContainsKey(suffix))
                {
                    if (File.Exists(MatchingDir + RootName + suffix + ".star"))
                        ParticleCounts.Add(suffix, Star.CountLines(MatchingDir + RootName + suffix + ".star"));
                    else
                        ParticleCounts.Add(suffix, -1);
                }
            }

            return ParticleCounts[suffix];
        }

        public void UpdateParticleCount(string suffix, int count = -1)
        {
            if (string.IsNullOrEmpty(suffix))
                return;

            int Result = Math.Max(-1, count);
            if (count < 0)
                if (File.Exists(MatchingDir + RootName + suffix + ".star"))
                    Result = Star.CountLines(MatchingDir + RootName + suffix + ".star");

            lock (ParticleCounts)
            {
                if (ParticleCounts.ContainsKey(suffix))
                    ParticleCounts[suffix] = Result;
                else
                    ParticleCounts.Add(suffix, Result);
            }
        }

        public void DiscoverParticleSuffixes()
        {
            ParticleCounts.Clear();

            if (Directory.Exists(MatchingDir))
            {
                foreach (var file in Directory.EnumerateFiles(MatchingDir, RootName + "*.star"))
                {
                    string Suffix = Helper.PathToName(file);
                    Suffix = Suffix.Substring(RootName.Length);

                    if (!string.IsNullOrEmpty(Suffix))
                        UpdateParticleCount(Suffix);
                }
            }
        }

        public IEnumerable<string> GetParticlesSuffixes()
        {
            return ParticleCounts.Keys;
        }

        public bool HasAnyParticleSuffixes()
        {
            return ParticleCounts.Count > 0;
        }

        public bool HasParticleSuffix(string suffix)
        {
            return ParticleCounts.ContainsKey(suffix);
        }

        private decimal _MaskPercentage = -1;
        public decimal MaskPercentage
        {
            get { return _MaskPercentage; }
            set { if (value != _MaskPercentage) { _MaskPercentage = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Events

        public event EventHandler ProcessingStarted;
        private void OnProcessingStarted() => ProcessingStarted?.Invoke(this, null);

        public event EventHandler ProcessingChanged;
        private void OnProcessingChanged() => ProcessingChanged?.Invoke(this, null);

        public event EventHandler ProcessingFinished;
        private void OnProcessingFinished() => ProcessingFinished?.Invoke(this, null);

        public event EventHandler CTF1DChanged;
        private void OnCTF1DChanged() => CTF1DChanged?.Invoke(this, null);

        public event EventHandler CTF2DChanged;
        private void OnCTF2DChanged() => CTF2DChanged?.Invoke(this, null);

        public event EventHandler PS2DChanged;
        private void OnPS2DChanged() => PS2DChanged?.Invoke(this, null);

        public event EventHandler MovementChanged;
        private void OnMovementChanged() => MovementChanged?.Invoke(this, null);

        public event EventHandler AverageChanged;
        private void OnAverageChanged() => AverageChanged?.Invoke(this, null);

        public event EventHandler ParticlesChanged;
        public void OnParticlesChanged() => ParticlesChanged?.Invoke(this, null);
        
        #endregion

        public Movie(string path)
        {
            Path = path;

            LoadMeta();
            DiscoverParticleSuffixes();
        }

        public virtual void LoadMeta()
        {
            if (!File.Exists(XMLPath))
                return;

            using (Stream SettingsStream = File.OpenRead(XMLPath))
            {
                XPathDocument Doc = new XPathDocument(SettingsStream);
                XPathNavigator Reader = Doc.CreateNavigator();
                Reader.MoveToRoot();
                Reader.MoveToFirstChild();

                //_UnselectFilter = XMLHelper.LoadAttribute(Reader, "UnselectFilter", _UnselectFilter);
                string UnselectManualString = XMLHelper.LoadAttribute(Reader, "UnselectManual", "null");
                if (UnselectManualString != "null")
                    _UnselectManual = bool.Parse(UnselectManualString);
                else
                    _UnselectManual = null;
                CTFResolutionEstimate = XMLHelper.LoadAttribute(Reader, "CTFResolutionEstimate", CTFResolutionEstimate);
                MeanFrameMovement = XMLHelper.LoadAttribute(Reader, "MeanFrameMovement", MeanFrameMovement);
                MaskPercentage = XMLHelper.LoadAttribute(Reader, "MaskPercentage", MaskPercentage);

                XPathNavigator NavPS1D = Reader.SelectSingleNode("//PS1D");
                if (NavPS1D != null)
                    PS1D = NavPS1D.InnerXml.Split(';').Select(v =>
                    {
                        string[] Pair = v.Split('|');
                        return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                    }).ToArray();

                XPathNavigator NavSimBackground = Reader.SelectSingleNode("//SimulatedBackground");
                if (NavSimBackground != null)
                    _SimulatedBackground = new Cubic1D(NavSimBackground.InnerXml.Split(';').Select(v =>
                    {
                        string[] Pair = v.Split('|');
                        return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                    }).ToArray());

                XPathNavigator NavSimScale = Reader.SelectSingleNode("//SimulatedScale");
                if (NavSimScale != null)
                    _SimulatedScale = new Cubic1D(NavSimScale.InnerXml.Split(';').Select(v =>
                    {
                        string[] Pair = v.Split('|');
                        return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                    }).ToArray());

                XPathNavigator NavCTF = Reader.SelectSingleNode("//CTF");
                if (NavCTF != null)
                    CTF.ReadFromXML(NavCTF);

                XPathNavigator NavGridCTF = Reader.SelectSingleNode("//GridCTF");
                if (NavGridCTF != null)
                    GridCTF = CubicGrid.Load(NavGridCTF);

                XPathNavigator NavGridCTFPhase = Reader.SelectSingleNode("//GridCTFPhase");
                if (NavGridCTFPhase != null)
                    GridCTFPhase = CubicGrid.Load(NavGridCTFPhase);

                XPathNavigator NavMoveX = Reader.SelectSingleNode("//GridMovementX");
                if (NavMoveX != null)
                    GridMovementX = CubicGrid.Load(NavMoveX);

                XPathNavigator NavMoveY = Reader.SelectSingleNode("//GridMovementY");
                if (NavMoveY != null)
                    GridMovementY = CubicGrid.Load(NavMoveY);

                XPathNavigator NavLocalX = Reader.SelectSingleNode("//GridLocalMovementX");
                if (NavLocalX != null)
                    GridLocalX = CubicGrid.Load(NavLocalX);

                XPathNavigator NavLocalY = Reader.SelectSingleNode("//GridLocalMovementY");
                if (NavLocalY != null)
                    GridLocalY = CubicGrid.Load(NavLocalY);

                PyramidShiftX.Clear();
                foreach (XPathNavigator NavShiftX in Reader.Select("//PyramidShiftX"))
                    PyramidShiftX.Add(CubicGrid.Load(NavShiftX));

                PyramidShiftY.Clear();
                foreach (XPathNavigator NavShiftY in Reader.Select("//PyramidShiftY"))
                    PyramidShiftY.Add(CubicGrid.Load(NavShiftY));

                XPathNavigator NavOptionsCTF = Reader.SelectSingleNode("//OptionsCTF");
                if (NavOptionsCTF != null)
                {
                    ProcessingOptionsMovieCTF Temp = new ProcessingOptionsMovieCTF();
                    Temp.ReadFromXML(NavOptionsCTF);
                    OptionsCTF = Temp;
                }

                XPathNavigator NavOptionsMovement = Reader.SelectSingleNode("//OptionsMovement");
                if (NavOptionsMovement != null)
                {
                    ProcessingOptionsMovieMovement Temp = new ProcessingOptionsMovieMovement();
                    Temp.ReadFromXML(NavOptionsMovement);
                    OptionsMovement = Temp;
                }

                XPathNavigator NavOptionsBoxNet = Reader.SelectSingleNode("//OptionsBoxNet");
                if (NavOptionsBoxNet != null)
                {
                    ProcessingOptionsBoxNet Temp = new ProcessingOptionsBoxNet();
                    Temp.ReadFromXML(NavOptionsBoxNet);
                    OptionsBoxNet = Temp;
                }

                XPathNavigator NavOptionsExport = Reader.SelectSingleNode("//OptionsMovieExport");
                if (NavOptionsExport != null)
                {
                    ProcessingOptionsMovieExport Temp = new ProcessingOptionsMovieExport();
                    Temp.ReadFromXML(NavOptionsExport);
                    OptionsMovieExport = Temp;
                }

                XPathNavigator NavOptionsParticlesExport = Reader.SelectSingleNode("//OptionsParticlesExport");
                if (NavOptionsParticlesExport != null)
                {
                    ProcessingOptionsParticlesExport Temp = new ProcessingOptionsParticlesExport();
                    Temp.ReadFromXML(NavOptionsParticlesExport);
                    OptionsParticlesExport = Temp;
                }

                XPathNavigator NavPickingThresholds = Reader.SelectSingleNode("//PickingThresholds");
                if (NavPickingThresholds != null)
                {
                    PickingThresholds.Clear();

                    foreach (XPathNavigator nav in NavPickingThresholds.SelectChildren("Threshold", ""))
                        try
                        {
                            PickingThresholds.Add(nav.GetAttribute("Suffix", ""), decimal.Parse(nav.GetAttribute("Value", ""), CultureInfo.InvariantCulture));
                        }
                        catch { }
                }
            }
        }

        public virtual void SaveMeta()
        {
            using (XmlTextWriter Writer = new XmlTextWriter(XMLPath, Encoding.Unicode))
            {
                Writer.Formatting = Formatting.Indented;
                Writer.IndentChar = '\t';
                Writer.Indentation = 1;
                Writer.WriteStartDocument();
                Writer.WriteStartElement("Movie");

                Writer.WriteAttributeString("UnselectFilter", UnselectFilter.ToString());
                Writer.WriteAttributeString("UnselectManual", UnselectManual != null ? UnselectManual.ToString() : "null");
                Writer.WriteAttributeString("CTFResolutionEstimate", CTFResolutionEstimate.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("MeanFrameMovement", MeanFrameMovement.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("MaskPercentage", MaskPercentage.ToString(CultureInfo.InvariantCulture));

                if (OptionsCTF != null)
                {
                    Writer.WriteStartElement("OptionsCTF");
                    OptionsCTF.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (PS1D != null)
                {
                    Writer.WriteStartElement("PS1D");
                    Writer.WriteString(string.Join(";", PS1D.Select(v => v.X.ToString(CultureInfo.InvariantCulture) + "|" + v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (SimulatedBackground != null)
                {
                    Writer.WriteStartElement("SimulatedBackground");
                    Writer.WriteString(string.Join(";",
                                                   _SimulatedBackground.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                         "|" +
                                                                                         v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (SimulatedScale != null)
                {
                    Writer.WriteStartElement("SimulatedScale");
                    Writer.WriteString(string.Join(";",
                                                   _SimulatedScale.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                    "|" +
                                                                                    v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                Writer.WriteStartElement("CTF");
                CTF.WriteToXML(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTF");
                GridCTF.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFPhase");
                GridCTFPhase.Save(Writer);
                Writer.WriteEndElement();

                if (OptionsMovement != null)
                {
                    Writer.WriteStartElement("OptionsMovement");
                    OptionsMovement.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                Writer.WriteStartElement("GridMovementX");
                GridMovementX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridMovementY");
                GridMovementY.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridLocalMovementX");
                GridLocalX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridLocalMovementY");
                GridLocalY.Save(Writer);
                Writer.WriteEndElement();

                foreach (var grid in PyramidShiftX)
                {
                    Writer.WriteStartElement("PyramidShiftX");
                    grid.Save(Writer);
                    Writer.WriteEndElement();
                }

                foreach (var grid in PyramidShiftY)
                {
                    Writer.WriteStartElement("PyramidShiftY");
                    grid.Save(Writer);
                    Writer.WriteEndElement();
                }

                if (OptionsBoxNet != null)
                {
                    Writer.WriteStartElement("OptionsBoxNet");
                    OptionsBoxNet.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (OptionsMovieExport != null)
                {
                    Writer.WriteStartElement("OptionsMovieExport");
                    OptionsMovieExport.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (OptionsParticlesExport != null)
                {
                    Writer.WriteStartElement("OptionsParticlesExport");
                    OptionsParticlesExport.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                if (PickingThresholds.Count > 0)
                {
                    Writer.WriteStartElement("PickingThresholds");
                    foreach (var pair in PickingThresholds)
                    {
                        Writer.WriteStartElement("Threshold");

                        XMLHelper.WriteAttribute(Writer, "Suffix", pair.Key);
                        XMLHelper.WriteAttribute(Writer, "Value", pair.Value);

                        Writer.WriteEndElement();
                    }
                    Writer.WriteEndElement();
                }

                Writer.WriteEndElement();
                Writer.WriteEndDocument();
            }
        }

        public virtual string GetDataHash()
        {
            FileInfo Info = new FileInfo(Path);
            byte[] DataBytes = new byte[Math.Min(1 << 19, Info.Length)];
            using (BinaryReader Reader = new BinaryReader(File.OpenRead(Path)))
            {
                Reader.Read(DataBytes, 0, DataBytes.Length);
            }

            DataBytes = Helper.Combine(Helper.ToBytes(RootName.ToCharArray()), DataBytes);

            return MathHelper.GetSHA1(DataBytes);
        }

        public virtual string GetProcessingHash()
        {
            List<byte[]> Arrays = new List<byte[]>();

            if (CTF != null)
                Arrays.Add(Helper.ToBytes(new []
                {
                    CTF.Amplitude,
                    CTF.Bfactor,
                    CTF.Cc,
                    CTF.Cs,
                    CTF.Defocus,
                    CTF.DefocusAngle,
                    CTF.DefocusDelta,
                    CTF.EnergySpread,
                    CTF.IllumAngle,
                    CTF.PhaseShift,
                    CTF.PixelSize,
                    CTF.PixelSizeAngle,
                    CTF.PixelSizeDelta,
                    CTF.Scale,
                    CTF.Voltage
                }));

            if (GridCTF != null)
            {
                Arrays.Add(GridCTF.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTF.FlatValues));
            }

            if (GridCTFPhase != null)
            {
                Arrays.Add(GridCTFPhase.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTFPhase.FlatValues));
            }

            if (GridMovementX != null)
            {
                Arrays.Add(GridMovementX.Dimensions);
                Arrays.Add(Helper.ToBytes(GridMovementX.FlatValues));
            }

            if (GridMovementY != null)
            {
                Arrays.Add(GridMovementY.Dimensions);
                Arrays.Add(Helper.ToBytes(GridMovementY.FlatValues));
            }

            if (GridLocalX != null)
            {
                Arrays.Add(GridLocalX.Dimensions);
                Arrays.Add(Helper.ToBytes(GridLocalX.FlatValues));
            }

            if (GridLocalY != null)
            {
                Arrays.Add(GridLocalY.Dimensions);
                Arrays.Add(Helper.ToBytes(GridLocalY.FlatValues));
            }

            if (PyramidShiftX != null)
                foreach (var grid in PyramidShiftX)
                {
                    Arrays.Add(grid.Dimensions);
                    Arrays.Add(Helper.ToBytes(grid.FlatValues));
                }

            if (PyramidShiftY != null)
                foreach (var grid in PyramidShiftY)
                {
                    Arrays.Add(grid.Dimensions);
                    Arrays.Add(Helper.ToBytes(grid.FlatValues));
                }

            byte[] ArraysCombined = Helper.Combine(Arrays);
            return MathHelper.GetSHA1(ArraysCombined);
        }

        public virtual void ProcessCTF(Image originalStack, ProcessingOptionsMovieCTF options)
        {
            IsProcessing = true;

            if (!Directory.Exists(PowerSpectrumDir))
                Directory.CreateDirectory(PowerSpectrumDir);

            //CTF = new CTF();
            PS1D = null;
            _SimulatedBackground = null;
            _SimulatedScale = new Cubic1D(new[] { new float2(0, 1), new float2(1, 1) });

            #region Dimensions and grids

            int NFrames = options.UseMovieSum ? 1 : originalStack.Dims.Z;
            int2 DimsImage = new int2(originalStack.Dims);
            int2 DimsRegion = new int2(options.Window);
            int2 DimsRegionLarge = DimsRegion * 2;

            float OverlapFraction = 0.5f;
            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, DimsRegionLarge, OverlapFraction, out DimsPositionGrid);
            int NPositions = (int)DimsPositionGrid.Elements();

            int CTFGridX = Math.Min(DimsPositionGrid.X, options.GridDims.X);
            int CTFGridY = Math.Min(DimsPositionGrid.Y, options.GridDims.Y);
            int CTFGridZ = Math.Min(NFrames, options.GridDims.Z);
            GridCTF = new CubicGrid(new int3(CTFGridX, CTFGridY, CTFGridZ));
            GridCTFPhase = new CubicGrid(new int3(1, 1, CTFGridZ));

            bool CTFSpace = CTFGridX * CTFGridY > 1;
            bool CTFTime = CTFGridZ > 1;
            int3 CTFSpectraGrid = new int3(CTFSpace ? DimsPositionGrid.X : 1,
                                           CTFSpace ? DimsPositionGrid.Y : 1,
                                           CTFTime ? CTFGridZ : 1);

            int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
            int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
            int NFreq = MaxFreqExclusive - MinFreqInclusive;

            #endregion

            #region Allocate GPU memory

            Image CTFSpectra = new Image(IntPtr.Zero, new int3(DimsRegion.X, DimsRegion.X, (int)CTFSpectraGrid.Elements()), true);
            Image CTFMean = new Image(IntPtr.Zero, new int3(DimsRegion), true);
            Image CTFCoordsCart = new Image(new int3(DimsRegion), true, true);
            Image CTFCoordsPolarTrimmed = new Image(new int3(NFreq, DimsRegion.X, 1), false, true);

            #endregion

            // Extract movie regions, create individual spectra in Cartesian coordinates and their mean.

            #region Create spectra

            if (options.UseMovieSum)
            {
                Image StackAverage = originalStack.AsReducedAlongZ();
                originalStack.FreeDevice();

                GPU.CreateSpectra(StackAverage.GetDevice(Intent.Read),
                                  DimsImage,
                                  1,
                                  PositionGrid,
                                  NPositions,
                                  DimsRegionLarge,
                                  CTFSpectraGrid,
                                  DimsRegion,
                                  CTFSpectra.GetDevice(Intent.Write),
                                  CTFMean.GetDevice(Intent.Write),
                                  0,
                                  0);

                StackAverage.Dispose();
            }
            else
            {
                GPU.CreateSpectra(originalStack.GetDevice(Intent.Read),
                                  DimsImage,
                                  NFrames,
                                  PositionGrid,
                                  NPositions,
                                  DimsRegionLarge,
                                  CTFSpectraGrid,
                                  DimsRegion,
                                  CTFSpectra.GetDevice(Intent.Write),
                                  CTFMean.GetDevice(Intent.Write),
                                  0,
                                  0);

                originalStack.FreeDevice(); // Won't need it in this method anymore.
            }

            CTFSpectra.WriteMRC("d_spectra.mrc", true);
            #endregion

            // Populate address arrays for later.

            #region Init addresses

            {
                float2[] CoordsData = new float2[CTFCoordsCart.ElementsSliceComplex];

                Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) => CoordsData[y * (DimsRegion.X / 2 + 1) + x] = new float2(r, a));
                CTFCoordsCart.UpdateHostWithComplex(new[] { CoordsData });

                CoordsData = new float2[NFreq * DimsRegion.X];
                Helper.ForEachElement(CTFCoordsPolarTrimmed.DimsSlice, (x, y) =>
                {
                    float Angle = (float)y / DimsRegion.X * (float)Math.PI;
                    float Ny = 1f / DimsRegion.X;
                    CoordsData[y * NFreq + x] = new float2((x + MinFreqInclusive) * Ny, Angle);
                });
                CTFCoordsPolarTrimmed.UpdateHostWithComplex(new[] { CoordsData });
            }

            #endregion

            // Retrieve average 1D spectrum from CTFMean (not corrected for astigmatism yet).

            #region Initial 1D spectrum

            {
                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                GPU.CTFMakeAverage(CTFMean.GetDevice(Intent.Read),
                                   CTFCoordsCart.GetDevice(Intent.Read),
                                   (uint)CTFMean.ElementsSliceReal,
                                   (uint)DimsRegion.X,
                                   new[] { new CTF().ToStruct() },
                                   new CTF().ToStruct(),
                                   0,
                                   (uint)DimsRegion.X / 2,
                                   1,
                                   CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                float[] CTFAverage1DData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[DimsRegion.X / 2];
                for (int i = 0; i < ForPS1D.Length; i++)
                    ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(CTFAverage1DData[i], 4));
                _PS1D = ForPS1D;

                CTFAverage1D.Dispose();
            }

            #endregion

            #region Background fitting methods

            Action UpdateBackgroundFit = () =>
            {
                float2[] ForPS1D = PS1D.Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray();
                Cubic1D.FitCTF(ForPS1D,
                               CTF.Get1DWithIce(PS1D.Length, true, true).Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray(),
                               CTF.GetZeros(true),
                               CTF.GetPeaks(true),
                               out _SimulatedBackground,
                               out _SimulatedScale);
            };

            Action<bool> UpdateRotationalAverage = keepbackground =>
            {
                float[] MeanData = CTFMean.GetHost(Intent.Read)[0];

                Image CTFMeanCorrected = new Image(new int3(DimsRegion), true);
                float[] MeanCorrectedData = CTFMeanCorrected.GetHost(Intent.Write)[0];

                // Subtract current background estimate from spectra, populate coords.
                Helper.ForEachElementFT(DimsRegion,
                                        (x, y, xx, yy, r, a) =>
                                        {
                                            int i = y * (DimsRegion.X / 2 + 1) + x;
                                            MeanCorrectedData[i] = MeanData[i] - _SimulatedBackground.Interp(r / DimsRegion.X);
                                        });

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                GPU.CTFMakeAverage(CTFMeanCorrected.GetDevice(Intent.Read),
                                   CTFCoordsCart.GetDevice(Intent.Read),
                                   (uint)CTFMeanCorrected.DimsEffective.ElementsSlice(),
                                   (uint)DimsRegion.X,
                                   new[] { CTF.ToStruct() },
                                   CTF.ToStruct(),
                                   0,
                                   (uint)DimsRegion.X / 2,
                                   1,
                                   CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[PS1D.Length];
                if (keepbackground)
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i] + _SimulatedBackground.Interp((float)i / DimsRegion.X));
                else
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                MathHelper.UnNaN(ForPS1D);

                _PS1D = ForPS1D;

                CTFMeanCorrected.Dispose();
                CTFAverage1D.Dispose();
            };

            #endregion

            // Fit background to currently best average (not corrected for astigmatism yet).
            {
                float2[] ForPS1D = PS1D.Skip(MinFreqInclusive).Take(Math.Max(2, NFreq)).ToArray();

                int NumNodes = Math.Max(3, (int)((options.RangeMax - options.RangeMin) * 5M));
                _SimulatedBackground = Cubic1D.Fit(ForPS1D, NumNodes); // This won't fit falloff and scale, because approx function is 0

                float[] CurrentBackground = _SimulatedBackground.Interp(PS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray();
                float[] Subtracted1D = Helper.ArrayOfFunction(i => ForPS1D[i].Y - CurrentBackground[i], ForPS1D.Length);
                MathHelper.NormalizeInPlace(Subtracted1D);

                float ZMin = (float)options.ZMin;
                float ZMax = (float)options.ZMax;
                float ZStep = (ZMax - ZMin) / 200f;
                
                float BestZ = 0, BestIceOffset = 0, BestPhase = 0, BestScore = -float.MaxValue;
                for (float z = ZMin; z <= ZMax + 1e-5f; z += 0.01f)
                {
                    for (float dz = (options.DoIce ? -0.06f : 0f); dz <= 0.0f + 1e-5f; dz += 0.005f)
                    {
                        for (float p = 0; p <= (options.DoPhase ? 1f : 0f); p += 0.01f)
                        {
                            CTF CurrentParams = new CTF
                            {
                                PixelSize = options.BinnedPixelSizeMean,

                                Defocus = (decimal)z,
                                PhaseShift = (decimal)p,

                                Cs = options.Cs,
                                Voltage = options.Voltage,
                                Amplitude = options.Amplitude,

                                IceOffset = (decimal)dz,
                                IceIntensity = 0.8M,
                                IceStd = new float2(0.5f)
                            };

                            float[] SimulatedCTF = CurrentParams.Get1DWithIce(PS1D.Length, true).Skip(MinFreqInclusive).Take(NFreq).ToArray();

                            MathHelper.NormalizeInPlace(SimulatedCTF);
                            float Score = MathHelper.CrossCorrelate(Subtracted1D, SimulatedCTF);

                            if (Score > BestScore)
                            {
                                BestScore = Score;
                                BestZ = z;
                                BestIceOffset = dz;
                                BestPhase = p;
                            }
                        }
                    }
                }

                CTF = new CTF
                {
                    PixelSize = options.BinnedPixelSizeMean,

                    Defocus = (decimal)BestZ,
                    PhaseShift = (decimal)BestPhase,

                    Cs = options.Cs,
                    Voltage = options.Voltage,
                    Amplitude = options.Amplitude,

                    IceOffset = (decimal)BestIceOffset,
                    IceIntensity = 0.8M,
                    IceStd = new float2(0.5f)
                };

                UpdateRotationalAverage(true); // This doesn't have a nice background yet.
                UpdateBackgroundFit(); // Now get a reasonably nice background.
            }

            // Do BFGS optimization of defocus, astigmatism and phase shift,
            // using 2D simulation for comparison

            #region BFGS

            GridCTF = new CubicGrid(GridCTF.Dimensions, (float)CTF.Defocus, (float)CTF.Defocus, Dimension.X);
            GridCTFPhase = new CubicGrid(GridCTFPhase.Dimensions, (float)CTF.PhaseShift, (float)CTF.PhaseShift, Dimension.X);

            {
                NFreq = MaxFreqExclusive - MinFreqInclusive;

                Image CTFSpectraPolarTrimmed = CTFSpectra.AsPolar((uint)MinFreqInclusive, (uint)(MinFreqInclusive + NFreq));
                CTFSpectra.FreeDevice(); // This will only be needed again for the final PS1D.

                #region Create background and scale

                float[] CurrentScale = _SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray());

                Image CTFSpectraScale = new Image(new int3(NFreq, DimsRegion.X, 1));
                float[] CTFSpectraScaleData = CTFSpectraScale.GetHost(Intent.Write)[0];

                // Trim polar to relevant frequencies, and populate coordinates.
                Parallel.For(0, DimsRegion.X, y =>
                {
                    float Angle = ((float)y / DimsRegion.X + 0.5f) * (float)Math.PI;
                    for (int x = 0; x < NFreq; x++)
                        CTFSpectraScaleData[y * NFreq + x] = CurrentScale[x + MinFreqInclusive];
                });
                //CTFSpectraScale.WriteMRC("ctfspectrascale.mrc");

                // Background is just 1 line since we're in polar.
                Image CurrentBackground = new Image(_SimulatedBackground.Interp(PS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray());

                #endregion

                CTFSpectraPolarTrimmed.SubtractFromLines(CurrentBackground);
                CurrentBackground.Dispose();

                // Normalize background-subtracted spectra.
                GPU.Normalize(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                              CTFSpectraPolarTrimmed.GetDevice(Intent.Write),
                              (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                              (uint)CTFSpectraGrid.Elements());
                //CTFSpectraPolarTrimmed.WriteMRC("ctfspectrapolartrimmed.mrc");
                
                // Wiggle weights show how the defocus on the spectra grid is altered 
                // by changes in individual anchor points of the spline grid.
                // They are used later to compute the dScore/dDefocus values for each spectrum 
                // only once, and derive the values for each anchor point from them.
                float[][] WiggleWeights = GridCTF.GetWiggleWeights(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 1f / (CTFGridZ + 1)));
                float[][] WiggleWeightsPhase = GridCTFPhase.GetWiggleWeights(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 1f / (CTFGridZ + 1)));

                // Helper method for getting CTFStructs for the entire spectra grid.
                Func<double[], CTF, float[], float[], CTFStruct[]> EvalGetCTF = (input, ctf, defocusValues, phaseValues) =>
                {
                    decimal AlteredDelta = (decimal)input[input.Length - 2];
                    decimal AlteredAngle = (decimal)(input[input.Length - 1] * 20 * Helper.ToDeg);

                    CTF Local = ctf.GetCopy();
                    Local.DefocusDelta = AlteredDelta;
                    Local.DefocusAngle = AlteredAngle;

                    CTFStruct LocalStruct = Local.ToStruct();
                    CTFStruct[] LocalParams = new CTFStruct[defocusValues.Length];
                    for (int i = 0; i < LocalParams.Length; i++)
                    {
                        LocalParams[i] = LocalStruct;
                        LocalParams[i].Defocus = defocusValues[i] * -1e-6f;
                        LocalParams[i].PhaseShift = phaseValues[i] * (float)Math.PI;
                    }

                    return LocalParams;
                };

                // Simulate with adjusted CTF, compare to originals

                #region Eval and Gradient methods

                float BorderZ = 0.5f / CTFGridZ;

                Func<double[], double> Eval = input =>
                {
                    CubicGrid Altered = new CubicGrid(GridCTF.Dimensions, input.Take((int)GridCTF.Dimensions.Elements()).Select(v => (float)v).ToArray());
                    float[] DefocusValues = Altered.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                    CubicGrid AlteredPhase = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTF.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());
                    float[] PhaseValues = AlteredPhase.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                    CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                    float IceIntensity = 1 / (1 + (float)Math.Exp(-input[input.Length - 3] * 10));
                    float2 IceStd = new float2((float)Math.Exp(input[input.Length - 5] * 10), (float)Math.Exp(input[input.Length - 4] * 10));
                    float IceOffset = (float)input[input.Length - 6] * (-1e4f);

                    CTF.IceIntensity = (decimal)IceIntensity;
                    CTF.IceStd = IceStd;
                    float[] IceMask1D = CTF.GetIceMask(DimsRegion.X / 2).Skip(MinFreqInclusive).Take(NFreq).ToArray();
                    float[] IceMaskData = Helper.Combine(Helper.ArrayOfConstant(IceMask1D, CTFSpectraScale.Dims.Y));
                    Image IceMask = new Image(IceMaskData, CTFSpectraScale.Dims);


                    float[] Result = new float[LocalParams.Length];

                    GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                        CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                        CTFSpectraScale.GetDevice(Intent.Read),
                                        IceMask.GetDevice(Intent.Read),
                                        IceOffset,
                                        (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                        LocalParams,
                                        Result,
                                        (uint)LocalParams.Length);

                    IceMask.Dispose();

                    float Score = Result.Sum();

                    if (float.IsNaN(Score) || float.IsInfinity(Score))
                        throw new Exception("Bad score.");

                    return Score;
                };

                Func<double[], double[]> Gradient = input =>
                {
                    const float Step = 0.005f;
                    double[] Result = new double[input.Length];

                    // In 0D grid case, just get gradient for all 4 parameters.
                    // In 1+D grid case, do simple gradient for ice ring, astigmatism, phase, ...
                    int StartComponent = input.Length - (options.DoIce ? 6 : 2);
                    //int StartComponent = 0;
                    for (int i = StartComponent; i < input.Length; i++)
                    {
                        double[] UpperInput = new double[input.Length];
                        input.CopyTo(UpperInput, 0);
                        UpperInput[i] += Step;
                        double UpperValue = Eval(UpperInput);

                        double[] LowerInput = new double[input.Length];
                        input.CopyTo(LowerInput, 0);
                        LowerInput[i] -= Step;
                        double LowerValue = Eval(LowerInput);

                        Result[i] = (UpperValue - LowerValue) / (2f * Step);
                    }

                    float IceIntensity = 1 / (1 + (float)Math.Exp(-input[input.Length - 3] * 10));
                    float2 IceStd = new float2((float)Math.Exp(input[input.Length - 5] * 10), (float)Math.Exp(input[input.Length - 4] * 10));
                    float IceOffset = (float)input[input.Length - 6] * (-1e4f);

                    CTF.IceIntensity = (decimal)IceIntensity;
                    CTF.IceStd = IceStd;
                    float[] IceMask1D = CTF.GetIceMask(DimsRegion.X / 2).Skip(MinFreqInclusive).Take(NFreq).ToArray();
                    float[] IceMaskData = Helper.Combine(Helper.ArrayOfConstant(IceMask1D, CTFSpectraScale.Dims.Y));
                    Image IceMask = new Image(IceMaskData, CTFSpectraScale.Dims);

                    float[] ResultPlus = new float[CTFSpectraGrid.Elements()];
                    float[] ResultMinus = new float[CTFSpectraGrid.Elements()];

                    // ... take shortcut for defoci, ...
                    {
                        CubicGrid AlteredPhase = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTF.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());
                        float[] PhaseValues = AlteredPhase.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                        {
                            CubicGrid AlteredPlus = new CubicGrid(GridCTF.Dimensions, input.Take((int)GridCTF.Dimensions.Elements()).Select(v => (float)v + Step).ToArray());
                            float[] DefocusValues = AlteredPlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                IceMask.GetDevice(Intent.Read),
                                                IceOffset,
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultPlus,
                                                (uint)LocalParams.Length);
                        }
                        {
                            CubicGrid AlteredMinus = new CubicGrid(GridCTF.Dimensions, input.Take((int)GridCTF.Dimensions.Elements()).Select(v => (float)v - Step).ToArray());
                            float[] DefocusValues = AlteredMinus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                IceMask.GetDevice(Intent.Read),
                                                IceOffset,
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultMinus,
                                                (uint)LocalParams.Length);
                        }
                        float[] LocalGradients = new float[ResultPlus.Length];
                        for (int i = 0; i < LocalGradients.Length; i++)
                            LocalGradients[i] = ResultPlus[i] - ResultMinus[i];

                        // Now compute gradients per grid anchor point using the precomputed individual gradients and wiggle factors.
                        Parallel.For(0, GridCTF.Dimensions.Elements(), i => Result[i] = MathHelper.ReduceWeighted(LocalGradients, WiggleWeights[i]) / (2f * Step));
                    }

                    // ... and take shortcut for phases.
                    if (options.DoPhase)
                    {
                        CubicGrid AlteredPlus = new CubicGrid(GridCTF.Dimensions, input.Take((int)GridCTF.Dimensions.Elements()).Select(v => (float)v).ToArray());
                        float[] DefocusValues = AlteredPlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                        {
                            CubicGrid AlteredPhasePlus = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTF.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v + Step).ToArray());
                            float[] PhaseValues = AlteredPhasePlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                IceMask.GetDevice(Intent.Read),
                                                IceOffset,
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultPlus,
                                                (uint)LocalParams.Length);
                        }
                        {
                            CubicGrid AlteredPhaseMinus = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTF.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v - Step).ToArray());
                            float[] PhaseValues = AlteredPhaseMinus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                            CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                            GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                                CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                                CTFSpectraScale.GetDevice(Intent.Read),
                                                IceMask.GetDevice(Intent.Read),
                                                IceOffset,
                                                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                                LocalParams,
                                                ResultMinus,
                                                (uint)LocalParams.Length);
                        }
                        float[] LocalGradients = new float[ResultPlus.Length];
                        for (int i = 0; i < LocalGradients.Length; i++)
                            LocalGradients[i] = ResultPlus[i] - ResultMinus[i];

                        // Now compute gradients per grid anchor point using the precomputed individual gradients and wiggle factors.
                        Parallel.For(0, GridCTFPhase.Dimensions.Elements(), i => Result[i + GridCTF.Dimensions.Elements()] = MathHelper.ReduceWeighted(LocalGradients, WiggleWeightsPhase[i]) / (2f * Step));
                    }

                    IceMask.Dispose();

                    foreach (var i in Result)
                        if (double.IsNaN(i) || double.IsInfinity(i))
                            throw new Exception("Bad score.");

                    return Result;
                };

                #endregion

                #region Optimize

                double[] StartParams = new double[GridCTF.Dimensions.Elements() + GridCTFPhase.Dimensions.Elements() + 6];
                for (int i = 0; i < GridCTF.Dimensions.Elements(); i++)
                    StartParams[i] = GridCTF.FlatValues[i];
                for (int i = 0; i < GridCTFPhase.Dimensions.Elements(); i++)
                    StartParams[i + GridCTF.Dimensions.Elements()] = GridCTFPhase.FlatValues[i];

                StartParams[StartParams.Length - 6] = (double)CTF.IceOffset;
                StartParams[StartParams.Length - 5] = Math.Log(0.5) / 10;
                StartParams[StartParams.Length - 4] = Math.Log(0.5) / 10;
                StartParams[StartParams.Length - 3] = 0 / 10;
                StartParams[StartParams.Length - 2] = (double)CTF.DefocusDelta;
                StartParams[StartParams.Length - 1] = (double)CTF.DefocusAngle / 20 * Helper.ToRad;

                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient)
                {
                    MaxIterations = 15
                };
                Optimizer.Maximize(StartParams);

                #endregion

                #region Retrieve parameters

                CTF.Defocus = (decimal)MathHelper.Mean(Optimizer.Solution.Take((int)GridCTF.Dimensions.Elements()).Select(v => (float)v));
                CTF.DefocusDelta = (decimal)Optimizer.Solution[StartParams.Length - 2];
                CTF.DefocusAngle = (decimal)(Optimizer.Solution[StartParams.Length - 1] * 20 * Helper.ToDeg);
                CTF.PhaseShift = (decimal)MathHelper.Mean(Optimizer.Solution.Skip((int)GridCTF.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v));

                CTF.IceIntensity = (decimal)(1 / (1 + Math.Exp(-Optimizer.Solution[Optimizer.Solution.Length - 3] * 10)));
                CTF.IceStd = new float2((float)Math.Exp(Optimizer.Solution[Optimizer.Solution.Length - 5] * 10), (float)Math.Exp(Optimizer.Solution[Optimizer.Solution.Length - 4] * 10));
                CTF.IceOffset = (decimal)Optimizer.Solution[Optimizer.Solution.Length - 6];

                if (CTF.DefocusDelta < 0)
                {
                    CTF.DefocusAngle += 90;
                    CTF.DefocusDelta *= -1;
                }
                CTF.DefocusAngle = ((int)CTF.DefocusAngle + 180 * 99) % 180;

                GridCTF = new CubicGrid(GridCTF.Dimensions, Optimizer.Solution.Take((int)GridCTF.Dimensions.Elements()).Select(v => (float)v).ToArray());
                GridCTFPhase = new CubicGrid(GridCTFPhase.Dimensions, Optimizer.Solution.Skip((int)GridCTF.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());

                #endregion

                // Dispose GPU resources manually because GC can't be bothered to do it in time.
                CTFSpectraPolarTrimmed.Dispose();
                CTFSpectraScale.Dispose();

                #region Get nicer envelope fit
                
                {
                    if (!CTFSpace && !CTFTime)
                    {
                        UpdateRotationalAverage(true);
                    }
                    else
                    {
                        Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                        float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                        // Construct background in Cartesian coordinates.
                        Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) =>
                        {
                            CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = _SimulatedBackground.Interp(r / DimsRegion.X);
                        });

                        CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                        float[] DefocusValues = GridCTF.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                        CTFStruct[] LocalParams = DefocusValues.Select(v =>
                        {
                            CTF Local = CTF.GetCopy();
                            Local.Defocus = (decimal)v + 0.0M;

                            return Local.ToStruct();
                        }).ToArray();

                        Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                        CTF CTFAug = CTF.GetCopy();
                        CTFAug.Defocus += 0.0M;
                        GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                           CTFCoordsCart.GetDevice(Intent.Read),
                                           (uint)CTFSpectra.ElementsSliceReal,
                                           (uint)DimsRegion.X,
                                           LocalParams,
                                           CTFAug.ToStruct(),
                                           0,
                                           (uint)DimsRegion.X / 2,
                                           (uint)CTFSpectraGrid.Elements(),
                                           CTFAverage1D.GetDevice(Intent.Write));

                        CTFSpectra.AddToSlices(CTFSpectraBackground);

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        float2[] ForPS1D = new float2[PS1D.Length];
                        for (int i = 0; i < ForPS1D.Length; i++)
                            ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(RotationalAverageData[i], 4) + _SimulatedBackground.Interp((float)i / DimsRegion.X));
                        MathHelper.UnNaN(ForPS1D);
                        _PS1D = ForPS1D;

                        CTFSpectraBackground.Dispose();
                        CTFAverage1D.Dispose();
                        CTFSpectra.FreeDevice();
                    }

                    CTF.Defocus = Math.Max(CTF.Defocus, 0);
                    UpdateBackgroundFit();
                }

                #endregion
            }

            #endregion

            // Subtract background from 2D average and write it to disk. 
            // This image is used for quick visualization purposes only.

            #region PS2D update
            {
                int3 DimsAverage = new int3(DimsRegion.X, DimsRegion.X / 2, 1);
                float[] Average2DData = new float[DimsAverage.Elements()];
                float[] OriginalAverageData = CTFMean.GetHost(Intent.Read)[0];
                int DimHalf = DimsRegion.X / 2;

                for (int y = 0; y < DimsAverage.Y; y++)
                {
                    int yy = y * y;
                    for (int x = 0; x < DimHalf; x++)
                    {
                        int xx = x;
                        xx *= xx;
                        float r = (float)Math.Sqrt(xx + yy) / DimsRegion.X;
                        Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x + DimHalf] = OriginalAverageData[(DimsRegion.X - 1 - y) * (DimsRegion.X / 2 + 1) + x] - SimulatedBackground.Interp(r);
                    }

                    for (int x = 1; x < DimHalf; x++)
                    {
                        int xx = -(x - DimHalf);
                        float r = (float)Math.Sqrt(xx * xx + yy) / DimsRegion.X;
                        Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x] = OriginalAverageData[y * (DimsRegion.X / 2 + 1) + xx] - SimulatedBackground.Interp(r);
                    }
                }

                IOHelper.WriteMapFloat(PowerSpectrumPath,
                                       new HeaderMRC
                                       {
                                           Dimensions = DimsAverage,
                                           MinValue = MathHelper.Min(Average2DData),
                                           MaxValue = MathHelper.Max(Average2DData)
                                       },
                                       Average2DData);
            }
            #endregion

            for (int i = 0; i < PS1D.Length; i++)
                PS1D[i].Y -= SimulatedBackground.Interp(PS1D[i].X);
            SimulatedBackground = new Cubic1D(SimulatedBackground.Data.Select(v => new float2(v.X, 0f)).ToArray());

            CTFSpectra.Dispose();
            CTFMean.Dispose();
            CTFCoordsCart.Dispose();
            CTFCoordsPolarTrimmed.Dispose();

            Simulated1D = GetSimulated1D();
            //CTFQuality = GetCTFQuality();

            #region Estimate fittable resolution
            {
                float[] Quality = CTF.EstimateQuality(PS1D.Select(p => p.Y).ToArray(), SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray()), (float)options.RangeMin, 16, true);
                int FirstFreq = MinFreqInclusive + NFreq / 2;
                //while ((float.IsNaN(Quality[FirstFreq]) || Quality[FirstFreq] < 0.8f) && FirstFreq < Quality.Length - 1)
                //    FirstFreq++;

                int LastFreq = FirstFreq;
                while (!float.IsNaN(Quality[LastFreq]) && Quality[LastFreq] > 0.3f && LastFreq < Quality.Length - 1)
                    LastFreq++;

                CTFResolutionEstimate = Math.Round(options.BinnedPixelSizeMean / ((decimal)LastFreq / options.Window), 1);
            }
            #endregion

            OptionsCTF = options;

            SaveMeta();

            IsProcessing = false;
        }

        public void ProcessShift(Image originalStack, ProcessingOptionsMovieMovement options)
        {
            IsProcessing = true;

            // Deal with dimensions and grids.

            int NFrames = originalStack.Dims.Z;
            int2 DimsImage = new int2(originalStack.Dims);
            int2 DimsRegion = new int2(512, 512);

            float OverlapFraction = 0.5f;
            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, DimsRegion, OverlapFraction, out DimsPositionGrid);
            //PositionGrid = new[] { new int3(0, 0, 0) };
            //DimsPositionGrid = new int2(1, 1);
            int NPositions = PositionGrid.Length;

            int ShiftGridX = options.GridDims.X;
            int ShiftGridY = options.GridDims.Y;
            int ShiftGridZ = Math.Min(NFrames, options.GridDims.Z);
            GridMovementX = new CubicGrid(new int3(1, 1, ShiftGridZ));
            GridMovementY = new CubicGrid(new int3(1, 1, ShiftGridZ));

            int LocalGridX = Math.Min(DimsPositionGrid.X, options.GridDims.X);
            int LocalGridY = Math.Min(DimsPositionGrid.Y, options.GridDims.Y);
            int LocalGridZ = LocalGridX * LocalGridY <= 1 ? 1 : 3;//Math.Max(3, (int)Math.Ceiling(options.GridDims.Z / (float)(LocalGridX * LocalGridY)));
            GridLocalX = new CubicGrid(new int3(LocalGridX, LocalGridY, LocalGridZ));
            GridLocalY = new CubicGrid(new int3(LocalGridX, LocalGridY, LocalGridZ));

            int3 ShiftGrid = new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames);

            int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
            int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
            int NFreq = MaxFreqExclusive - MinFreqInclusive;

            int CentralFrame = NFrames / 2;

            int MaskExpansions = Math.Max(1, (int)Math.Ceiling(Math.Log(ShiftGridZ - 0.01, 3)));    // Each expansion doubles the temporal resolution
            int[] MaskSizes = new int[MaskExpansions];

            // Allocate memory and create all prerequisites:
            int MaskLength;
            Image ShiftFactors;
            Image Patches;
            Image PatchesAverage;
            Image Shifts;
            {
                List<long> Positions = new List<long>();
                List<float2> Factors = new List<float2>();
                List<float2> Freq = new List<float2>();
                int Min2 = MinFreqInclusive * MinFreqInclusive;
                int Max2 = MaxFreqExclusive * MaxFreqExclusive;
                float PixelSize = (float)options.BinnedPixelSizeMean;

                for (int y = 0; y < DimsRegion.Y; y++)
                {
                    int yy = y > DimsRegion.Y / 2 ? y - DimsRegion.Y : y;
                    for (int x = 0; x < DimsRegion.X / 2 + 1; x++)
                    {
                        int xx = x;
                        int r2 = xx * xx + yy * yy;
                        if (r2 >= Min2 && r2 < Max2)
                        {
                            Positions.Add(y * (DimsRegion.X / 2 + 1) + x);
                            Factors.Add(new float2((float)xx / DimsRegion.X * 2f * (float)Math.PI,
                                                   (float)yy / DimsRegion.Y * 2f * (float)Math.PI));

                            float Angle = (float)Math.Atan2(yy, xx);
                            float r = (float)Math.Sqrt(r2);
                            Freq.Add(new float2(r, Angle));
                        }
                    }
                }

                // Sort everyone by ascending distance from center.
                List<KeyValuePair<float, int>> FreqIndices = Freq.Select((v, i) => new KeyValuePair<float, int>(v.X, i)).ToList();
                FreqIndices.Sort((a, b) => a.Key.CompareTo(b.Key));
                int[] SortedIndices = FreqIndices.Select(v => v.Value).ToArray();

                Helper.Reorder(Positions, SortedIndices);
                Helper.Reorder(Factors, SortedIndices);
                Helper.Reorder(Freq, SortedIndices);

                float Bfac = (float)options.Bfactor * 0.25f;
                float[] BfacWeightsData = Freq.Select(v =>
                {
                    float r2 = v.X / PixelSize / DimsRegion.X;
                    r2 *= r2;
                    return (float)Math.Exp(r2 * Bfac);
                }).ToArray();
                Image BfacWeights = new Image(BfacWeightsData);

                long[] RelevantMask = Positions.ToArray();
                ShiftFactors = new Image(Helper.ToInterleaved(Factors.ToArray()));
                MaskLength = RelevantMask.Length;

                // Get mask sizes for different expansion steps.
                for (int i = 0; i < MaskExpansions; i++)
                {
                    float CurrentMaxFreq = MinFreqInclusive + (MaxFreqExclusive - MinFreqInclusive) / (float)MaskExpansions * (i + 1);
                    MaskSizes[i] = Freq.Count(v => v.X * v.X < CurrentMaxFreq * CurrentMaxFreq);
                }

                Patches = new Image(IntPtr.Zero, new int3(MaskLength, DimsPositionGrid.X * DimsPositionGrid.Y, NFrames), false, true, false);

                GPU.CreateShift(originalStack.GetDevice(Intent.Read),
                                DimsImage,
                                originalStack.Dims.Z,
                                PositionGrid,
                                PositionGrid.Length,
                                DimsRegion,
                                RelevantMask,
                                (uint)MaskLength,
                                Patches.GetDevice(Intent.Write));

                Patches.MultiplyLines(BfacWeights);
                BfacWeights.Dispose();

                originalStack.FreeDevice();
                PatchesAverage = new Image(IntPtr.Zero, new int3(MaskLength, NPositions, 1), false, true);
                Shifts = new Image(new float[NPositions * NFrames * 2]);
            }

            #region Fit movement

            {
                int MinXSteps = 1, MinYSteps = 1;
                int MinZSteps = Math.Min(NFrames, 3);
                int3 ExpansionGridSize = new int3(MinXSteps, MinYSteps, MinZSteps);
                int3 LocalGridSize = new int3(LocalGridX, LocalGridY, LocalGridZ);
                int LocalGridParams = (int)LocalGridSize.Elements() * 2;

                // Get wiggle weights for global and local shifts, will need latter in last iteration
                float[][] WiggleWeights = new CubicGrid(ExpansionGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                float[][] WiggleWeightsLocal = new CubicGrid(LocalGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                double[] StartParams = new double[ExpansionGridSize.Elements() * 2 + LocalGridParams];

                for (int m = 0; m < MaskExpansions; m++)
                {
                    bool LastIter = m == MaskExpansions - 1;
                    double[] LastAverage = null;

                    int ExpansionGridParams = (int)ExpansionGridSize.Elements() * 2;

                    #region Helper methods

                    Action<double[]> SetPositions = input =>
                    {
                        // Construct CubicGrids and get interpolated shift values.
                        CubicGrid AlteredGridX = new CubicGrid(ExpansionGridSize, input.Take(ExpansionGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                        CubicGrid AlteredGridY = new CubicGrid(ExpansionGridSize, input.Take(ExpansionGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                        float[] AlteredX = AlteredGridX.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                              new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                        float[] AlteredY = AlteredGridY.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                              new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                        // In last iteration, also model local motion
                        //if (LastIter)
                        {
                            CubicGrid AlteredGridLocalX = new CubicGrid(LocalGridSize, input.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                            CubicGrid AlteredGridLocalY = new CubicGrid(LocalGridSize, input.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                            float[] AlteredLocalX = AlteredGridLocalX.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                                  new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));
                            float[] AlteredLocalY = AlteredGridLocalY.GetInterpolatedNative(new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames),
                                                                                  new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                            for (int i = 0; i < AlteredX.Length; i++)
                            {
                                AlteredX[i] += AlteredLocalX[i];
                                AlteredY[i] += AlteredLocalY[i];
                            }
                        }
                        
                        // Finally, set the shift values in the device array.
                        float[] ShiftData = Shifts.GetHost(Intent.Write)[0];
                        Parallel.For(0, AlteredX.Length, i =>
                        {
                            ShiftData[i * 2] = AlteredX[i];// - CenterFrameOffsets[i % NPositions].X;
                            ShiftData[i * 2 + 1] = AlteredY[i];// - CenterFrameOffsets[i % NPositions].Y;
                        });
                    };

                    Action<double[]> DoAverage = input =>
                    {
                        if (LastAverage == null || input.Where((t, i) => t != LastAverage[i]).Any())
                        {
                            SetPositions(input);
                            GPU.ShiftGetAverage(Patches.GetDevice(Intent.Read),
                                                PatchesAverage.GetDevice(Intent.Write),
                                                ShiftFactors.GetDevice(Intent.Read),
                                                (uint)MaskLength,
                                                (uint)MaskSizes[m],
                                                Shifts.GetDevice(Intent.Read),
                                                (uint)NPositions,
                                                (uint)NFrames);

                            if (LastAverage == null)
                                LastAverage = new double[input.Length];
                            Array.Copy(input, LastAverage, input.Length);
                        }
                    };

                    #endregion

                    #region Eval and gradient methods

                    Func<double[], double> Eval = input =>
                    {
                        DoAverage(input);

                        float[] Diff = new float[NPositions * NFrames];
                        GPU.ShiftGetDiff(Patches.GetDevice(Intent.Read),
                                         PatchesAverage.GetDevice(Intent.Read),
                                         ShiftFactors.GetDevice(Intent.Read),
                                         (uint)MaskLength,
                                         (uint)MaskSizes[m],
                                         Shifts.GetDevice(Intent.Read),
                                         Diff,
                                         (uint)NPositions,
                                         (uint)NFrames);

                        for (int i = 0; i < Diff.Length; i++)
                            Diff[i] = Diff[i];

                        return Diff.Sum();
                    };

                    Func<double[], double[]> Grad = input =>
                    {
                        DoAverage(input);

                        float[] GradX = new float[NPositions * NFrames], GradY = new float[NPositions * NFrames];

                        float[] Diff = new float[NPositions * NFrames * 2];
                        GPU.ShiftGetGrad(Patches.GetDevice(Intent.Read),
                                         PatchesAverage.GetDevice(Intent.Read),
                                         ShiftFactors.GetDevice(Intent.Read),
                                         (uint)MaskLength,
                                         (uint)MaskSizes[m],
                                         Shifts.GetDevice(Intent.Read),
                                         Diff,
                                         (uint)NPositions,
                                         (uint)NFrames);
                        
                        for (int i = 0; i < GradX.Length; i++)
                        {
                            GradX[i] = Diff[i * 2];
                            GradY[i] = Diff[i * 2 + 1];
                        }

                        double[] Result = new double[input.Length];

                        Parallel.For(0, ExpansionGridParams / 2, i =>
                        {
                            Result[i * 2] = MathHelper.ReduceWeighted(GradX, WiggleWeights[i]);
                            Result[i * 2 + 1] = MathHelper.ReduceWeighted(GradY, WiggleWeights[i]);
                        });

                        //if (LastIter)
                            Parallel.For(0, LocalGridParams / 2, i =>
                            {
                                Result[ExpansionGridParams + i * 2] = MathHelper.ReduceWeighted(GradX, WiggleWeightsLocal[i]);
                                Result[ExpansionGridParams + i * 2 + 1] = MathHelper.ReduceWeighted(GradY, WiggleWeightsLocal[i]);
                            });

                        return Result;
                    };

                    #endregion

                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Grad);
                    Optimizer.MaxIterations = 10;
                    Optimizer.Minimize(StartParams);

                    // Anything should be quite centered anyway

                    //float MeanX = MathHelper.Mean(Optimizer.Solution.Where((v, i) => i % 2 == 0).Select(v => (float)v));
                    //float MeanY = MathHelper.Mean(Optimizer.Solution.Where((v, i) => i % 2 == 1).Select(v => (float)v));
                    //for (int i = 0; i < ExpansionGridSize.Elements(); i++)
                    //{
                    //    Optimizer.Solution[i * 2] -= MeanX;
                    //    Optimizer.Solution[i * 2 + 1] -= MeanY;
                    //}

                    // Store coarse values in grids.
                    GridMovementX = new CubicGrid(ExpansionGridSize, Optimizer.Solution.Take(ExpansionGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                    GridMovementY = new CubicGrid(ExpansionGridSize, Optimizer.Solution.Take(ExpansionGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());

                    //if (LastIter)
                    {
                        GridLocalX = new CubicGrid(LocalGridSize, Optimizer.Solution.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 0).Select(v => (float)v).ToArray());
                        GridLocalY = new CubicGrid(LocalGridSize, Optimizer.Solution.Skip(ExpansionGridParams).Take(LocalGridParams).Where((v, i) => i % 2 == 1).Select(v => (float)v).ToArray());
                    }

                    if (!LastIter)
                    {
                        // Refine sampling.
                        ExpansionGridSize = new int3(1, //(int)Math.Round((float)(ShiftGridX - MinXSteps) / (MaskExpansions - 1) * (m + 1) + MinXSteps),
                                                     1, //(int)Math.Round((float)(ShiftGridY - MinYSteps) / (MaskExpansions - 1) * (m + 1) + MinYSteps),
                                                     (int)Math.Round((float)Math.Min(ShiftGridZ, Math.Pow(3, m + 2))));
                        ExpansionGridParams = (int)ExpansionGridSize.Elements() * 2;

                        WiggleWeights = new CubicGrid(ExpansionGridSize).GetWiggleWeights(ShiftGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 0f));

                        // Resize the grids to account for finer sampling.
                        GridMovementX = GridMovementX.Resize(ExpansionGridSize);
                        GridMovementY = GridMovementY.Resize(ExpansionGridSize);

                        // Construct start parameters for next optimization iteration.
                        StartParams = new double[ExpansionGridParams + LocalGridParams];
                        for (int i = 0; i < ExpansionGridParams / 2; i++)
                        {
                            StartParams[i * 2] = GridMovementX.FlatValues[i];
                            StartParams[i * 2 + 1] = GridMovementY.FlatValues[i];
                        }
                        for (int i = 0; i < LocalGridParams / 2; i++)
                        {
                            StartParams[ExpansionGridParams + i * 2] = GridLocalX.FlatValues[i];
                            StartParams[ExpansionGridParams + i * 2 + 1] = GridLocalY.FlatValues[i];
                        }
                        // Local shifts will be initialized with 0 for last iteration
                    }
                }
            }

            #endregion

            // Center the global shifts
            {
                float2 AverageShift = new float2(MathHelper.Mean(GridMovementX.FlatValues),
                                                  MathHelper.Mean(GridMovementY.FlatValues));

                GridMovementX = new CubicGrid(GridMovementX.Dimensions, GridMovementX.FlatValues.Select(v => v - AverageShift.X).ToArray());
                GridMovementY = new CubicGrid(GridMovementY.Dimensions, GridMovementY.FlatValues.Select(v => v - AverageShift.Y).ToArray());
            }

            // Scale everything from (binned) pixels to Angstrom
            GridMovementX = new CubicGrid(GridMovementX.Dimensions, GridMovementX.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());
            GridMovementY = new CubicGrid(GridMovementY.Dimensions, GridMovementY.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());

            GridLocalX = new CubicGrid(GridLocalX.Dimensions, GridLocalX.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());
            GridLocalY = new CubicGrid(GridLocalY.Dimensions, GridLocalY.FlatValues.Select(v => v * (float)options.BinnedPixelSizeMean).ToArray());

            ShiftFactors.Dispose();
            Patches.Dispose();
            PatchesAverage.Dispose();
            Shifts.Dispose();

            OptionsMovement = options;

            // Calculate mean per-frame shift
            {
                float2[] Track = GetMotionTrack(new float2(0.5f, 0.5f), 1);
                float[] Diff = MathHelper.Diff(Track).Select(v => v.Length()).ToArray();
                MeanFrameMovement = (decimal)MathHelper.Mean(Diff.Take(Math.Max(1, Diff.Length / 3)));
            }

            SaveMeta();

            IsProcessing = false;
        }

        public float2 GetShiftFromPyramid(float3 coords, bool localOnly = false)
        {
            float2 Result = new float2(0, 0);

            Result.X = localOnly ? 0 : GridMovementX.GetInterpolated(coords);
            Result.Y = localOnly ? 0 : GridMovementY.GetInterpolated(coords);

            Result.X += GridLocalX.GetInterpolated(coords);
            Result.Y += GridLocalY.GetInterpolated(coords);

            for (int i = 0; i < PyramidShiftX.Count; i++)
            {
                Result.X += PyramidShiftX[i].GetInterpolated(coords);
                Result.Y += PyramidShiftY[i].GetInterpolated(coords);
            }

            return Result;
        }

        public float2[] GetShiftFromPyramid(float3[] coords, bool localOnly = false)
        {
            float2[] Result = new float2[coords.Length];

            if (!localOnly)
            {
                float[] X = GridMovementX.GetInterpolated(coords);
                float[] Y = GridMovementY.GetInterpolated(coords);
                for (int i = 0; i < Result.Length; i++)
                {
                    Result[i].X += X[i];
                    Result[i].Y += Y[i];
                }
            }

            {
                float[] X = GridLocalX.GetInterpolated(coords);
                float[] Y = GridLocalY.GetInterpolated(coords);
                for (int i = 0; i < Result.Length; i++)
                {
                    Result[i].X += X[i];
                    Result[i].Y += Y[i];
                }
            }

            for (int p = 0; p < PyramidShiftX.Count; p++)
            {
                float[] X = PyramidShiftX[p].GetInterpolated(coords);
                float[] Y = PyramidShiftY[p].GetInterpolated(coords);
                for (int i = 0; i < Result.Length; i++)
                {
                    Result[i].X += X[i];
                    Result[i].Y += Y[i];
                }
            }

            return Result;
        }

        public void ExportMovie(Image originalStack, ProcessingOptionsMovieExport options)
        {
            IsProcessing = true;

            #region Make sure all directories are there

            if (!Directory.Exists(AverageDir))
                Directory.CreateDirectory(AverageDir);

            if (options.DoStack && !Directory.Exists(ShiftedStackDir))
                Directory.CreateDirectory(ShiftedStackDir);

            if (options.DoDeconv && !Directory.Exists(DeconvolvedDir))
                Directory.CreateDirectory(DeconvolvedDir);

            #endregion

            #region Helper variables

            int3 Dims = originalStack.Dims;
            int FirstFrame = Math.Max(0, Math.Min(Dims.Z- 1, options.SkipFirstN));
            int LastFrameExclusive = Math.Min(Dims.Z, Dims.Z - options.SkipLastN);
            Dims.Z = LastFrameExclusive - FirstFrame;

            Task WriteAverageAsync = null;
            Task WriteDeconvAsync = null;
            Task WriteStackAsync = null;

            #endregion

            #region Warp, and get FTs of all relevant frames

            Image ShiftedStackFT = new Image(IntPtr.Zero, Dims, true, true);
            int PlanForw = GPU.CreateFFTPlan(Dims.Slice(), 1);

            Image WarpedFrame = new Image(IntPtr.Zero, Dims.Slice());

            float StepZ = 1f / Math.Max(originalStack.Dims.Z - 1, 1);

            for (int z = 0; z < Dims.Z; z++)
            {
                int2 DimsWarp = new int2(16);
                float3[] InterpPoints = new float3[DimsWarp.Elements()];
                for (int y = 0; y < DimsWarp.Y; y++)
                    for (int x = 0; x < DimsWarp.X; x++)
                        InterpPoints[y * DimsWarp.X + x] = new float3((float)x / (DimsWarp.X - 1), (float)y / (DimsWarp.Y - 1), (z + FirstFrame) * StepZ);

                float2[] WarpXY = GetShiftFromPyramid(InterpPoints);
                float[] WarpX = WarpXY.Select(v => v.X / (float)options.BinnedPixelSizeMean).ToArray();
                float[] WarpY = WarpXY.Select(v => v.Y / (float)options.BinnedPixelSizeMean).ToArray();

                GPU.WarpImage(originalStack.GetDeviceSlice(z + FirstFrame, Intent.Read),
                              WarpedFrame.GetDevice(Intent.Write),
                              new int2(Dims),
                              WarpX,
                              WarpY,
                              DimsWarp);

                GPU.FFT(WarpedFrame.GetDevice(Intent.Read),
                        ShiftedStackFT.GetDeviceSlice(z, Intent.Write),
                        Dims.Slice(),
                        1,
                        PlanForw);
            }

            WarpedFrame.Dispose();
            originalStack.FreeDevice();
            GPU.DestroyFFTPlan(PlanForw);

            #endregion

            #region In case shifted stack is needed, IFFT everything and async write to disk

            if (options.DoStack)
            {
                int PlanBack = GPU.CreateIFFTPlan(Dims.Slice(), 1);

                Image ShiftedStack = new Image(IntPtr.Zero, Dims);
                for (int i = 0; i < Dims.Z; i++)
                    GPU.IFFT(ShiftedStackFT.GetDeviceSlice(i, Intent.Read),
                             ShiftedStack.GetDeviceSlice(i, Intent.Write),
                             Dims.Slice(),
                             1,
                             PlanBack,
                             true);

                GPU.DestroyFFTPlan(PlanBack);
                ShiftedStack.FreeDevice();

                WriteStackAsync = new Task(() =>
                {
                    if (options.StackGroupSize <= 1)
                    {
                        ShiftedStack.WriteMRC(ShiftedStackPath, (float)options.BinnedPixelSizeMean, true);
                        ShiftedStack.Dispose();
                    }
                    else
                    {
                        int NGroups = ShiftedStack.Dims.Z / options.StackGroupSize;
                        Image GroupedStack = new Image(new int3(ShiftedStack.Dims.X, ShiftedStack.Dims.Y, NGroups));
                        float[][] GroupedStackData = GroupedStack.GetHost(Intent.Write);
                        float[][] ShiftedStackData = ShiftedStack.GetHost(Intent.Read);

                        Parallel.For(0, NGroups, g =>
                        {
                            for (int f = 0; f < options.StackGroupSize; f++)
                                GroupedStackData[g] = MathHelper.Add(GroupedStackData[g], ShiftedStackData[g * options.StackGroupSize + f]);
                        });

                        GroupedStack.WriteMRC(ShiftedStackPath, (float)options.BinnedPixelSizeMean, true);

                        GroupedStack.Dispose();
                        ShiftedStack.Dispose();
                    }
                });
                WriteStackAsync.Start();
            }

            #endregion

            #region Prepare spectral coordinates

            float PixelSize = (float)options.BinnedPixelSizeMean;
            float PixelDelta = (float)options.BinnedPixelSizeDelta;
            float PixelAngle = (float)options.PixelSizeAngle * Helper.ToRad;
            Image CTFCoordsWeighting = CTF.GetCTFCoords(new int2(Dims), new int2(Dims));
            Image Wiener = null;
            {
                float2[] CTFCoordsData = new float2[Dims.Slice().ElementsFFT()];
                Helper.ForEachElementFT(new int2(Dims), (x, y, xx, yy) =>
                {
                    float xs = xx / (float)Dims.X;
                    float ys = yy / (float)Dims.Y;
                    float r = (float)Math.Sqrt(xs * xs + ys * ys);
                    float angle = (float)Math.Atan2(yy, xx);
                    float CurrentPixelSize = PixelSize + PixelDelta * (float)Math.Cos(2f * (angle - PixelAngle));

                    CTFCoordsData[y * (Dims.X / 2 + 1) + x] = new float2(r / CurrentPixelSize, angle);
                });

                Image CTFCoords = new Image(CTFCoordsData, Dims.Slice(), true);
                CTFCoords.Dispose();

                if (options.DoDeconv)
                {
                    float[] CTF2D = CTF.Get2DFromScaledCoords(CTFCoordsData, false);
                    float HighPassNyquist = PixelSize * 2 / 100;
                    float Strength = (float)Math.Pow(10, 3 * (double)options.DeconvolutionStrength);
                    float Falloff = (float)options.DeconvolutionFalloff * 100 / PixelSize;

                    Helper.ForEachElementFT(new int2(Dims), (x, y, xx, yy) =>
                    {
                        float xs = xx / (float)Dims.X * 2;
                        float ys = yy / (float)Dims.Y * 2;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);

                        float HighPass = 1 - (float)Math.Cos(Math.Min(1, r / HighPassNyquist) * Math.PI);
                        float SNR = (float)Math.Exp(-r * Falloff) * Strength * HighPass;
                        float CTFVal = CTF2D[y * (Dims.X / 2 + 1) + x];
                        CTF2D[y * (Dims.X / 2 + 1) + x] = CTFVal / (CTFVal * CTFVal + 1 / SNR);
                    });

                    Wiener = new Image(CTF2D, Dims.Slice(), true);
                }
            }

            #endregion
            Image Weights = new Image(Dims.Slice(), true, false);
            Weights.Fill(1e-15f);

            #region Apply spectral filter to every frame in stack

            for (int nframe = FirstFrame; nframe < LastFrameExclusive; nframe++)
            {
                Image PS = new Image(Dims.Slice(), true);
                PS.Fill(1f);

                #region Apply motion blur filter.
                /*{
                    float StartZ = (nframe - 0.5f) * StepZ;
                    float StopZ = (nframe + 0.5f) * StepZ;

                    float2[] Shifts = new float2[21];
                    for (int z = 0; z < Shifts.Length; z++)
                    {
                        float zp = StartZ + (StopZ - StartZ) / (Shifts.Length - 1) * z;
                        Shifts[z] = new float2(CollapsedMovementX.GetInterpolated(new float3(0.5f, 0.5f, zp)),
                                               CollapsedMovementY.GetInterpolated(new float3(0.5f, 0.5f, zp)));
                    }
                    // Center the shifts around 0
                    float2 ShiftMean = MathHelper.Mean(Shifts);
                    Shifts = Shifts.Select(v => v - ShiftMean).ToArray();

                    Image MotionFilter = new Image(IntPtr.Zero, Dims.Slice(), true);
                    GPU.CreateMotionBlur(MotionFilter.GetDevice(Intent.Write), 
                                         MotionFilter.Dims, 
                                         Helper.ToInterleaved(Shifts.Select(v => new float3(v.X, v.Y, 0)).ToArray()), 
                                         (uint)Shifts.Length, 
                                         1);
                    PS.Multiply(MotionFilter);
                    //MotionFilter.WriteMRC("motion.mrc");
                    MotionFilter.Dispose();
                }*/
                #endregion

                // Apply dose weighting.
                {
                    float3 NikoConst = new float3(0.245f, -1.665f, 2.81f);

                    // Niko's formula expects e-/A2/frame.

                    Image DoseImage = new Image(IntPtr.Zero, Dims.Slice(), true);
                    //GPU.DoseWeighting(CTFFreq.GetDevice(Intent.Read),
                    //                  DoseImage.GetDevice(Intent.Write),
                    //                  (uint)DoseImage.ElementsSliceComplex,
                    //                  new[] { (float)options.DosePerAngstromFrame * nframe, (float)options.DosePerAngstromFrame * (nframe + 0.5f) },
                    //                  NikoConst,
                    //                  options.Voltage > 250 ? 1 : 0.8f, // It's only defined for 300 and 200 kV, but let's not throw an exception
                    //                  1);
                    CTF CTFBfac = new CTF()
                    {
                        PixelSize = (decimal)PixelSize,
                        Defocus = 0,
                        Amplitude = 1,
                        Cs = 0,
                        Cc = 0,
                        IllumAngle = 0,
                        Bfactor = -(decimal)((float)options.DosePerAngstromFrame * (nframe + 0.5f) * 3)
                    };
                    GPU.CreateCTF(DoseImage.GetDevice(Intent.Write),
                                  CTFCoordsWeighting.GetDevice(Intent.Read),
                                  (uint)CTFCoordsWeighting.ElementsSliceComplex,
                                  new[] { CTFBfac.ToStruct() },
                                  false,
                                  1);

                    PS.Multiply(DoseImage);
                    //DoseImage.WriteMRC("dose.mrc");
                    DoseImage.Dispose();
                }
                //PS.WriteMRC("ps.mrc");

                GPU.MultiplyComplexSlicesByScalar(ShiftedStackFT.GetDeviceSlice(nframe - FirstFrame, Intent.Read),
                                                  PS.GetDevice(Intent.Read),
                                                  ShiftedStackFT.GetDeviceSlice(nframe - FirstFrame, Intent.Write),
                                                  PS.ElementsSliceReal,
                                                  1);
                
                Weights.Add(PS);

                PS.Dispose();
            }

            #endregion

            Image AverageFT = new Image(Dims.Slice(), true, true);
            GPU.ReduceAdd(ShiftedStackFT.GetDevice(Intent.Read),
                          AverageFT.GetDevice(Intent.Write),
                          (uint)ShiftedStackFT.ElementsSliceReal,
                          (uint)ShiftedStackFT.Dims.Z,
                          1);
            ShiftedStackFT.Dispose();

            //AverageFT.Divide(Weights);
            //AverageFT.WriteMRC("averageft.mrc");
            //Weights.WriteMRC("weights.mrc");
            Weights.Dispose();

            Image Average;
            if (options.DoAverage)
            {
                Average = AverageFT.AsIFFT(false, 0, true);

                // Previous division by weight sum brought values to stack average, multiply by number of frame to go back to sum

                Average.Multiply(Dims.Z);
                Average.FreeDevice();

                // Write average async to disk
                WriteAverageAsync = new Task(() =>
                {
                    Average.WriteMRC(AveragePath, (float)options.BinnedPixelSizeMean, true);
                    Average.Dispose();
                    OnAverageChanged();
                });
                WriteAverageAsync.Start();
            }

            Image Deconvolved = null;
            if (options.DoDeconv)
            {
                AverageFT.Multiply(Wiener);
                Wiener.Dispose();
                Deconvolved = AverageFT.AsIFFT(false, 0, true);
                Deconvolved.Multiply(Dims.Z);   // It's only the average at this point, needs to be sum

                // Write deconv async to disk
                WriteDeconvAsync = new Task(() =>
                {
                    Deconvolved.WriteMRC(DeconvolvedPath, (float)options.BinnedPixelSizeMean, true);
                    Deconvolved.Dispose();
                });
                WriteDeconvAsync.Start();
            }

            AverageFT.Dispose();
            CTFCoordsWeighting.Dispose();
            
            using (TextWriter Writer = File.CreateText(AverageDir + RootName + "_ctffind3.log"))
            {
                decimal Mag = 50000M / options.BinnedPixelSizeMean;

                Writer.WriteLine("CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]");
                Writer.WriteLine($"{CTF.Cs} {CTF.Voltage} {CTF.Amplitude} {Mag} {50000}");
                
                Writer.WriteLine($"{(CTF.Defocus + CTF.DefocusDelta / 2M) * 1e4M} {(CTF.Defocus - CTF.DefocusDelta / 2M) * 1e4M} {CTF.DefocusAngle} {1.0} {CTF.PhaseShift * 180M} Final Values");
            }

            // Wait for all async IO to finish
            WriteStackAsync?.Wait();
            WriteDeconvAsync?.Wait();
            WriteAverageAsync?.Wait();

            OptionsMovieExport = options;
            SaveMeta();

            IsProcessing = false;
        }

        public void CreateThumbnail(int size, float stddevRange)
        {
            if (!File.Exists(AveragePath))
                return;

            Directory.CreateDirectory(ThumbnailsDir);

            Image Average = Image.FromFile(AveragePath);
            float ScaleFactor = (float)size / Math.Max(Average.Dims.X, Average.Dims.Y);
            int2 DimsScaled = new int2(new float2(Average.Dims.X, Average.Dims.Y) * ScaleFactor + 0.5f) / 2 * 2;

            Image AverageScaled = Average.AsScaled(DimsScaled);
            Average.Dispose();

            float2 MeanStd = MathHelper.MeanAndStd(AverageScaled.GetHost(Intent.Read)[0]);
            float Min = MeanStd.X - MeanStd.Y * stddevRange;
            float Range = MeanStd.Y * stddevRange * 2;

            AverageScaled.TransformValues(v => (v - Min) / Range * 255);

            AverageScaled.WritePNG(ThumbnailsPath);
            AverageScaled.Dispose();
        }

        public float2[] GetMotionTrack(float2 position, int samples, bool localOnly = false)
        {
            if (OptionsMovement == null || OptionsMovement.Dimensions.Z <= 1)
                return null;

            int NFrames = (int)OptionsMovement.Dimensions.Z;
            float2[] Result = new float2[NFrames * samples];

            float StepZ = 1f / Math.Max(NFrames * samples - 1, 1);
            for (int z = 0; z < NFrames * samples; z++)
                Result[z] = GetShiftFromPyramid(new float3(position.X, position.Y, z * StepZ), localOnly);

            return Result;
        }

        public void ExportParticles(Image originalStack, float2[] positions, ProcessingOptionsParticlesExport options)
        {
            IsProcessing = true;

            options.CorrectAnisotropy &= options.PixelSizeDelta != 0;
            options.PreflipPhases &= OptionsCTF != null;

            #region Make sure all directories are there

            if (options.DoAverage && !Directory.Exists(ParticlesDir))
                Directory.CreateDirectory(ParticlesDir);

            if (options.DoStack && !Directory.Exists(ParticleMoviesDir))
                Directory.CreateDirectory(ParticleMoviesDir);

            #endregion

            #region Helper variables

            int NParticles = positions.Length;

            int3 DimsMovie = originalStack.Dims;
            int FirstFrame = Math.Max(0, Math.Min(DimsMovie.Z - 1, options.SkipFirstN));
            int LastFrameExclusive = Math.Min(DimsMovie.Z, DimsMovie.Z - options.SkipLastN);
            DimsMovie.Z = LastFrameExclusive - FirstFrame;

            int3 DimsParticle = new int3(options.BoxSize, options.BoxSize, NParticles);
            int3 DimsPreflip = DimsParticle.MultXY(2);
            int3 DimsExtraction = options.PreflipPhases ? DimsPreflip : DimsParticle;

            Task WriteAverageAsync = null;
            Task WriteStackAsync = null;

            float PixelSize = (float)options.BinnedPixelSizeMean;
            float PixelDelta = (float)options.BinnedPixelSizeDelta;
            float PixelAngle = (float)options.PixelSizeAngle * Helper.ToRad;

            #endregion

            #region Figure out where to extract, and how much to shift afterwards

            float3[] ParticleCenters = positions.Select(p => new float3(p) / (float)options.BinnedPixelSizeMean).ToArray(); // From Angstrom to binned pixels

            float3[][] ParticleOrigins = Helper.ArrayOfFunction(z =>
            {
                float Z = (z + FirstFrame) / (float)Math.Max(1, originalStack.Dims.Z - 1);
                return ParticleCenters.Select(p =>
                {
                    float2 LocalShift = GetShiftFromPyramid(new float3(p.X * PixelSize / options.Dimensions.X,                  // ParticleCenters are in binned pixels, Dimensions are in Angstrom
                                                                       p.Y * PixelSize / options.Dimensions.Y, Z)) / PixelSize; // Shifts are in physical Angstrom, convert to binned pixels
                    return new float3(p.X - LocalShift.X - DimsExtraction.X / 2, p.Y - LocalShift.Y - DimsExtraction.Y / 2, 0);
                }).ToArray();
            }, DimsMovie.Z);

            int3[][] ParticleIntegerOrigins = ParticleOrigins.Select(a => a.Select(p => new int3(p.Floor())).ToArray()).ToArray();
            float3[][] ParticleResidualShifts = Helper.ArrayOfFunction(z =>
                                                                       Helper.ArrayOfFunction(i =>
                                                                                              new float3(ParticleIntegerOrigins[z][i]) - ParticleOrigins[z][i],
                                                                                              NParticles),
                                                                       ParticleOrigins.Length);

            #endregion

            #region Pre-calc phase flipping and dose weighting

            Image CTFSign = null;
            Image CTFCoords = CTF.GetCTFCoords(DimsExtraction.X, DimsExtraction.X);
            Image[] DoseWeights = null;

            if (options.PreflipPhases)
            {
                CTFStruct[] Params = positions.Select(p =>
                {
                    CTF Local = CTF.GetCopy();
                    Local.Defocus = (decimal)GridCTF.GetInterpolated(new float3(p.X / options.Dimensions.X, p.Y / options.Dimensions.Y, 0));
                    return Local.ToStruct();
                }).ToArray();

                CTFSign = new Image(DimsExtraction, true);
                GPU.CreateCTF(CTFSign.GetDevice(Intent.Write),
                              CTFCoords.GetDevice(Intent.Read),
                              (uint)CTFCoords.ElementsSliceComplex,
                              Params,
                              false,
                              (uint)CTFSign.Dims.Z);

                GPU.Sign(CTFSign.GetDevice(Intent.Read),
                         CTFSign.GetDevice(Intent.Write),
                         CTFSign.ElementsReal);
            }

            if (options.DosePerAngstromFrame > 0)
            {

                DoseWeights = Helper.ArrayOfFunction(z =>
                {
                    Image Weights = new Image(IntPtr.Zero, DimsExtraction.Slice(), true);

                    CTF CTFBfac = new CTF
                    {
                        PixelSize = (decimal)PixelSize,
                        Defocus = 0,
                        Amplitude = 1,
                        Cs = 0,
                        Cc = 0,
                        IllumAngle = 0,
                        Bfactor = -(decimal)((float)options.DosePerAngstromFrame * (z + FirstFrame + 0.5f) * 3)
                    };
                    GPU.CreateCTF(Weights.GetDevice(Intent.Write),
                                  CTFCoords.GetDevice(Intent.Read),
                                  (uint)CTFCoords.ElementsSliceComplex,
                                  new[] { CTFBfac.ToStruct() },
                                  false,
                                  1);

                    return Weights;
                }, DimsMovie.Z);
            }

            CTFCoords.Dispose();

            #endregion

            #region Make FFT plans and memory

            Image Extracted = new Image(IntPtr.Zero, DimsExtraction);
            Image ExtractedFT = new Image(IntPtr.Zero, DimsExtraction, true, true);

            Image AverageFT = options.DoAverage ? new Image(DimsExtraction, true, true) : null;
            Image Stack = options.DoStack ? new Image(new int3(DimsParticle.X, DimsParticle.Y, DimsParticle.Z * DimsMovie.Z)) : null;
            float[][] StackData = options.DoStack ? Stack.GetHost(Intent.Write) : null;
            
            int PlanForw = GPU.CreateFFTPlan(DimsExtraction.Slice(), (uint)NParticles);
            int PlanBack = GPU.CreateIFFTPlan(DimsExtraction.Slice(), (uint)NParticles);

            #endregion

            #region Extract and process everything

            for (int nframe = 0; nframe < DimsMovie.Z; nframe++)
            {
                GPU.Extract(originalStack.GetDeviceSlice(nframe + FirstFrame, Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            DimsMovie.Slice(),
                            DimsExtraction.Slice(),
                            Helper.ToInterleaved(ParticleIntegerOrigins[nframe]),
                            (uint)NParticles);

                if (options.CorrectAnisotropy && options.DoStack)
                    GPU.CorrectMagAnisotropy(Extracted.GetDevice(Intent.Read),
                                             new int2(DimsExtraction),
                                             Extracted.GetDevice(Intent.Write),
                                             new int2(DimsExtraction),
                                             (float)(options.PixelSizeMean + options.PixelSizeDelta / 2),
                                             (float)(options.PixelSizeMean - options.PixelSizeDelta / 2),
                                             (float)options.PixelSizeAngle * Helper.ToRad,
                                             1,
                                             (uint)NParticles);

                GPU.FFT(Extracted.GetDevice(Intent.Read),
                        ExtractedFT.GetDevice(Intent.Write),
                        DimsExtraction.Slice(),
                        (uint)NParticles,
                        PlanForw);

                if (options.PreflipPhases)
                    ExtractedFT.Multiply(CTFSign);

                ExtractedFT.ShiftSlices(ParticleResidualShifts[nframe]);

                if (options.DoStack)    // IFFT frames, crop, normalize, and put them in the big stack
                {
                    GPU.IFFT(ExtractedFT.GetDevice(Intent.Read),
                             Extracted.GetDevice(Intent.Write),
                             DimsExtraction.Slice(),
                             (uint)NParticles,
                             PlanBack,
                             true);

                    Image Cropped = DimsParticle != DimsExtraction ? Extracted.AsPadded(new int2(DimsParticle)) : Extracted;

                    if (options.Normalize)
                        GPU.NormParticles(Cropped.GetDevice(Intent.Read),
                                          Cropped.GetDevice(Intent.Write),
                                          DimsParticle.Slice(),
                                          (uint)(options.Diameter / PixelSize / 2),
                                          options.Invert,
                                          (uint)NParticles);
                    else if (options.Invert)
                        Cropped.Multiply(-1f);

                    float[][] CroppedData = Cropped.GetHost(Intent.Read);
                    for (int p = 0; p < NParticles; p++)
                        StackData[nframe * NParticles + p] = CroppedData[p];

                    if (DimsParticle != DimsExtraction)
                        Cropped.Dispose();
                }

                if (options.DoAverage)
                {
                    if (options.DosePerAngstromFrame > 0)
                        ExtractedFT.MultiplySlices(DoseWeights[nframe]);

                    AverageFT.Add(ExtractedFT);
                }
            }

            if (options.DoStack)
            {
                WriteStackAsync = new Task(() =>
                {
                    Stack.WriteMRC(ParticleMoviesDir + RootName + options.Suffix + ".mrcs", (float)options.BinnedPixelSizeMean);
                    Stack.Dispose();
                });
                WriteStackAsync.Start();
            }

            if (options.DoAverage)
            {
                Image Average = AverageFT.AsIFFT(false, PlanBack, true);
                AverageFT.Dispose();

                if (!options.DoStack && options.CorrectAnisotropy) // In case of DoStack, the individual frames have already been corrected
                    GPU.CorrectMagAnisotropy(Average.GetDevice(Intent.Read),
                                             new int2(DimsExtraction),
                                             Average.GetDevice(Intent.Write),
                                             new int2(DimsExtraction),
                                             (float)(options.PixelSizeMean + options.PixelSizeDelta / 2),
                                             (float)(options.PixelSizeMean - options.PixelSizeDelta / 2),
                                             (float)options.PixelSizeAngle * Helper.ToRad,
                                             1,
                                             (uint)NParticles);

                Image AverageCropped = Average.AsPadded(new int2(DimsParticle));
                Average.Dispose();

                #region Subtract background plane

                AverageCropped.FreeDevice();
                float[][] AverageData = AverageCropped.GetHost(Intent.ReadWrite);
                for (int p = 0; p < NParticles; p++)
                {
                    float[] ParticleData = AverageData[p];
                    float[] Background = MathHelper.FitAndGeneratePlane(ParticleData, new int2(AverageCropped.Dims));
                    for (int i = 0; i < ParticleData.Length; i++)
                        ParticleData[i] -= Background[i];
                }

                #endregion

                if (options.Normalize)
                    GPU.NormParticles(AverageCropped.GetDevice(Intent.Read),
                                      AverageCropped.GetDevice(Intent.Write),
                                      DimsParticle.Slice(),
                                      (uint)(options.Diameter / PixelSize / 2),
                                      options.Invert,
                                      (uint)NParticles);
                else if (options.Invert)
                    AverageCropped.Multiply(-1f);

                AverageCropped.FreeDevice();

                WriteAverageAsync = new Task(() =>
                {
                    AverageCropped.WriteMRC(ParticlesDir + RootName + options.Suffix + ".mrcs", (float)options.BinnedPixelSizeMean, true);
                    AverageCropped.Dispose();
                });
                WriteAverageAsync.Start();
            }

            #endregion

            #region Clean up

            Extracted.Dispose();
            ExtractedFT.Dispose();
            GPU.DestroyFFTPlan(PlanForw);
            GPU.DestroyFFTPlan(PlanBack);

            CTFSign?.Dispose();
            if (DoseWeights != null)
                foreach (var doseWeight in DoseWeights)
                    doseWeight.Dispose();

            #endregion

            // Wait for all async IO to finish
            WriteStackAsync?.Wait();
            WriteAverageAsync?.Wait();

            OptionsParticlesExport = options;
            SaveMeta();

            IsProcessing = false;
        }

        public void MatchFull(Image originalStack, ProcessingOptionsFullMatch options, Image template, Func<int3, int, string, bool> progressCallback)
        {
            bool IsCanceled = false;
            if (!Directory.Exists(MatchingDir))
                Directory.CreateDirectory(MatchingDir);

            string NameWithRes = RootName + $"_{options.BinnedPixelSizeMean:F2}Apx";

            float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();

            Image CorrImage = null;
            Image AngleImage = null;
            float[] CorrData;
            float[] AngleData;

            GPU.CheckGPUExceptions();

            #region Dimensions

            int SizeSub = options.SubPatchSize;
            int SizeSubPadded = SizeSub * 2;
            int SizeParticle = (int)(options.TemplateDiameter / options.BinnedPixelSizeMean);
            int SizeUseful = Math.Min(SizeSub / 2, SizeSub - SizeParticle * 2);// Math.Min(SizeSub - SizeParticle, SizeSub / 2);
            if (SizeUseful < 2)
                throw new DimensionMismatchException("Particle diameter is bigger than the box.");

            int3 DimsMicrographCropped = new int3(originalStack.Dims.X, originalStack.Dims.Y, 1);

            int3 Grid = (DimsMicrographCropped - SizeParticle + SizeUseful - 1) / SizeUseful;
            Grid.Z = 1;
            List<float3> GridCoords = new List<float3>();
            for (int y = 0; y < Grid.Y; y++)
                for (int x = 0; x < Grid.X; x++)
                    GridCoords.Add(new float3(x * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                              y * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                              0));

            int3 DimsExtraction = new int3(SizeSubPadded, SizeSubPadded, GridCoords.Count);
            int3 DimsParticle = new int3(SizeSub, SizeSub, GridCoords.Count);

            progressCallback?.Invoke(Grid, 0, "Preparing...");

            #endregion

            #region Figure out where to extract, and how much to shift afterwards

            float3[] ParticleCenters = GridCoords.Select(p => new float3(p)).ToArray();

            float3[][] ParticleOrigins = Helper.ArrayOfFunction(z =>
            {
                float Z = z / (float)Math.Max(1, originalStack.Dims.Z - 1);
                return ParticleCenters.Select(p =>
                {
                    float2 LocalShift = GetShiftFromPyramid(new float3(p.X / DimsMicrographCropped.X, p.Y / DimsMicrographCropped.Y, Z)) / (float)options.BinnedPixelSizeMean; // Shifts are in physical Angstrom, convert to binned pixels
                    return new float3(p.X - LocalShift.X - SizeSubPadded / 2, p.Y - LocalShift.Y - SizeSubPadded / 2, 0);
                }).ToArray();
            }, originalStack.Dims.Z);

            int3[][] ParticleIntegerOrigins = ParticleOrigins.Select(a => a.Select(p => new int3(p.Floor())).ToArray()).ToArray();
            float3[][] ParticleResidualShifts = Helper.ArrayOfFunction(z =>
                                                                           Helper.ArrayOfFunction(i =>
                                                                                                      new float3(ParticleIntegerOrigins[z][i]) - ParticleOrigins[z][i],
                                                                                                  GridCoords.Count),
                                                                       ParticleOrigins.Length);

            #endregion

            #region CTF, phase flipping & dose weighting

            Image CTFCoords = CTF.GetCTFCoords(SizeSub, (int)(SizeSub * options.DownsampleFactor));
            Image CTFCoordsPadded = CTF.GetCTFCoords(SizeSub * 2, (int)(SizeSub * 2 * options.DownsampleFactor));
            Image[] DoseWeights = null;
            
            #region Dose

            if (options.DosePerAngstromFrame > 0)
            {
                float3 NikoConst = new float3(0.245f, -1.665f, 2.81f);
                Image CTFFreq = CTFCoordsPadded.AsReal();

                GPU.CheckGPUExceptions();

                DoseWeights = Helper.ArrayOfFunction(z =>
                {
                    Image Weights = new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, 1), true);

                    GPU.DoseWeighting(CTFFreq.GetDevice(Intent.Read),
                                      Weights.GetDevice(Intent.Write),
                                      (uint)Weights.ElementsSliceComplex,
                                      new[] { (float)options.DosePerAngstromFrame * z, (float)options.DosePerAngstromFrame * (z + 1) },
                                      NikoConst,
                                      options.Voltage > 250 ? 1 : 0.8f, // It's only defined for 300 and 200 kV, but let's not throw an exception
                                      1);

                    return Weights;
                }, originalStack.Dims.Z);

                GPU.CheckGPUExceptions();

                Image WeightSum = new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, 1), true);
                WeightSum.Fill(1e-15f);
                for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
                    WeightSum.Add(DoseWeights[nframe]);
                WeightSum.Multiply(1f / originalStack.Dims.Z);

                GPU.CheckGPUExceptions();

                for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
                {
                    DoseWeights[nframe].Divide(WeightSum);
                    //DoseWeights[nframe].WriteMRC($"d_doseweights_{GPU.GetDevice()}_{nframe}.mrc", true);
                }

                GPU.CheckGPUExceptions();

                WeightSum.Dispose();
                CTFFreq.Dispose();

                GPU.CheckGPUExceptions();
            }

            GPU.CheckGPUExceptions();

            #endregion

            #region Create CTF for themplate and padded phase flipping

            Image ExperimentalCTF = new Image(new int3(SizeSub, SizeSub, 1), true);
            Image ExperimentalCTFPadded = new Image(new int3(SizeSub * 2, SizeSub * 2, 1), true);
            CTF CTFParams = CTF.GetCopy();

            GPU.CreateCTF(ExperimentalCTF.GetDevice(Intent.Write),
                          CTFCoords.GetDevice(Intent.Read),
                          (uint)CTFCoords.ElementsComplex,
                          new[] { CTFParams.ToStruct() },
                          false,
                          1);
            ExperimentalCTF.Abs();

            GPU.CreateCTF(ExperimentalCTFPadded.GetDevice(Intent.Write),
                          CTFCoordsPadded.GetDevice(Intent.Read),
                          (uint)CTFCoordsPadded.ElementsComplex,
                          new[] { CTFParams.ToStruct() },
                          false,
                          1);
            ExperimentalCTFPadded.Sign();

            #endregion

            #endregion

            #region Whiten spectrum in images

            Image FlatteningFactors = new Image(new int3(SizeSub, SizeSub, 1), true);
            FlatteningFactors.Fill(1f);

            if (options.WhitenSpectrum)
            {
                progressCallback?.Invoke(Grid, 0, "Whitening spectral noise...");

                Image OriginalStackFlat = originalStack.AsSpectrumFlattened(false, 0.99f, 256);
                float[] AS1D = originalStack.AsAmplitudes1D(false, 0.99f, SizeSub / 2);
                originalStack.FreeDevice();
                originalStack = OriginalStackFlat;

                float[] FlatteningFactorsData = FlatteningFactors.GetHost(Intent.Write)[0];
                Helper.ForEachElementFT(new int2(FlatteningFactors.Dims), (x, y, xx, yy) =>
                {
                    int R = (int)((float)Math.Sqrt(xx * xx + yy * yy) / (SizeSub / 2) * AS1D.Length);
                    R = Math.Min(AS1D.Length - 1, R);

                    FlatteningFactorsData[y * (SizeSub / 2 + 1) + x] = AS1D[R] > 0 ? 1 / AS1D[R] : 0;
                });
            }

            ExperimentalCTF.MultiplySlices(FlatteningFactors);
            FlatteningFactors.Dispose();

            #endregion

            #region Extract and preprocess all patches

            Image AllPatchesFT;
            {
                Image AverageFT = new Image(DimsExtraction, true, true);
                Image Extracted = new Image(IntPtr.Zero, DimsExtraction);
                Image ExtractedFT = new Image(IntPtr.Zero, DimsExtraction, true, true);

                int PlanForw = GPU.CreateFFTPlan(DimsExtraction.Slice(), (uint)GridCoords.Count);
                int PlanBack = GPU.CreateIFFTPlan(DimsExtraction.Slice(), (uint)GridCoords.Count);

                for (int nframe = 0; nframe < originalStack.Dims.Z; nframe++)
                {
                    GPU.Extract(originalStack.GetDeviceSlice(nframe, Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                originalStack.Dims.Slice(),
                                DimsExtraction.Slice(),
                                Helper.ToInterleaved(ParticleIntegerOrigins[nframe]),
                                (uint)GridCoords.Count);

                    GPU.FFT(Extracted.GetDevice(Intent.Read),
                            ExtractedFT.GetDevice(Intent.Write),
                            DimsExtraction.Slice(),
                            (uint)GridCoords.Count,
                            PlanForw);

                    ExtractedFT.MultiplySlices(ExperimentalCTFPadded);

                    ExtractedFT.ShiftSlices(ParticleResidualShifts[nframe]);

                    if (options.DosePerAngstromFrame > 0)
                        ExtractedFT.MultiplySlices(DoseWeights[nframe]);

                    AverageFT.Add(ExtractedFT);
                }

                Image Average = AverageFT.AsIFFT(false, PlanBack, true);
                AverageFT.Dispose();

                Image AllPatches = Average.AsPadded(new int2(DimsParticle));
                Average.Dispose();

                GPU.Normalize(AllPatches.GetDevice(Intent.Read),
                              AllPatches.GetDevice(Intent.Write),
                              (uint)AllPatches.ElementsSliceReal,
                              (uint)GridCoords.Count);
                if (options.Invert)
                    AllPatches.Multiply(-1f);

                AllPatchesFT = AllPatches.AsFFT();
                AllPatches.Dispose();

                GPU.DestroyFFTPlan(PlanBack);
                GPU.DestroyFFTPlan(PlanForw);
                Extracted.Dispose();
                ExtractedFT.Dispose();
            }

            #endregion

            originalStack.FreeDevice();

            #region Get correlation and angles

            //if (false)
            {
                #region Scale and pad/crop the template to the right size, create projector

                progressCallback?.Invoke(Grid, 0, "Preparing template...");

                Projector ProjectorReference;
                {
                    int SizeBinned = (int)Math.Round(template.Dims.X * (options.TemplatePixel / options.BinnedPixelSizeMean) / 2) * 2;

                    Image TemplateScaled = template.AsScaled(new int3(SizeBinned));
                    template.FreeDevice();

                    GPU.SphereMask(TemplateScaled.GetDevice(Intent.Read),
                                   TemplateScaled.GetDevice(Intent.Write),
                                   TemplateScaled.Dims,
                                   SizeParticle / 2,
                                   Math.Max(5, 20 / (float)options.BinnedPixelSizeMean),
                                   1);

                    Image TemplatePadded = TemplateScaled.AsPadded(new int3(SizeSub));
                    TemplateScaled.Dispose();

                    ProjectorReference = new Projector(TemplatePadded, 1, 2);
                    TemplatePadded.Dispose();
                    ProjectorReference.PutTexturesOnDevice();
                }

                #endregion

                CorrData = new float[DimsMicrographCropped.ElementsSlice()];
                AngleData = new float[DimsMicrographCropped.ElementsSlice()];

                progressCallback?.Invoke(Grid, 0, "Matching...");

                int BatchSize = 1;
                for (int b = 0; b < GridCoords.Count; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, GridCoords.Count - b);

                    #region Perform correlation

                    Image BestCorrelation = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, CurBatch));
                    Image BestAngle = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, CurBatch));

                    GPU.CorrelateSubTomos(ProjectorReference.t_DataRe,
                                          ProjectorReference.t_DataIm,
                                          ProjectorReference.Oversampling,
                                          ProjectorReference.Data.Dims,
                                          AllPatchesFT.GetDeviceSlice(b, Intent.Read),
                                          ExperimentalCTF.GetDevice(Intent.Read),
                                          new int3(SizeSub, SizeSub, 1),
                                          (uint)CurBatch,
                                          Helper.ToInterleaved(HealpixAngles),
                                          (uint)HealpixAngles.Length,
                                          SizeParticle / 2,
                                          BestCorrelation.GetDevice(Intent.Write),
                                          BestAngle.GetDevice(Intent.Write));

                    #endregion

                    #region Put correlation values and best angle IDs back into the large volume

                    for (int st = 0; st < CurBatch; st++)
                    {
                        Image ThisCorrelation = new Image(BestCorrelation.GetDeviceSlice(st, Intent.Read), new int3(SizeSub, SizeSub, 1));
                        Image CroppedCorrelation = ThisCorrelation.AsPadded(new int2(SizeUseful));

                        Image ThisAngle = new Image(BestAngle.GetDeviceSlice(st, Intent.Read), new int3(SizeSub, SizeSub, 1));
                        Image CroppedAngle = ThisAngle.AsPadded(new int2(SizeUseful));

                        float[] SubCorr = CroppedCorrelation.GetHostContinuousCopy();
                        float[] SubAngle = CroppedAngle.GetHostContinuousCopy();
                        int3 Origin = new int3(GridCoords[b + st]) - SizeUseful / 2;
                        for (int y = 0; y < SizeUseful; y++)
                        {
                            int yVol = Origin.Y + y;
                            if (yVol >= DimsMicrographCropped.Y - SizeParticle / 2)
                                continue;

                            for (int x = 0; x < SizeUseful; x++)
                            {
                                int xVol = Origin.X + x;
                                if (xVol >= DimsMicrographCropped.X - SizeParticle / 2)
                                    continue;

                                CorrData[yVol * DimsMicrographCropped.X + xVol] = SubCorr[y * SizeUseful + x];// / (SizeSub * SizeSub);
                                AngleData[yVol * DimsMicrographCropped.X + xVol] = SubAngle[y * SizeUseful + x];
                            }
                        }

                        CroppedCorrelation.Dispose();
                        ThisCorrelation.Dispose();
                        CroppedAngle.Dispose();
                        ThisAngle.Dispose();
                    }

                    #endregion

                    BestCorrelation.Dispose();
                    BestAngle.Dispose();

                    if (progressCallback != null)
                        IsCanceled = progressCallback(Grid, b + CurBatch, "Matching...");
                }

                #region Postflight

                CTFCoords.Dispose();
                CTFCoordsPadded.Dispose();
                ExperimentalCTF.Dispose();
                ExperimentalCTFPadded.Dispose();
                AllPatchesFT.Dispose();
                ProjectorReference.Dispose();

                if (options.Supersample > 1)
                {
                    progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Looking for sub-pixel peaks...");

                    Image NormalSampled = new Image(CorrData, DimsMicrographCropped);
                    Image SuperSampled = new Image(NormalSampled.GetDevice(Intent.Read), NormalSampled.Dims);

                    GPU.SubpixelMax(NormalSampled.GetDevice(Intent.Read),
                                    SuperSampled.GetDevice(Intent.Write),
                                    NormalSampled.Dims,
                                    options.Supersample);

                    CorrData = SuperSampled.GetHost(Intent.Read)[0];

                    NormalSampled.Dispose();
                    SuperSampled.Dispose();
                }

                //if (options.KeepOnlyFullVoxels)
                {
                    progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Trimming...");
                    
                    float Margin = (float)options.TemplateDiameter / (float)options.BinnedPixelSizeMean;
                    
                    for (int y = 0; y < DimsMicrographCropped.Y; y++)
                        for (int x = 0; x < DimsMicrographCropped.X; x++)
                        {
                            if (x < Margin || y < Margin ||
                                x > DimsMicrographCropped.X - Margin ||
                                y > DimsMicrographCropped.Y - Margin)
                            {
                                CorrData[y * DimsMicrographCropped.X + x] = 0;
                            }
                        }
                }

                progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Saving global scores...");

                // Store correlation values and angle IDs for re-use later
                CorrImage = new Image(CorrData, DimsMicrographCropped);
                CorrImage.WriteMRC(MatchingDir + NameWithRes + "_" + options.TemplateName + "_corr.mrc", (float)options.BinnedPixelSizeMean, true);

                #endregion
            }

            #endregion

            #region Get peak list that has at most nPeaks values

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Extracting best peaks...");

            int3[] InitialPeaks = new int3[0];
            {
                float2 MeanAndStd = MathHelper.MeanAndStdNonZero(CorrImage.GetHostContinuousCopy());

                for (float s = 4; s > 0.5f; s -= 0.05f)
                {
                    float Threshold = MeanAndStd.X + MeanAndStd.Y * s;
                    InitialPeaks = CorrImage.GetLocalPeaks(SizeParticle * 2 / 3, Threshold);

                    if (InitialPeaks.Length >= options.NResults)
                        break;
                }
            }

            CorrImage.Dispose();

            #endregion

            #region Write peak positions and angles into table

            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit"
            });

            {
                for (int n = 0; n < Math.Min(InitialPeaks.Length, options.NResults); n++)
                {
                    float3 Position = new float3(InitialPeaks[n]) * (float)options.DownsampleFactor;
                    float Score = CorrData[DimsMicrographCropped.ElementFromPosition(InitialPeaks[n])];
                    float3 Angle = HealpixAngles[(int)AngleData[DimsMicrographCropped.ElementFromPosition(InitialPeaks[n])]] * Helper.ToDeg;

                    TableOut.AddRow(new List<string>()
                    {
                        Position.X.ToString(CultureInfo.InvariantCulture),
                        Position.Y.ToString(CultureInfo.InvariantCulture),
                        Angle.X.ToString(CultureInfo.InvariantCulture),
                        Angle.Y.ToString(CultureInfo.InvariantCulture),
                        Angle.Z.ToString(CultureInfo.InvariantCulture),
                        RootName + ".mrc",
                        Score.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            TableOut.Save(MatchingDir + NameWithRes + "_" + options.TemplateName + ".star");
            UpdateParticleCount("_" + options.TemplateName);

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done.");

            #endregion
        }

        public void MatchBoxNet(BoxNet[] networks, Image average, ProcessingOptionsBoxNet options, Func<int3, int, string, bool> progressCallback)
        {
            IsProcessing = true;

            Directory.CreateDirectory(MatchingDir);

            float PixelSizeBN = networks[0].PixelSize;
            int2 DimsRegionBN = networks[0].BoxDimensions;

            int2 DimsBN = (new int2(average.Dims * average.PixelSize / PixelSizeBN) + 1) / 2 * 2;
            Image AverageBN = average.AsScaled(DimsBN);
            average.FreeDevice();

            if (options.PickingInvert)
                AverageBN.Multiply(-1f);

            float[] Predictions = new float[DimsBN.Elements()];
            bool[] NeedRefinement = new bool[Predictions.Length];

            int BatchSize = networks[0].BatchSize;
            
            {
                int NPositions = (int)DimsBN.Elements();
                int NBatches = (NPositions + BatchSize - 1) / BatchSize;

                int3[][] Positions = Helper.ArrayOfFunction(i => new int3[BatchSize], NBatches);
                for (int p = 0; p < NPositions; p++)
                {
                    int X = (p % DimsBN.X) - DimsRegionBN.X / 2;
                    int Y = (p / DimsBN.X) - DimsRegionBN.Y / 2;
                    Positions[p / BatchSize][p % BatchSize] = new int3(X, Y, 0);
                }

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                int NGPUs = networks.Length;
                int NGPUThreads = networks[0].MaxThreads;

                Image[] AverageBNLocal = new Image[NGPUs];
                Image[] Extracted = new Image[NGPUs * NGPUThreads];
                int DeviceID = GPU.GetDevice();
                AverageBN.FreeDevice();

                int BatchesDone = 0;
                float Threshold = (float)options.MinimumScore;

                Helper.ForCPU(0, NBatches, NGPUs * NGPUThreads,

                              threadID =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  GPU.SetDevice((DeviceID + GPUID) % GPU.GetDeviceCount());
                                  if (GPUThreadID == 0)
                                  {
                                      AverageBNLocal[GPUID] = AverageBN.GetCopy();
                                      AverageBNLocal[GPUID].GetDevice(Intent.Read);
                                  }
                                  Extracted[threadID] = new Image(IntPtr.Zero, new int3(DimsRegionBN.X, DimsRegionBN.Y, BatchSize));
                              },

                              (b, threadID) =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  #region Extract and normalize windows

                                  GPU.Extract(AverageBNLocal[GPUID].GetDevice(Intent.Read),
                                              Extracted[threadID].GetDevice(Intent.Write),
                                              AverageBNLocal[GPUID].Dims,
                                              new int3(DimsRegionBN),
                                              Helper.ToInterleaved(Positions[b]),
                                              (uint)BatchSize);

                                  GPU.Normalize(Extracted[threadID].GetDevice(Intent.Read),
                                                Extracted[threadID].GetDevice(Intent.Write),
                                                (uint)Extracted[threadID].ElementsSliceReal,
                                                (uint)BatchSize);

                                  #endregion

                                  #region Predict

                                  int BatchStart = b * BatchSize;
                                  int CurBatch = Math.Min(BatchSize, NPositions - BatchStart);

                                  long[] BatchArgMax;
                                  float[] BatchProbability;
                                  networks[GPUID].Predict(Extracted[threadID].GetDevice(Intent.Read),
                                                          GPUThreadID,
                                                          out BatchArgMax,
                                                          out BatchProbability);

                                  for (int bb = 0; bb < CurBatch; bb++)
                                  {
                                      int X = Positions[b][bb].X + DimsRegionBN.X / 2;
                                      int Y = Positions[b][bb].Y + DimsRegionBN.X / 2;
                                      Predictions[Y * DimsBN.X + X] = BatchProbability[bb * 2 + BatchArgMax[bb]] >= Threshold ? BatchArgMax[bb] * BatchProbability[bb * 2 + BatchArgMax[bb]] : 0;
                                  }

                                  #endregion

                                  lock (networks)
                                      progressCallback?.Invoke(new int3(NBatches, 1, 1), ++BatchesDone, "");
                              },

                              threadID =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  if (GPUThreadID == 0)
                                      AverageBNLocal[GPUID].Dispose();
                                  Extracted[threadID].Dispose();
                              });

                Watch.Stop();
                Debug.WriteLine(Watch.ElapsedMilliseconds / 1000.0);
                
                AverageBN.FreeDevice();
            }


            AverageBN.Dispose();

            #region Apply Gaussian and find peaks

            Image PredictionsImage = new Image(Predictions, new int3(DimsBN));

            //Image PredictionsConvolved = PredictionsImage.AsConvolvedGaussian((float)options.ExpectedDiameter / PixelSizeBN / 6);
            //PredictionsConvolved.Multiply(PredictionsImage);
            //PredictionsImage.Dispose();

            //PredictionsImage.WriteMRC(MatchingDir + RootName + "_boxnet.mrc", PixelSizeBN, true);

            int3[] Peaks = PredictionsImage.GetLocalPeaks((int)((float)options.ExpectedDiameter / PixelSizeBN / 4 + 0.5f), 1e-6f);
            PredictionsImage.Dispose();

            int BorderDist = (int)((float)options.ExpectedDiameter / PixelSizeBN * 0.8f + 0.5f);
            Peaks = Peaks.Where(p => p.X > BorderDist && p.Y > BorderDist && p.X < DimsBN.X - BorderDist && p.Y < DimsBN.Y - BorderDist).ToArray();

            #endregion

            #region Label connected components and get centroids

            List<List<int2>> Components = new List<List<int2>>();
            int[] PixelLabels = Helper.ArrayOfConstant(-1, Predictions.Length);

            foreach (var peak in Peaks.Select(v => new int2(v)))
            {
                if (PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                    continue;

                List<int2> Component = new List<int2>() { peak };
                int CN = Components.Count;

                PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                Queue<int2> Expansion = new Queue<int2>(100);
                Expansion.Enqueue(peak);

                while (Expansion.Count > 0)
                {
                    int2 pos = Expansion.Dequeue();
                    int PosElement = DimsBN.ElementFromPosition(pos);

                    if (pos.X > 0 && Predictions[PosElement - 1] > 0 && PixelLabels[PosElement - 1] < 0)
                    {
                        PixelLabels[PosElement - 1] = CN;
                        Component.Add(pos + new int2(-1, 0));
                        Expansion.Enqueue(pos + new int2(-1, 0));
                    }
                    if (pos.X < DimsBN.X - 1 && Predictions[PosElement + 1] > 0 && PixelLabels[PosElement + 1] < 0)
                    {
                        PixelLabels[PosElement + 1] = CN;
                        Component.Add(pos + new int2(1, 0));
                        Expansion.Enqueue(pos + new int2(1, 0));
                    }

                    if (pos.Y > 0 && Predictions[PosElement - DimsBN.X] > 0 && PixelLabels[PosElement - DimsBN.X] < 0)
                    {
                        PixelLabels[PosElement - DimsBN.X] = CN;
                        Component.Add(pos + new int2(0, -1));
                        Expansion.Enqueue(pos + new int2(0, -1));
                    }
                    if (pos.Y < DimsBN.Y - 1 && Predictions[PosElement + DimsBN.X] > 0 && PixelLabels[PosElement + DimsBN.X] < 0)
                    {
                        PixelLabels[PosElement + DimsBN.X] = CN;
                        Component.Add(pos + new int2(0, 1));
                        Expansion.Enqueue(pos + new int2(0, 1));
                    }
                }

                Components.Add(Component);
            }

            List<float2> Centroids = Components.Select(c => MathHelper.Mean(c.Select(v => new float2(v)))).ToList();
            int[] Extents = Components.Select(c => c.Count).ToArray();

            List<int> ToDelete = new List<int>();
            for (int c1 = 0; c1 < Centroids.Count - 1; c1++)
            {
                float2 P1 = Centroids[c1];

                for (int c2 = c1 + 1; c2 < Centroids.Count; c2++)
                {
                    if ((P1 - Centroids[c2]).Length() < (float)options.ExpectedDiameter / PixelSizeBN / 1.5f)
                    {
                        int D = Extents[c1] < Extents[c2] ? c1 : c2;

                        if (!ToDelete.Contains(D))
                            ToDelete.Add(D);
                    }
                }
            }

            ToDelete.Sort();
            for (int i = ToDelete.Count - 1; i > 0; i--)
                Centroids.RemoveAt(ToDelete[i]);

            #endregion

            #region Write peak positions and angles into table

            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit"
            });

            {
                foreach (float2 peak in Centroids)
                {
                    float2 Position = peak * PixelSizeBN / average.PixelSize;
                    float Score = Predictions[DimsBN.ElementFromPosition(new int2(peak))];

                    TableOut.AddRow(new List<string>
                    {
                        Position.X.ToString(CultureInfo.InvariantCulture),
                        Position.Y.ToString(CultureInfo.InvariantCulture),
                        RootName + ".mrc",
                        Score.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            TableOut.Save(MatchingDir + RootName + "_" + Helper.PathToNameWithExtension(options.ModelName) + ".star");
            UpdateParticleCount("_" + Helper.PathToNameWithExtension(options.ModelName));

            #endregion

            OptionsBoxNet = options;
            SaveMeta();

            IsProcessing = false;
        }

        public void MatchBoxNet2(BoxNet2[] networks, Image average, ProcessingOptionsBoxNet options, Func<int3, int, string, bool> progressCallback)
        {
            IsProcessing = true;

            Directory.CreateDirectory(MatchingDir);

            float PixelSizeBN = BoxNet2.PixelSize;
            int2 DimsRegionBN = BoxNet2.BoxDimensionsPredict;
            int2 DimsRegionValidBN = BoxNet2.BoxDimensionsValidPredict;
            int BorderBN = (DimsRegionBN.X - DimsRegionValidBN.X) / 2;

            int2 DimsBN = (new int2(average.Dims * average.PixelSize / PixelSizeBN) + 1) / 2 * 2;

            Image AverageBN = average.AsScaled(DimsBN);
            AverageBN.SubtractMeanPlane();
            GPU.Normalize(AverageBN.GetDevice(Intent.Read),
                          AverageBN.GetDevice(Intent.Write),
                          (uint)AverageBN.ElementsSliceReal,
                          1);
            
            average.FreeDevice();

            if (options.PickingInvert)
                AverageBN.Multiply(-1f);

            float[] Predictions = new float[DimsBN.Elements()];
            float[] Mask = new float[DimsBN.Elements()];
            

            {
                int2 DimsPositions = (DimsBN + DimsRegionValidBN - 1) / DimsRegionValidBN;
                float2 PositionStep = new float2(DimsBN - DimsRegionBN) / new float2(Math.Max(DimsPositions.X - 1, 1),
                                                                                     Math.Max(DimsPositions.Y - 1, 1));

                int NPositions = (int)DimsPositions.Elements();

                int3[] Positions = new int3[NPositions];
                for (int p = 0; p < NPositions; p++)
                {
                    int X = p % DimsPositions.X;
                    int Y = p / DimsPositions.X;
                    Positions[p] = new int3((int)(X * PositionStep.X + DimsRegionBN.X / 2),
                                            (int)(Y * PositionStep.Y + DimsRegionBN.Y / 2),
                                            0);
                }

                float[][] PredictionTiles = Helper.ArrayOfFunction(i => new float[DimsRegionBN.Elements()], NPositions);
                float[][] MaskTiles = Helper.ArrayOfFunction(i => new float[DimsRegionBN.Elements()], NPositions);
                
                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                int NGPUs = networks.Length;
                int NGPUThreads = networks[0].MaxThreads;

                Image[] AverageBNLocal = new Image[NGPUs];
                Image[] Extracted = new Image[NGPUs * NGPUThreads];
                int DeviceID = GPU.GetDevice();
                AverageBN.FreeDevice();

                int BatchesDone = 0;
                float Threshold = (float)options.MinimumScore;

                Helper.ForCPU(0, NPositions, NGPUs * NGPUThreads,

                              threadID =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  GPU.SetDevice((DeviceID + GPUID) % GPU.GetDeviceCount());
                                  if (GPUThreadID == 0)
                                  {
                                      AverageBNLocal[GPUID] = AverageBN.GetCopy();
                                      AverageBNLocal[GPUID].GetDevice(Intent.Read);
                                  }
                                  Extracted[threadID] = new Image(IntPtr.Zero, new int3(DimsRegionBN.X, DimsRegionBN.Y, 1));
                              },

                              (b, threadID) =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  #region Extract and normalize windows

                                  GPU.Extract(AverageBNLocal[GPUID].GetDevice(Intent.Read),
                                              Extracted[threadID].GetDevice(Intent.Write),
                                              AverageBNLocal[GPUID].Dims,
                                              new int3(DimsRegionBN),
                                              new int[]{Positions[b].X - DimsRegionBN.X / 2, Positions[b].Y - DimsRegionBN.Y / 2, 0},
                                              1);

                                  GPU.Normalize(Extracted[threadID].GetDevice(Intent.Read),
                                                Extracted[threadID].GetDevice(Intent.Write),
                                                (uint)Extracted[threadID].ElementsSliceReal,
                                                1);

                                  #endregion

                                  //Extracted[threadID].WriteMRC("d_extracted.mrc", true);

                                  #region Predict
                                  

                                  long[] BatchArgMax;
                                  float[] BatchProbability;
                                  networks[GPUID].Predict(Extracted[threadID].GetDevice(Intent.Read),
                                                          GPUThreadID,
                                                          out BatchArgMax,
                                                          out BatchProbability);

                                  //new Image(BatchArgMax.Select(v => (float)v).ToArray(), new int3(DimsRegionBN)).WriteMRC("d_labels.mrc", true);
                                  
                                  for (int i = 0; i < BatchArgMax.Length; i++)
                                  {
                                      int Label = (int)BatchArgMax[i];
                                      float Probability = BatchProbability[i * 3 + Label];

                                      PredictionTiles[b][i] = (Label == 1 && Probability >= Threshold ? Probability : 0);
                                      MaskTiles[b][i] = Label == 2 && Probability >= 0.0f ? 1 : 0;
                                  }

                                  #endregion

                                  lock (networks)
                                      progressCallback?.Invoke(new int3(NPositions, 1, 1), ++BatchesDone, "");
                              },

                              threadID =>
                              {
                                  int GPUID = threadID / NGPUThreads;
                                  int GPUThreadID = threadID % NGPUThreads;

                                  if (GPUThreadID == 0)
                                      AverageBNLocal[GPUID].Dispose();
                                  Extracted[threadID].Dispose();
                              });

                for (int y = 0; y < DimsBN.Y; y++)
                {
                    for (int x = 0; x < DimsBN.X; x++)
                    {
                        int ClosestX = (int)Math.Max(0, Math.Min(DimsPositions.X - 1, (int)(((float)x - DimsRegionBN.X / 2) / PositionStep.X + 0.5f)));
                        int ClosestY = (int)Math.Max(0, Math.Min(DimsPositions.Y - 1, (int)(((float)y - DimsRegionBN.Y / 2) / PositionStep.Y + 0.5f)));
                        int ClosestID = ClosestY * DimsPositions.X + ClosestX;

                        int3 Position = Positions[ClosestID];
                        int LocalX = Math.Max(0, Math.Min(DimsRegionBN.X - 1, x - Position.X + DimsRegionBN.X / 2));
                        int LocalY = Math.Max(0, Math.Min(DimsRegionBN.Y - 1, y - Position.Y + DimsRegionBN.Y / 2));

                        Predictions[y * DimsBN.X + x] = PredictionTiles[ClosestID][LocalY * DimsRegionBN.X + LocalX];
                        Mask[y * DimsBN.X + x] = MaskTiles[ClosestID][LocalY * DimsRegionBN.X + LocalX];
                    }
                }

                Watch.Stop();
                Debug.WriteLine(Watch.ElapsedMilliseconds / 1000.0);

                AverageBN.FreeDevice();
            }


            AverageBN.Dispose();

            #region Rescale and save mask

            Image MaskImage = new Image(Mask, new int3(DimsBN));

            // Get rid of all connected components in the mask that are too small
            {
                float[] MaskData = MaskImage.GetHost(Intent.ReadWrite)[0];

                List<List<int2>> Components = new List<List<int2>>();
                int[] PixelLabels = Helper.ArrayOfConstant(-1, MaskData.Length);

                for (int y = 0; y < DimsBN.Y; y++)
                {
                    for (int x = 0; x < DimsBN.X; x++)
                    {
                        int2 peak = new int2(x, y);

                        if (MaskData[DimsBN.ElementFromPosition(peak)] != 1 || PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                            continue;

                        List<int2> Component = new List<int2>() { peak };
                        int CN = Components.Count;

                        PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                        Queue<int2> Expansion = new Queue<int2>(100);
                        Expansion.Enqueue(peak);

                        while (Expansion.Count > 0)
                        {
                            int2 pos = Expansion.Dequeue();
                            int PosElement = DimsBN.ElementFromPosition(pos);

                            if (pos.X > 0 && MaskData[PosElement - 1] == 1 && PixelLabels[PosElement - 1] < 0)
                            {
                                PixelLabels[PosElement - 1] = CN;
                                Component.Add(pos + new int2(-1, 0));
                                Expansion.Enqueue(pos + new int2(-1, 0));
                            }
                            if (pos.X < DimsBN.X - 1 && MaskData[PosElement + 1] == 1 && PixelLabels[PosElement + 1] < 0)
                            {
                                PixelLabels[PosElement + 1] = CN;
                                Component.Add(pos + new int2(1, 0));
                                Expansion.Enqueue(pos + new int2(1, 0));
                            }

                            if (pos.Y > 0 && MaskData[PosElement - DimsBN.X] == 1 && PixelLabels[PosElement - DimsBN.X] < 0)
                            {
                                PixelLabels[PosElement - DimsBN.X] = CN;
                                Component.Add(pos + new int2(0, -1));
                                Expansion.Enqueue(pos + new int2(0, -1));
                            }
                            if (pos.Y < DimsBN.Y - 1 && MaskData[PosElement + DimsBN.X] == 1 && PixelLabels[PosElement + DimsBN.X] < 0)
                            {
                                PixelLabels[PosElement + DimsBN.X] = CN;
                                Component.Add(pos + new int2(0, 1));
                                Expansion.Enqueue(pos + new int2(0, 1));
                            }
                        }

                        Components.Add(Component);
                    }
                }

                foreach (var component in Components)
                    if (component.Count < 20)
                        foreach (var pos in component)
                            MaskData[DimsBN.ElementFromPosition(pos)] = 0;

                MaskPercentage = (decimal)MaskData.Sum() / MaskData.Length * 100;
            }

            Image MaskImage8 = MaskImage.AsScaled(new int2(new float2(DimsBN) * BoxNet2.PixelSize / 8) / 2 * 2);
            MaskImage8.Binarize(0.5f);

            int MaxHitTestDistance = (int)((options.ExpectedDiameter / 2 + options.MinimumMaskDistance) / (decimal)BoxNet2.PixelSize) + 2;
            Image MaskDistance = MaskImage.AsDistanceMapExact(MaxHitTestDistance);
            MaskImage.Dispose();
            MaskDistance.Binarize(MaxHitTestDistance - 2);
            float[] MaskHitTest = MaskDistance.GetHostContinuousCopy();
            MaskDistance.Dispose();

            Directory.CreateDirectory(MaskDir);
            MaskImage8.WriteTIFF(MaskPath, 8, typeof(float));
            MaskImage8.Dispose();

            #endregion

            #region Apply Gaussian and find peaks

            Image PredictionsImage = new Image(Predictions, new int3(DimsBN));

            //Image PredictionsConvolved = PredictionsImage.AsConvolvedGaussian((float)options.ExpectedDiameter / PixelSizeBN / 6);
            //PredictionsConvolved.Multiply(PredictionsImage);
            //PredictionsImage.Dispose();

            //PredictionsImage.WriteMRC(MatchingDir + RootName + "_boxnet.mrc", PixelSizeBN, true);

            int3[] Peaks = PredictionsImage.GetLocalPeaks((int)((float)options.ExpectedDiameter / PixelSizeBN / 4 + 0.5f), 1e-6f);
            PredictionsImage.Dispose();

            int BorderDist = (int)((float)options.ExpectedDiameter / PixelSizeBN * 0.8f + 0.5f);
            Peaks = Peaks.Where(p => p.X > BorderDist && p.Y > BorderDist && p.X < DimsBN.X - BorderDist && p.Y < DimsBN.Y - BorderDist).ToArray();

            #endregion

            #region Label connected components and get centroids

            List<float2> Centroids;
            int[] Extents;
            {
                List<List<int2>> Components = new List<List<int2>>();
                int[] PixelLabels = Helper.ArrayOfConstant(-1, Predictions.Length);

                foreach (var peak in Peaks.Select(v => new int2(v)))
                {
                    if (PixelLabels[DimsBN.ElementFromPosition(peak)] >= 0)
                        continue;

                    List<int2> Component = new List<int2>() { peak };
                    int CN = Components.Count;

                    PixelLabels[DimsBN.ElementFromPosition(peak)] = CN;
                    Queue<int2> Expansion = new Queue<int2>(100);
                    Expansion.Enqueue(peak);

                    while (Expansion.Count > 0)
                    {
                        int2 pos = Expansion.Dequeue();
                        int PosElement = DimsBN.ElementFromPosition(pos);

                        if (pos.X > 0 && Predictions[PosElement - 1] > 0 && PixelLabels[PosElement - 1] < 0)
                        {
                            PixelLabels[PosElement - 1] = CN;
                            Component.Add(pos + new int2(-1, 0));
                            Expansion.Enqueue(pos + new int2(-1, 0));
                        }
                        if (pos.X < DimsBN.X - 1 && Predictions[PosElement + 1] > 0 && PixelLabels[PosElement + 1] < 0)
                        {
                            PixelLabels[PosElement + 1] = CN;
                            Component.Add(pos + new int2(1, 0));
                            Expansion.Enqueue(pos + new int2(1, 0));
                        }

                        if (pos.Y > 0 && Predictions[PosElement - DimsBN.X] > 0 && PixelLabels[PosElement - DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement - DimsBN.X] = CN;
                            Component.Add(pos + new int2(0, -1));
                            Expansion.Enqueue(pos + new int2(0, -1));
                        }
                        if (pos.Y < DimsBN.Y - 1 && Predictions[PosElement + DimsBN.X] > 0 && PixelLabels[PosElement + DimsBN.X] < 0)
                        {
                            PixelLabels[PosElement + DimsBN.X] = CN;
                            Component.Add(pos + new int2(0, 1));
                            Expansion.Enqueue(pos + new int2(0, 1));
                        }
                    }

                    Components.Add(Component);
                }

                Centroids = Components.Select(c => MathHelper.Mean(c.Select(v => new float2(v)))).ToList();
                Extents = Components.Select(c => c.Count).ToArray();
            }

            List<int> ToDelete = new List<int>();

            // Hit test with crap mask
            for (int c1 = 0; c1 < Centroids.Count; c1++)
            {
                float2 P1 = Centroids[c1];
                if (MaskHitTest[(int)P1.Y * DimsBN.X + (int)P1.X] == 0)
                {
                    ToDelete.Add(c1);
                    continue;
                }
            }

            for (int c1 = 0; c1 < Centroids.Count - 1; c1++)
            {
                float2 P1 = Centroids[c1];

                for (int c2 = c1 + 1; c2 < Centroids.Count; c2++)
                {
                    if ((P1 - Centroids[c2]).Length() < (float)options.ExpectedDiameter / PixelSizeBN / 1.5f)
                    {
                        int D = Extents[c1] < Extents[c2] ? c1 : c2;

                        if (!ToDelete.Contains(D))
                            ToDelete.Add(D);
                    }
                }
            }

            ToDelete.Sort();
            for (int i = ToDelete.Count - 1; i >= 0; i--)
                Centroids.RemoveAt(ToDelete[i]);

            #endregion

            #region Write peak positions and angles into table

            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit"
            });

            {
                foreach (float2 peak in Centroids)
                {
                    float2 Position = peak * PixelSizeBN / average.PixelSize;
                    float Score = Predictions[DimsBN.ElementFromPosition(new int2(peak))];

                    TableOut.AddRow(new List<string>
                    {
                        Position.X.ToString(CultureInfo.InvariantCulture),
                        Position.Y.ToString(CultureInfo.InvariantCulture),
                        RootName + ".mrc",
                        Score.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            TableOut.Save(MatchingDir + RootName + "_" + Helper.PathToNameWithExtension(options.ModelName) + ".star");
            UpdateParticleCount("_" + Helper.PathToNameWithExtension(options.ModelName));

            #endregion

            OptionsBoxNet = options;
            SaveMeta();

            IsProcessing = false;
        }
    }

    public class ProcessingOptionsMovieCTF : ProcessingOptionsBase
    {
        [WarpSerializable]
        public int Window { get; set; }
        [WarpSerializable]
        public decimal RangeMin { get; set; }
        [WarpSerializable]
        public decimal RangeMax { get; set; }
        [WarpSerializable]
        public int Voltage { get; set; }
        [WarpSerializable]
        public decimal Cs { get; set; }
        [WarpSerializable]
        public decimal Cc { get; set; }
        [WarpSerializable]
        public decimal IllumAngle { get; set; }
        [WarpSerializable]
        public decimal EnergySpread { get; set; }
        [WarpSerializable]
        public decimal Thickness { get; set; }
        [WarpSerializable]
        public decimal Amplitude { get; set; }
        [WarpSerializable]
        public bool DoPhase { get; set; }
        [WarpSerializable]
        public bool DoIce { get; set; }
        [WarpSerializable]
        public bool DoSimultaneous { get; set; }
        [WarpSerializable]
        public bool UseMovieSum { get; set; }
        [WarpSerializable]
        public decimal Astigmatism { get; set; }
        [WarpSerializable]
        public decimal AstigmatismAngle { get; set; }
        [WarpSerializable]
        public decimal ZMin { get; set; }
        [WarpSerializable]
        public decimal ZMax { get; set; }
        [WarpSerializable]
        public int3 GridDims { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsMovieCTF)obj);
        }

        protected bool Equals(ProcessingOptionsMovieCTF other)
        {
            return base.Equals(other) &&
                   Window == other.Window &&
                   RangeMin == other.RangeMin &&
                   RangeMax == other.RangeMax &&
                   Voltage == other.Voltage &&
                   Cs == other.Cs &&
                   Cc == other.Cc &&
                   IllumAngle == other.IllumAngle &&
                   EnergySpread == other.EnergySpread &&
                   Thickness == other.Thickness &&
                   Amplitude == other.Amplitude &&
                   DoPhase == other.DoPhase &&
                   DoIce == other.DoIce &&
                   DoSimultaneous == other.DoSimultaneous &&
                   UseMovieSum == other.UseMovieSum &&
                   Astigmatism == other.Astigmatism &&
                   AstigmatismAngle == other.AstigmatismAngle &&
                   ZMin == other.ZMin &&
                   ZMax == other.ZMax &&
                   GridDims == other.GridDims;
        }

        public static bool operator ==(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
        {
            return !Equals(left, right);
        }
    }

    public class ProcessingOptionsMovieMovement : ProcessingOptionsBase
    {
        [WarpSerializable]
        public decimal RangeMin { get; set; }
        [WarpSerializable]
        public decimal RangeMax { get; set; }
        [WarpSerializable]
        public decimal Bfactor { get; set; }
        [WarpSerializable]
        public int3 GridDims { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsMovieMovement)obj);
        }

        protected bool Equals(ProcessingOptionsMovieMovement other)
        {
            return base.Equals(other) &&
                   RangeMin == other.RangeMin &&
                   RangeMax == other.RangeMax &&
                   Bfactor == other.Bfactor &&
                   GridDims == other.GridDims;
        }

        public static bool operator ==(ProcessingOptionsMovieMovement left, ProcessingOptionsMovieMovement right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsMovieMovement left, ProcessingOptionsMovieMovement right)
        {
            return !Equals(left, right);
        }
    }

    public class ProcessingOptionsMovieExport : ProcessingOptionsBase
    {
        [WarpSerializable]
        public bool DoAverage { get; set; }
        [WarpSerializable]
        public bool DoStack { get; set; }
        [WarpSerializable]
        public bool DoDeconv { get; set; }
        [WarpSerializable]
        public decimal DeconvolutionStrength { get; set; }
        [WarpSerializable]
        public decimal DeconvolutionFalloff { get; set; }
        [WarpSerializable]
        public int StackGroupSize { get; set; }
        [WarpSerializable]
        public int SkipFirstN { get; set; }
        [WarpSerializable]
        public int SkipLastN { get; set; }
        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }
        [WarpSerializable]
        public int Voltage { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsMovieExport)obj);
        }

        protected bool Equals(ProcessingOptionsMovieExport other)
        {
            return base.Equals(other) &&
                   DoAverage == other.DoAverage &&
                   DoStack == other.DoStack &&
                   DoDeconv == other.DoDeconv &&
                   DeconvolutionStrength == other.DeconvolutionStrength &&
                   DeconvolutionFalloff == other.DeconvolutionFalloff &&
                   StackGroupSize == other.StackGroupSize &&
                   SkipFirstN == other.SkipFirstN &&
                   SkipLastN == other.SkipLastN &&
                   DosePerAngstromFrame == other.DosePerAngstromFrame &&
                   Voltage == other.Voltage;
        }

        public static bool operator ==(ProcessingOptionsMovieExport left, ProcessingOptionsMovieExport right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsMovieExport left, ProcessingOptionsMovieExport right)
        {
            return !Equals(left, right);
        }
    }

    public class ProcessingOptionsParticlesExport : ProcessingOptionsBase
    {
        [WarpSerializable]
        public string Suffix { get; set; }
        [WarpSerializable]
        public int BoxSize { get; set; }
        [WarpSerializable]
        public int Diameter { get; set; }
        [WarpSerializable]
        public bool Invert { get; set; }
        [WarpSerializable]
        public bool Normalize { get; set; }
        [WarpSerializable]
        public bool DoAverage { get; set; }
        [WarpSerializable]
        public bool DoStack { get; set; }
        [WarpSerializable]
        public int StackGroupSize { get; set; }
        [WarpSerializable]
        public int SkipFirstN { get; set; }
        [WarpSerializable]
        public int SkipLastN { get; set; }
        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }
        [WarpSerializable]
        public int Voltage { get; set; }
        [WarpSerializable]
        public bool CorrectAnisotropy { get; set; }
        [WarpSerializable]
        public bool PreflipPhases { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsParticlesExport)obj);
        }

        protected bool Equals(ProcessingOptionsParticlesExport other)
        {
            return base.Equals(other) &&
                   DoAverage == other.DoAverage &&
                   DoStack == other.DoStack &&
                   StackGroupSize == other.StackGroupSize &&
                   SkipFirstN == other.SkipFirstN &&
                   SkipLastN == other.SkipLastN &&
                   DosePerAngstromFrame == other.DosePerAngstromFrame &&
                   Voltage == other.Voltage &&
                   CorrectAnisotropy == other.CorrectAnisotropy &&
                   PreflipPhases == other.PreflipPhases;
        }

        public static bool operator ==(ProcessingOptionsParticlesExport left, ProcessingOptionsParticlesExport right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsParticlesExport left, ProcessingOptionsParticlesExport right)
        {
            return !Equals(left, right);
        }
    }

    public class ProcessingOptionsFullMatch : ProcessingOptionsBase
    {
        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public bool Invert { get; set; }

        [WarpSerializable]
        public int SubPatchSize { get; set; }

        [WarpSerializable]
        public string TemplateName { get; set; }

        [WarpSerializable]
        public decimal TemplatePixel { get; set; }

        [WarpSerializable]
        public decimal TemplateDiameter { get; set; }

        [WarpSerializable]
        public decimal TemplateFraction { get; set; }

        [WarpSerializable]
        public bool WhitenSpectrum { get; set; }

        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }

        [WarpSerializable]
        public int Voltage { get; set; }

        [WarpSerializable]
        public string Symmetry { get; set; }

        [WarpSerializable]
        public int HealpixOrder { get; set; }

        [WarpSerializable]
        public int Supersample { get; set; }

        [WarpSerializable]
        public int NResults { get; set; }
    }

    public class ProcessingOptionsBoxNet : ProcessingOptionsBase
    {
        [WarpSerializable]
        public string ModelName { get; set; }

        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public bool PickingInvert { get; set; }

        [WarpSerializable]
        public decimal ExpectedDiameter { get; set; }

        [WarpSerializable]
        public decimal MinimumScore { get; set; }

        [WarpSerializable]
        public decimal MinimumMaskDistance { get; set; }

        [WarpSerializable]
        public bool ExportParticles { get; set; }

        [WarpSerializable]
        public int ExportBoxSize { get; set; }

        [WarpSerializable]
        public bool ExportInvert { get; set; }

        [WarpSerializable]
        public bool ExportNormalize { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsBoxNet)obj);
        }

        protected bool Equals(ProcessingOptionsBoxNet other)
        {
            return ModelName == other.ModelName &&
                   OverwriteFiles == other.OverwriteFiles &&
                   PickingInvert == other.PickingInvert &&
                   ExpectedDiameter == other.ExpectedDiameter &&
                   MinimumScore == other.MinimumScore &&
                   ExportParticles == other.ExportParticles &&
                   ExportBoxSize == other.ExportBoxSize &&
                   ExportInvert == other.ExportInvert &&
                   ExportNormalize == other.ExportNormalize;
        }

        public static bool operator ==(ProcessingOptionsBoxNet left, ProcessingOptionsBoxNet right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsBoxNet left, ProcessingOptionsBoxNet right)
        {
            return !Equals(left, right);
        }
    }
}
