using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Warp.Tools;

namespace Warp
{
    public class Options : WarpBase
    {        
        #region Pixel size

        private decimal _PixelSizeX = 1.35M;
        [WarpSerializable]
        public decimal PixelSizeX
        {
            get { return _PixelSizeX; }
            set
            {
                if (value != _PixelSizeX)
                {
                    _PixelSizeX = value;
                    OnPropertyChanged();
                    RecalcBinnedPixelSize();
                }
            }
        }

        private decimal _PixelSizeY = 1.35M;
        [WarpSerializable]
        public decimal PixelSizeY
        {
            get { return _PixelSizeY; }
            set
            {
                if (value != _PixelSizeY)
                {
                    _PixelSizeY = value;
                    OnPropertyChanged();
                    RecalcBinnedPixelSize();
                }
            }
        }

        private decimal _PixelSizeAngle = 0M;
        [WarpSerializable]
        public decimal PixelSizeAngle
        {
            get { return _PixelSizeAngle; }
            set
            {
                if (value != _PixelSizeAngle)
                {
                    _PixelSizeAngle = value;
                    OnPropertyChanged();
                }
            }
        }

        #endregion

        #region Things to process

        private bool _ProcessCTF = true;
        [WarpSerializable]
        public bool ProcessCTF
        {
            get { return _ProcessCTF; }
            set { if (value != _ProcessCTF) { _ProcessCTF = value; OnPropertyChanged(); } }
        }

        private bool _ProcessMovement = true;
        [WarpSerializable]
        public bool ProcessMovement
        {
            get { return _ProcessMovement; }
            set { if (value != _ProcessMovement) { _ProcessMovement = value; OnPropertyChanged(); } }
        }

        private bool _ProcessPicking = false;
        [WarpSerializable]
        public bool ProcessPicking
        {
            get { return _ProcessPicking; }
            set { if (value != _ProcessPicking) { _ProcessPicking = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Sub-categories

        private OptionsImport _Import = new OptionsImport();
        public OptionsImport Import
        {
            get { return _Import; }
            set { if (value != _Import) { _Import = value; OnPropertyChanged(); } }
        }

        private OptionsCTF _CTF = new OptionsCTF();
        public OptionsCTF CTF
        {
            get { return _CTF; }
            set { if (value != _CTF) { _CTF = value; OnPropertyChanged(); } }
        }

        private OptionsMovement _Movement = new OptionsMovement();
        public OptionsMovement Movement
        {
            get { return _Movement; }
            set { if (value != _Movement) { _Movement = value; OnPropertyChanged(); } }
        }

        private OptionsGrids _Grids = new OptionsGrids();
        public OptionsGrids Grids
        {
            get { return _Grids; }
            set { if (value != _Grids) { _Grids = value; OnPropertyChanged(); } }
        }

        private OptionsPicking _Picking = new OptionsPicking();
        public OptionsPicking Picking
        {
            get { return _Picking; }
            set { if (value != _Picking) { _Picking = value; OnPropertyChanged(); } }
        }

        private OptionsTomo _Tomo = new OptionsTomo();
        public OptionsTomo Tomo
        {
            get { return _Tomo; }
            set { if (value != _Tomo) { _Tomo = value; OnPropertyChanged(); } }
        }

        private OptionsExport _Export = new OptionsExport();
        public OptionsExport Export
        {
            get { return _Export; }
            set { if (value != _Export) { _Export = value; OnPropertyChanged(); } }
        }

        private OptionsTasks _Tasks = new OptionsTasks();
        public OptionsTasks Tasks
        {
            get { return _Tasks; }
            set { if (value != _Tasks) { _Tasks = value; OnPropertyChanged(); } }
        }

        private OptionsFilter _Filter = new OptionsFilter();
        public OptionsFilter Filter
        {
            get { return _Filter; }
            set { if (value != _Filter) { _Filter = value; OnPropertyChanged(); } }
        }

        private OptionsAdvanced _Advanced = new OptionsAdvanced();
        public OptionsAdvanced Advanced
        {
            get { return _Advanced; }
            set { if (value != _Advanced) { _Advanced = value; OnPropertyChanged(); } }
        }

        private OptionsRuntime _Runtime = new OptionsRuntime();
        public OptionsRuntime Runtime
        {
            get { return _Runtime; }
            set { if (value != _Runtime) { _Runtime = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Runtime

        private void RecalcBinnedPixelSize()
        {
            Runtime.BinnedPixelSizeMean = PixelSizeMean * (decimal)Math.Pow(2.0, (double)Import.BinTimes);
        }

        public decimal PixelSizeMean => (PixelSizeX + PixelSizeY) * 0.5M;
        public decimal BinnedPixelSizeX => PixelSizeX * (decimal)Math.Pow(2.0, (double)Import.BinTimes);
        public decimal BinnedPixelSizeY => PixelSizeY * (decimal)Math.Pow(2.0, (double)Import.BinTimes);
        public decimal BinnedPixelSizeMean => (BinnedPixelSizeX + BinnedPixelSizeY) * 0.5M;

        public float2 AstigmatismMean = new float2();
        public float AstigmatismStd = 0.1f;

        public void UpdateGPUStats()
        {
            int NDevices = GPU.GetDeviceCount();
            string[] Stats = new string[NDevices];
            for (int i = 0; i < NDevices; i++)
                Stats[i] = "GPU" + i + ": " + GPU.GetFreeMemory(i) + " MB";
            Runtime.GPUStats = string.Join(", ", Stats);
        }

        #endregion

        public Options()
        {
        }

        public void Load(string path)
        {
            try
            {
                using (Stream SettingsStream = File.OpenRead(path))
                {
                    XPathDocument Doc = new XPathDocument(SettingsStream);
                    XPathNavigator Reader = Doc.CreateNavigator();
                    Reader.MoveToRoot();

                    Reader.MoveToRoot();
                    Reader.MoveToChild("Settings", "");

                    ReadFromXML(Reader);

                    Import.ReadFromXML(Reader.SelectSingleNode("Import"));
                    CTF.ReadFromXML(Reader.SelectSingleNode("CTF"));
                    Movement.ReadFromXML(Reader.SelectSingleNode("Movement"));
                    Grids.ReadFromXML(Reader.SelectSingleNode("Grids"));
                    Tomo.ReadFromXML(Reader.SelectSingleNode("Tomo"));
                    Picking.ReadFromXML(Reader.SelectSingleNode("Picking"));
                    Export.ReadFromXML(Reader.SelectSingleNode("Export"));
                    Tasks.ReadFromXML(Reader.SelectSingleNode("Tasks"));
                    Filter.ReadFromXML(Reader.SelectSingleNode("Filter"));
                    Advanced.ReadFromXML(Reader.SelectSingleNode("Advanced"));

                    RecalcBinnedPixelSize();
                }
            }
            catch { }
        }

        #region 2D processing settings creation and adoption

        public ProcessingOptionsMovieCTF GetProcessingMovieCTF()
        {
            return new ProcessingOptionsMovieCTF
            {
                PixelSizeX = PixelSizeX,
                PixelSizeY = PixelSizeY,
                PixelSizeAngle = PixelSizeAngle,
                BinTimes = Import.BinTimes,
                EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0,
                GainPath = Import.CorrectGain ? Import.GainPath : "",
                GainHash = Import.CorrectGain ? Runtime.GainReferenceHash : "",
                DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "",
                DefectsHash = Import.CorrectDefects ? Runtime.DefectMapHash : "",
                GainFlipX = Import.GainFlipX,
                GainFlipY = Import.GainFlipY,
                GainTranspose = Import.GainTranspose,
                Window = CTF.Window,
                RangeMin = CTF.RangeMin,
                RangeMax = CTF.RangeMax,
                Voltage = CTF.Voltage,
                Cs = CTF.Cs,
                Cc = CTF.Cc,
                IllumAngle = CTF.IllAperture,
                EnergySpread = CTF.DeltaE,
                Thickness = CTF.Thickness,
                Amplitude = CTF.Amplitude,
                DoPhase = CTF.DoPhase,
                DoIce = false, //CTF.DoIce,
                UseMovieSum = CTF.UseMovieSum,
                ZMin = CTF.ZMin,
                ZMax = CTF.ZMax,
                GridDims = new int3(Grids.CTFX, Grids.CTFY, Grids.CTFZ)
            };
        }

        public ProcessingOptionsMovieMovement GetProcessingMovieMovement()
        {
            return new ProcessingOptionsMovieMovement
            {
                PixelSizeX = PixelSizeX,
                PixelSizeY = PixelSizeY,
                PixelSizeAngle = PixelSizeAngle,
                BinTimes = Import.BinTimes,
                EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0,
                GainPath = Import.CorrectGain ? Import.GainPath : "",
                GainHash = Import.CorrectGain ? Runtime.GainReferenceHash : "",
                DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "",
                DefectsHash = Import.CorrectDefects ? Runtime.DefectMapHash : "",
                GainFlipX = Import.GainFlipX,
                GainFlipY = Import.GainFlipY,
                GainTranspose = Import.GainTranspose,
                RangeMin = Movement.RangeMin,
                RangeMax = Movement.RangeMax,
                Bfactor = Movement.Bfactor,
                GridDims = new int3(Grids.MovementX, Grids.MovementY, Grids.MovementZ)
            };
        }

        public ProcessingOptionsMovieExport GetProcessingMovieExport()
        {
            return new ProcessingOptionsMovieExport
            {
                PixelSizeX = PixelSizeX,
                PixelSizeY = PixelSizeY,
                PixelSizeAngle = PixelSizeAngle,

                BinTimes = Import.BinTimes,
                EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0,
                GainPath = Import.CorrectGain ? Import.GainPath : "",
                GainHash = Import.CorrectGain ? Runtime.GainReferenceHash : "",
                DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "",
                DefectsHash = Import.CorrectDefects ? Runtime.DefectMapHash : "",
                GainFlipX = Import.GainFlipX,
                GainFlipY = Import.GainFlipY,
                GainTranspose = Import.GainTranspose,
                DosePerAngstromFrame = Import.DosePerAngstromFrame,

                DoAverage = true, //Export.DoAverage,
                DoStack = Export.DoStack,
                DoDeconv = Export.DoDeconvolve,
                DeconvolutionStrength = Export.DeconvolutionStrength,
                DeconvolutionFalloff = Export.DeconvolutionFalloff,
                StackGroupSize = Export.StackGroupSize,
                SkipFirstN = Export.SkipFirstN,
                SkipLastN = Export.SkipLastN,

                Voltage = CTF.Voltage
            };
        }

        public ProcessingOptionsBoxNet GetProcessingBoxNet()
        {
            return new ProcessingOptionsBoxNet
            {
                PixelSizeX = PixelSizeX,
                PixelSizeY = PixelSizeY,
                PixelSizeAngle = PixelSizeAngle,

                BinTimes = Import.BinTimes,
                EERGroupFrames = Import.ExtensionEER ? Import.EERGroupFrames : 0,
                GainPath = Import.CorrectGain ? Import.GainPath : "",
                GainHash = Import.CorrectGain ? Runtime.GainReferenceHash : "",
                DefectsPath = Import.CorrectDefects ? Import.DefectsPath : "",
                DefectsHash = Import.CorrectDefects ? Runtime.DefectMapHash : "",
                GainFlipX = Import.GainFlipX,
                GainFlipY = Import.GainFlipY,
                GainTranspose = Import.GainTranspose,

                OverwriteFiles = true,

                ModelName = Picking.ModelPath,

                PickingInvert = Picking.DataStyle != "cryo",
                ExpectedDiameter = Picking.Diameter,
                MinimumScore = Picking.MinimumScore,
                MinimumMaskDistance = Picking.MinimumMaskDistance,

                ExportParticles = Picking.DoExport,
                ExportBoxSize = Picking.BoxSize,
                ExportInvert = Picking.Invert,
                ExportNormalize = Picking.Normalize
            };
        }

        #endregion
    }

    public class OptionsImport : WarpBase
    {
        private string _Folder = "";
        [WarpSerializable]
        public string Folder
        {
            get { return _Folder; }
            set
            {
                if (value != _Folder)
                {
                    _Folder = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _Extension = "*.mrc";
        [WarpSerializable]
        public string Extension
        {
            get { return _Extension; }
            set
            {
                if (value != _Extension)
                {
                    _Extension = value;
                    OnPropertyChanged();

                    OnPropertyChanged("ExtensionMRC");
                    OnPropertyChanged("ExtensionMRCS");
                    OnPropertyChanged("ExtensionEM");
                    OnPropertyChanged("ExtensionTIFF");
                    OnPropertyChanged("ExtensionDAT");
                    OnPropertyChanged("ExtensionTomoSTAR");
                }
            }
        }
        
        public bool ExtensionMRC
        {
            get { return Extension == "*.mrc"; }
            set
            {
                if (value != (Extension == "*.mrc"))
                {
                    if (value)
                        Extension = "*.mrc";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionMRCS
        {
            get { return Extension == "*.mrcs"; }
            set
            {
                if (value != (Extension == "*.mrcs"))
                {
                    if (value)
                        Extension = "*.mrcs";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionEM
        {
            get { return Extension == "*.em"; }
            set
            {
                if (value != (Extension == "*.em"))
                {
                    if (value)
                        Extension = "*.em";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionTIFF
        {
            get { return Extension == "*.tif"; }
            set
            {
                if (value != (Extension == "*.tif"))
                {
                    if (value)
                        Extension = "*.tif";
                    OnPropertyChanged();
                }
            }
        }

        public bool ExtensionTIFFF
        {
            get { return Extension == "*.tiff"; }
            set
            {
                if (value != (Extension == "*.tiff"))
                {
                    if (value)
                        Extension = "*.tiff";
                    OnPropertyChanged();
                }
            }
        }

        public bool ExtensionEER
        {
            get { return Extension == "*.eer"; }
            set
            {
                if (value != (Extension == "*.eer"))
                {
                    if (value)
                        Extension = "*.eer";
                    OnPropertyChanged();
                }
            }
        }

        public bool ExtensionTomoSTAR
        {
            get { return Extension == "*.tomostar"; }
            set
            {
                if (value != (Extension == "*.tomostar"))
                {
                    if (value)
                        Extension = "*.tomostar";
                    OnPropertyChanged();
                }
            }
        }
        
        public bool ExtensionDAT
        {
            get { return Extension == "*.dat"; }
            set
            {
                if (value != (Extension == "*.dat"))
                {
                    if (value)
                        Extension = "*.dat";
                    OnPropertyChanged();
                }
            }
        }

        private int _HeaderlessWidth = 7676;
        [WarpSerializable]
        public int HeaderlessWidth
        {
            get { return _HeaderlessWidth; }
            set { if (value != _HeaderlessWidth) { _HeaderlessWidth = value; OnPropertyChanged(); } }
        }

        private int _HeaderlessHeight = 7420;
        [WarpSerializable]
        public int HeaderlessHeight
        {
            get { return _HeaderlessHeight; }
            set { if (value != _HeaderlessHeight) { _HeaderlessHeight = value; OnPropertyChanged(); } }
        }

        private string _HeaderlessType = "int8";
        [WarpSerializable]
        public string HeaderlessType
        {
            get { return _HeaderlessType; }
            set { if (value != _HeaderlessType) { _HeaderlessType = value; OnPropertyChanged(); } }
        }

        private long _HeaderlessOffset = 0;
        [WarpSerializable]
        public long HeaderlessOffset
        {
            get { return _HeaderlessOffset; }
            set { if (value != _HeaderlessOffset) { _HeaderlessOffset = value; OnPropertyChanged(); } }
        }

        private decimal _BinTimes = 0;
        [WarpSerializable]
        public decimal BinTimes
        {
            get { return _BinTimes; }
            set
            {
                if (value != _BinTimes)
                {
                    _BinTimes = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _GainPath = "";
        [WarpSerializable]
        public string GainPath
        {
            get { return _GainPath; }
            set
            {
                if (value != _GainPath)
                {
                    _GainPath = value;
                    OnPropertyChanged();
                }
            }
        }

        private string _DefectsPath = "";
        [WarpSerializable]
        public string DefectsPath
        {
            get { return _DefectsPath; }
            set { if (value != _DefectsPath) { _DefectsPath = value; OnPropertyChanged(); } }
        }

        private bool _GainFlipX = false;
        [WarpSerializable]
        public bool GainFlipX
        {
            get { return _GainFlipX; }
            set { if (value != _GainFlipX) { _GainFlipX = value; OnPropertyChanged(); } }
        }

        private bool _GainFlipY = false;
        [WarpSerializable]
        public bool GainFlipY
        {
            get { return _GainFlipY; }
            set { if (value != _GainFlipY) { _GainFlipY = value; OnPropertyChanged(); } }
        }

        private bool _GainTranspose = false;
        [WarpSerializable]
        public bool GainTranspose
        {
            get { return _GainTranspose; }
            set { if (value != _GainTranspose) { _GainTranspose = value; OnPropertyChanged(); } }
        }

        private bool _CorrectGain = false;
        [WarpSerializable]
        public bool CorrectGain
        {
            get { return _CorrectGain; }
            set
            {
                if (value != _CorrectGain)
                {
                    _CorrectGain = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _CorrectDefects = false;
        [WarpSerializable]
        public bool CorrectDefects
        {
            get { return _CorrectDefects; }
            set { if (value != _CorrectDefects) { _CorrectDefects = value; OnPropertyChanged(); } }
        }

        private decimal _DosePerAngstromFrame = 0;
        [WarpSerializable]
        public decimal DosePerAngstromFrame
        {
            get { return _DosePerAngstromFrame; }
            set { if (value != _DosePerAngstromFrame) { _DosePerAngstromFrame = value; OnPropertyChanged(); } }
        }

        private int _EERGroupFrames = 10;
        [WarpSerializable]
        public int EERGroupFrames
        {
            get { return _EERGroupFrames; }
            set { if (value != _EERGroupFrames) { _EERGroupFrames = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsCTF : WarpBase
    {
        private int _Window = 512;
        [WarpSerializable]
        public int Window
        {
            get { return _Window; }
            set
            {
                if (value != _Window)
                {
                    _Window = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _RangeMin = 0.10M;
        [WarpSerializable]
        public decimal RangeMin
        {
            get { return _RangeMin; }
            set
            {
                if (value != _RangeMin)
                {
                    _RangeMin = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _RangeMax = 0.6M;
        [WarpSerializable]
        public decimal RangeMax
        {
            get { return _RangeMax; }
            set
            {
                if (value != _RangeMax)
                {
                    _RangeMax = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _MinQuality = 0.8M;
        [WarpSerializable]
        public decimal MinQuality
        {
            get { return _MinQuality; }
            set
            {
                if (value != _MinQuality)
                {
                    _MinQuality = value;
                    OnPropertyChanged();
                }
            }
        }

        private int _Voltage = 300;
        [WarpSerializable]
        public int Voltage
        {
            get { return _Voltage; }
            set
            {
                if (value != _Voltage)
                {
                    _Voltage = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _Cs = 2.7M;
        [WarpSerializable]
        public decimal Cs
        {
            get { return _Cs; }
            set
            {
                if (value != _Cs)
                {
                    _Cs = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _Cc = 2.7M;
        [WarpSerializable]
        public decimal Cc
        {
            get { return _Cc; }
            set { if (value != _Cc) { _Cc = value; OnPropertyChanged(); } }
        }

        private decimal _Amplitude = 0.07M;
        [WarpSerializable]
        public decimal Amplitude
        {
            get { return _Amplitude; }
            set
            {
                if (value != _Amplitude)
                {
                    _Amplitude = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _IllAperture = 30;
        [WarpSerializable]
        public decimal IllAperture
        {
            get { return _IllAperture; }
            set { if (value != _IllAperture) { _IllAperture = value; OnPropertyChanged(); } }
        }

        private decimal _DeltaE = 0.7M;
        [WarpSerializable]
        public decimal DeltaE
        {
            get { return _DeltaE; }
            set { if (value != _DeltaE) { _DeltaE = value; OnPropertyChanged(); } }
        }

        private decimal _Thickness = 0;
        [WarpSerializable]
        public decimal Thickness
        {
            get { return _Thickness; }
            set { if (value != _Thickness) { _Thickness = value; OnPropertyChanged(); } }
        }

        private bool _DoPhase = true;
        [WarpSerializable]
        public bool DoPhase
        {
            get { return _DoPhase; }
            set { if (value != _DoPhase) { _DoPhase = value; OnPropertyChanged(); } }
        }

        //private bool _DoIce = false;
        //[WarpSerializable]
        //public bool DoIce
        //{
        //    get { return _DoIce; }
        //    set { if (value != _DoIce) { _DoIce = value; OnPropertyChanged(); } }
        //}

        private bool _DoSimultaneous = false;
        [WarpSerializable]
        public bool DoSimultaneous
        {
            get { return _DoSimultaneous; }
            set { if (value != _DoSimultaneous) { _DoSimultaneous = value; OnPropertyChanged(); } }
        }

        private bool _UseMovieSum = false;
        [WarpSerializable]
        public bool UseMovieSum
        {
            get { return _UseMovieSum; }
            set { if (value != _UseMovieSum) { _UseMovieSum = value; OnPropertyChanged(); } }
        }

        private decimal _Astigmatism = 0M;
        [WarpSerializable]
        public decimal Astigmatism
        {
            get { return _Astigmatism; }
            set
            {
                if (value != _Astigmatism)
                {
                    _Astigmatism = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _AstigmatismAngle = 0M;
        [WarpSerializable]
        public decimal AstigmatismAngle
        {
            get { return _AstigmatismAngle; }
            set
            {
                if (value != _AstigmatismAngle)
                {
                    _AstigmatismAngle = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _ZMin = 0M;
        [WarpSerializable]
        public decimal ZMin
        {
            get { return _ZMin; }
            set
            {
                if (value != _ZMin)
                {
                    _ZMin = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _ZMax = 5M;
        [WarpSerializable]
        public decimal ZMax
        {
            get { return _ZMax; }
            set
            {
                if (value != _ZMax)
                {
                    _ZMax = value;
                    OnPropertyChanged();
                }
            }
        }
    }

    public class OptionsMovement : WarpBase
    {
        private decimal _RangeMin = 0.05M;
        [WarpSerializable]
        public decimal RangeMin
        {
            get { return _RangeMin; }
            set
            {
                if (value != _RangeMin)
                {
                    _RangeMin = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _RangeMax = 0.25M;
        [WarpSerializable]
        public decimal RangeMax
        {
            get { return _RangeMax; }
            set
            {
                if (value != _RangeMax)
                {
                    _RangeMax = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _Bfactor = -500;
        [WarpSerializable]
        public decimal Bfactor
        {
            get { return _Bfactor; }
            set { if (value != _Bfactor) { _Bfactor = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsGrids : WarpBase
    {
        private int _CTFX = 5;
        [WarpSerializable]
        public int CTFX
        {
            get { return _CTFX; }
            set { if (value != _CTFX) { _CTFX = value; OnPropertyChanged(); } }
        }

        private int _CTFY = 5;
        [WarpSerializable]
        public int CTFY
        {
            get { return _CTFY; }
            set { if (value != _CTFY) { _CTFY = value; OnPropertyChanged(); } }
        }

        private int _CTFZ = 1;
        [WarpSerializable]
        public int CTFZ
        {
            get { return _CTFZ; }
            set { if (value != _CTFZ) { _CTFZ = value; OnPropertyChanged(); } }
        }

        private int _MovementX = 5;
        [WarpSerializable]
        public int MovementX
        {
            get { return _MovementX; }
            set { if (value != _MovementX) { _MovementX = value; OnPropertyChanged(); } }
        }

        private int _MovementY = 5;
        [WarpSerializable]
        public int MovementY
        {
            get { return _MovementY; }
            set { if (value != _MovementY) { _MovementY = value; OnPropertyChanged(); } }
        }

        private int _MovementZ = 20;
        [WarpSerializable]
        public int MovementZ
        {
            get { return _MovementZ; }
            set { if (value != _MovementZ) { _MovementZ = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsPicking : WarpBase
    {
        private string _ModelPath = "";
        [WarpSerializable]
        public string ModelPath
        {
            get { return _ModelPath; }
            set { if (value != _ModelPath) { _ModelPath = value; OnPropertyChanged(); } }
        }

        private string _DataStyle = "cryo";
        [WarpSerializable]
        public string DataStyle
        {
            get { return _DataStyle; }
            set { if (value != _DataStyle) { _DataStyle = value; OnPropertyChanged(); } }
        }

        private int _Diameter = 200;
        [WarpSerializable]
        public int Diameter
        {
            get { return _Diameter; }
            set { if (value != _Diameter) { _Diameter = value; OnPropertyChanged(); } }
        }

        private decimal _MinimumScore = 0.95M;
        [WarpSerializable]
        public decimal MinimumScore
        {
            get { return _MinimumScore; }
            set { if (value != _MinimumScore) { _MinimumScore = value; OnPropertyChanged(); } }
        }

        private decimal _MinimumMaskDistance = 0;
        [WarpSerializable]
        public decimal MinimumMaskDistance
        {
            get { return _MinimumMaskDistance; }
            set { if (value != _MinimumMaskDistance) { _MinimumMaskDistance = value; OnPropertyChanged(); } }
        }

        private bool _DoExport = false;
        [WarpSerializable]
        public bool DoExport
        {
            get { return _DoExport; }
            set { if (value != _DoExport) { _DoExport = value; OnPropertyChanged(); } }
        }

        private int _BoxSize = 128;
        [WarpSerializable]
        public int BoxSize
        {
            get { return _BoxSize; }
            set { if (value != _BoxSize) { _BoxSize = value; OnPropertyChanged(); } }
        }

        private bool _Invert = true;
        [WarpSerializable]
        public bool Invert
        {
            get { return _Invert; }
            set { if (value != _Invert) { _Invert = value; OnPropertyChanged(); } }
        }

        private bool _Normalize = true;
        [WarpSerializable]
        public bool Normalize
        {
            get { return _Normalize; }
            set { if (value != _Normalize) { _Normalize = value; OnPropertyChanged(); } }
        }

        private bool _DoRunningWindow = true;
        [WarpSerializable]
        public bool DoRunningWindow
        {
            get { return _DoRunningWindow; }
            set { if (value != _DoRunningWindow) { _DoRunningWindow = value; OnPropertyChanged(); } }
        }

        private int _RunningWindowLength = 10000;
        [WarpSerializable]
        public int RunningWindowLength
        {
            get { return _RunningWindowLength; }
            set { if (value != _RunningWindowLength) { _RunningWindowLength = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsTomo : WarpBase
    {
        private decimal _DimensionsX = 3712;
        [WarpSerializable]
        public decimal DimensionsX
        {
            get { return _DimensionsX; }
            set { if (value != _DimensionsX) { _DimensionsX = value; OnPropertyChanged(); } }
        }

        private decimal _DimensionsY = 3712;
        [WarpSerializable]
        public decimal DimensionsY
        {
            get { return _DimensionsY; }
            set { if (value != _DimensionsY) { _DimensionsY = value; OnPropertyChanged(); } }
        }

        private decimal _DimensionsZ = 1400;
        [WarpSerializable]
        public decimal DimensionsZ
        {
            get { return _DimensionsZ; }
            set { if (value != _DimensionsZ) { _DimensionsZ = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsExport : WarpBase
    {
        private bool _DoAverage = true;
        [WarpSerializable]
        public bool DoAverage
        {
            get { return _DoAverage; }
            set
            {
                if (value != _DoAverage)
                {
                    _DoAverage = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _DoStack = false;
        [WarpSerializable]
        public bool DoStack
        {
            get { return _DoStack; }
            set
            {
                if (value != _DoStack)
                {
                    _DoStack = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _DoDeconvolve = false;
        [WarpSerializable]
        public bool DoDeconvolve
        {
            get { return _DoDeconvolve; }
            set { if (value != _DoDeconvolve) { _DoDeconvolve = value; OnPropertyChanged(); } }
        }

        private decimal _DeconvolutionStrength = 1;
        [WarpSerializable]
        public decimal DeconvolutionStrength
        {
            get { return _DeconvolutionStrength; }
            set { if (value != _DeconvolutionStrength) { _DeconvolutionStrength = value; OnPropertyChanged(); } }
        }

        private decimal _DeconvolutionFalloff = 1;
        [WarpSerializable]
        public decimal DeconvolutionFalloff
        {
            get { return _DeconvolutionFalloff; }
            set { if (value != _DeconvolutionFalloff) { _DeconvolutionFalloff = value; OnPropertyChanged(); } }
        }

        private int _StackGroupSize = 1;
        [WarpSerializable]
        public int StackGroupSize
        {
            get { return _StackGroupSize; }
            set
            {
                if (value != _StackGroupSize)
                {
                    _StackGroupSize = value;
                    OnPropertyChanged();
                }
            }
        }

        private int _SkipFirstN = 0;
        [WarpSerializable]
        public int SkipFirstN
        {
            get { return _SkipFirstN; }
            set { if (value != _SkipFirstN) { _SkipFirstN = value; OnPropertyChanged(); } }
        }

        private int _SkipLastN = 0;
        [WarpSerializable]
        public int SkipLastN
        {
            get { return _SkipLastN; }
            set { if (value != _SkipLastN) { _SkipLastN = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsTasks : WarpBase
    {
        #region Common

        private bool _UseRelativePaths = true;
        [WarpSerializable]
        public bool UseRelativePaths
        {
            get { return _UseRelativePaths; }
            set { if (value != _UseRelativePaths) { _UseRelativePaths = value; OnPropertyChanged(); } }
        }

        private bool _IncludeFilteredOut = false;
        [WarpSerializable]
        public bool IncludeFilteredOut
        {
            get { return _IncludeFilteredOut; }
            set { if (value != _IncludeFilteredOut) { _IncludeFilteredOut = value; OnPropertyChanged(); } }
        }

        private bool _IncludeUnselected = false;
        [WarpSerializable]
        public bool IncludeUnselected
        {
            get { return _IncludeUnselected; }
            set { if (value != _IncludeUnselected) { _IncludeUnselected = value; OnPropertyChanged(); } }
        }

        private bool _InputOnePerItem = false;
        [WarpSerializable]
        public bool InputOnePerItem
        {
            get { return _InputOnePerItem; }
            set { if (value != _InputOnePerItem) { _InputOnePerItem = value; OnPropertyChanged(); } }
        }

        private decimal _InputPixelSize = 1;
        [WarpSerializable]
        public decimal InputPixelSize
        {
            get { return _InputPixelSize; }
            set { if (value != _InputPixelSize) { _InputPixelSize = value; OnPropertyChanged(); } }
        }

        private decimal _InputShiftPixelSize = 1;
        [WarpSerializable]
        public decimal InputShiftPixelSize
        {
            get { return _InputShiftPixelSize; }
            set { if (value != _InputShiftPixelSize) { _InputShiftPixelSize = value; OnPropertyChanged(); } }
        }

        private decimal _OutputPixelSize = 1;
        [WarpSerializable]
        public decimal OutputPixelSize
        {
            get { return _OutputPixelSize; }
            set { if (value != _OutputPixelSize) { _OutputPixelSize = value; OnPropertyChanged(); } }
        }

        private string _OutputSuffix = "";
        [WarpSerializable]
        public string OutputSuffix
        {
            get { return _OutputSuffix; }
            set { if (value != _OutputSuffix) { _OutputSuffix = value; OnPropertyChanged(); } }
        }

        private bool _InputInvert = true;
        [WarpSerializable]
        public bool InputInvert
        {
            get { return _InputInvert; }
            set { if (value != _InputInvert) { _InputInvert = value; OnPropertyChanged(); } }
        }

        private bool _InputNormalize = true;
        [WarpSerializable]
        public bool InputNormalize
        {
            get { return _InputNormalize; }
            set { if (value != _InputNormalize) { _InputNormalize = value; OnPropertyChanged(); } }
        }

        private bool _InputFlipX = false;
        [WarpSerializable]
        public bool InputFlipX
        {
            get { return _InputFlipX; }
            set { if (value != _InputFlipX) { _InputFlipX = value; OnPropertyChanged(); } }
        }

        private bool _InputFlipY = false;
        [WarpSerializable]
        public bool InputFlipY
        {
            get { return _InputFlipY; }
            set { if (value != _InputFlipY) { _InputFlipY = value; OnPropertyChanged(); } }
        }

        private bool _OutputNormalize = true;
        [WarpSerializable]
        public bool OutputNormalize
        {
            get { return _OutputNormalize; }
            set { if (value != _OutputNormalize) { _OutputNormalize = value; OnPropertyChanged(); } }
        }

        #endregion

        #region 2D

        private bool _MicListMakePolishing = false;
        [WarpSerializable]
        public bool MicListMakePolishing
        {
            get { return _MicListMakePolishing; }
            set { if (value != _MicListMakePolishing) { _MicListMakePolishing = value; OnPropertyChanged(); } }
        }

        private bool _AdjustDefocusSkipExcluded = true;
        [WarpSerializable]
        public bool AdjustDefocusSkipExcluded
        {
            get { return _AdjustDefocusSkipExcluded; }
            set { if (value != _AdjustDefocusSkipExcluded) { _AdjustDefocusSkipExcluded = value; OnPropertyChanged(); } }
        }

        private bool _AdjustDefocusDeleteExcluded = false;
        [WarpSerializable]
        public bool AdjustDefocusDeleteExcluded
        {
            get { return _AdjustDefocusDeleteExcluded; }
            set { if (value != _AdjustDefocusDeleteExcluded) { _AdjustDefocusDeleteExcluded = value; OnPropertyChanged(); } }
        }

        private decimal _Export2DPixel = 1M;
        [WarpSerializable]
        public decimal Export2DPixel
        {
            get { return _Export2DPixel; }
            set { if (value != _Export2DPixel) { _Export2DPixel = value; OnPropertyChanged(); } }
        }

        private decimal _Export2DBoxSize = 128;
        [WarpSerializable]
        public decimal Export2DBoxSize
        {
            get { return _Export2DBoxSize; }
            set { if (value != _Export2DBoxSize) { _Export2DBoxSize = value; OnPropertyChanged(); } }
        }

        private decimal _Export2DParticleDiameter = 100;
        [WarpSerializable]
        public decimal Export2DParticleDiameter
        {
            get { return _Export2DParticleDiameter; }
            set { if (value != _Export2DParticleDiameter) { _Export2DParticleDiameter = value; OnPropertyChanged(); } }
        }

        private bool _Export2DDoAverages = true;
        [WarpSerializable]
        public bool Export2DDoAverages
        {
            get { return _Export2DDoAverages; }
            set { if (value != _Export2DDoAverages) { _Export2DDoAverages = value; OnPropertyChanged(); } }
        }

        private bool _Export2DDoMovies = false;
        [WarpSerializable]
        public bool Export2DDoMovies
        {
            get { return _Export2DDoMovies; }
            set { if (value != _Export2DDoMovies) { _Export2DDoMovies = value; OnPropertyChanged(); } }
        }

        private bool _Export2DDoOnlyTable = false;
        [WarpSerializable]
        public bool Export2DDoOnlyTable
        {
            get { return _Export2DDoOnlyTable; }
            set { if (value != _Export2DDoOnlyTable) { _Export2DDoOnlyTable = value; OnPropertyChanged(); } }
        }

        private bool _Export2DPreflip = false;
        [WarpSerializable]
        public bool Export2DPreflip
        {
            get { return _Export2DPreflip; }
            set { if (value != _Export2DPreflip) { _Export2DPreflip = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Tomo

        #region Full reconstruction

        private decimal _TomoFullReconstructPixel = 1M;
        [WarpSerializable]
        public decimal TomoFullReconstructPixel
        {
            get { return _TomoFullReconstructPixel; }
            set { if (value != _TomoFullReconstructPixel) { _TomoFullReconstructPixel = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructDoDeconv = false;
        [WarpSerializable]
        public bool TomoFullReconstructDoDeconv
        {
            get { return _TomoFullReconstructDoDeconv; }
            set { if (value != _TomoFullReconstructDoDeconv) { _TomoFullReconstructDoDeconv = value; OnPropertyChanged(); } }
        }

        private decimal _TomoFullReconstructDeconvStrength = 1M;
        [WarpSerializable]
        public decimal TomoFullReconstructDeconvStrength
        {
            get { return _TomoFullReconstructDeconvStrength; }
            set { if (value != _TomoFullReconstructDeconvStrength) { _TomoFullReconstructDeconvStrength = value; OnPropertyChanged(); } }
        }

        private decimal _TomoFullReconstructDeconvFalloff = 1M;
        [WarpSerializable]
        public decimal TomoFullReconstructDeconvFalloff
        {
            get { return _TomoFullReconstructDeconvFalloff; }
            set { if (value != _TomoFullReconstructDeconvFalloff) { _TomoFullReconstructDeconvFalloff = value; OnPropertyChanged(); } }
        }

        private decimal _TomoFullReconstructDeconvHighpass = 300;
        [WarpSerializable]
        public decimal TomoFullReconstructDeconvHighpass
        {
            get { return _TomoFullReconstructDeconvHighpass; }
            set { if (value != _TomoFullReconstructDeconvHighpass) { _TomoFullReconstructDeconvHighpass = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructInvert = true;
        [WarpSerializable]
        public bool TomoFullReconstructInvert
        {
            get { return _TomoFullReconstructInvert; }
            set { if (value != _TomoFullReconstructInvert) { _TomoFullReconstructInvert = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructNormalize = true;
        [WarpSerializable]
        public bool TomoFullReconstructNormalize
        {
            get { return _TomoFullReconstructNormalize; }
            set { if (value != _TomoFullReconstructNormalize) { _TomoFullReconstructNormalize = value; OnPropertyChanged(); } }
        }

        private bool _TomoFullReconstructOnlyFullVoxels = false;
        [WarpSerializable]
        public bool TomoFullReconstructOnlyFullVoxels
        {
            get { return _TomoFullReconstructOnlyFullVoxels; }
            set { if (value != _TomoFullReconstructOnlyFullVoxels) { _TomoFullReconstructOnlyFullVoxels = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Sub reconstruction

        private bool _TomoSubReconstructNormalizedCoords = false;
        [WarpSerializable]
        public bool TomoSubReconstructNormalizedCoords
        {
            get { return _TomoSubReconstructNormalizedCoords; }
            set { if (value != _TomoSubReconstructNormalizedCoords) { _TomoSubReconstructNormalizedCoords = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructPixel = 1M;
        [WarpSerializable]
        public decimal TomoSubReconstructPixel
        {
            get { return _TomoSubReconstructPixel; }
            set { if (value != _TomoSubReconstructPixel) { _TomoSubReconstructPixel = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructBox = 128;
        [WarpSerializable]
        public decimal TomoSubReconstructBox
        {
            get { return _TomoSubReconstructBox; }
            set { if (value != _TomoSubReconstructBox) { _TomoSubReconstructBox = value; OnPropertyChanged(); } }
        }

        private decimal _TomoSubReconstructDiameter = 100;
        [WarpSerializable]
        public decimal TomoSubReconstructDiameter
        {
            get { return _TomoSubReconstructDiameter; }
            set { if (value != _TomoSubReconstructDiameter) { _TomoSubReconstructDiameter = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructVolume = true;
        [WarpSerializable]
        public bool TomoSubReconstructVolume
        {
            get { return _TomoSubReconstructVolume; }
            set { if (value != _TomoSubReconstructVolume) { _TomoSubReconstructVolume = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructSeries = false;
        [WarpSerializable]
        public bool TomoSubReconstructSeries
        {
            get { return _TomoSubReconstructSeries; }
            set { if (value != _TomoSubReconstructSeries) { _TomoSubReconstructSeries = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructPrerotated = false;
        [WarpSerializable]
        public bool TomoSubReconstructPrerotated
        {
            get { return _TomoSubReconstructPrerotated; }
            set { if (value != _TomoSubReconstructPrerotated) { _TomoSubReconstructPrerotated = value; OnPropertyChanged(); } }
        }

        private bool _TomoSubReconstructDoLimitDose = false;
        [WarpSerializable]
        public bool TomoSubReconstructDoLimitDose
        {
            get { return _TomoSubReconstructDoLimitDose; }
            set { if (value != _TomoSubReconstructDoLimitDose) { _TomoSubReconstructDoLimitDose = value; OnPropertyChanged(); } }
        }

        private int _TomoSubReconstructNTilts = 1;
        [WarpSerializable]
        public int TomoSubReconstructNTilts
        {
            get { return _TomoSubReconstructNTilts; }
            set { if (value != _TomoSubReconstructNTilts) { _TomoSubReconstructNTilts = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Template matching

        private decimal _TomoMatchTemplatePixel = 1M;
        [WarpSerializable]
        public decimal TomoMatchTemplatePixel
        {
            get { return _TomoMatchTemplatePixel; }
            set { if (value != _TomoMatchTemplatePixel) { _TomoMatchTemplatePixel = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchTemplateDiameter = 100;
        [WarpSerializable]
        public decimal TomoMatchTemplateDiameter
        {
            get { return _TomoMatchTemplateDiameter; }
            set
            {
                if (value != _TomoMatchTemplateDiameter)
                {
                    _TomoMatchTemplateDiameter = value;
                    OnPropertyChanged();
                    TomoUpdateMatchRecommendation();
                }
            }
        }

        private decimal _TomoMatchTemplateFraction = 100M;
        [WarpSerializable]
        public decimal TomoMatchTemplateFraction
        {
            get { return _TomoMatchTemplateFraction; }
            set { if (value != _TomoMatchTemplateFraction) { _TomoMatchTemplateFraction = value; OnPropertyChanged(); } }
        }

        private bool _TomoMatchWhitenSpectrum = true;
        [WarpSerializable]
        public bool TomoMatchWhitenSpectrum
        {
            get { return _TomoMatchWhitenSpectrum; }
            set { if (value != _TomoMatchWhitenSpectrum) { _TomoMatchWhitenSpectrum = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchHealpixOrder = 1;
        [WarpSerializable]
        public decimal TomoMatchHealpixOrder
        {
            get { return _TomoMatchHealpixOrder; }
            set
            {
                if (value != _TomoMatchHealpixOrder)
                {
                    _TomoMatchHealpixOrder = value;
                    OnPropertyChanged();

                    TomoMatchHealpixAngle = Math.Round(60M / (decimal)Math.Pow(2, (double)value), 3);
                }
            }
        }

        private decimal _TomoMatchHealpixAngle = 30;
        public decimal TomoMatchHealpixAngle
        {
            get { return _TomoMatchHealpixAngle; }
            set
            {
                if (value != _TomoMatchHealpixAngle)
                {
                    _TomoMatchHealpixAngle = value;
                    OnPropertyChanged();
                    TomoUpdateMatchRecommendation();
                }
            }
        }

        private string _TomoMatchSymmetry = "C1";
        [WarpSerializable]
        public string TomoMatchSymmetry
        {
            get { return _TomoMatchSymmetry; }
            set { if (value != _TomoMatchSymmetry) { _TomoMatchSymmetry = value; OnPropertyChanged(); } }
        }

        private decimal _TomoMatchRecommendedAngPix = 25.88M;
        public decimal TomoMatchRecommendedAngPix
        {
            get { return _TomoMatchRecommendedAngPix; }
            set { if (value != _TomoMatchRecommendedAngPix) { _TomoMatchRecommendedAngPix = value; OnPropertyChanged(); } }
        }

        private void TomoUpdateMatchRecommendation()
        {
            float2 AngularSampling = new float2((float)Math.Sin((float)TomoMatchHealpixAngle * Helper.ToRad),
                                                1f - (float)Math.Cos((float)TomoMatchHealpixAngle * Helper.ToRad));
            decimal AtLeast = TomoMatchTemplateDiameter / 2 * (decimal)AngularSampling.Length();
            AtLeast = Math.Round(AtLeast, 2);
            TomoMatchRecommendedAngPix = AtLeast;
        }

        private decimal _TomoMatchNResults = 1000;
        [WarpSerializable]
        public decimal TomoMatchNResults
        {
            get { return _TomoMatchNResults; }
            set { if (value != _TomoMatchNResults) { _TomoMatchNResults = value; OnPropertyChanged(); } }
        }

        #endregion

        #endregion
    }

    public class OptionsFilter : WarpBase
    {
        private decimal _AstigmatismMax = 3;
        [WarpSerializable]
        public decimal AstigmatismMax
        {
            get { return _AstigmatismMax; }
            set { if (value != _AstigmatismMax) { _AstigmatismMax = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusMin = 0;
        [WarpSerializable]
        public decimal DefocusMin
        {
            get { return _DefocusMin; }
            set { if (value != _DefocusMin) { _DefocusMin = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusMax = 5;
        [WarpSerializable]
        public decimal DefocusMax
        {
            get { return _DefocusMax; }
            set { if (value != _DefocusMax) { _DefocusMax = value; OnPropertyChanged(); } }
        }

        private decimal _PhaseMin = 0;
        [WarpSerializable]
        public decimal PhaseMin
        {
            get { return _PhaseMin; }
            set { if (value != _PhaseMin) { _PhaseMin = value; OnPropertyChanged(); } }
        }

        private decimal _PhaseMax = 1;
        [WarpSerializable]
        public decimal PhaseMax
        {
            get { return _PhaseMax; }
            set { if (value != _PhaseMax) { _PhaseMax = value; OnPropertyChanged(); } }
        }

        private decimal _ResolutionMax = 5;
        [WarpSerializable]
        public decimal ResolutionMax
        {
            get { return _ResolutionMax; }
            set { if (value != _ResolutionMax) { _ResolutionMax = value; OnPropertyChanged(); } }
        }

        private decimal _MotionMax = 5;
        [WarpSerializable]
        public decimal MotionMax
        {
            get { return _MotionMax; }
            set { if (value != _MotionMax) { _MotionMax = value; OnPropertyChanged(); } }
        }

        private string _ParticlesSuffix = "";
        [WarpSerializable]
        public string ParticlesSuffix
        {
            get { return _ParticlesSuffix; }
            set { if (value != _ParticlesSuffix) { _ParticlesSuffix = value; OnPropertyChanged(); } }
        }

        private int _ParticlesMin = 1;
        [WarpSerializable]
        public int ParticlesMin
        {
            get { return _ParticlesMin; }
            set { if (value != _ParticlesMin) { _ParticlesMin = value; OnPropertyChanged(); } }
        }

        private decimal _MaskPercentage = 10;
        [WarpSerializable]
        public decimal MaskPercentage
        {
            get { return _MaskPercentage; }
            set { if (value != _MaskPercentage) { _MaskPercentage = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsAdvanced : WarpBase
    {
        private int _ProjectionOversample = 2;
        [WarpSerializable]
        public int ProjectionOversample
        {
            get { return _ProjectionOversample; }
            set { if (value != _ProjectionOversample) { _ProjectionOversample = value; OnPropertyChanged(); } }
        }
    }

    public class OptionsRuntime : WarpBase
    {
        private int _DeviceCount = 0;
        public int DeviceCount
        {
            get { return _DeviceCount; }
            set { if (value != _DeviceCount) { _DeviceCount = value; OnPropertyChanged(); } }
        }

        private decimal _BinnedPixelSizeMean = 1M;
        public decimal BinnedPixelSizeMean
        {
            get { return _BinnedPixelSizeMean; }
            set { if (value != _BinnedPixelSizeMean) { _BinnedPixelSizeMean = value; OnPropertyChanged(); } }
        }

        private string _GainReferenceHash = "";
        public string GainReferenceHash
        {
            get { return _GainReferenceHash; }
            set { if (value != _GainReferenceHash) { _GainReferenceHash = value; OnPropertyChanged(); } }
        }

        private string _DefectMapHash = "";
        public string DefectMapHash
        {
            get { return _DefectMapHash; }
            set { if (value != _DefectMapHash) { _DefectMapHash = value; OnPropertyChanged(); } }
        }

        private string _GPUStats = "";
        public string GPUStats
        {
            get { return _GPUStats; }
            set { if (value != _GPUStats) { _GPUStats = value; OnPropertyChanged(); } }
        }

        private Movie _DisplayedMovie = null;
        public Movie DisplayedMovie
        {
            get { return _DisplayedMovie; }
            set
            {
                if (value != _DisplayedMovie)
                {
                    _DisplayedMovie = value;
                    OnPropertyChanged();
                }
            }
        }

        private int _OverviewPlotHighlightID = -1;
        public int OverviewPlotHighlightID
        {
            get { return _OverviewPlotHighlightID; }
            set { if (value != _OverviewPlotHighlightID) { _OverviewPlotHighlightID = value; OnPropertyChanged(); } }
        }
    }
}