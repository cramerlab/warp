using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Warp.Tools;

namespace Warp.Sociology
{
    [Serializable]
    public class DataSource : WarpBase
    {
        private Guid _GUID = Guid.NewGuid();
        [WarpSerializable]
        public Guid GUID
        {
            get { return _GUID; }
            set { if (value != _GUID) { _GUID = value; OnPropertyChanged(); } }
        }

        private string _Version = "";
        [WarpSerializable]
        public string Version
        {
            get { return _Version; }
            set { if (value != _Version) { _Version = value; OnPropertyChanged(); } }
        }

        private string _PreviousVersion = "";
        [WarpSerializable]
        public string PreviousVersion
        {
            get { return _PreviousVersion; }
            set { if (value != _PreviousVersion) { _PreviousVersion = value; OnPropertyChanged(); } }
        }

        private string _Name = "New Source";
        [WarpSerializable]
        public string Name
        {
            get { return _Name; }
            set
            {
                if (value != _Name)
                {
                    _Name = value;
                    OnPropertyChanged();
                }
            }
        }

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

        public string FolderPath => Path.Substring(0, Math.Max(Path.LastIndexOf("\\"), Path.LastIndexOf("/")) + 1);

        public bool IsRemote
        {
            get
            {
                try
                {
                    return !new Uri(Path).IsFile;
                }
                catch
                {
                    return false;
                }
            }
        }

        public bool IsTiltSeries => Files.Count > 0 && Files.First().Value.Contains(".tomostar");

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

        public decimal PixelSizeMean => (PixelSizeX + PixelSizeY) * 0.5M;

        #endregion

        #region Dimensions

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

        private int _FrameLimit = -1;
        [WarpSerializable]
        public int FrameLimit
        {
            get { return _FrameLimit; }
            set { if (value != _FrameLimit) { _FrameLimit = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Gain

        private string _GainPath = "";
        public string GainPath
        {
            get { return _GainPath; }
            set { if (value != _GainPath) { _GainPath = value; OnPropertyChanged(); } }
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

        private string _DefectsPath = "";
        public string DefectsPath
        {
            get { return _DefectsPath; }
            set { if (value != _DefectsPath) { _DefectsPath = value; OnPropertyChanged(); } }
        }

        #endregion

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

        /// <summary>
        /// Keys are hashes, values are file names
        /// </summary>
        public Dictionary<string, string> Files = new Dictionary<string, string>();

        public Dictionary<Guid, string> UsedSpecies = new Dictionary<Guid, string>();

        public DataSource()
        {

        }

        #region Load & save

        public void Save()
        {
            XmlTextWriter Writer = new XmlTextWriter(File.Create(Path), Encoding.Unicode);
            Writer.Formatting = Formatting.Indented;
            Writer.IndentChar = '\t';
            Writer.Indentation = 1;
            Writer.WriteStartDocument();
            Writer.WriteStartElement("DataSource");

            WriteToXML(Writer);

            XMLHelper.WriteParamNode(Writer, "GainPath", Helper.MakePathRelativeTo(GainPath, FolderPath));
            XMLHelper.WriteParamNode(Writer, "DefectsPath", Helper.MakePathRelativeTo(DefectsPath, FolderPath));

            Writer.WriteStartElement("UsedSpecies");
            foreach (var source in UsedSpecies)
            {
                Writer.WriteStartElement("Species");
                XMLHelper.WriteAttribute(Writer, "GUID", source.Key.ToString());
                XMLHelper.WriteAttribute(Writer, "Version", source.Value);
                Writer.WriteEndElement();
            }
            Writer.WriteEndElement();

            Writer.WriteStartElement("Files");
            foreach (var file in Files)
            {
                Writer.WriteStartElement("File");
                XMLHelper.WriteAttribute(Writer, "Hash", file.Key);
                XMLHelper.WriteAttribute(Writer, "Name", file.Value);
                Writer.WriteEndElement();
            }
            Writer.WriteEndElement();

            Writer.WriteEndElement(); 
            Writer.WriteEndDocument();
            Writer.Flush();
            Writer.Close();
        }

        public void Load(string path)
        {
            Path = path;

            if (!IsRemote)
            {
                using (Stream SettingsStream = File.OpenRead(path))
                {
                    XPathDocument Doc = new XPathDocument(SettingsStream);
                    XPathNavigator Reader = Doc.CreateNavigator();
                    Reader.MoveToRoot();

                    Reader.MoveToRoot();
                    Reader.MoveToChild("DataSource", "");

                    ReadFromXML(Reader);

                    GainPath = XMLHelper.LoadParamNode(Reader, "GainPath", GainPath);
                    if (!string.IsNullOrEmpty(GainPath))
                        GainPath = System.IO.Path.Combine(FolderPath, GainPath);

                    DefectsPath = XMLHelper.LoadParamNode(Reader, "DefectsPath", DefectsPath);
                    if (!string.IsNullOrEmpty(DefectsPath))
                        DefectsPath = System.IO.Path.Combine(FolderPath, DefectsPath);

                    UsedSpecies = new Dictionary<Guid, string>();
                    foreach (XPathNavigator nav in Reader.Select("UsedSpecies/Species"))
                    {
                        Guid SpeciesGUID = Guid.Parse(nav.GetAttribute("GUID", ""));
                        string SpeciesVersion = nav.GetAttribute("Version", "");

                        UsedSpecies.Add(SpeciesGUID, SpeciesVersion);
                    }

                    Files = new Dictionary<string, string>();
                    foreach (XPathNavigator nav in Reader.Select("Files/File"))
                        Files.Add(nav.GetAttribute("Hash", ""), nav.GetAttribute("Name", ""));
                }
            }
        }

        #endregion

        #region Version control

        public string GetDataHash()
        {
            StringBuilder Builder = new StringBuilder();
            foreach (var file in Files)
                Builder.Append(file.Key);

            return MathHelper.GetSHA1(Helper.ToBytes(Builder.ToString().ToCharArray()));
        }

        public void ComputeVersionHash()
        {
            StringBuilder Builder = new StringBuilder();

            foreach (var file in Files)
            {
                Movie Movie = IsTiltSeries ? new TiltSeries(FolderPath + file.Value) : new Movie(FolderPath + file.Value);
                Builder.Append(Movie.GetProcessingHash());
            }

            foreach (var species in UsedSpecies)
            {
                Builder.Append(species.Key);
                Builder.Append(species.Value);
            }

            Version = MathHelper.GetSHA1(Helper.ToBytes(Builder.ToString().ToCharArray()));
        }

        public void Commit()
        {
            PreviousVersion = Version;
            ComputeVersionHash();

            if (PreviousVersion != null && Version == PreviousVersion)
                return;
            
            string VersionFolderPath = FolderPath + "versions/" + Version + "/";
            Directory.CreateDirectory(VersionFolderPath);

            string FileName = Helper.PathToNameWithExtension(Path);
            string OriginalFolderPath = FolderPath;

            Path = VersionFolderPath + FileName;
            Save();

            foreach (var file in Files)
            {
                string NameXML = Helper.PathToName(file.Value) + ".xml";
                if (!File.Exists(VersionFolderPath + NameXML))
                    File.Copy(OriginalFolderPath + NameXML, VersionFolderPath + NameXML);
            }

            Path = OriginalFolderPath + FileName;
            Save();
        }

        #endregion

        public Image LoadAndPrepareGainReference()
        {
            if (string.IsNullOrEmpty(GainPath))
                return null;

            Image Gain = Image.FromFile(GainPath, new int2(1), 0, typeof(float));

            float Mean = MathHelper.Mean(Gain.GetHost(Intent.Read)[0]);
            Gain.TransformValues(v => v == 0 ? 1 : v / Mean);

            if (GainFlipX)
                Gain = Gain.AsFlippedX();
            if (GainFlipY)
                Gain = Gain.AsFlippedY();
            if (GainTranspose)
                Gain = Gain.AsTransposed();

            return Gain;
        }

        public static DataSource FromFile(string path)
        {
            DataSource Result = new DataSource();
            Result.Load(path);

            return Result;
        }
    }
}
