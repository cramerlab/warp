using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Management;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Warp.WarpAnalytics;

namespace Warp
{
    public class GlobalOptions : WarpBase
    {
        private Guid Secret = Guid.Parse("5527e951-beab-46d3-ba75-73ea94d1a9df");

        private bool _PromptShown = false;
        [WarpSerializable]
        public bool PromptShown
        {
            get { return _PromptShown; }
            set
            {
                if (value != _PromptShown)
                {
                    _PromptShown = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _AllowCollection = false;
        [WarpSerializable]
        public bool AllowCollection
        {
            get { return _AllowCollection; }
            set
            {
                if (value != _AllowCollection)
                {
                    _AllowCollection = value;
                    OnPropertyChanged();
                }
            }
        }

        private bool _ShowBoxNetReminder = true;
        [WarpSerializable]
        public bool ShowBoxNetReminder
        {
            get { return _ShowBoxNetReminder; }
            set { if (value != _ShowBoxNetReminder) { _ShowBoxNetReminder = value; OnPropertyChanged(); } }
        }

        private bool _CheckForUpdates = true;
        [WarpSerializable]
        public bool CheckForUpdates
        {
            get { return _CheckForUpdates; }
            set { if (value != _CheckForUpdates) { _CheckForUpdates = value; OnPropertyChanged(); } }
        }

        private bool _ShowTiffReminder = true;
        [WarpSerializable]
        public bool ShowTiffReminder
        {
            get { return _ShowTiffReminder; }
            set { if (value != _ShowTiffReminder) { _ShowTiffReminder = value; OnPropertyChanged(); } }
        }

        private int _ProcessesPerDevice = 1;
        [WarpSerializable]
        public int ProcessesPerDevice
        {
            get { return _ProcessesPerDevice; }
            set { if (value != _ProcessesPerDevice) { _ProcessesPerDevice = value; OnPropertyChanged(); } }
        }

        private int _APIPort = -1;
        [WarpSerializable]
        public int APIPort
        {
            get { return _APIPort; }
            set { if (value != _APIPort) { _APIPort = value; OnPropertyChanged(); } }
        }

        public Version GetLatestVersion()
        {
            return new WarpAnalyticsClient().GetLatestVersion();
        }

        public void LogEnvironment()
        {
            if (!AllowCollection)
                return;

            Task.Run(() =>
            {
                try
                {
                    int CPUCores = Environment.ProcessorCount;

                    ulong CPUMemory = 0;
                    ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT Capacity FROM Win32_PhysicalMemory");
                    foreach (ManagementObject WniPART in searcher.Get())
                        CPUMemory += Convert.ToUInt64(WniPART.Properties["Capacity"].Value);
                    CPUMemory >>= 20;

                    int CPUClock = 0;
                    using (ManagementObject Mo = new ManagementObject("Win32_Processor.DeviceID='CPU0'"))
                        CPUClock = (int)(uint)(Mo["MaxClockSpeed"]);

                    int GPUCores = GPU.GetDeviceCount();
                    int GPUMemory = (int)GPU.GetTotalMemory(0);
                    IntPtr NamePtr = GPU.GetDeviceName(0);
                    string GPUName = new string(Marshal.PtrToStringAnsi(NamePtr).Take(48).ToArray());
                    CPU.HostFree(NamePtr);

                    Version Version = Assembly.GetExecutingAssembly().GetName().Version;

                    new WarpAnalyticsClient().LogEnvironment(Secret,
                                                             DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                                                             Version.ToString(),
                                                             CPUCores,
                                                             CPUClock,
                                                             (int)CPUMemory,
                                                             GPUCores,
                                                             GPUMemory,
                                                             GPUName);
                }
                catch
                {
                }
            });
        }

        public void LogProcessingMovement(ProcessingOptionsMovieMovement options, float meanShift)
        {
            if (!AllowCollection)
                return;

            Task.Run(() =>
            {
                try
                {
                    new WarpAnalyticsClient().LogProcessingMotion(Secret,
                                                                  DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                                                                  (float)options.PixelSizeX,
                                                                  (float)options.PixelSizeY,
                                                                  (float)options.PixelSizeAngle,
                                                                  (float)options.BinTimes,
                                                                  !string.IsNullOrEmpty(options.GainPath),
                                                                  (int)(options.Dimensions.X / (float)options.PixelSizeMean),
                                                                  (int)(options.Dimensions.Y / (float)options.PixelSizeMean),
                                                                  (int)options.Dimensions.Z,
                                                                  (float)options.RangeMin,
                                                                  (float)options.RangeMax,
                                                                  (int)options.Bfactor,
                                                                  options.GridDims.X,
                                                                  options.GridDims.Y,
                                                                  options.GridDims.Z,
                                                                  meanShift);
                }
                catch
                {
                }
            });
        }

        public void LogProcessingCTF(ProcessingOptionsMovieCTF options, CTF ctf, float resolution)
        {
            if (!AllowCollection)
                return;

            Task.Run(() =>
            {
                try
                {
                    new WarpAnalyticsClient().LogProcessingCTF(Secret,
                                                               DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                                                               (float)options.PixelSizeX,
                                                               (float)options.PixelSizeY,
                                                               (float)options.PixelSizeAngle,
                                                               (float)options.BinTimes,
                                                               !string.IsNullOrEmpty(options.GainPath),
                                                               (int)(options.Dimensions.X / (float)options.PixelSizeMean),
                                                               (int)(options.Dimensions.Y / (float)options.PixelSizeMean),
                                                               (int)options.Dimensions.Z,
                                                               options.Window,
                                                               (float)options.RangeMin,
                                                               (float)options.RangeMax,
                                                               options.Voltage,
                                                               (float)options.Cs,
                                                               (float)options.Cc,
                                                               (float)options.IllumAngle,
                                                               (float)options.EnergySpread,
                                                               (float)options.Thickness,
                                                               (float)options.Amplitude,
                                                               options.DoPhase,
                                                               options.DoIce,
                                                               options.UseMovieSum,
                                                               (float)options.ZMin,
                                                               (float)options.ZMax,
                                                               options.GridDims.X,
                                                               options.GridDims.Y,
                                                               options.GridDims.Z,
                                                               (float)ctf.Defocus,
                                                               (float)ctf.DefocusDelta,
                                                               (float)ctf.DefocusAngle,
                                                               (float)ctf.PhaseShift,
                                                               resolution);
                }
                catch
                {
                }
            });
        }

        public void LogProcessingBoxNet(ProcessingOptionsBoxNet options, int nparticles)
        {
            if (!AllowCollection)
                return;

            Task.Run(() =>
            {
                try
                {
                    new WarpAnalyticsClient().LogProcessingBoxNet(Secret,
                                                                  DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                                                                  2,
                                                                  (int)options.ExpectedDiameter,
                                                                  (float)options.MinimumScore,
                                                                  (float)options.MinimumMaskDistance,
                                                                  options.ExportParticles,
                                                                  options.ExportBoxSize,
                                                                  options.ExportInvert,
                                                                  options.ExportNormalize,
                                                                  options.ModelName.ToLower().Contains("mask"),
                                                                  nparticles);
                }
                catch
                {
                }
            });
        }

        public void LogCrash(Exception exception)
        {
            if (!AllowCollection)
                return;

            try
            {
                int CPUCores = Environment.ProcessorCount;

                ulong CPUMemory = 0;
                ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT Capacity FROM Win32_PhysicalMemory");
                foreach (ManagementObject WniPART in searcher.Get())
                    CPUMemory += Convert.ToUInt64(WniPART.Properties["Capacity"].Value);
                CPUMemory >>= 20;

                int CPUClock = 0;
                using (ManagementObject Mo = new ManagementObject("Win32_Processor.DeviceID='CPU0'"))
                    CPUClock = (int)(uint)(Mo["MaxClockSpeed"]);

                int GPUCores = GPU.GetDeviceCount();
                int GPUMemory = (int)GPU.GetTotalMemory(0);
                IntPtr NamePtr = GPU.GetDeviceName(0);
                string GPUName = new string(Marshal.PtrToStringAnsi(NamePtr).Take(48).ToArray());
                CPU.HostFree(NamePtr);

                Version Version = Assembly.GetExecutingAssembly().GetName().Version;

                new WarpAnalyticsClient().LogCrash(Secret,
                                                    DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                                                    Version.ToString(),
                                                    CPUCores,
                                                    CPUClock,
                                                    (int)CPUMemory,
                                                    GPUCores,
                                                    GPUMemory,
                                                    GPUName,
                                                    exception.ToString());
            }
            catch
            {
            }
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
                }
            }
            catch { }
        }

        public void Save(string path)
        {
            XmlTextWriter Writer = new XmlTextWriter(File.Create(path), Encoding.Unicode);
            Writer.Formatting = Formatting.Indented;
            Writer.IndentChar = '\t';
            Writer.Indentation = 1;
            Writer.WriteStartDocument();
            Writer.WriteStartElement("Settings");

            WriteToXML(Writer);

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
            Writer.Flush();
            Writer.Close();
        }
    }
}
