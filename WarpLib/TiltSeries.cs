using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Xml;
using System.Xml.XPath;
using Accord;
using Accord.Math.Optimization;
using MathNet.Numerics;
using Warp.Headers;
using Warp.Tools;

namespace Warp
{
    public class TiltSeries : Movie
    {
        #region Directories

        public string ReconstructionDir => DirectoryName + "reconstruction\\";

        public string ReconstructionDeconvDir => ReconstructionDir + "deconv\\";

        public string SubtomoDir => DirectoryName + "subtomo\\" + RootName + "\\";

        public string WeightOptimizationDir => DirectoryName + "weightoptimization\\";

        #endregion

        public float GlobalWeight = 1;
        public float GlobalBfactor = 0;

        #region Runtime dimensions
        // These must be populated before most operations, otherwise exceptions will be thrown.
        // Not an elegant solution, but it avoids passing them to a lot of methods.
        // Given in Angstrom.

        public float3 VolumeDimensionsPhysical;
        public float2[] ImageDimensions;

        #endregion

        private bool _AreAnglesInverted = false;
        public bool AreAnglesInverted
        {
            get { return _AreAnglesInverted; }
            set { if (value != _AreAnglesInverted) { _AreAnglesInverted = value; OnPropertyChanged(); } }
        }

        public float3 PlaneNormal;

        #region Grids

        private CubicGrid _GridAngleWeights = new CubicGrid(new int3(1, 1, 1), 1, 1, Dimension.X);
        public CubicGrid GridAngleWeights
        {
            get { return _GridAngleWeights; }
            set { if (value != _GridAngleWeights) { _GridAngleWeights = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridDoseBfacs = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridDoseBfacs
        {
            get { return _GridDoseBfacs; }
            set { if (value != _GridDoseBfacs) { _GridDoseBfacs = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridCTFDefocusDelta
        {
            get { return _GridCTFDefocusDelta; }
            set { if (value != _GridCTFDefocusDelta) { _GridCTFDefocusDelta = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridCTFDefocusAngle
        {
            get { return _GridCTFDefocusAngle; }
            set { if (value != _GridCTFDefocusAngle) { _GridCTFDefocusAngle = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridLocalZ = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridLocalZ
        {
            get { return _GridLocalZ; }
            set { if (value != _GridLocalZ) { _GridLocalZ = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridAngleX = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridAngleX
        {
            get { return _GridAngleX; }
            set { if (value != _GridAngleX) { _GridAngleX = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridAngleY = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridAngleY
        {
            get { return _GridAngleY; }
            set { if (value != _GridAngleY) { _GridAngleY = value; OnPropertyChanged(); } }
        }

        private CubicGrid _GridAngleZ = new CubicGrid(new int3(1, 1, 1));
        public CubicGrid GridAngleZ
        {
            get { return _GridAngleZ; }
            set { if (value != _GridAngleZ) { _GridAngleZ = value; OnPropertyChanged(); } }
        }

        #endregion

        #region Per-tilt CTF data

        private ObservableCollection<float2[]> _TiltPS1D = new ObservableCollection<float2[]>();
        public ObservableCollection<float2[]> TiltPS1D
        {
            get { return _TiltPS1D; }
            set { if (value != _TiltPS1D) { _TiltPS1D = value; OnPropertyChanged(); } }
        }

        private ObservableCollection<Cubic1D> _TiltSimulatedBackground = new ObservableCollection<Cubic1D>();
        public ObservableCollection<Cubic1D> TiltSimulatedBackground
        {
            get { return _TiltSimulatedBackground; }
            set { if (value != _TiltSimulatedBackground) { _TiltSimulatedBackground = value; OnPropertyChanged(); } }
        }

        private ObservableCollection<Cubic1D> _TiltSimulatedScale = new ObservableCollection<Cubic1D>();
        public ObservableCollection<Cubic1D> TiltSimulatedScale
        {
            get { return _TiltSimulatedScale; }
            set { if (value != _TiltSimulatedScale) { _TiltSimulatedScale = value; OnPropertyChanged(); } }
        }

        public float GetTiltDefocus(int tiltID)
        {
            if (GridCTF != null && GridCTF.FlatValues.Length > tiltID)
                return GridCTF.FlatValues[tiltID];
            return 0;
        }

        public float GetTiltDefocusDelta(int tiltID)
        {
            if (GridCTFDefocusDelta != null && GridCTFDefocusDelta.FlatValues.Length > tiltID)
                return GridCTFDefocusDelta.FlatValues[tiltID];
            return 0;
        }

        public float GetTiltDefocusAngle(int tiltID)
        {
            if (GridCTFDefocusAngle != null && GridCTFDefocusAngle.FlatValues.Length > tiltID)
                return GridCTFDefocusAngle.FlatValues[tiltID];
            return 0;
        }

        public float GetTiltPhase(int tiltID)
        {
            if (GridCTFPhase != null && GridCTFPhase.FlatValues.Length > tiltID)
                return GridCTFPhase.FlatValues[tiltID];
            return 0;
        }

        public CTF GetTiltCTF(int tiltID)
        {
            CTF Result = CTF.GetCopy();
            Result.Defocus = (decimal)GetTiltDefocus(tiltID);
            Result.DefocusDelta = (decimal)GetTiltDefocusDelta(tiltID);
            Result.DefocusAngle = (decimal)GetTiltDefocusAngle(tiltID);
            Result.PhaseShift = (decimal)GetTiltPhase(tiltID);

            return Result;
        }

        public float2[] GetTiltSimulated1D(int tiltID)
        {
            if (TiltPS1D.Count <= tiltID || 
                TiltPS1D[tiltID] == null ||
                TiltSimulatedScale.Count <= tiltID ||
                TiltSimulatedScale[tiltID] == null)
                return null;

            CTF TiltCTF = GetTiltCTF(tiltID);

            float[] SimulatedCTF = TiltCTF.Get1DWithIce(TiltPS1D[tiltID].Length, true);

            float2[] Result = new float2[SimulatedCTF.Length];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = new float2(TiltPS1D[tiltID][i].X, SimulatedCTF[i] *
                                                              TiltSimulatedScale[tiltID].Interp(TiltPS1D[tiltID][i].X));

            return Result;
        }

        public event Action TiltCTFProcessed;

        #endregion

        #region Per-tilt parameters

        public float[] Angles = { 0 };
        public float[] Dose = { 0 };
        public bool[] UseTilt = { true };

        public float[] TiltAxisAngles = { 0 };
        public float[] TiltAxisOffsetX = { 0 };
        public float[] TiltAxisOffsetY = { 0 };
        public string[] TiltMoviePaths = { "" };

        public float[] AnglesCorrect => Angles.Select(v => AreAnglesInverted ? v : v).ToArray();

        public int[] IndicesSortedAngle
        {
            get
            {
                if (Angles == null)
                    return null;

                List<int> Sorted = new List<int>(Angles.Length);
                for (int i = 0; i < Angles.Length; i++)
                    Sorted.Add(i);

                Sorted.Sort((a, b) => Angles[a].CompareTo(Angles[b]));

                return Sorted.ToArray();
            }
        }

        public int[] IndicesSortedAbsoluteAngle
        {
            get
            {
                if (Angles == null)
                    return null;

                List<int> Sorted = new List<int>(Angles.Length);
                for (int i = 0; i < Angles.Length; i++)
                    Sorted.Add(i);

                Sorted.Sort((a, b) => Math.Abs(Angles[a]).CompareTo(Math.Abs(Angles[b])));

                return Sorted.ToArray();
            }
        }

        private int[] _IndicesSortedDose;
        public int[] IndicesSortedDose
        {
            get
            {
                if (Dose == null)
                    return null;

                if (_IndicesSortedDose == null)
                {
                    List<int> Sorted = new List<int>(Dose.Length);
                    for (int i = 0; i < Dose.Length; i++)
                        Sorted.Add(i);

                    Sorted.Sort((a, b) => Dose[a].CompareTo(Dose[b]));

                    _IndicesSortedDose = Sorted.ToArray();
                }

                return _IndicesSortedDose;
            }
        }

        public int NUniqueTilts
        {
            get
            {
                HashSet<float> UniqueAngles = new HashSet<float>();
                foreach (var angle in Angles)
                    if (!UniqueAngles.Contains(angle))
                        UniqueAngles.Add(angle);

                return UniqueAngles.Count;
            }
        }

        public int NTilts => Angles.Length;

        public float MinTilt => MathHelper.Min(Angles);
        public float MaxTilt => MathHelper.Max(Angles);

        public float MinDose => MathHelper.Min(Dose);
        public float MaxDose => MathHelper.Max(Dose);

        #endregion

        public TiltSeries(string path) : base(path)
        {
            if (Angles.Length <= 1)   // In case angles and dose haven't been read and stored in .xml yet.
            {
                Star Table = new Star(path);

                if (!Table.HasColumn("wrpDose") || !Table.HasColumn("wrpAngleTilt"))
                    throw new Exception("STAR file has no wrpDose or wrpTilt column.");

                List<float> TempAngles = new List<float>();
                List<float> TempDose = new List<float>();
                List<float> TempAxisAngles = new List<float>();
                List<float> TempOffsetX = new List<float>();
                List<float> TempOffsetY = new List<float>();
                List<string> TempMoviePaths = new List<string>();

                for (int i = 0; i < Table.RowCount; i++)
                {
                    TempAngles.Add(float.Parse(Table.GetRowValue(i, "wrpAngleTilt")));
                    TempDose.Add(float.Parse(Table.GetRowValue(i, "wrpDose")));

                    if (Table.HasColumn("wrpAxisAngle"))
                        TempAxisAngles.Add(float.Parse(Table.GetRowValue(i, "wrpAxisAngle"), CultureInfo.InvariantCulture));
                    else
                        TempAxisAngles.Add(0);

                    if (Table.HasColumn("wrpAxisOffsetX") && Table.HasColumn("wrpAxisOffsetY"))
                    {
                        TempOffsetX.Add(float.Parse(Table.GetRowValue(i, "wrpAxisOffsetX"), CultureInfo.InvariantCulture));
                        TempOffsetY.Add(float.Parse(Table.GetRowValue(i, "wrpAxisOffsetY"), CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        TempOffsetX.Add(0);
                        TempOffsetY.Add(0);
                    }

                    if (Table.HasColumn("wrpMovieName"))
                        TempMoviePaths.Add(Table.GetRowValue(i, "wrpMovieName"));
                }

                if (TempAngles.Count == 0 || TempMoviePaths.Count == 0)
                    throw new Exception("Metadata must contain at least 3 values per tilt: movie paths, tilt angles, and accumulated dose.");

                Angles = TempAngles.ToArray();
                Dose = TempDose.ToArray();
                TiltAxisAngles = TempAxisAngles.Count > 0 ? TempAxisAngles.ToArray() : Helper.ArrayOfConstant(0f, NTilts);
                TiltAxisOffsetX = TempOffsetX.Count > 0 ? TempOffsetX.ToArray() : Helper.ArrayOfConstant(0f, NTilts);
                TiltAxisOffsetY = TempOffsetY.Count > 0 ? TempOffsetY.ToArray() : Helper.ArrayOfConstant(0f, NTilts);

                TiltMoviePaths = TempMoviePaths.ToArray();

                UseTilt = Helper.ArrayOfConstant(true, NTilts);
            }
        }

        #region Processing tasks

        #region CTF fitting

        public override void ProcessCTF(Image originalStack, ProcessingOptionsMovieCTF options)
        {
            if (!Directory.Exists(PowerSpectrumDir))
                Directory.CreateDirectory(PowerSpectrumDir);

            AreAnglesInverted = false;
            float LastFittedAngle = 9999f;
            float AverageDose = Dose[IndicesSortedDose.Last()] / NTilts;
            List<int> ProcessedIndices = new List<int>();

            CTF[] FitCTF = new CTF[NTilts];
            float2[] FitPlaneAngles = new float2[NTilts];
            float2[][] FitPS1D = new float2[NTilts][];
            Cubic1D[] FitBackground = new Cubic1D[NTilts];
            Cubic1D[] FitScale = new Cubic1D[NTilts];
            Image[] FitPS2D = new Image[NTilts];

            float[][] StackData = originalStack.GetHost(Intent.Read);

            TiltPS1D.Clear();
            TiltSimulatedBackground.Clear();
            TiltSimulatedScale.Clear();
            for (int i = 0; i < NTilts; i++)
            {
                TiltPS1D.Add(null);
                TiltSimulatedBackground.Add(null);
                TiltSimulatedScale.Add(null);
            }
            TiltCTFProcessed?.Invoke();

            #region Get astigmatism from lower tilts

            List<float> AstigmatismDeltas = new List<float>();
            List<float> AstigmatismAngles = new List<float>();

            List<float3> PlaneNormals = new List<float3>();
            List<float> PlaneNormalTilts = new List<float>();

            for (int i = 0; i < Math.Min(NTilts, 14); i++)
            {
                int AngleID = IndicesSortedAbsoluteAngle[i];
                Image UncroppedAngleImage = new Image(StackData[AngleID], originalStack.Dims.Slice());
                Image AngleImage = UncroppedAngleImage.AsPadded(UncroppedAngleImage.Dims - new int3(200, 200, 0));
                UncroppedAngleImage.Dispose();

                int BestPrevious = -1;
                if (Math.Abs(LastFittedAngle - Angles[AngleID]) <= 5.1f)
                    BestPrevious = IndicesSortedAbsoluteAngle[i - 1];
                else if (ProcessedIndices.Count > 0)
                {
                    List<int> SortedProcessed = new List<int>(ProcessedIndices);
                    SortedProcessed.Sort((a, b) => Math.Abs(Angles[AngleID] - Angles[a]).CompareTo(Math.Abs(Angles[AngleID] - Angles[b])));
                    if (Math.Abs(Dose[SortedProcessed.First()] - Dose[AngleID]) < AverageDose * 5f)
                        BestPrevious = SortedProcessed.First();
                }

                CTF ThisCTF;
                float2 ThisPlaneAngles;
                float2[] ThisPS1D;
                Cubic1D ThisBackground, ThisScale;
                Image ThisPS2D;

                CTF PrevCTF = BestPrevious >= 0 ? FitCTF[BestPrevious] : null;
                float2 PrevPlaneAngles = BestPrevious >= 0 ? FitPlaneAngles[BestPrevious] : new float2();
                Cubic1D PrevBackground = BestPrevious >= 0 ? FitBackground[BestPrevious] : null;
                Cubic1D PrevScale = BestPrevious >= 0 ? FitScale[BestPrevious] : null;

                ProcessCTFOneAngle(options,
                                   AngleImage,
                                   Angles[AngleID],
                                   BestPrevious < 0,
                                   false,
                                   new float2(0, 0),
                                   PrevCTF,
                                   PrevPlaneAngles,
                                   PrevBackground,
                                   PrevScale,
                                   new float3(0, 0, 0),
                                   false,
                                   out ThisCTF,
                                   out ThisPlaneAngles,
                                   out ThisPS1D,
                                   out ThisBackground,
                                   out ThisScale,
                                   out ThisPS2D);
                AngleImage.Dispose();

                FitCTF[AngleID] = ThisCTF;
                FitPlaneAngles[AngleID] = ThisPlaneAngles;
                FitPS1D[AngleID] = ThisPS1D;
                FitBackground[AngleID] = ThisBackground;
                FitScale[AngleID] = ThisScale;
                FitPS2D[AngleID] = ThisPS2D;

                LastFittedAngle = Angles[AngleID];
                ProcessedIndices.Add(AngleID);

                AstigmatismDeltas.Add((float)ThisCTF.DefocusDelta);
                AstigmatismAngles.Add((float)ThisCTF.DefocusAngle);

                float3 Normal = (Matrix3.RotateX(ThisPlaneAngles.X * Helper.ToRad) * Matrix3.RotateY(ThisPlaneAngles.Y * Helper.ToRad)) * new float3(0, 0, 1);
                PlaneNormals.Add(Normal);
                PlaneNormalTilts.Add(Angles[AngleID]);
            }

            ProcessedIndices.Clear();
            LastFittedAngle = 9999;
            int[] GoodIndices = MathHelper.WithinNStdFromMedianIndices(AstigmatismDeltas.ToArray(), 1f);
            float2 MeanAstigmatismVector = MathHelper.Mean(GoodIndices.Select(i => new float2((float)Math.Cos(AstigmatismAngles[i] * Helper.ToRad),
                                                                                              (float)Math.Sin(AstigmatismAngles[i] * Helper.ToRad)) * AstigmatismDeltas[i]));
            float MeanAstigmatismAngle = (float)Math.Atan2(MeanAstigmatismVector.Y, MeanAstigmatismVector.X) * Helper.ToDeg;
            float MeanAstigmatismDelta = MeanAstigmatismVector.Length();

            #region Determine if angles are inverted compared to actual defocus, compute plane normal based on that

            {
                float3[] NormalsOriginal = Helper.ArrayOfFunction(i => Matrix3.Euler(0, -PlaneNormalTilts[i] * Helper.ToRad, 0) * PlaneNormals[i], PlaneNormals.Count);
                float3[] NormalsInverted = Helper.ArrayOfFunction(i => Matrix3.Euler(0,  PlaneNormalTilts[i] * Helper.ToRad, 0) * PlaneNormals[i], PlaneNormals.Count);

                if (float3.RMSD(NormalsInverted) < float3.RMSD(NormalsOriginal))
                    AreAnglesInverted = true;

                PlaneNormal = (AreAnglesInverted ? MathHelper.Mean(NormalsInverted) : MathHelper.Mean(NormalsOriginal)).Normalized();
            }

            #endregion

            #endregion

            #region Fit every tilt

            for (int i = 0; i < NTilts; i++)
            {
                int AngleID = IndicesSortedDose[i];
                Image UncroppedAngleImage = new Image(StackData[AngleID], originalStack.Dims.Slice());
                Image AngleImage = UncroppedAngleImage.AsPadded(UncroppedAngleImage.Dims - new int3(200, 200, 0));
                UncroppedAngleImage.Dispose();

                int BestPrevious = -1;
                if (Math.Abs(LastFittedAngle - Angles[AngleID]) <= 5.1f)
                    BestPrevious = IndicesSortedDose[i - 1];
                else if (ProcessedIndices.Count > 0)
                {
                    List<int> SortedProcessed = new List<int>(ProcessedIndices);
                    SortedProcessed.Sort((a, b) => Math.Abs(Angles[AngleID] - Angles[a]).CompareTo(Math.Abs(Angles[AngleID] - Angles[b])));
                    if (Math.Abs(Dose[SortedProcessed.First()] - Dose[AngleID]) < AverageDose * 5f)
                        BestPrevious = SortedProcessed.First();
                }

                CTF ThisCTF;
                float2 ThisPlaneAngles;
                float2[] ThisPS1D;
                Cubic1D ThisBackground, ThisScale;
                Image ThisPS2D;

                CTF PrevCTF = BestPrevious >= 0 ? FitCTF[BestPrevious] : null;
                float2 PrevPlaneAngles = BestPrevious >= 0 ? FitPlaneAngles[BestPrevious] : new float2(0);
                Cubic1D PrevBackground = BestPrevious >= 0 ? FitBackground[BestPrevious] : null;
                Cubic1D PrevScale = BestPrevious >= 0 ? FitScale[BestPrevious] : null;

                ProcessCTFOneAngle(options,
                                   AngleImage,
                                   Angles[AngleID],
                                   BestPrevious < 0,
                                   true,
                                   new float2(MeanAstigmatismDelta, MeanAstigmatismAngle),
                                   PrevCTF,
                                   PrevPlaneAngles,
                                   PrevBackground,
                                   PrevScale,
                                   Matrix3.Euler(0, Angles[AngleID] * Helper.ToRad * (AreAnglesInverted ? -1 : 1), 0) * PlaneNormal,
                                   true,
                                   out ThisCTF,
                                   out ThisPlaneAngles,
                                   out ThisPS1D,
                                   out ThisBackground,
                                   out ThisScale,
                                   out ThisPS2D);
                AngleImage.Dispose();

                FitCTF[AngleID] = ThisCTF;
                FitPlaneAngles[AngleID] = ThisPlaneAngles;
                FitPS1D[AngleID] = ThisPS1D;
                FitBackground[AngleID] = ThisBackground;
                FitScale[AngleID] = ThisScale;
                FitPS2D[AngleID] = ThisPS2D;

                LastFittedAngle = Angles[AngleID];
                ProcessedIndices.Add(AngleID);

                TiltPS1D[AngleID] = FitPS1D[AngleID];
                TiltSimulatedBackground[AngleID] = new Cubic1D(FitBackground[AngleID].Data.Select(v => new float2(v.X, 0)).ToArray());
                TiltSimulatedScale[AngleID] = FitScale[AngleID];

                TiltCTFProcessed?.Invoke();
            }

            #endregion

            CTF = FitCTF[IndicesSortedDose[0]];

            #region Create grids for fitted CTF params
            {
                float[] DefocusValues = new float[NTilts];
                float[] DeltaValues = new float[NTilts];
                float[] AngleValues = new float[NTilts];
                for (int i = 0; i < NTilts; i++)
                {
                    DefocusValues[i] = (float)FitCTF[i].Defocus;
                    DeltaValues[i] = (float)FitCTF[i].DefocusDelta;
                    AngleValues[i] = (float)FitCTF[i].DefocusAngle;
                }

                GridCTF = new CubicGrid(new int3(1, 1, NTilts), DefocusValues);
                GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, NTilts), DeltaValues);
                GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, NTilts), AngleValues);
            }
            #endregion

            #region Put all 2D spectra into one stack and write it to disk for display purposes
            {
                Image AllPS2D = new Image(new int3(FitPS2D[0].Dims.X, FitPS2D[0].Dims.Y, NTilts));
                float[][] AllPS2DData = AllPS2D.GetHost(Intent.Write);
                for (int i = 0; i < NTilts; i++)
                {
                    AllPS2DData[i] = FitPS2D[i].GetHost(Intent.Read)[0];
                    FitPS2D[i].Dispose();
                }

                AllPS2D.WriteMRC(PowerSpectrumPath, true);
            }
            #endregion

            #region Estimate fittable resolution

            {
                float[] Quality = CTF.EstimateQuality(TiltPS1D[IndicesSortedDose[0]].Select(p => p.Y).ToArray(),
                                                      SimulatedScale.Interp(TiltPS1D[IndicesSortedDose[0]].Select(p => p.X).ToArray()),
                                                      (float)options.RangeMin, 6, true);
                int FirstFreq = 0;
                while ((float.IsNaN(Quality[FirstFreq]) || Quality[FirstFreq] < 0.8f) && FirstFreq < Quality.Length - 1)
                    FirstFreq++;

                int LastFreq = FirstFreq;
                while (!float.IsNaN(Quality[LastFreq]) && Quality[LastFreq] > 0.3f && LastFreq < Quality.Length - 1)
                    LastFreq++;

                CTFResolutionEstimate = Math.Round(options.BinnedPixelSizeMean / ((decimal)LastFreq / options.Window), 1);
            }
            #endregion

            OptionsCTF = options;

            SaveMeta();
        }

        public void ProcessCTFOneAngle(ProcessingOptionsMovieCTF options,
                                       Image angleImage,
                                       float angle,
                                       bool fromScratch,
                                       bool fixAstigmatism,
                                       float2 astigmatism,
                                       CTF previousCTF,
                                       float2 previousPlaneAngles,
                                       Cubic1D previousBackground,
                                       Cubic1D previousScale,
                                       float3 fixedPlaneNormal,
                                       bool fixPlaneNormal,
                                       out CTF thisCTF,
                                       out float2 thisPlaneAngle,
                                       out float2[] thisPS1D,
                                       out Cubic1D thisBackground,
                                       out Cubic1D thisScale,
                                       out Image thisPS2D)
        {
            CTF TempCTF = previousCTF != null ? previousCTF.GetCopy() : new CTF();
            float2[] TempPS1D = null;
            Cubic1D TempBackground = null, TempScale = null;
            
            #region Dimensions and grids

            int NFrames = angleImage.Dims.Z;
            int2 DimsImage = angleImage.DimsSlice;
            int2 DimsRegionBig = new int2(1024);
            int2 DimsRegion = new int2(options.Window, options.Window);

            float OverlapFraction = 0.5f;
            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, DimsRegionBig, OverlapFraction, out DimsPositionGrid);
            float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X + DimsRegionBig.X / 2 - DimsImage.X / 2,
                                                                                v.Y + DimsRegionBig.Y / 2 - DimsImage.Y / 2,
                                                                                0) * (float)options.BinnedPixelSizeMean * 1e-4f).ToArray();
            int NPositions = (int)DimsPositionGrid.Elements();
            
            bool CTFSpace = true;
            bool CTFTime = false;
            int3 CTFSpectraGrid = new int3(DimsPositionGrid.X, DimsPositionGrid.Y, 1);

            int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
            int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
            int NFreq = MaxFreqExclusive - MinFreqInclusive;

            float2 TempPlaneAngle = previousPlaneAngles;

            #endregion

            #region Allocate GPU memory

            Image CTFSpectra = new Image(IntPtr.Zero, new int3(DimsRegion.X, DimsRegion.X, (int)CTFSpectraGrid.Elements()), true);
            Image CTFMean = new Image(IntPtr.Zero, new int3(DimsRegion), true);
            Image CTFCoordsCart = new Image(new int3(DimsRegion), true, true);
            Image CTFCoordsPolarTrimmed = new Image(new int3(NFreq, DimsRegion.X, 1), false, true);

            #endregion

            // Extract movie regions, create individual spectra in Cartesian coordinates and their mean.

            #region Create spectra

            GPU.CreateSpectra(angleImage.GetDevice(Intent.Read),
                              DimsImage,
                              NFrames,
                              PositionGrid,
                              NPositions,
                              DimsRegionBig,
                              CTFSpectraGrid,
                              DimsRegion,
                              CTFSpectra.GetDevice(Intent.Write),
                              CTFMean.GetDevice(Intent.Write),
                              0,
                              0);
            angleImage.FreeDevice(); // Won't need it in this method anymore.

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
                    float Angle = ((float)y / DimsRegion.X + 0.5f) * (float)Math.PI;
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
                TempPS1D = ForPS1D;

                CTFAverage1D.Dispose();
            }

            #endregion

            #region Background fitting methods

            Action UpdateBackgroundFit = () =>
            {
                float2[] ForPS1D = TempPS1D.Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray();
                Cubic1D.FitCTF(ForPS1D,
                               TempCTF.Get1DWithIce(TempPS1D.Length, true, true).Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray(),
                               TempCTF.GetZeros(),
                               TempCTF.GetPeaks(),
                               out TempBackground,
                               out TempScale);
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
                                            MeanCorrectedData[i] = MeanData[i] - TempBackground.Interp(r / DimsRegion.X);
                                        });

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                GPU.CTFMakeAverage(CTFMeanCorrected.GetDevice(Intent.Read),
                                   CTFCoordsCart.GetDevice(Intent.Read),
                                   (uint)CTFMeanCorrected.DimsEffective.ElementsSlice(),
                                   (uint)DimsRegion.X,
                                   new[] { TempCTF.ToStruct() },
                                   TempCTF.ToStruct(),
                                   0,
                                   (uint)DimsRegion.X / 2,
                                   1,
                                   CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[TempPS1D.Length];
                if (keepbackground)
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i] + TempBackground.Interp((float)i / DimsRegion.X));
                else
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                MathHelper.UnNaN(ForPS1D);

                TempPS1D = ForPS1D;

                CTFMeanCorrected.Dispose();
                CTFAverage1D.Dispose();
            };

            #endregion

            #region Do initial fit on mean 1D PS
            {
                float2[] ForPS1D = TempPS1D.Skip(MinFreqInclusive).Take(Math.Max(2, NFreq)).ToArray();

                float[] CurrentBackground;

                // Get a very rough background spline fit with 3-5 nodes
                int NumNodes = Math.Max(3, (int)((options.RangeMax - options.RangeMin) * 5M));
                TempBackground = Cubic1D.Fit(ForPS1D, NumNodes);

                CurrentBackground = TempBackground.Interp(TempPS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray();
                float[] Subtracted1D = Helper.ArrayOfFunction(i => ForPS1D[i].Y - CurrentBackground[i], ForPS1D.Length);
                MathHelper.NormalizeInPlace(Subtracted1D);

                float ZMin = (float)options.ZMin;
                float ZMax = (float)options.ZMax;
                float PhaseMin = 0f;
                float PhaseMax = options.DoPhase ? 1f : 0f;

                if (previousCTF != null)
                {
                    ZMin = (float)previousCTF.Defocus - 1f;
                    ZMax = (float)previousCTF.Defocus + 1f;
                    if (PhaseMax > 0)
                    {
                        PhaseMin = (float)previousCTF.PhaseShift - 0.3f;
                        PhaseMax = (float)previousCTF.PhaseShift + 0.3f;
                    }
                }

                float ZStep = (ZMax - ZMin) / 100f;

                float BestZ = 0, BestPhase = 0, BestScore = -999;
                for (float z = ZMin; z <= ZMax + 1e-5f; z += ZStep)
                {
                    for (float p = PhaseMin; p <= PhaseMax; p += 0.01f)
                    {
                        CTF CurrentParams = new CTF
                        {
                            PixelSize = options.BinnedPixelSizeMean,

                            Defocus = (decimal)z,
                            PhaseShift = (decimal)p,

                            Cs = options.Cs,
                            Voltage = options.Voltage,
                            Amplitude = options.Amplitude
                        };
                        float[] SimulatedCTF = CurrentParams.Get1D(TempPS1D.Length, true).Skip(MinFreqInclusive).Take(Math.Max(2, NFreq)).ToArray();
                        MathHelper.NormalizeInPlace(SimulatedCTF);
                        float Score = MathHelper.CrossCorrelate(Subtracted1D, SimulatedCTF);
                        if (Score > BestScore)
                        {
                            BestScore = Score;
                            BestZ = z;
                            BestPhase = p;
                        }
                    }
                }

                TempCTF = new CTF
                {
                    PixelSize = options.BinnedPixelSizeMean,

                    Defocus = (decimal)BestZ,
                    PhaseShift = (decimal)BestPhase,

                    Cs = options.Cs,
                    Voltage = options.Voltage,
                    Amplitude = options.Amplitude
                };

                UpdateRotationalAverage(true);  // This doesn't have a nice background yet.
                UpdateBackgroundFit();          // Now get a reasonably nice background.
            }
            #endregion

            if (previousCTF != null)
            {
                TempCTF.DefocusDelta = previousCTF.DefocusDelta;
                TempCTF.DefocusAngle = previousCTF.DefocusAngle;
            }

            if (fixAstigmatism)
            {
                TempCTF.DefocusDelta = (decimal)astigmatism.X;
                TempCTF.DefocusAngle = (decimal)astigmatism.Y;
            }
            
            // Do BFGS optimization of defocus, astigmatism and phase shift,
            // using 2D simulation for comparison

            #region BFGS

            {
                Image CTFSpectraPolarTrimmed = CTFSpectra.AsPolar((uint)MinFreqInclusive, (uint)(MinFreqInclusive + NFreq));
                CTFSpectra.FreeDevice(); // This will only be needed again for the final PS1D.

                #region Create background and scale

                float[] CurrentScale = TempScale.Interp(TempPS1D.Select(p => p.X).ToArray());

                Image CTFSpectraScale = new Image(new int3(NFreq, DimsRegion.X, 1));
                float[] CTFSpectraScaleData = CTFSpectraScale.GetHost(Intent.Write)[0];

                // Trim polar to relevant frequencies, and populate coordinates.
                Parallel.For(0, DimsRegion.X, y =>
                {
                    for (int x = 0; x < NFreq; x++)
                        CTFSpectraScaleData[y * NFreq + x] = CurrentScale[x + MinFreqInclusive];
                });
                //CTFSpectraScale.WriteMRC("ctfspectrascale.mrc");

                // Background is just 1 line since we're in polar.
                Image CurrentBackground = new Image(TempBackground.Interp(TempPS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray());

                #endregion

                CTFSpectraPolarTrimmed.SubtractFromLines(CurrentBackground);
                CurrentBackground.Dispose();

                // Normalize background-subtracted spectra.
                //GPU.Normalize(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),      REVERT
                //              CTFSpectraPolarTrimmed.GetDevice(Intent.Write),
                //              (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                //              (uint)CTFSpectraGrid.Elements());
                //CTFSpectraPolarTrimmed.WriteMRC("ctfspectrapolartrimmed.mrc");

                Image IceMask = new Image(CTFSpectraScale.Dims);   // Not doing ice ring modeling in tomo

                Func<float2, float, float[]> GetDefocusGrid = (planeAngles, defocus) =>
                {
                    float[] Result = new float[PositionGridPhysical.Length];

                    float3 Normal = fixPlaneNormal ? fixedPlaneNormal : (Matrix3.RotateX(planeAngles.X) * Matrix3.RotateY(planeAngles.Y) * new float3(0, 0, 1));
                    for (int i = 0; i < Result.Length; i++)
                        Result[i] = defocus - float3.Dot(Normal, PositionGridPhysical[i]) / Normal.Z;

                    return Result;
                };
                
                // Helper method for getting CTFStructs for the entire spectra grid.
                Func<double[], CTF, float[], CTFStruct[]> EvalGetCTF = (input, ctf, defocusValues) =>
                {
                    decimal AlteredPhase = options.DoPhase ? (decimal)input[3] : 0;
                    decimal AlteredDelta = (decimal)input[4];
                    decimal AlteredAngle = (decimal)(input[5] * 20 / (Math.PI / 180));

                    CTF Local = ctf.GetCopy();
                    Local.PhaseShift = AlteredPhase;
                    Local.DefocusDelta = AlteredDelta;
                    Local.DefocusAngle = AlteredAngle;

                    CTFStruct LocalStruct = Local.ToStruct();
                    CTFStruct[] LocalParams = new CTFStruct[defocusValues.Length];
                    for (int i = 0; i < LocalParams.Length; i++)
                    {
                        LocalParams[i] = LocalStruct;
                        LocalParams[i].Defocus = defocusValues[i] * -1e-6f;
                    }

                    return LocalParams;
                };

                // Simulate with adjusted CTF, compare to originals

                #region Eval and Gradient methods

                Func<double[], double> Eval = input =>
                {
                    float2 AlteredPlaneAngles = new float2((float)input[1], (float)input[2]);
                    float[] DefocusValues = GetDefocusGrid(AlteredPlaneAngles, (float)input[0]);

                    CTFStruct[] LocalParams = EvalGetCTF(input, TempCTF, DefocusValues);

                    float[] Result = new float[LocalParams.Length];

                    GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                        CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                        CTFSpectraScale.GetDevice(Intent.Read),
                                        IceMask.GetDevice(Intent.Read),
                                        0,
                                        (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                        LocalParams,
                                        Result,
                                        (uint)LocalParams.Length);

                    float Score = 0;
                    for (int i = 0; i < Result.Length; i++)
                        Score += Result[i];

                    if (float.IsNaN(Score) || float.IsInfinity(Score))
                        throw new Exception("Bad score.");

                    return Score;
                };

                Func<double[], double[]> Gradient = input =>
                {
                    const float Step = 0.0025f;
                    double[] Result = new double[input.Length];
                    
                    for (int i = 0; i < input.Length; i++)
                    {
                        if (fixPlaneNormal && (i == 1 || i == 2))
                            continue;

                        if (fixAstigmatism && i > 3)
                            continue;

                        if (!options.DoPhase && i == 3)
                            continue;

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

                    if (Result.Any(i => double.IsNaN(i) || double.IsInfinity(i)))
                        throw new Exception("Bad score.");

                    return Result;
                };

                #endregion

                #region Do optimization

                double[] StartParams = new double[6];
                StartParams[0] = (double)TempCTF.Defocus;
                StartParams[1] = previousPlaneAngles.X * Helper.ToRad;
                StartParams[2] = previousPlaneAngles.Y * Helper.ToRad;
                StartParams[3] = (double)TempCTF.PhaseShift;
                StartParams[4] = (double)TempCTF.DefocusDelta;
                StartParams[5] = (double)TempCTF.DefocusAngle / 20 * Helper.ToRad;

                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient)
                {
                    MaxIterations = 10
                };
                Optimizer.Maximize(StartParams);

                #endregion

                #region Retrieve parameters

                TempCTF.Defocus = (decimal)Optimizer.Solution[0];
                TempCTF.PhaseShift = (decimal)Optimizer.Solution[3];
                TempCTF.DefocusDelta = (decimal)Optimizer.Solution[4];
                TempCTF.DefocusAngle = (decimal)(Optimizer.Solution[5] * 20 * Helper.ToDeg);

                if (TempCTF.DefocusDelta < 0)
                {
                    TempCTF.DefocusAngle += 90;
                    TempCTF.DefocusDelta *= -1;
                }
                TempCTF.DefocusAngle = ((int)TempCTF.DefocusAngle + 180 * 99) % 180;

                TempPlaneAngle = new float2((float)Optimizer.Solution[1],
                                            (float)Optimizer.Solution[2]) * Helper.ToDeg;

                #endregion

                // Dispose GPU resources manually because GC can't be bothered to do it in time.
                CTFSpectraPolarTrimmed.Dispose();
                CTFSpectraScale.Dispose();
                IceMask.Dispose();

                #region Get nicer envelope fit

                {
                    {
                        Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                        float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                        // Construct background in Cartesian coordinates.
                        Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) =>
                        {
                            CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = TempBackground.Interp(r / DimsRegion.X);
                        });

                        CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                        float[] DefocusValues = GetDefocusGrid(TempPlaneAngle * Helper.ToRad, (float)TempCTF.Defocus);
                        CTFStruct[] LocalParams = DefocusValues.Select(v =>
                        {
                            CTF Local = TempCTF.GetCopy();
                            Local.Defocus = (decimal)v + 0.0M;

                            return Local.ToStruct();
                        }).ToArray();

                        Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                        CTF CTFAug = TempCTF.GetCopy();
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
                        float2[] ForPS1D = new float2[TempPS1D.Length];
                        for (int i = 0; i < ForPS1D.Length; i++)
                            ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(RotationalAverageData[i], 4) + TempBackground.Interp((float)i / DimsRegion.X));
                        MathHelper.UnNaN(ForPS1D);
                        TempPS1D = ForPS1D;

                        CTFSpectraBackground.Dispose();
                        CTFAverage1D.Dispose();
                        CTFSpectra.FreeDevice();
                    }

                    //TempCTF.Defocus = Math.Max(TempCTF.Defocus, options.ZMin);

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
                        Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x + DimHalf] = OriginalAverageData[(DimsRegion.X - 1 - y) * (DimsRegion.X / 2 + 1) + x] - TempBackground.Interp(r);
                    }

                    for (int x = 1; x < DimHalf; x++)
                    {
                        int xx = -(x - DimHalf);
                        float r = (float)Math.Sqrt(xx * xx + yy) / DimsRegion.X;
                        Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x] = OriginalAverageData[y * (DimsRegion.X / 2 + 1) + xx] - TempBackground.Interp(r);
                    }
                }

                thisPS2D = new Image(Average2DData, DimsAverage);
            }
            #endregion

            for (int i = 0; i < TempPS1D.Length; i++)
                TempPS1D[i].Y -= TempBackground.Interp(TempPS1D[i].X);

            CTFSpectra.Dispose();
            CTFMean.Dispose();
            CTFCoordsCart.Dispose();
            CTFCoordsPolarTrimmed.Dispose();

            thisPS1D = TempPS1D;
            thisBackground = TempBackground;
            thisScale = TempScale;
            thisCTF = TempCTF;
            thisPlaneAngle = TempPlaneAngle;
        }

        public void ProcessCTFSimultaneous(ProcessingOptionsMovieCTF options)
        {
            IsProcessing = true;

            if (!Directory.Exists(PowerSpectrumDir))
                Directory.CreateDirectory(PowerSpectrumDir);

            int2 DimsFrame;
            {
                MapHeader HeaderMovie = MapHeader.ReadFromFile(DirectoryName + TiltMoviePaths[0]);
                DimsFrame = new int2(new float2(HeaderMovie.Dimensions.X, HeaderMovie.Dimensions.Y) / (float)options.DownsampleFactor + 1) / 2 * 2;
            }

            #region Dimensions and grids

            int NFrames = NTilts;
            int2 DimsImage = DimsFrame;
            int2 DimsRegionBig = new int2(1536);
            int2 DimsRegion = new int2(options.Window, options.Window);

            float OverlapFraction = 0.5f;
            int2 DimsPositionGrid;
            int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, new int2(DimsRegionBig.X, DimsRegionBig.Y), OverlapFraction, out DimsPositionGrid);
            float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X + DimsRegionBig.X / 2 - DimsImage.X / 2,
                                                                                v.Y + DimsRegionBig.Y / 2 - DimsImage.Y / 2,
                                                                                0) * 
                                                                                (float)options.BinnedPixelSizeMean * 1e-4f).ToArray();
            int NPositions = (int)DimsPositionGrid.Elements();

            bool CTFSpace = true;
            bool CTFTime = false;
            int3 CTFSpectraGrid = new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames);

            int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
            int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
            int NFreq = MaxFreqExclusive - MinFreqInclusive;

            #endregion

            #region Allocate memory

            // GPU
            Image CTFSpectra = new Image(IntPtr.Zero, new int3(DimsRegion.X, DimsRegion.X, (int)CTFSpectraGrid.Elements()), true);
            Image CTFMean = new Image(new int3(DimsRegion), true);
            Image CTFCoordsCart = new Image(new int3(DimsRegion), true, true);
            Image CTFCoordsPolarTrimmed = new Image(new int3(NFreq, DimsRegion.X, 1), false, true);

            // CPU
            float2[] GlobalPS1D = null;
            float[][] LocalPS1D = new float[NPositions * NFrames][];
            Cubic1D GlobalBackground = null, GlobalScale = null;
            CTF GlobalCTF = null;
            float2 GlobalPlaneAngle = new float2();

            #endregion

            #region Helper methods

            Func<float[], float[]> GetDefocusGrid = (defoci) =>
            {
                float[] Result = new float[NPositions * NFrames];

                for (int t = 0; t < NFrames; t++)
                {
                    float3 Normal = (Matrix3.RotateX(GlobalPlaneAngle.X * Helper.ToRad) * Matrix3.RotateY(GlobalPlaneAngle.Y * Helper.ToRad)) * new float3(0, 0, 1);
                    Normal = Matrix3.Euler(0, Angles[t] * Helper.ToRad, 0) * Normal;
                    Normal = Matrix3.Euler(0, 0, -TiltAxisAngles[t] * Helper.ToRad) * Normal;
                    for (int i = 0; i < NPositions; i++)
                        Result[t * NPositions + i] = defoci[t] - float3.Dot(Normal, PositionGridPhysical[i]) / Normal.Z;
                }

                return Result;
            };

            #region Background fitting methods

            Action UpdateBackgroundFit = () =>
            {
                float2[] ForPS1D = GlobalPS1D.Skip(Math.Max(5, MinFreqInclusive / 1)).ToArray();
                Cubic1D.FitCTF(ForPS1D,
                               GlobalCTF.Get1DWithIce(GlobalPS1D.Length, true, true).Skip(Math.Max(5, MinFreqInclusive / 1)).ToArray(),
                               GlobalCTF.GetZeros(),
                               GlobalCTF.GetPeaks(),
                               out GlobalBackground,
                               out GlobalScale);
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
                                            MeanCorrectedData[i] = MeanData[i] - GlobalBackground.Interp(r / DimsRegion.X);
                                        });

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                GPU.CTFMakeAverage(CTFMeanCorrected.GetDevice(Intent.Read),
                                   CTFCoordsCart.GetDevice(Intent.Read),
                                   (uint)CTFMeanCorrected.DimsEffective.ElementsSlice(),
                                   (uint)DimsRegion.X,
                                   new[] { GlobalCTF.ToStruct() },
                                   GlobalCTF.ToStruct(),
                                   0,
                                   (uint)DimsRegion.X / 2,
                                   1,
                                   CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                float2[] ForPS1D = new float2[GlobalPS1D.Length];
                if (keepbackground)
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i] + GlobalBackground.Interp((float)i / DimsRegion.X));
                else
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                MathHelper.UnNaN(ForPS1D);

                GlobalPS1D = ForPS1D;

                CTFMeanCorrected.Dispose();
                CTFAverage1D.Dispose();
            };

            #endregion

            #endregion

            // Extract movie regions, create individual spectra in Cartesian coordinates and their mean.

            #region Create spectra

            int PlanForw = GPU.CreateFFTPlan(new int3(DimsRegionBig), (uint)NPositions);
            int PlanBack = GPU.CreateIFFTPlan(new int3(DimsRegion), (uint)NPositions);

            Movie[] TiltMovies;
            Image[] TiltMovieData;
            LoadMovieData(options, true, true, true, out TiltMovies, out TiltMovieData);

            for (int t = 0; t < NTilts; t++)
            {
                Image TiltMovieAverage = TiltMovieData[t];

                GPU.Normalize(TiltMovieAverage.GetDevice(Intent.Read),
                              TiltMovieAverage.GetDevice(Intent.Write),
                              (uint)TiltMovieAverage.ElementsReal,
                              1);

                Image MovieCTFMean = new Image(new int3(DimsRegion), true);

                GPU.CreateSpectra(TiltMovieAverage.GetDevice(Intent.Read),
                                  DimsImage,
                                  TiltMovieAverage.Dims.Z,
                                  PositionGrid,
                                  NPositions,
                                  DimsRegionBig,
                                  CTFSpectraGrid.Slice(),
                                  DimsRegion,
                                  CTFSpectra.GetDeviceSlice(t * (int)CTFSpectraGrid.ElementsSlice(), Intent.Write),
                                  MovieCTFMean.GetDevice(Intent.Write),
                                  PlanForw,
                                  PlanBack);

                CTFMean.Add(MovieCTFMean);

                MovieCTFMean.Dispose();
                TiltMovieAverage.Dispose();
            }

            GPU.DestroyFFTPlan(PlanBack);
            GPU.DestroyFFTPlan(PlanForw);

            CTFMean.Multiply(1f / NTilts);

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

            #region Initial 1D spectra

            // Mean spectrum to fit background
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
                GlobalPS1D = ForPS1D;

                CTFAverage1D.Dispose();
            }

            // Individual 1D spectra for initial grid search below
            {
                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                for (int s = 0; s < NPositions * NFrames; s++)
                {
                    GPU.CTFMakeAverage(CTFSpectra.GetDeviceSlice(s, Intent.Read),
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

                    LocalPS1D[s] = CTFAverage1D.GetHostContinuousCopy();
                }

                CTFAverage1D.Dispose();
            }

            #endregion

            #region Do initial fit on mean 1D PS
            {
                float2[] ForPS1D = GlobalPS1D.Skip(MinFreqInclusive).Take(Math.Max(2, NFreq * 2 / 3)).ToArray();

                float[] CurrentBackground;

                // Get a very rough background spline fit with 3-5 nodes
                int NumNodes = Math.Max(3, (int)((options.RangeMax - options.RangeMin) * 5M * 2 / 3));
                GlobalBackground = Cubic1D.Fit(ForPS1D, NumNodes);

                CurrentBackground = GlobalBackground.Interp(ForPS1D.Select(p => p.X).ToArray());
                float[][] SubtractedLocal1D = new float[LocalPS1D.Length][];
                for (int s = 0; s < LocalPS1D.Length; s++)
                {
                    SubtractedLocal1D[s] = new float[NFreq * 2 / 3];
                    for (int f = 0; f < NFreq * 2 / 3; f++)
                        SubtractedLocal1D[s][f] = LocalPS1D[s][f + MinFreqInclusive] - CurrentBackground[f];
                }

                float[] GridDeltas = GetDefocusGrid(Helper.ArrayOfConstant(0f, NFrames));

                float ZMin = (float)options.ZMin;
                float ZMax = (float)options.ZMax;
                float PhaseMin = 0f;
                float PhaseMax = options.DoPhase ? 1f : 0f;
                
                float ZStep = Math.Max(0.01f, (ZMax - ZMin) / 200f);

                float BestZ = 0, BestPhase = 0, BestScore = -999;
                Parallel.For(0, (int)((ZMax - ZMin + ZStep - 1e-6f) / ZStep), zi =>
                {
                    float z = ZMin + zi * ZStep;

                    for (float p = PhaseMin; p <= PhaseMax; p += 0.01f)
                    {
                        float Score = 0;

                        for (int s = 0; s < NPositions * NFrames; s++)
                        {
                            CTF CurrentParams = new CTF
                            {
                                PixelSize = options.BinnedPixelSizeMean,

                                Defocus = (decimal)(z + GridDeltas[s]),
                                PhaseShift = (decimal)p,

                                Cs = options.Cs,
                                Voltage = options.Voltage,
                                Amplitude = options.Amplitude
                            };
                            float[] SimulatedCTF = CurrentParams.Get1D(GlobalPS1D.Length, true).Skip(MinFreqInclusive).Take(Math.Max(2, NFreq * 2 / 3)).ToArray();
                            MathHelper.NormalizeInPlace(SimulatedCTF);

                            Score += MathHelper.CrossCorrelate(SubtractedLocal1D[s], SimulatedCTF);
                        }

                        lock (ForPS1D)
                            if (Score > BestScore)
                            {
                                BestScore = Score;
                                BestZ = z;
                                BestPhase = p;
                            }
                    }
                });

                GlobalCTF = new CTF
                {
                    PixelSize = options.BinnedPixelSizeMean,

                    Defocus = (decimal)BestZ,
                    PhaseShift = (decimal)BestPhase,

                    Cs = options.Cs,
                    Voltage = options.Voltage,
                    Amplitude = options.Amplitude
                };

                //UpdateRotationalAverage(true);  // This doesn't have a nice background yet.

                // Scale everything to one common defocus value
                {
                    CTFStruct[] LocalParams = GridDeltas.Select(v =>
                    {
                        CTF Local = GlobalCTF.GetCopy();
                        Local.Defocus += (decimal)v;

                        return Local.ToStruct();
                    }).ToArray();

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                    CTF CTFAug = GlobalCTF.GetCopy();

                    GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                       CTFCoordsCart.GetDevice(Intent.Read),
                                       (uint)CTFSpectra.ElementsSliceReal,
                                       (uint)DimsRegion.X,
                                       LocalParams,
                                       CTFAug.ToStruct(),
                                       0,
                                       (uint)DimsRegion.X / 2,
                                       (uint)LocalParams.Length,
                                       CTFAverage1D.GetDevice(Intent.Write));

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    for (int i = 0; i < RotationalAverageData.Length; i++)
                        GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                    MathHelper.UnNaN(GlobalPS1D);
                    
                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();
                }

                UpdateBackgroundFit();          // Now get a reasonably nice background.

                #region For debug purposes, check what the background-subtracted average looks like at this point

                // Scale everything to one common defocus value
                if (false)
                {
                    Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                    float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                    // Construct background in Cartesian coordinates.
                    Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) =>
                    {
                        CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = GlobalBackground.Interp(r / DimsRegion.X);
                    });

                    CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                    CTFStruct[] LocalParams = GridDeltas.Select(v =>
                    {
                        CTF Local = GlobalCTF.GetCopy();
                        Local.Defocus += (decimal)v;

                        return Local.ToStruct();
                    }).ToArray();

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                    CTF CTFAug = GlobalCTF.GetCopy();

                    GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                       CTFCoordsCart.GetDevice(Intent.Read),
                                       (uint)CTFSpectra.ElementsSliceReal,
                                       (uint)DimsRegion.X,
                                       LocalParams,
                                       CTFAug.ToStruct(),
                                       0,
                                       (uint)DimsRegion.X / 2,
                                       (uint)LocalParams.Length,
                                       CTFAverage1D.GetDevice(Intent.Write));

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    for (int i = 0; i < RotationalAverageData.Length; i++)
                        GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                    MathHelper.UnNaN(GlobalPS1D);

                    CTFSpectra.AddToSlices(CTFSpectraBackground);

                    CTFSpectraBackground.Dispose();
                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();
                }

                #endregion
            }
            #endregion

            // Do BFGS optimization of defocus, astigmatism and phase shift,
            // using 2D simulation for comparison

            double[] StartParams = new double[7];
            StartParams[0] = 0;
            StartParams[1] = 0;
            StartParams[2] = (double)GlobalCTF.PhaseShift;
            StartParams[3] = (double)GlobalCTF.PhaseShift;
            StartParams[4] = (double)GlobalCTF.DefocusDelta;
            StartParams[5] = (double)GlobalCTF.DefocusDelta;
            StartParams[6] = (double)GlobalCTF.DefocusAngle / 20 * Helper.ToRad;
            StartParams = Helper.Combine(StartParams, Helper.ArrayOfConstant((double)GlobalCTF.Defocus, NFrames / 1));

            float3[] GridCoordsByAngle = Helper.ArrayOfFunction(i => new float3((float)i / (NFrames - 1), 0, 0), NFrames);
            float3[] GridCoordsByDose = Helper.ArrayOfFunction(i => new float3(Dose[i] / MathHelper.Max(Dose), 0, 0), NFrames);

            #region BFGS

            {
                // Second iteration will have a nicer background
                for (int opt = 0; opt < 1; opt++)
                {
                    if (opt > 0)
                        NFreq = Math.Min(NFreq + 10, DimsRegion.X / 2 - MinFreqInclusive - 1);

                    Image CTFSpectraPolarTrimmed = CTFSpectra.AsPolar((uint)MinFreqInclusive, (uint)(MinFreqInclusive + NFreq));
                    CTFSpectra.FreeDevice(); // This will only be needed again for the final PS1D.

                    #region Create background and scale

                    float[] CurrentScale = Helper.ArrayOfConstant(1f, GlobalPS1D.Length);// GlobalScale.Interp(GlobalPS1D.Select(p => p.X).ToArray());

                    Image CTFSpectraScale = new Image(new int3(NFreq, DimsRegion.X, 1));
                    float[] CTFSpectraScaleData = CTFSpectraScale.GetHost(Intent.Write)[0];

                    // Trim polar to relevant frequencies, and populate coordinates.
                    Parallel.For(0, DimsRegion.X, y =>
                    {
                        for (int x = 0; x < NFreq; x++)
                            CTFSpectraScaleData[y * NFreq + x] = CurrentScale[x + MinFreqInclusive];
                    });
                    //CTFSpectraScale.WriteMRC("ctfspectrascale.mrc");

                    // Background is just 1 line since we're in polar.
                    Image CurrentBackground = new Image(GlobalBackground.Interp(GlobalPS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray());

                    CTFSpectraPolarTrimmed.SubtractFromLines(CurrentBackground);
                    CurrentBackground.Dispose();
                    
                    //CTFSpectraPolarTrimmed.WriteMRC("ctfspectrapolartrimmed.mrc");

                    Image IceMask = new Image(CTFSpectraScale.Dims);   // Not doing ice ring modeling in tomo

                    #endregion

                    #region Eval and Gradient methods

                    // Helper method for getting CTFStructs for the entire spectra grid.
                    Func<double[], CTF, float[], float[], float[], CTFStruct[]> EvalGetCTF = (input, ctf, phaseValues, defocusValues, defocusDeltaValues) =>
                    {
                        CTF Local = ctf.GetCopy();
                        Local.DefocusAngle = (decimal)(input[6] * 20 / (Math.PI / 180));

                        CTFStruct LocalStruct = Local.ToStruct();
                        CTFStruct[] LocalParams = new CTFStruct[defocusValues.Length];
                        for (int f = 0; f < NFrames; f++)
                            for (int p = 0; p < NPositions; p++)
                            {
                                LocalParams[f * NPositions + p] = LocalStruct;
                                LocalParams[f * NPositions + p].Defocus = defocusValues[f * NPositions + p] * -1e-6f;
                                LocalParams[f * NPositions + p].DefocusDelta = defocusDeltaValues[f] * -1e-6f;
                                LocalParams[f * NPositions + p].PhaseShift = phaseValues[f] * (float)Math.PI;
                            }

                        return LocalParams;
                    };

                    Func<double[], double> Eval = input =>
                    {
                        GlobalPlaneAngle = new float2((float)input[0], (float)input[1]) * Helper.ToDeg;

                        CubicGrid TempGridPhase = new CubicGrid(new int3(2, 1, 1), new[] { (float)input[2], (float)input[3] });
                        CubicGrid TempGridDefocus = new CubicGrid(new int3(input.Length - 7, 1, 1), input.Skip(7).Select(v => (float)v).ToArray());
                        CubicGrid TempGridDefocusDelta = new CubicGrid(new int3(1, 1, 1), new[] { (float)input[4] });

                        float[] PhaseValues = TempGridPhase.GetInterpolated(GridCoordsByDose);
                        float[] DefocusValues = GetDefocusGrid(TempGridDefocus.GetInterpolated(GridCoordsByAngle));
                        float[] DefocusDeltaValues = TempGridDefocusDelta.GetInterpolated(GridCoordsByDose);

                        CTFStruct[] LocalParams = EvalGetCTF(input, GlobalCTF, PhaseValues, DefocusValues, DefocusDeltaValues);

                        float[] Result = new float[LocalParams.Length];

                        GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                                            CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                                            CTFSpectraScale.GetDevice(Intent.Read),
                                            IceMask.GetDevice(Intent.Read),
                                            0,
                                            (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                                            LocalParams,
                                            Result,
                                            (uint)LocalParams.Length);

                        float Score = Result.Sum();

                        if (float.IsNaN(Score) || float.IsInfinity(Score))
                            throw new Exception("Bad score.");

                        return Score;
                    };

                    Func<double[], double[]> Gradient = input =>
                    {
                        const float Step = 0.0025f;
                        double[] Result = new double[input.Length];

                        for (int i = 0; i < input.Length; i++)
                        {
                            if (!options.DoPhase && (i == 2 || i == 3))
                                continue;

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

                        if (Result.Any(i => double.IsNaN(i) || double.IsInfinity(i)))
                            throw new Exception("Bad score.");

                        return Result;
                    };

                    #endregion

                    #region Do optimization

                    // StartParams are initialized above, before the optimization loop

                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient)
                    {
                        MaxIterations = 15
                    };
                    Optimizer.Maximize(StartParams);

                    #endregion

                    #region Retrieve parameters

                    GlobalCTF.Defocus = (decimal)MathHelper.Mean(Optimizer.Solution.Skip(7).Select(v => (float)v));
                    GlobalCTF.PhaseShift = (decimal)(Optimizer.Solution[2] + Optimizer.Solution[3]) / 2;
                    GlobalCTF.DefocusDelta = (decimal)(Optimizer.Solution[4]) / 1;
                    GlobalCTF.DefocusAngle = (decimal)(Optimizer.Solution[6] * 20 * Helper.ToDeg);

                    if (GlobalCTF.DefocusDelta < 0)
                    {
                        GlobalCTF.DefocusAngle += 90;
                        GlobalCTF.DefocusDelta *= -1;
                    }
                    GlobalCTF.DefocusAngle = ((int)GlobalCTF.DefocusAngle + 180 * 99) % 180;

                    GlobalPlaneAngle = new float2((float)Optimizer.Solution[0],
                                                  (float)Optimizer.Solution[1]) * Helper.ToDeg;

                    {
                        CubicGrid TempGridPhase = new CubicGrid(new int3(2, 1, 1), new[] { (float)StartParams[2], (float)StartParams[3] });
                        CubicGrid TempGridDefocusDelta = new CubicGrid(new int3(1, 1, 1), new[] { (float)GlobalCTF.DefocusDelta });
                        CubicGrid TempGridDefocus = new CubicGrid(new int3(StartParams.Length - 7, 1, 1), StartParams.Skip(7).Select(v => (float)v).ToArray());
                        
                        GridCTF = new CubicGrid(new int3(1, 1, NTilts), TempGridDefocus.GetInterpolated(GridCoordsByAngle));
                        GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, NTilts), TempGridDefocusDelta.GetInterpolated(GridCoordsByDose));
                        GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, NTilts), Helper.ArrayOfConstant((float)GlobalCTF.DefocusAngle, NTilts));
                        GridCTFPhase = new CubicGrid(new int3(1, 1, NTilts), TempGridPhase.GetInterpolated(GridCoordsByDose));
                    }

                    #endregion

                    // Dispose GPU resources manually because GC can't be bothered to do it in time.
                    CTFSpectraPolarTrimmed.Dispose();
                    CTFSpectraScale.Dispose();
                    IceMask.Dispose();

                    #region Get nicer envelope fit

                    // Scale everything to one common defocus value
                    {
                        float3[] GridCoords = Helper.ArrayOfFunction(i => new float3(0, 0, (float)i / (NFrames - 1)), NFrames);

                        float[] DefocusValues = GetDefocusGrid(GridCTF.GetInterpolated(GridCoords));
                        float[] DefocusDeltaValues = GridCTFDefocusDelta.GetInterpolated(GridCoords);
                        float[] DefocusAngleValues = GridCTFDefocusAngle.GetInterpolated(GridCoords);
                        float[] PhaseValues = GridCTFPhase.GetInterpolated(GridCoords);
                        
                        CTFStruct[] LocalParams = new CTFStruct[DefocusValues.Length];
                        for (int f = 0; f < NFrames; f++)
                        {
                            for (int p = 0; p < NPositions; p++)
                            {
                                CTF Local = GlobalCTF.GetCopy();
                                Local.Defocus = (decimal)DefocusValues[f * NPositions + p];
                                Local.DefocusDelta = (decimal)DefocusDeltaValues[f];
                                Local.DefocusAngle = (decimal)DefocusAngleValues[f];
                                Local.PhaseShift = (decimal)PhaseValues[f];

                                LocalParams[f * NPositions + p] = Local.ToStruct();
                            }
                        }

                        Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                        CTF CTFAug = GlobalCTF.GetCopy();

                        GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                            CTFCoordsCart.GetDevice(Intent.Read),
                                            (uint)CTFSpectra.ElementsSliceReal,
                                            (uint)DimsRegion.X,
                                            LocalParams,
                                            CTFAug.ToStruct(),
                                            0,
                                            (uint)DimsRegion.X / 2,
                                            (uint)LocalParams.Length,
                                            CTFAverage1D.GetDevice(Intent.Write));

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        for (int i = 0; i < RotationalAverageData.Length; i++)
                            GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                        MathHelper.UnNaN(GlobalPS1D);

                        CTFAverage1D.Dispose();
                        CTFSpectra.FreeDevice();

                        UpdateBackgroundFit(); // Now get a nice background.
                    }

                    #endregion
                }
            }

            #endregion
                
            #region Create global, and per-tilt average spectra
                
            {
                TiltPS1D = new ObservableCollection<float2[]>();
                TiltSimulatedBackground = new ObservableCollection<Cubic1D>();
                TiltSimulatedScale = new ObservableCollection<Cubic1D>();
                Image AllPS2D = new Image(new int3(DimsRegion.X, DimsRegion.X / 2, NTilts));

                float3[] GridCoords = Helper.ArrayOfFunction(i => new float3(0, 0, (float)i / (NFrames - 1)), NFrames);

                float[] DefocusValues = GetDefocusGrid(GridCTF.GetInterpolated(GridCoords));
                float[] DefocusDeltaValues = GridCTFDefocusDelta.GetInterpolated(GridCoords);
                float[] DefocusAngleValues = GridCTFDefocusAngle.GetInterpolated(GridCoords);
                float[] PhaseValues = GridCTFPhase.GetInterpolated(GridCoords);
                        
                // Scale everything to one common defocus value
                {
                    Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                    float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                    // Construct background in Cartesian coordinates.
                    Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) =>
                    {
                        CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = GlobalBackground.Interp(r / DimsRegion.X);
                    });

                    CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                    CTFStruct[] LocalParams = new CTFStruct[DefocusValues.Length];
                    for (int f = 0; f < NFrames; f++)
                    {
                        for (int p = 0; p < NPositions; p++)
                        {
                            CTF Local = GlobalCTF.GetCopy();
                            Local.Defocus = (decimal)DefocusValues[f * NPositions + p];
                            Local.DefocusDelta = (decimal)DefocusDeltaValues[f];
                            Local.DefocusAngle = (decimal)DefocusAngleValues[f];
                            Local.PhaseShift = (decimal)PhaseValues[f];

                            LocalParams[f * NPositions + p] = Local.ToStruct();
                        }
                    }

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                    CTF CTFAug = GlobalCTF.GetCopy();

                    {
                        GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                                           CTFCoordsCart.GetDevice(Intent.Read),
                                           (uint)CTFSpectra.ElementsSliceReal,
                                           (uint)DimsRegion.X,
                                           LocalParams,
                                           CTFAug.ToStruct(),
                                           0,
                                           (uint)DimsRegion.X / 2,
                                           (uint)LocalParams.Length,
                                           CTFAverage1D.GetDevice(Intent.Write));

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        for (int i = 0; i < RotationalAverageData.Length; i++)
                            GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                        MathHelper.UnNaN(GlobalPS1D);

                        PS1D = GlobalPS1D.ToArray();
                    }

                    #region Now go through all tilts

                    for (int t = 0; t < NTilts; t++)
                    {
                        CTFAug.Defocus = (decimal)GridCTF.FlatValues[t];

                        GPU.CTFMakeAverage(CTFSpectra.GetDeviceSlice(t * NPositions, Intent.Read),
                                           CTFCoordsCart.GetDevice(Intent.Read),
                                           (uint)CTFSpectra.ElementsSliceReal,
                                           (uint)DimsRegion.X,
                                           LocalParams.Skip(t * NPositions).Take(NPositions).ToArray(),
                                           CTFAug.ToStruct(),
                                           0,
                                           (uint)DimsRegion.X / 2,
                                           (uint)NPositions,
                                           CTFAverage1D.GetDevice(Intent.Write));

                        float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                        for (int i = 0; i < RotationalAverageData.Length; i++)
                            GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                        MathHelper.UnNaN(GlobalPS1D);

                        TiltPS1D.Add(GlobalPS1D.ToArray());
                        TiltSimulatedBackground.Add(new Cubic1D(GlobalBackground.Data.ToArray()));
                        TiltSimulatedScale.Add(new Cubic1D(GlobalScale.Data.ToArray()));

                        #region Make 2D power spectrum for display

                        Image Sum2D = new Image(CTFSpectra.Dims.Slice(), true);
                        GPU.ReduceMean(CTFSpectra.GetDeviceSlice(t * NPositions, Intent.Read),
                                       Sum2D.GetDevice(Intent.Write),
                                       (uint)Sum2D.ElementsReal,
                                       (uint)NPositions,
                                       1);

                        float[] Sum2DData = Sum2D.GetHostContinuousCopy();
                        Sum2D.Dispose();

                        int3 DimsAverage = new int3(DimsRegion.X, DimsRegion.X / 2, 1);
                        float[] Average2DData = new float[DimsAverage.Elements()];
                        int DimHalf = DimsRegion.X / 2;

                        for (int y = 0; y < DimsAverage.Y; y++)
                        {
                            int yy = y * y;
                            for (int x = 0; x < DimHalf; x++)
                            {
                                int xx = x;
                                xx *= xx;
                                float r = (float)Math.Sqrt(xx + yy) / DimsRegion.X;
                                Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x + DimHalf] = Sum2DData[(DimsRegion.X - 1 - y) * (DimsRegion.X / 2 + 1) + x];
                            }

                            for (int x = 1; x < DimHalf; x++)
                            {
                                int xx = -(x - DimHalf);
                                float r = (float)Math.Sqrt(xx * xx + yy) / DimsRegion.X;
                                Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x] = Sum2DData[y * (DimsRegion.X / 2 + 1) + xx];
                            }
                        }

                        AllPS2D.GetHost(Intent.Write)[t] = Average2DData;

                        #endregion
                    }

                    #endregion

                    AllPS2D.WriteMRC(PowerSpectrumPath, true);

                    CTFSpectraBackground.Dispose();
                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();
                }
            }

            #endregion

            CTF = GlobalCTF;
            SimulatedScale = GlobalScale;
            PlaneNormal = (Matrix3.RotateX(GlobalPlaneAngle.X * Helper.ToRad) * Matrix3.RotateY(GlobalPlaneAngle.Y * Helper.ToRad)) * new float3(0, 0, 1);
            
            #region Estimate fittable resolution

            {
                float[] Quality = CTF.EstimateQuality(PS1D.Select(p => p.Y).ToArray(),
                                                      SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray()),
                                                      (float)options.RangeMin, 16, true);
                int FirstFreq = 0;
                while ((float.IsNaN(Quality[FirstFreq]) || Quality[FirstFreq] < 0.8f) && FirstFreq < Quality.Length - 1)
                    FirstFreq++;

                int LastFreq = FirstFreq;
                while (!float.IsNaN(Quality[LastFreq]) && Quality[LastFreq] > 0.3f && LastFreq < Quality.Length - 1)
                    LastFreq++;

                CTFResolutionEstimate = Math.Round(options.BinnedPixelSizeMean / ((decimal)LastFreq / options.Window), 1);
            }

            #endregion

            CTFSpectra.Dispose();
            CTFMean.Dispose();
            CTFCoordsCart.Dispose();
            CTFCoordsPolarTrimmed.Dispose();
                    
            Simulated1D = GetSimulated1D();

            OptionsCTF = options;

            SaveMeta();

            IsProcessing = false;
            TiltCTFProcessed?.Invoke();
        }

        #endregion

        public void Export2DParticles(float3[] positions, float3[] angles)
        {
            //VolumeDimensions = volumeDimensions;

            //#region Get rows from table

            //List<int> RowIndices = new List<int>();
            //string[] ColumnMicrographName = tableIn.GetColumn("rlnMicrographName");
            //for (int i = 0; i < ColumnMicrographName.Length; i++)
            //    if (ColumnMicrographName[i].Contains(RootName + "."))
            //        RowIndices.Add(i);

            ////if (RowIndices.Count == 0)
            ////    return;

            //int NParticles = RowIndices.Count;

            //#endregion

            //#region Make sure all columns and directories are there

            //lock (tableOut)
            //{
            //    if (!tableOut.HasColumn("rlnAngleRot"))
            //        tableOut.AddColumn("rlnAngleRot");
            //    if (!tableOut.HasColumn("rlnAngleTilt"))
            //        tableOut.AddColumn("rlnAngleTilt");
            //    if (!tableOut.HasColumn("rlnAnglePsi"))
            //        tableOut.AddColumn("rlnAnglePsi");
            //    if (!tableOut.HasColumn("rlnImageName"))
            //        tableOut.AddColumn("rlnImageName");
            //    if (!tableOut.HasColumn("rlnCtfImage"))
            //        tableOut.AddColumn("rlnCtfImage");

            //    if (!Directory.Exists(ParticleTiltsDir))
            //        Directory.CreateDirectory(ParticleTiltsDir);
            //    if (!Directory.Exists(ParticleTiltsCTFDir))
            //        Directory.CreateDirectory(ParticleTiltsCTFDir);
            //}

            //#endregion

            //#region Get subtomo positions from table

            //float3[] Origins = new float3[NParticles];
            //float3[] ParticleAngles = new float3[NParticles];
            //{
            //    string[] ColumnPosX = tableIn.GetColumn("rlnCoordinateX");
            //    string[] ColumnPosY = tableIn.GetColumn("rlnCoordinateY");
            //    string[] ColumnPosZ = tableIn.GetColumn("rlnCoordinateZ");
            //    string[] ColumnOriginX = tableIn.GetColumn("rlnOriginX");
            //    string[] ColumnOriginY = tableIn.GetColumn("rlnOriginY");
            //    string[] ColumnOriginZ = tableIn.GetColumn("rlnOriginZ");
            //    string[] ColumnAngleRot = tableIn.GetColumn("rlnAngleRot");
            //    string[] ColumnAngleTilt = tableIn.GetColumn("rlnAngleTilt");
            //    string[] ColumnAnglePsi = tableIn.GetColumn("rlnAnglePsi");

            //    for (int i = 0; i < NParticles; i++)
            //    {
            //        float3 Pos = new float3(float.Parse(ColumnPosX[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                float.Parse(ColumnPosY[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                float.Parse(ColumnPosZ[RowIndices[i]], CultureInfo.InvariantCulture));
            //        float3 Shift = new float3(float.Parse(ColumnOriginX[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnOriginY[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnOriginZ[RowIndices[i]], CultureInfo.InvariantCulture));
                    
            //        Origins[i] = Pos - Shift;

            //        float3 Angle = new float3(0, 0, 0);
            //        if (ColumnAngleRot != null && ColumnAngleTilt != null && ColumnAnglePsi != null)
            //            Angle = new float3(float.Parse(ColumnAngleRot[RowIndices[i]], CultureInfo.InvariantCulture),
            //                               float.Parse(ColumnAngleTilt[RowIndices[i]], CultureInfo.InvariantCulture),
            //                               float.Parse(ColumnAnglePsi[RowIndices[i]], CultureInfo.InvariantCulture));
            //        ParticleAngles[i] = Angle;
            //    }
            //}

            //#endregion

            //tiltStack.FreeDevice();

            //if (NParticles == 0)
            //    return;

            //int DownSize = size / downsample;

            //Image CTFCoords = CTF.GetCTFCoords(DownSize, size);

            //int PlanForw, PlanBack, PlanForwCTF;
            //Projector.GetPlans(new int3(DownSize, DownSize, DownSize), 2, out PlanForw, out PlanBack, out PlanForwCTF);
            
            //for (int p = 0; p < NParticles; p++)
            //{
            //    if (File.Exists(ParticlesDir + RootName + "_" + p.ToString("D5") + ".mrc"))
            //        return;
                
            //    float3[] ImageAngles = GetParticleAngleInImages(Origins[p], ParticleAngles[p]);

            //    Image ImagesFT = GetSubtomoImages(tiltStack, size, Origins[p], true);
            //    Image ImagesFTCropped = ImagesFT.AsPadded(new int2(DownSize, DownSize));
            //    ImagesFT.Dispose();
            //    Image Images = ImagesFTCropped.AsIFFT();
            //    ImagesFTCropped.Dispose();
            //    Images.RemapFromFT();

            //    Images.WriteMRC(ParticleTiltsDir + RootName + "_" + p.ToString("D5") + ".mrcs");
            //    Images.Dispose();

            //    Image CTFs = GetSubtomoCTFs(Origins[p], CTFCoords, false, false, false);
            //    CTFs.WriteMRC(ParticleTiltsCTFDir + RootName + "_" + p.ToString("D5") + ".mrcs");
            //    CTFs.Dispose();

            //    lock (tableOut)
            //        for (int a = 0; a < ImageAngles.Length; a++)
            //        {
            //            string[] Row = new string[tableOut.ColumnCount].Select(v => "0").ToArray();
            //            Row[tableOut.GetColumnID("rlnAngleRot")] = (ImageAngles[a].X * Helper.ToDeg).ToString(CultureInfo.InvariantCulture);
            //            Row[tableOut.GetColumnID("rlnAngleTilt")] = (ImageAngles[a].Y * Helper.ToDeg).ToString(CultureInfo.InvariantCulture);
            //            Row[tableOut.GetColumnID("rlnAnglePsi")] = (ImageAngles[a].Z * Helper.ToDeg).ToString(CultureInfo.InvariantCulture);

            //            Row[tableOut.GetColumnID("rlnImageName")] = (a + 1) + "@particles2d/" + RootName + "_" + p.ToString("D5") + ".mrcs";
            //            Row[tableOut.GetColumnID("rlnCtfImage")] = (a + 1) + "@particlectf2d/" + RootName + "_" + p.ToString("D5") + ".mrcs";

            //            tableOut.AddRow(new List<string>(Row));
            //        }
            //}

            //GPU.DestroyFFTPlan(PlanForw);
            //GPU.DestroyFFTPlan(PlanBack);
            //GPU.DestroyFFTPlan(PlanForwCTF);

            //CTFCoords.Dispose();
        }

        public void ExportSubtomos(Star tableIn, Image tiltStack)
        {
            //VolumeDimensions = volumeDimensions;

            //#region Get rows from table

            //List<int> RowIndices = new List<int>();
            //string[] ColumnMicrographName = tableIn.GetColumn("rlnMicrographName");
            //for (int i = 0; i < ColumnMicrographName.Length; i++)
            //    if (ColumnMicrographName[i].Contains(RootName + "."))
            //        RowIndices.Add(i);

            ////if (RowIndices.Count == 0)
            ////    return;

            //int NParticles = RowIndices.Count;

            //#endregion

            //#region Make sure all columns and directories are there

            //if (!tableIn.HasColumn("rlnImageName"))
            //    tableIn.AddColumn("rlnImageName");
            //if (!tableIn.HasColumn("rlnCtfImage"))
            //    tableIn.AddColumn("rlnCtfImage");

            //if (!Directory.Exists(ParticlesDir))
            //    Directory.CreateDirectory(ParticlesDir);
            //if (!Directory.Exists(ParticleCTFDir))
            //    Directory.CreateDirectory(ParticleCTFDir);

            //#endregion

            //#region Get subtomo positions from table

            //float3[] Origins = new float3[NParticles];
            //float3[] ParticleAngles = new float3[NParticles];
            //{
            //    string[] ColumnPosX = tableIn.GetColumn("rlnCoordinateX");
            //    string[] ColumnPosY = tableIn.GetColumn("rlnCoordinateY");
            //    string[] ColumnPosZ = tableIn.GetColumn("rlnCoordinateZ");
            //    string[] ColumnOriginX = tableIn.GetColumn("rlnOriginX");
            //    string[] ColumnOriginY = tableIn.GetColumn("rlnOriginY");
            //    string[] ColumnOriginZ = tableIn.GetColumn("rlnOriginZ");
            //    string[] ColumnAngleRot = tableIn.GetColumn("rlnAngleRot");
            //    string[] ColumnAngleTilt = tableIn.GetColumn("rlnAngleTilt");
            //    string[] ColumnAnglePsi = tableIn.GetColumn("rlnAnglePsi");

            //    for (int i = 0; i < NParticles; i++)
            //    {
            //        float3 Pos = new float3(float.Parse(ColumnPosX[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                float.Parse(ColumnPosY[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                float.Parse(ColumnPosZ[RowIndices[i]], CultureInfo.InvariantCulture));
            //        float3 Shift = new float3(float.Parse(ColumnOriginX[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnOriginY[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnOriginZ[RowIndices[i]], CultureInfo.InvariantCulture));

            //        //Origins[i] *= new float3(3838f / 959f, 3710f / 927f, 4f);
            //        Origins[i] = Pos - Shift;

            //        float3 Angle = new float3(0, 0, 0);
            //        if (ColumnAngleRot != null && ColumnAngleTilt != null && ColumnAnglePsi != null)
            //            Angle = new float3(float.Parse(ColumnAngleRot[RowIndices[i]], CultureInfo.InvariantCulture),
            //                               float.Parse(ColumnAngleTilt[RowIndices[i]], CultureInfo.InvariantCulture),
            //                               float.Parse(ColumnAnglePsi[RowIndices[i]], CultureInfo.InvariantCulture));
            //        ParticleAngles[i] = Angle;

            //        tableIn.SetRowValue(RowIndices[i], "rlnCoordinateX", Origins[i].X.ToString(CultureInfo.InvariantCulture));
            //        tableIn.SetRowValue(RowIndices[i], "rlnCoordinateY", Origins[i].Y.ToString(CultureInfo.InvariantCulture));
            //        tableIn.SetRowValue(RowIndices[i], "rlnCoordinateZ", Origins[i].Z.ToString(CultureInfo.InvariantCulture));
            //        tableIn.SetRowValue(RowIndices[i], "rlnOriginX", "0.0");
            //        tableIn.SetRowValue(RowIndices[i], "rlnOriginY", "0.0");
            //        tableIn.SetRowValue(RowIndices[i], "rlnOriginZ", "0.0");

            //        //tableIn.SetRowValue(RowIndices[i], "rlnAngleRot", "0.0");
            //        //tableIn.SetRowValue(RowIndices[i], "rlnAngleTilt", "0.0");
            //        //tableIn.SetRowValue(RowIndices[i], "rlnAnglePsi", "0.0");
            //    }
            //}

            //#endregion

            //tiltStack?.FreeDevice();

            ///*int SizeCropped = 50;//size / 2;

            //int3 Grid = (VolumeDimensions + SizeCropped - 1) / SizeCropped;
            //List<float3> GridCoords = new List<float3>();
            //for (int z = 0; z < Grid.Z; z++)
            //    for (int y = 0; y < Grid.Y; y++)
            //        for (int x = 0; x < Grid.X; x++)
            //            GridCoords.Add(new float3(x * SizeCropped + SizeCropped / 2,
            //                                      y * SizeCropped + SizeCropped / 2,
            //                                      z * SizeCropped + SizeCropped / 2));

            //Origins = GridCoords.ToArray();
            //NParticles = Origins.Length;*/

            //if (NParticles == 0)
            //    return;

            //int DownSize = size / downsample;

            //Image CTFCoords = CTF.GetCTFCoords(DownSize, size);

            //int PlanForw, PlanBack, PlanForwCTF;
            //Projector.GetPlans(new int3(DownSize, DownSize, DownSize), 2, out PlanForw, out PlanBack, out PlanForwCTF);
            
            ////Parallel.For(0, NParticles, new ParallelOptions() { MaxDegreeOfParallelism = 1 }, p =>
            //for (int p = 0; p < NParticles; p++)
            //{
            //    lock (tableIn)
            //    {
            //        tableIn.SetRowValue(RowIndices[p], "rlnImageName", "particles/" + RootName + "_" + p.ToString("D5") + ".mrc");
            //        tableIn.SetRowValue(RowIndices[p], "rlnCtfImage", "particlectf/" + RootName + "_" + p.ToString("D5") + ".mrc");

            //        //tableIn.Save("D:\\rubisco\\luisexported.star");
            //    }

            //    if (File.Exists(ParticlesDir + RootName + "_" + p.ToString("D5") + ".mrc"))
            //        continue;

            //    Image Subtomo, SubtomoCTF;
            //    GetSubtomo(tiltStack, Origins[p], ParticleAngles[p], CTFCoords, downsample, out Subtomo, out SubtomoCTF, PlanForw, PlanBack, PlanForwCTF);
            //    //Image SubtomoCropped = Subtomo.AsPadded(new int3(SizeCropped, SizeCropped, SizeCropped));

            //    Subtomo.WriteMRC(ParticlesDir + RootName + "_" + p.ToString("D5") + ".mrc");
            //    SubtomoCTF.WriteMRC(ParticleCTFDir + RootName + "_" + p.ToString("D5") + ".mrc");
            //    //SubtomoCropped.Dispose();

            //    Subtomo?.Dispose();
            //    SubtomoCTF?.Dispose();
            //}//);

            //GPU.DestroyFFTPlan(PlanForw);
            //GPU.DestroyFFTPlan(PlanBack);
            //GPU.DestroyFFTPlan(PlanForwCTF);

            //CTFCoords.Dispose();
        }

        public void MatchFull(ProcessingOptionsTomoFullMatch options, Image template, Func<int3, int, string, bool> progressCallback)
        {
            bool IsCanceled = false;
            if (!Directory.Exists(MatchingDir))
                Directory.CreateDirectory(MatchingDir);

            string NameWithRes = RootName + $"_{options.BinnedPixelSizeMean:F2}Apx";

            float3[] HealpixAngles = Helper.GetHealpixAngles(options.HealpixOrder, options.Symmetry).Select(a => a * Helper.ToRad).ToArray();
            LoadMovieSizes(options);

            Image CorrImage = null;
            float[][] CorrData;
            float[][] AngleData;

            #region Dimensions

            int SizeSub = options.SubVolumeSize;
            int SizeParticle = (int)(options.TemplateDiameter / options.BinnedPixelSizeMean);
            int SizeUseful = Math.Min(SizeSub / 2, SizeSub - SizeParticle * 2);// Math.Min(SizeSub - SizeParticle, SizeSub / 2);
            if (SizeUseful < 2)
                throw new DimensionMismatchException("Particle diameter is bigger than the box.");

            VolumeDimensionsPhysical = options.DimensionsPhysical;

            int3 DimsVolumeCropped = new int3((int)Math.Round(options.DimensionsPhysical.X / (float)options.BinnedPixelSizeMean / 2) * 2,
                                                (int)Math.Round(options.DimensionsPhysical.Y / (float)options.BinnedPixelSizeMean / 2) * 2,
                                                (int)Math.Round(options.DimensionsPhysical.Z / (float)options.BinnedPixelSizeMean / 2) * 2);

            int3 Grid = (DimsVolumeCropped - SizeParticle + SizeUseful - 1) / SizeUseful;
            List<float3> GridCoords = new List<float3>();
            for (int z = 0; z < Grid.Z; z++)
                for (int x = 0; x < Grid.X; x++)
                    for (int y = 0; y < Grid.Y; y++)
                        GridCoords.Add(new float3(x * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                                    y * SizeUseful + SizeUseful / 2 + SizeParticle / 2,
                                                    z * SizeUseful + SizeUseful / 2 + SizeParticle / 2));

            #endregion

            #region Get correlation and angles either by calculating them from scratch, or by loading precalculated volumes

            CorrData = Helper.ArrayOfFunction(i => new float[DimsVolumeCropped.ElementsSlice()], DimsVolumeCropped.Z);
            AngleData = Helper.ArrayOfFunction(i => new float[DimsVolumeCropped.ElementsSlice()], DimsVolumeCropped.Z);

            if (true)
            {
                if (!File.Exists(ReconstructionDir + NameWithRes + ".mrc"))
                    throw new FileNotFoundException("A reconstruction at the desired resolution was not found.");

                progressCallback?.Invoke(Grid, 0, "Loading...");

                Image TomoRec = Image.FromFile(ReconstructionDir + NameWithRes + ".mrc");

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

                    ProjectorReference = new Projector(TemplatePadded, 2, 3);
                    TemplatePadded.Dispose();
                    ProjectorReference.PutTexturesOnDevice();
                }

                #endregion

                #region Preflight

                if (TomoRec.Dims != DimsVolumeCropped)
                    throw new DimensionMismatchException("Reconstruction resolution doesn't match desired correlation resolution.");

                if (options.WhitenSpectrum)
                {
                    progressCallback?.Invoke(Grid, 0, "Whitening spectral noise...");

                    Image TomoRecFlat = TomoRec.AsSpectrumFlattened(true, 0.99f);
                    TomoRec.Dispose();
                    TomoRec = TomoRecFlat;
                }

                float[][] TomoRecData = TomoRec.GetHost(Intent.Read);

                int PlanForw, PlanBack, PlanForwCTF;
                Projector.GetPlans(new int3(SizeSub), 2, out PlanForw, out PlanBack, out PlanForwCTF);

                Image CTFCoords = CTF.GetCTFCoords(SizeSub, (int)(SizeSub * options.DownsampleFactor));

                #endregion

                progressCallback?.Invoke(Grid, 0, "Matching...");

                int BatchSize = Grid.Y;
                for (int b = 0; b < GridCoords.Count; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, GridCoords.Count - b);

                    Image CorrSum = new Image(new int3(SizeSub));
                    Image Subtomos = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch), true, true);

                    #region Create CTF for this column of subvolumes (X = const, Z = const)

                    Image SubtomoCTF;
                    Image CTFs = GetCTFsForOneParticle(options, GridCoords[b], CTFCoords, true, false, false);
                    Image CTFsAbs = CTFs.GetCopyGPU();
                    CTFsAbs.Abs();
                    {
                        // CTF has to be converted to complex numbers with imag = 0, and weighted by itself
                        float2[] CTFsComplexData = new float2[CTFs.ElementsComplex];
                        float[] CTFsContinuousData = CTFs.GetHostContinuousCopy();
                        for (int i = 0; i < CTFsComplexData.Length; i++)
                            CTFsComplexData[i] = new float2(CTFsContinuousData[i] * CTFsContinuousData[i], 0);
                        //CTFsAbs.Fill(1f);

                        Image CTFsComplex = new Image(CTFsComplexData, CTFs.Dims, true);

                        // Back-project and reconstruct
                        Projector ProjCTF = new Projector(new int3(SizeSub), 2);

                        ProjCTF.BackProject(CTFsComplex, CTFsAbs, GetAngleInAllTilts(GridCoords[b]));
                        CTFsComplex.Dispose();

                        SubtomoCTF = ProjCTF.Reconstruct(true, PlanForw, PlanBack, PlanForwCTF);
                        ProjCTF.Dispose();
                    }
                    //SubtomoCTF.Fill(1f);

                    CTFs.Dispose();
                    CTFsAbs.Dispose();

                    #endregion

                    #region Extract subvolumes and store their FFTs

                    for (int st = 0; st < CurBatch; st++)
                    {
                        float[][] SubtomoData = new float[SizeSub][];

                        int XStart = (int)GridCoords[b + st].X - SizeSub / 2;
                        int YStart = (int)GridCoords[b + st].Y - SizeSub / 2;
                        int ZStart = (int)GridCoords[b + st].Z - SizeSub / 2;
                        for (int z = 0; z < SizeSub; z++)
                        {
                            SubtomoData[z] = new float[SizeSub * SizeSub];
                            int zz = (ZStart + z + TomoRec.Dims.Z) % TomoRec.Dims.Z;
                            
                            for (int y = 0; y < SizeSub; y++)
                            {
                                int yy = (YStart + y + TomoRec.Dims.Y) % TomoRec.Dims.Y;
                                for (int x = 0; x < SizeSub; x++)
                                {
                                    int xx = (XStart + x + TomoRec.Dims.X) % TomoRec.Dims.X;
                                    SubtomoData[z][y * SizeSub + x] = TomoRecData[zz][yy * TomoRec.Dims.X + xx];
                                }
                            }
                        }

                        Image Subtomo = new Image(SubtomoData, new int3(SizeSub));

                        // Re-use FFT plan created previously for CTF reconstruction since it has the right size
                        GPU.FFT(Subtomo.GetDevice(Intent.Read),
                                Subtomos.GetDeviceSlice(SizeSub * st, Intent.Write),
                                Subtomo.Dims,
                                1,
                                PlanForwCTF);

                        Subtomo.Dispose();
                    }

                    #endregion

                    #region Perform correlation

                    Image BestCorrelation = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));
                    Image BestAngle = new Image(IntPtr.Zero, new int3(SizeSub, SizeSub, SizeSub * CurBatch));

                    GPU.CorrelateSubTomos(ProjectorReference.t_DataRe,
                                          ProjectorReference.t_DataIm,
                                          ProjectorReference.Oversampling,
                                          ProjectorReference.Data.Dims,
                                          Subtomos.GetDevice(Intent.Read),
                                          SubtomoCTF.GetDevice(Intent.Read),
                                          new int3(SizeSub),
                                          (uint)CurBatch,
                                          Helper.ToInterleaved(HealpixAngles),
                                          (uint)HealpixAngles.Length,
                                          SizeParticle / 2,
                                          BestCorrelation.GetDevice(Intent.Write),
                                          BestAngle.GetDevice(Intent.Write));

                    #endregion

                    #region Put correlation values and best angle IDs back into the large volume
                    
                    // Compensate for the fact I'm too stupid to figure out why the real-space result is attenuated in weird ways
                    float[] CorrSumData;
                    //{
                    //    GPU.ReduceAdd(BestCorrelation.GetDevice(Intent.Read),
                    //                  CorrSum.GetDevice(Intent.Write),
                    //                  (uint)(SizeSub * SizeSub * SizeSub),
                    //                  (uint)CurBatch,
                    //                  1);

                    //    CorrSum.Multiply(1f / CurBatch / (SizeSub * SizeSub * SizeSub));
                    //    CorrSum.Bandpass(0, 4f / SizeSub * 2, true);
                    //    Image CorrSumCropped = CorrSum.AsPadded(new int3(SizeUseful));
                    //    CorrSum.Dispose();

                    //    CorrSumData = CorrSumCropped.GetHostContinuousCopy();
                    //    CorrSumCropped.Dispose();
                    //}

                    for (int st = 0; st < CurBatch; st++)
                    {
                        Image ThisCorrelation = new Image(BestCorrelation.GetDeviceSlice(SizeSub * st, Intent.Read), new int3(SizeSub));
                        Image CroppedCorrelation = ThisCorrelation.AsPadded(new int3(SizeUseful));

                        Image ThisAngle = new Image(BestAngle.GetDeviceSlice(SizeSub * st, Intent.Read), new int3(SizeSub));
                        Image CroppedAngle = ThisAngle.AsPadded(new int3(SizeUseful));

                        float[] SubCorr = CroppedCorrelation.GetHostContinuousCopy();
                        float[] SubAngle = CroppedAngle.GetHostContinuousCopy();
                        int3 Origin = new int3(GridCoords[b + st]) - SizeUseful / 2;
                        for (int z = 0; z < SizeUseful; z++)
                        {
                            int zVol = Origin.Z + z;
                            if (zVol >= DimsVolumeCropped.Z - SizeParticle / 2)
                                continue;

                            for (int y = 0; y < SizeUseful; y++)
                            {
                                int yVol = Origin.Y + y;
                                if (yVol >= DimsVolumeCropped.Y - SizeParticle / 2)
                                    continue;

                                for (int x = 0; x < SizeUseful; x++)
                                {
                                    int xVol = Origin.X + x;
                                    if (xVol >= DimsVolumeCropped.X - SizeParticle / 2)
                                        continue;

                                    CorrData[zVol][yVol * DimsVolumeCropped.X + xVol] = SubCorr[(z * SizeUseful + y) * SizeUseful + x];// / (SizeSub * SizeSub * SizeSub);// / CorrSumData[(z * SizeUseful + y) * SizeUseful + x];
                                    AngleData[zVol][yVol * DimsVolumeCropped.X + xVol] = SubAngle[(z * SizeUseful + y) * SizeUseful + x];
                                }
                            }
                        }

                        CroppedCorrelation.Dispose();
                        ThisCorrelation.Dispose();
                        CroppedAngle.Dispose();
                        ThisAngle.Dispose();
                    }

                    #endregion

                    Subtomos.Dispose();
                    SubtomoCTF.Dispose();

                    BestCorrelation.Dispose();
                    BestAngle.Dispose();

                    if (progressCallback != null)
                        IsCanceled = progressCallback(Grid, b + CurBatch, "Matching...");
                }

                #region Postflight

                TomoRec.Dispose();

                GPU.DestroyFFTPlan(PlanForw);
                GPU.DestroyFFTPlan(PlanBack);
                GPU.DestroyFFTPlan(PlanForwCTF);

                CTFCoords.Dispose();
                ProjectorReference.Dispose();

                if (options.Supersample > 1)
                {
                    progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Looking for sub-pixel peaks...");

                    Image NormalSampled = new Image(CorrData, DimsVolumeCropped);
                    Image SuperSampled = new Image(NormalSampled.GetDevice(Intent.Read), NormalSampled.Dims);

                    GPU.SubpixelMax(NormalSampled.GetDevice(Intent.Read),
                                    SuperSampled.GetDevice(Intent.Write),
                                    NormalSampled.Dims,
                                    options.Supersample);

                    CorrData = SuperSampled.GetHost(Intent.Read);

                    NormalSampled.Dispose();
                    SuperSampled.Dispose();
                }

                if (options.KeepOnlyFullVoxels)
                {
                    progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Trimming...");

                    float BinnedAngPix = (float)options.BinnedPixelSizeMean;
                    float Margin = (float)options.TemplateDiameter;

                    Parallel.For(0, DimsVolumeCropped.Z, z =>
                    {
                        float3[] VolumePositions = new float3[DimsVolumeCropped.ElementsSlice()];
                        for (int y = 0; y < DimsVolumeCropped.Y; y++)
                        for (int x = 0; x < DimsVolumeCropped.X; x++)
                            VolumePositions[y * DimsVolumeCropped.X + x] = new float3(x * BinnedAngPix, y * BinnedAngPix, z * BinnedAngPix);

                        float3[] ImagePositions = GetPositionInAllTiltsNoLocalWarp(VolumePositions);

                        for (int i = 0; i < ImagePositions.Length; i++)
                        {
                            int ii = i / NTilts;
                            int t = i % NTilts;

                            if (ImagePositions[i].X < Margin || ImagePositions[i].Y < Margin ||
                                ImagePositions[i].X > ImageDimensions[t].X - BinnedAngPix - Margin ||
                                ImagePositions[i].Y > ImageDimensions[t].Y - BinnedAngPix - Margin)
                            {
                                CorrData[z][ii] = 0;
                            }
                        }
                    });
                }

                progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Saving global scores...");

                // Store correlation values and angle IDs for re-use later
                CorrImage = new Image(CorrData, DimsVolumeCropped);
                CorrImage.WriteMRC(MatchingDir + NameWithRes + "_" + options.TemplateName + "_corr.mrc", (float)options.BinnedPixelSizeMean, true);

                #endregion
            }

            CorrImage = Image.FromFile(MatchingDir + NameWithRes + "_" + options.TemplateName + "_corr.mrc");
            CorrData = CorrImage.GetHost(Intent.Read);

            //CorrImage?.Dispose();

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

            CorrImage?.Dispose();

            #endregion

            //Star TableIn = new Star("D:/alextomo/august/tiltseries/tiltseries0003/matching/series0003019_7.72_aldolase.star");
            //int[] ColumnCoordX = TableIn.GetColumn("rlnCoordinateX").Select(v => int.Parse(v)).ToArray();
            //int[] ColumnCoordY = TableIn.GetColumn("rlnCoordinateY").Select(v => int.Parse(v)).ToArray();
            //int[] ColumnCoordZ = TableIn.GetColumn("rlnCoordinateZ").Select(v => int.Parse(v)).ToArray();
            //float[] ColumnAngleRot = TableIn.GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            //float[] ColumnAngleTilt = TableIn.GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            //float[] ColumnAnglePsi = TableIn.GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            //InitialPeaks = Helper.ArrayOfFunction(i => new int3(ColumnCoordX[i], ColumnCoordY[i], ColumnCoordZ[i]), ColumnCoordX.Length);
            //float3[] RefinedAngles = Helper.ArrayOfFunction(i => new float3(ColumnAngleRot[i], ColumnAngleTilt[i], ColumnAnglePsi[i]) * Helper.ToRad, ColumnCoordX.Length);

            #region Refine parameters for the best positions

            //options.NResults = Math.Min(InitialPeaks.Length, options.NResults);
            //float3[] RefinedPositions = InitialPeaks.Take(options.NResults).Select(v => new float3(v) / new float3(DimsVolumeCropped)).ToArray();
            ////float3[] RefinedAngles = InitialPeaks.Take(options.NResults).Select(v => HealpixAngles[(int)AngleData[v.Z][v.Y * DimsVolumeCropped.X + v.X]]).ToArray();
            //float[] RefinedScores = new float[options.NResults];
            //{
            //    float FineAngPix = Math.Max((float)options.BinnedPixelSizeMean / 2f, (float)options.TemplatePixel);
            //    int SizeFine = (int)Math.Round(template.Dims.X * (float)options.TemplatePixel / FineAngPix / 2) * 2;
            //    FineAngPix = (float)options.TemplatePixel * template.Dims.X / SizeFine;
            //    options.BinTimes = (decimal)Math.Log(FineAngPix / (double)options.PixelSizeMean, 2.0);

            //    DimsVolumeCropped = new int3((int)Math.Round(options.DimensionsPhysical.X / (float)options.BinnedPixelSizeMean / 2) * 2,
            //                                 (int)Math.Round(options.DimensionsPhysical.Y / (float)options.BinnedPixelSizeMean / 2) * 2,
            //                                 (int)Math.Round(options.DimensionsPhysical.Z / (float)options.BinnedPixelSizeMean / 2) * 2);

            //    float FineRadius = (float)(options.TemplateDiameter / 2 / options.BinnedPixelSizeMean);

            //    RefinedPositions = RefinedPositions.Select(v => v * new float3(DimsVolumeCropped)).ToArray();

            //    Projector ProjectorReference;
            //    {
            //        Image TemplateWhite = template.AsSpectrumFlattened(true, 0.99f);
            //        template.FreeDevice();

            //        GPU.SphereMask(TemplateWhite.GetDevice(Intent.Read),
            //                       TemplateWhite.GetDevice(Intent.Write),
            //                       TemplateWhite.Dims,
            //                       (float)(options.TemplateDiameter / 2 / options.TemplatePixel),
            //                       10,
            //                       1);

            //        Image TemplateFine = TemplateWhite.AsScaled(new int3(SizeFine));
            //        TemplateWhite.Dispose();

            //        ProjectorReference = new Projector(TemplateFine, 2, 3);
            //        TemplateFine.Dispose();
            //    }

            //    #region Load original tilt images

            //    Movie[] TiltMovies;
            //    Image[] TiltData;
            //    LoadMovieData(options, true, true, true, out TiltMovies, out TiltData);
            //    for (int z = 0; z < NTilts; z++)
            //    {
            //        GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
            //                        TiltData[z].GetDevice(Intent.Write),
            //                        (uint)TiltData[z].ElementsReal,
            //                        1);
                    
            //        TiltData[z].Multiply(-1f);
            //    }

            //    #endregion

            //    #region Make particle sub-tomos and local CTFs

            //    Image[] Experimental = new Image[options.NResults];
            //    Image[] ExperimentalCTF = new Image[options.NResults];
            //    {
            //        #region Make reconstructions

            //        int PlanForwRec, PlanBackRec, PlanForwCTF;
            //        Projector.GetPlans(new int3(SizeFine), 2, out PlanForwRec, out PlanBackRec, out PlanForwCTF);
            //        int PlanForwParticle = GPU.CreateFFTPlan(new int3(SizeFine, SizeFine, 1), (uint)NTilts);

            //        Image CTFCoords = CTF.GetCTFCoords(SizeFine, template.Dims.X);

            //        Image AmpSum = new Image(new int3(SizeFine, SizeFine, 1), true);

            //        for (int n = 0; n < options.NResults; n++)
            //        {
            //            float3 CoordsPhysical = RefinedPositions[n] * (float)options.BinnedPixelSizeMean;

            //            Image ImagesFT = GetImagesForOneParticle(options, TiltMovies, TiltData, SizeFine, CoordsPhysical, PlanForwParticle);

            //            #region Add 2D amplitudes to average

            //            Image ImagesFTAbs = ImagesFT.AsAmplitudes();
            //            Image ImagesFTAbsAvg = ImagesFTAbs.AsReducedAlongZ();
            //            ImagesFTAbs.Dispose();
            //            AmpSum.Add(ImagesFTAbsAvg);
            //            ImagesFTAbsAvg.Dispose();

            //            #endregion

            //            Image CTFs = GetCTFsForOneParticle(CoordsPhysical, CTFCoords, true, false, false);
            //            Image CTFsAbs = GetCTFsForOneParticle(CoordsPhysical, CTFCoords, false, false, false);
            //            CTFsAbs.Abs();

            //            #region Sub-tomo

            //            ImagesFT.Multiply(CTFs);    // Weight and phase-flip image FTs

            //            Projector ProjSubtomo = new Projector(new int3(SizeFine), 2);
            //            ProjSubtomo.BackProject(ImagesFT, CTFsAbs, GetAngleInAllTilts(CoordsPhysical));
            //            Image Subtomo = ProjSubtomo.Reconstruct(false, PlanForwRec, PlanBackRec);

            //            ProjSubtomo.Dispose();
            //            ImagesFT.Dispose();

            //            Subtomo.FreeDevice();
            //            Experimental[n] = Subtomo;

            //            #endregion

            //            #region CTF

            //            // CTF has to be converted to complex numbers with imag = 0, and weighted by itself
            //            float2[] CTFsComplexData = new float2[CTFs.ElementsComplex];
            //            float[] CTFsContinuousData = CTFs.GetHostContinuousCopy();
            //            for (int i = 0; i < CTFsComplexData.Length; i++)
            //                CTFsComplexData[i] = new float2(CTFsContinuousData[i] * CTFsContinuousData[i], 0);

            //            Image CTFsComplex = new Image(CTFsComplexData, CTFs.Dims, true);

            //            // Back-project and reconstruct
            //            Projector ProjCTF = new Projector(new int3(SizeFine), 2);

            //            ProjCTF.BackProject(CTFsComplex, CTFsAbs, GetAngleInAllTilts(CoordsPhysical));
            //            CTFsComplex.Dispose();

            //            Image SubtomoCTF = ProjCTF.Reconstruct(true, PlanForwRec, PlanBackRec, PlanForwCTF);
            //            ProjCTF.Dispose();

            //            SubtomoCTF.FreeDevice();
            //            ExperimentalCTF[n] = SubtomoCTF;

            //            CTFs.Dispose();
            //            CTFsAbs.Dispose();

            //            #endregion
            //        }

            //        GPU.DestroyFFTPlan(PlanForwParticle);
            //        GPU.DestroyFFTPlan(PlanBackRec);
            //        GPU.DestroyFFTPlan(PlanForwRec);

            //        #endregion

            //        #region Calculate 1D amp spectrum

            //        {
            //            AmpSum.Multiply(1f / options.NResults);

            //            int SpectrumLength = SizeFine / 2;
            //            float[] Spectrum = new float[SpectrumLength];
            //            float[] Samples = new float[SpectrumLength];

            //            float[][] FTAmpData = AmpSum.GetHost(Intent.ReadWrite);
            //            for (int y = 0; y < SizeFine; y++)
            //            {
            //                int yy = y < SizeFine / 2 ? y : y - SizeFine;
            //                yy *= yy;

            //                for (int x = 0; x < SizeFine / 2 + 1; x++)
            //                {
            //                    int xx = x * x;

            //                    float r = (float)Math.Sqrt(xx + yy);
                                    
            //                    if (r > SpectrumLength - 1)
            //                        continue;

            //                    float WeightLow = 1f - (r - (int)r);
            //                    float WeightHigh = 1f - WeightLow;
            //                    float Val = FTAmpData[0][y * (SizeFine / 2 + 1) + x];

            //                    Spectrum[(int)r] += WeightLow * Val;
            //                    Samples[(int)r] += WeightLow;

            //                    if ((int)r < SpectrumLength - 1)
            //                    {
            //                        Spectrum[(int)r + 1] += WeightHigh * Val;
            //                        Samples[(int)r + 1] += WeightHigh;
            //                    }
            //                }
            //            }

            //            for (int i = 0; i < Spectrum.Length; i++)
            //                Spectrum[i] = (float)(Spectrum[i] / Math.Max(1e-5f, Samples[i]));

            //            AmpSum.Dispose();
            //            AmpSum = new Image(new int3(SizeFine), true);
            //            FTAmpData = AmpSum.GetHost(Intent.ReadWrite);

            //            for (int z = 0; z < SizeFine; z++)
            //            {
            //                int zz = z < SizeFine / 2 ? z : z - SizeFine;
            //                float fz = (float)zz / (SizeFine / 2);
            //                fz *= fz;

            //                for (int y = 0; y < SizeFine; y++)
            //                {
            //                    int yy = y < SizeFine / 2 ? y : y - SizeFine;
            //                    float fy = (float)yy / (SizeFine / 2);
            //                    fy *= fy;

            //                    for (int x = 0; x < SizeFine / 2 + 1; x++)
            //                    {
            //                        float fx = (float)x / (SizeFine / 2);
            //                        fx *= fx;

            //                        float r = (float)Math.Sqrt(fx + fy + fz) * SpectrumLength;
            //                        r = Math.Min(SpectrumLength - 2, r);

            //                        float WeightLow = 1f - (r - (int)r);
            //                        float WeightHigh = 1f - WeightLow;
            //                        float Val = Spectrum[(int)r] * WeightLow + Spectrum[(int)r + 1] * WeightHigh;

            //                        FTAmpData[z][y * (SizeFine / 2 + 1) + x] = Val > 1e-10f ? 1f / (float)Math.Sqrt(Val) : 0;
            //                    }
            //                }
            //            }
            //        }

            //        #endregion

            //        #region Whiten spectral noise in sub-tomos

            //        int PlanBackCTF = GPU.CreateIFFTPlan(new int3(SizeFine), 1);

            //        Image DebugSum = new Image(new int3(SizeFine));

            //        for (int n = 0; n < options.NResults; n++)
            //        {
            //            Image ExperimentalFT = Experimental[n].AsFFT(true, PlanForwCTF);
            //            ExperimentalFT.Multiply(AmpSum);

            //            Experimental[n].Dispose();
            //            Experimental[n] = ExperimentalFT.AsIFFT(true, PlanBackCTF, false);
            //            ExperimentalFT.Dispose();

            //            GPU.Normalize(Experimental[n].GetDevice(Intent.Read),
            //                          Experimental[n].GetDevice(Intent.Write),
            //                          (uint)Experimental[n].ElementsReal,
            //                          1);

            //            DebugSum.Add(Experimental[n]);

            //            Experimental[n].FreeDevice();
            //        }

            //        GPU.DestroyFFTPlan(PlanBackCTF);
            //        GPU.DestroyFFTPlan(PlanForwCTF);

            //        #endregion

            //        AmpSum.Dispose();
            //    }

            //    #endregion

            //    int PlanBackGrad = GPU.CreateIFFTPlan(new int3(SizeFine), 12);
            //    int PlanBackEval = GPU.CreateIFFTPlan(new int3(SizeFine), 1);
            //    Image CorrResult = new Image(new int3(12, 1, 1));

            //    for (int n = 0; n < options.NResults; n++)
            //    {
            //        Image Subtomo = new Image(Helper.Combine(Helper.ArrayOfConstant(Experimental[n].GetHostContinuousCopy(), 12)), new int3(SizeFine, SizeFine, SizeFine * 12));

            //        Image ParticleMask = new Image(new int3(SizeFine));
            //        ParticleMask.Fill(1f);
            //        GPU.SphereMask(ParticleMask.GetDevice(Intent.Read),
            //                       ParticleMask.GetDevice(Intent.Write),
            //                       new int3(SizeFine),
            //                       FineRadius,
            //                       3,
            //                       1);

            //        Image ParticleMaskFT = ParticleMask.AsFFT(true);
            //        ParticleMask.Dispose();
            //        //ParticleMaskFT.Multiply(ExperimentalCTF[n]);

            //        ParticleMask = ParticleMaskFT.AsIFFT(true, 0, true);
            //        ParticleMaskFT.Dispose();
            //        ParticleMask.Abs();
            //        GPU.SphereMask(ParticleMask.GetDevice(Intent.Read),
            //                       ParticleMask.GetDevice(Intent.Write),
            //                       new int3(SizeFine),
            //                       FineRadius * 1.5f,
            //                       0,
            //                       1);

            //        Image ParticleMasks = new Image(Helper.Combine(Helper.ArrayOfConstant(ParticleMask.GetHostContinuousCopy(), 12)), new int3(SizeFine, SizeFine, SizeFine * 12));
            //        ParticleMask.Dispose();

            //        Func<double[], float3> GetPos = (input) =>
            //        {
            //            return new float3((float)input[0],
            //                              (float)input[1],
            //                              (float)input[2]);
            //        };
            //        Func<double[], float3> GetAng = (input) =>
            //        {
            //            float Conditioning = 180 / SizeFine * Helper.ToRad;
            //            return Matrix3.EulerFromMatrix(Matrix3.RotateX((float)input[3] * Conditioning) *
            //                                           Matrix3.RotateY((float)input[4] * Conditioning) *
            //                                           Matrix3.RotateZ((float)input[5] * Conditioning) *
            //                                           Matrix3.Euler(RefinedAngles[n]));
            //        };

            //        Func<double[], double > Eval = (input) =>
            //        {
            //            Image ProjFT = ProjectorReference.Project(new int3(SizeFine), new[] { GetAng(input) }, new[] { GetPos(input) + SizeFine / 2 }, new[] { 1f });
            //            GPU.MultiplyComplexSlicesByScalar(ProjFT.GetDevice(Intent.Read),
            //                                              ExperimentalCTF[n].GetDevice(Intent.Read),
            //                                              ProjFT.GetDevice(Intent.Write),
            //                                              new int3(SizeFine).ElementsFFT(),
            //                                              1);
            //            Image Proj = ProjFT.AsIFFT(true, PlanBackEval);
            //            ProjFT.Dispose();

            //            //Image Mask = new Image(Proj.GetDevice(Intent.Read), Proj.Dims);
            //            //Mask.Abs();
            //            //GPU.SphereMask(Mask.GetDevice(Intent.Read),
            //            //               Mask.GetDevice(Intent.Write),
            //            //               new int3(SizeFine),
            //            //               FineRadius,
            //            //               0,
            //            //               1);

            //            GPU.CorrelateRealspace(Proj.GetDevice(Intent.Read),
            //                                   Subtomo.GetDevice(Intent.Read),
            //                                   Proj.Dims,
            //                                   ParticleMasks.GetDevice(Intent.Read),
            //                                   CorrResult.GetDevice(Intent.Write),
            //                                   1);

            //            //Mask.Dispose();
            //            Proj.Dispose();

            //            return CorrResult.GetHost(Intent.Read)[0][0];
            //        };

            //        Func<double[], double[]> Grad = (input) =>
            //        {
            //            double[] Result = new double[input.Length];
            //            float Delta = 0.01f;

            //            List<double[]> DeltaInputs = new List<double[]>();
            //            for (int i = 0; i < input.Length; i++)
            //            {
            //                double[] PlusInput = new double[input.Length];
            //                Array.Copy(input, 0, PlusInput, 0, input.Length);
            //                PlusInput[i] += Delta;
            //                DeltaInputs.Add(PlusInput);

            //                double[] MinusInput = new double[input.Length];
            //                Array.Copy(input, 0, MinusInput, 0, input.Length);
            //                MinusInput[i] -= Delta;
            //                DeltaInputs.Add(MinusInput);
            //            }

            //            Image ProjFT = ProjectorReference.Project(new int3(SizeFine),
            //                                                      DeltaInputs.Select(v => GetAng(v)).ToArray(),
            //                                                      DeltaInputs.Select(v => GetPos(v) + SizeFine / 2).ToArray(),
            //                                                      Helper.ArrayOfConstant(1f, 12));
            //            GPU.MultiplyComplexSlicesByScalar(ProjFT.GetDevice(Intent.Read),
            //                                              ExperimentalCTF[n].GetDevice(Intent.Read),
            //                                              ProjFT.GetDevice(Intent.Write),
            //                                              new int3(SizeFine).ElementsFFT(),
            //                                              12);
            //            Image Proj = ProjFT.AsIFFT(true, PlanBackGrad);
            //            ProjFT.Dispose();

            //            //Image Mask = new Image(Proj.GetDevice(Intent.Read), Proj.Dims);
            //            //Mask.Abs();
            //            //GPU.SphereMask(Mask.GetDevice(Intent.Read),
            //            //               Mask.GetDevice(Intent.Write),
            //            //               new int3(SizeFine),
            //            //               FineRadius,
            //            //               0,
            //            //               12);

            //            GPU.CorrelateRealspace(Proj.GetDevice(Intent.Read),
            //                                   Subtomo.GetDevice(Intent.Read),
            //                                   new int3(SizeFine),
            //                                   ParticleMasks.GetDevice(Intent.Read),
            //                                   CorrResult.GetDevice(Intent.Write),
            //                                   12);

            //            //Mask.Dispose();
            //            Proj.Dispose();

            //            float[] CorrResultData = CorrResult.GetHost(Intent.Read)[0];
            //            for (int i = 0; i < input.Length; i++)
            //                Result[i] = (CorrResultData[i * 2] - CorrResultData[i * 2 + 1]) / (Delta * 2);

            //            return Result;
            //        };

            //        double[] StartInput = new double[6];
            //        BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartInput.Length, Eval, Grad);
            //        Optimizer.MaxIterations = 6;

            //        Optimizer.Maximize(StartInput);

            //        RefinedPositions[n] = RefinedPositions[n] + GetPos(StartInput);
            //        RefinedAngles[n] = GetAng(StartInput);
            //        RefinedScores[n] = (float)Eval(StartInput);

            //        Subtomo.Dispose();
            //        ParticleMasks.Dispose();

            //        Experimental[n].Dispose();
            //        ExperimentalCTF[n].Dispose();
            //    }

            //    CorrResult.Dispose();
            //    GPU.DestroyFFTPlan(PlanBackEval);
            //    GPU.DestroyFFTPlan(PlanBackGrad);
            //}

            #endregion

            #region Write peak positions and angles into table

            Star TableOut = new Star(new string[]
            {
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnMicrographName",
                "rlnAutopickFigureOfMerit"
            });

            {
                for (int n = 0; n < InitialPeaks.Length; n++)
                {
                    //float3 Position = RefinedPositions[n] / new float3(DimsVolumeCropped);
                    //float Score = RefinedScores[n];
                    //float3 Angle = RefinedAngles[n] * Helper.ToDeg;

                    float3 Position = new float3(InitialPeaks[n]);
                    float Score = CorrData[(int)Position.Z][(int)Position.Y * DimsVolumeCropped.X + (int)Position.X];
                    float3 Angle = HealpixAngles[(int)AngleData[(int)Position.Z][(int)Position.Y * DimsVolumeCropped.X + (int)Position.X]] * Helper.ToDeg;
                    Position /= new float3(DimsVolumeCropped);

                    TableOut.AddRow(new List<string>()
                    {
                        Position.X.ToString(CultureInfo.InvariantCulture),
                        Position.Y.ToString(CultureInfo.InvariantCulture),
                        Position.Z.ToString(CultureInfo.InvariantCulture),
                        Angle.X.ToString(CultureInfo.InvariantCulture),
                        Angle.Y.ToString(CultureInfo.InvariantCulture),
                        Angle.Z.ToString(CultureInfo.InvariantCulture),
                        RootName + ".mrc",
                        Score.ToString(CultureInfo.InvariantCulture)
                    });
                }
            }

            TableOut.Save(MatchingDir + NameWithRes + "_" + options.TemplateName + ".star");

            progressCallback?.Invoke(Grid, (int)Grid.Elements(), "Done.");

            #endregion
        }

        public void ReconstructFull(ProcessingOptionsTomoFullReconstruction options, Func<int3, int, string, bool> progressCallback)
        {
            bool IsCanceled = false;
            string NameWithRes = RootName + $"_{options.BinnedPixelSizeMean:F2}Apx";

            if (!Directory.Exists(ReconstructionDir))
                Directory.CreateDirectory(ReconstructionDir);

            if (options.DoDeconv && !Directory.Exists(ReconstructionDeconvDir))
                Directory.CreateDirectory(ReconstructionDeconvDir);

            if (File.Exists(ReconstructionDir + NameWithRes + ".mrc") && !options.OverwriteFiles)
                return;

            #region Dimensions

            VolumeDimensionsPhysical = options.DimensionsPhysical;

            int3 DimsVolumeCropped = new int3((int)Math.Round(options.DimensionsPhysical.X / (float)options.BinnedPixelSizeMean / 2) * 2,
                                              (int)Math.Round(options.DimensionsPhysical.Y / (float)options.BinnedPixelSizeMean / 2) * 2,
                                              (int)Math.Round(options.DimensionsPhysical.Z / (float)options.BinnedPixelSizeMean / 2) * 2);
            int SizeSub = options.SubVolumeSize;
            int SizeSubPadded = (int)(SizeSub * options.SubVolumePadding);

            #endregion

            #region Establish reconstruction positions

            int3 Grid = (DimsVolumeCropped + SizeSub - 1) / SizeSub;
            List<float3> GridCoords = new List<float3>();
            for (int z = 0; z < Grid.Z; z++)
                for (int y = 0; y < Grid.Y; y++)
                    for (int x = 0; x < Grid.X; x++)
                        GridCoords.Add(new float3(x * SizeSub + SizeSub / 2,
                                                  y * SizeSub + SizeSub / 2,
                                                  z * SizeSub + SizeSub / 2));

            progressCallback?.Invoke(Grid, 0, "Loading...");

            #endregion

            #region Load and preprocess tilt series

            Movie[] TiltMovies;
            Image[] TiltData;
            Image[] TiltMasks;
            LoadMovieData(options, true, true, true, out TiltMovies, out TiltData);
            LoadMovieMasks(options, 2, 4, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                if (options.Normalize)
                {
                    TiltData[z].SubtractMeanPlane();
                    GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                                  TiltData[z].GetDevice(Intent.Write),
                                  (uint)TiltData[z].ElementsReal,
                                  1);
                }

                if (options.Invert)
                    TiltData[z].Multiply(-1f);

                TiltData[z].Multiply(TiltMasks[z]);
            }

            #endregion

            #region Memory and FFT plan allocation

            Image CTFCoords = CTF.GetCTFCoords(SizeSubPadded, (int)(SizeSubPadded * options.DownsampleFactor));

            float[][] OutputRec = Helper.ArrayOfFunction(i => new float[DimsVolumeCropped.ElementsSlice()], DimsVolumeCropped.Z);
            float[][] OutputRecDeconv = null;

            int PlanForw, PlanBack, PlanForwCTF;
            Projector.GetPlans(new int3(SizeSubPadded), 2, out PlanForw, out PlanBack, out PlanForwCTF);
            int PlanForwParticle = GPU.CreateFFTPlan(new int3(SizeSubPadded, SizeSubPadded, 1), (uint)NTilts);

            #endregion

            #region Reconstruction

            for (int p = 0; p < GridCoords.Count; p++)
            {
                float3 CoordsPhysical = GridCoords[p] * (float)options.BinnedPixelSizeMean;

                Image ImagesFT = GetImagesForOneParticle(options, TiltMovies, TiltData, SizeSubPadded, CoordsPhysical, PlanForwParticle);
                Image CTFs = GetCTFsForOneParticle(options, CoordsPhysical, CTFCoords, true, false, false);

                ImagesFT.Multiply(CTFs);    // Weight and phase-flip image FTs
                CTFs.Abs();                 // No need for Wiener, just phase flipping

                Projector ProjSubtomo = new Projector(new int3(SizeSubPadded), 2);
                ProjSubtomo.BackProject(ImagesFT, CTFs, GetAngleInAllTilts(CoordsPhysical));
                Image Subtomo = ProjSubtomo.Reconstruct(false, PlanForw, PlanBack, PlanForwCTF);
                ProjSubtomo.Dispose();

                Image SubtomoCropped = Subtomo.AsPadded(new int3(SizeSub));
                Subtomo.Dispose();

                ImagesFT.Dispose();
                CTFs.Dispose();

                float[] SubtomoData = SubtomoCropped.GetHostContinuousCopy();
                SubtomoCropped.Dispose();

                int3 Origin = new int3(GridCoords[p]) - SizeSub / 2;
                for (int z = 0; z < SizeSub; z++)
                {
                    int zVol = Origin.Z + z;
                    if (zVol >= DimsVolumeCropped.Z)
                        continue;

                    for (int y = 0; y < SizeSub; y++)
                    {
                        int yVol = Origin.Y + y;
                        if (yVol >= DimsVolumeCropped.Y)
                            continue;

                        for (int x = 0; x < SizeSub; x++)
                        {
                            int xVol = Origin.X + x;
                            if (xVol >= DimsVolumeCropped.X)
                                continue;

                            OutputRec[zVol][yVol * DimsVolumeCropped.X + xVol] = SubtomoData[(z * SizeSub + y) * SizeSub + x];
                        }
                    }
                }

                if (progressCallback != null)
                    IsCanceled = progressCallback(Grid, p + 1, "Reconstructing...");

                if (IsCanceled)
                {
                    GPU.DestroyFFTPlan(PlanForw);
                    GPU.DestroyFFTPlan(PlanBack);
                    GPU.DestroyFFTPlan(PlanForwCTF);
                    GPU.DestroyFFTPlan(PlanForwParticle);

                    CTFCoords.Dispose();
                    foreach (var image in TiltData)
                        image.Dispose();
                    foreach (var tiltMask in TiltMasks)
                        tiltMask?.Dispose();

                    return;
                }
            }

            if (options.DoDeconv)
            {
                IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Deconvolving...");

                Image FullRec = new Image(OutputRec, DimsVolumeCropped);

                Image FullRecFT = FullRec.AsFFT_CPU();
                FullRec.Dispose();

                CTF SubtomoCTF = CTF.GetCopy();
                SubtomoCTF.Defocus = (decimal)GetTiltDefocus(NTilts / 2);
                SubtomoCTF.PixelSize = options.BinnedPixelSizeMean;

                GPU.DeconvolveCTF(FullRecFT.GetDevice(Intent.Read),
                                    FullRecFT.GetDevice(Intent.Write),
                                    FullRecFT.Dims,
                                    SubtomoCTF.ToStruct(),
                                    (float)options.DeconvStrength,
                                    (float)options.DeconvFalloff,
                                    (float)(options.BinnedPixelSizeMean * 2 / options.DeconvHighpass));

                Image FullRecDeconv = FullRecFT.AsIFFT_CPU();
                FullRecFT.Dispose();

                OutputRecDeconv = FullRecDeconv.GetHost(Intent.Read);
                FullRecDeconv.Dispose();
            }

            if (options.KeepOnlyFullVoxels)
            {
                IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Trimming...");

                float BinnedAngPix = (float)options.BinnedPixelSizeMean;

                Parallel.For(0, DimsVolumeCropped.Z, z =>
                {
                    float3[] VolumePositions = new float3[DimsVolumeCropped.ElementsSlice()];
                    for (int y = 0; y < DimsVolumeCropped.Y; y++)
                        for (int x = 0; x < DimsVolumeCropped.X; x++)
                            VolumePositions[y * DimsVolumeCropped.X + x] = new float3(x * BinnedAngPix, y * BinnedAngPix, z * BinnedAngPix);

                    float3[] ImagePositions = GetPositionInAllTiltsNoLocalWarp(VolumePositions);

                    for (int i = 0; i < ImagePositions.Length; i++)
                    {
                        int ii = i / NTilts;
                        int t = i % NTilts;

                        if (ImagePositions[i].X < 0 || ImagePositions[i].Y < 0 ||
                            ImagePositions[i].X > ImageDimensions[t].X - BinnedAngPix ||
                            ImagePositions[i].Y > ImageDimensions[t].Y - BinnedAngPix)
                        {
                            OutputRec[z][ii] = 0;
                            if (options.DoDeconv)
                                OutputRecDeconv[z][ii] = 0;
                        }
                    }
                });
            }

            #endregion

            #region Teardown

            GPU.DestroyFFTPlan(PlanForw);
            GPU.DestroyFFTPlan(PlanBack);
            GPU.DestroyFFTPlan(PlanForwCTF);
            GPU.DestroyFFTPlan(PlanForwParticle);

            CTFCoords.Dispose();
            foreach (var image in TiltData)
                image.Dispose();
            foreach (var tiltMask in TiltMasks)
                tiltMask?.Dispose();

            #endregion

            IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Writing...");

            Image OutputRecImage = new Image(OutputRec, DimsVolumeCropped);
            OutputRecImage.WriteMRC(ReconstructionDir + NameWithRes + ".mrc", (float)options.BinnedPixelSizeMean, true);
            OutputRecImage.Dispose();

            if (options.DoDeconv)
            {
                Image OutputRecDeconvImage = new Image(OutputRecDeconv, DimsVolumeCropped);
                OutputRecDeconvImage.WriteMRC(ReconstructionDeconvDir + NameWithRes + ".mrc", (float)options.BinnedPixelSizeMean, true);
                OutputRecDeconvImage.Dispose();
            }

            IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Done.");
        }

        public void ReconstructSubtomos(ProcessingOptionsTomoSubReconstruction options, float3[] positions, float3[] angles)
        {
            bool IsCanceled = false;

            if (!Directory.Exists(SubtomoDir))
                Directory.CreateDirectory(SubtomoDir);
            
            #region Dimensions

            VolumeDimensionsPhysical = options.DimensionsPhysical;
            
            int SizeSub = options.BoxSize;
            int SizeSubPadded = SizeSub * 2;

            #endregion

            #region Load and preprocess tilt series

            Movie[] TiltMovies;
            Image[] TiltData;
            Image[] TiltMasks;
            LoadMovieData(options, true, true, true, out TiltMovies, out TiltData);
            LoadMovieMasks(options, 3, 4, out TiltMasks);
            for (int z = 0; z < NTilts; z++)
            {
                if (options.NormalizeInput)
                {
                    TiltData[z].SubtractMeanPlane();
                    GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                                  TiltData[z].GetDevice(Intent.Write),
                                  (uint)TiltData[z].ElementsReal,
                                  1);
                }

                if (options.Invert)
                    TiltData[z].Multiply(-1f);

                TiltData[z].Multiply(TiltMasks[z]);
            }

            #endregion

            #region Memory and FFT plan allocation

            Image CTFCoords = CTF.GetCTFCoords(SizeSub, (int)(SizeSub * options.DownsampleFactor));
            Image CTFCoordsPadded = CTF.GetCTFCoords(SizeSubPadded, (int)(SizeSubPadded * options.DownsampleFactor));

            int PlanForwRec, PlanBackRec, PlanForwCTF;
            Projector.GetPlans(new int3(SizeSub), 2, out PlanForwRec, out PlanBackRec, out PlanForwCTF);
            int PlanForwParticlePadded = GPU.CreateFFTPlan(new int3(SizeSubPadded, SizeSubPadded, 1), (uint)NTilts);
            int PlanBackParticlePadded = GPU.CreateIFFTPlan(new int3(SizeSubPadded, SizeSubPadded, 1), (uint)NTilts);
            int PlanForwParticle = GPU.CreateFFTPlan(new int3(SizeSub, SizeSub, 1), (uint)NTilts);

            #endregion

            for (int p = 0; p < positions.Length / NTilts; p++)
            {
                float3[] ParticlePositions = positions.Skip(p * NTilts).Take(NTilts).ToArray();
                float3[] ParticleAngles = options.PrerotateParticles ? angles.Skip(p * NTilts).Take(NTilts).ToArray() : null;

                #region Flip phases at double the size

                Image ImagesPaddedFT = GetImagesForOneParticle(options, TiltMovies, TiltData, SizeSubPadded, ParticlePositions, PlanForwParticlePadded);
                Image CTFsPadded = GetCTFsForOneParticle(options, ParticlePositions, CTFCoordsPadded, false, false, false);
                CTFsPadded.Sign();
                ImagesPaddedFT.Multiply(CTFsPadded);

                Image ImagesPadded = ImagesPaddedFT.AsIFFT(false, PlanBackParticlePadded, true);
                ImagesPadded.RemapFromFT();
                CTFsPadded.Dispose();
                ImagesPaddedFT.Dispose();

                Image Images = ImagesPadded.AsPadded(new int2(SizeSub));
                Images.RemapToFT();
                ImagesPadded.Dispose();

                Image ImagesFT = Images.AsFFT(false, PlanForwParticle);
                Images.Dispose();

                #endregion

                Image CTFsUnweighted = GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, false, false, false);
                Image CTFs = GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, true, false, false);

                #region Sub-tomo

                ImagesFT.Multiply(CTFs);    // Weight and phase-flip image FTs
                CTFsUnweighted.Abs();       // Divide by unweighted CTF during reconstruction to actually downweight the amplitudes in the result
                
                Projector ProjSubtomo = new Projector(new int3(SizeSub), 2);
                ProjSubtomo.BackProject(ImagesFT, CTFsUnweighted, !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles));

                Image Subtomo = ProjSubtomo.Reconstruct(false, PlanForwRec, PlanBackRec, PlanForwCTF);
                ProjSubtomo.Dispose();

                #endregion

                #region CTF

                // CTF has to be converted to complex numbers with imag = 0, and weighted by itself
                float2[] CTFsComplexData = new float2[CTFs.ElementsComplex];
                float[] CTFsContinuousData = CTFs.GetHostContinuousCopy();
                for (int i = 0; i < CTFsComplexData.Length; i++)
                    CTFsComplexData[i] = new float2(CTFsContinuousData[i] * CTFsContinuousData[i], 0);

                Image CTFsComplex = new Image(CTFsComplexData, CTFs.Dims, true);

                // Back-project and reconstruct
                Projector ProjCTF = new Projector(new int3(SizeSub), 2);

                ProjCTF.BackProject(CTFsComplex, CTFsUnweighted, !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles));
                CTFsComplex.Dispose();

                Image SubtomoCTF = ProjCTF.Reconstruct(true, PlanForwRec, PlanBackRec, PlanForwCTF);
                ProjCTF.Dispose();

                #endregion

                if (options.NormalizeOutput)
                    GPU.NormParticles(Subtomo.GetDevice(Intent.Read),
                                      Subtomo.GetDevice(Intent.Write),
                                      new int3(SizeSub),
                                      (uint)Math.Round(options.ParticleDiameter / options.BinnedPixelSizeMean / 2),
                                      false,
                                      1);

                Subtomo.WriteMRC(SubtomoDir + $"{RootName}_{p:D7}_{options.BinnedPixelSizeMean:F2}A.mrc", true);
                Subtomo.Dispose();

                SubtomoCTF.WriteMRC(SubtomoDir + $"{RootName}_{p:D7}_ctf_{options.BinnedPixelSizeMean:F2}A.mrc", true);
                SubtomoCTF.Dispose();

                ImagesFT.Dispose();
                CTFs.Dispose();
                CTFsUnweighted.Dispose();

                if (IsCanceled)
                    break;
            }

            #region Teardown

            GPU.DestroyFFTPlan(PlanForwRec);
            GPU.DestroyFFTPlan(PlanBackRec);
            GPU.DestroyFFTPlan(PlanForwCTF);
            GPU.DestroyFFTPlan(PlanForwParticlePadded);
            GPU.DestroyFFTPlan(PlanBackParticlePadded);
            GPU.DestroyFFTPlan(PlanForwParticle);

            CTFCoords.Dispose();
            CTFCoordsPadded.Dispose();
            foreach (var image in TiltData)
                image.Dispose();
            foreach (var tiltMask in TiltMasks)
                tiltMask?.Dispose();

            #endregion
        }

        #endregion

        #region Refinement

        public void PerformOptimizationStep(Star tableIn,
                                            Image tiltStack,
                                            int size,
                                            int3 volumeDimensions,
                                            Dictionary<int, Projector> references,
                                            float lowpassAngstrom,
                                            float highpassAngstrom,
                                            int nIterations,
                                            bool doAnyAlignment,
                                            bool doImageAlignment,
                                            bool doPerParticleMotion,
                                            bool updateReconstructions,
                                            Dictionary<int, Projector> outReconstructions,
                                            Dictionary<int, Projector> outCTFReconstructions)
        {
            //VolumeDimensions = volumeDimensions;

            //#region Get rows from table

            //List<int> RowIndices = new List<int>();
            //string[] ColumnMicrographName = tableIn.GetColumn("rlnMicrographName");
            //for (int i = 0; i < ColumnMicrographName.Length; i++)
            //    if (ColumnMicrographName[i].Contains(RootName + "."))
            //        RowIndices.Add(i);

            //if (RowIndices.Count == 0)
            //    return;

            //int NParticles = RowIndices.Count;

            //#endregion

            //#region Make sure all columns and directories are there

            //if (!tableIn.HasColumn("rlnParticleSelectZScore"))
            //    tableIn.AddColumn("rlnParticleSelectZScore");

            //if (!Directory.Exists(ParticlesDir))
            //    Directory.CreateDirectory(ParticlesDir);
            //if (!Directory.Exists(ParticleCTFDir))
            //    Directory.CreateDirectory(ParticleCTFDir);

            //#endregion

            //#region Get subtomo positions from table

            //float3[] ParticleOrigins = new float3[NParticles];
            //float3[] ParticleOrigins2 = new float3[NParticles];
            //float3[] ParticleAngles = new float3[NParticles];
            //float3[] ParticleAngles2 = new float3[NParticles];
            //int[] ParticleSubset = new int[NParticles];
            //{
            //    string[] ColumnPosX = tableIn.GetColumn("rlnCoordinateX");
            //    string[] ColumnPosY = tableIn.GetColumn("rlnCoordinateY");
            //    string[] ColumnPosZ = tableIn.GetColumn("rlnCoordinateZ");
            //    string[] ColumnOriginX = tableIn.GetColumn("rlnOriginX");
            //    string[] ColumnOriginY = tableIn.GetColumn("rlnOriginY");
            //    string[] ColumnOriginZ = tableIn.GetColumn("rlnOriginZ");
            //    string[] ColumnAngleRot = tableIn.GetColumn("rlnAngleRot");
            //    string[] ColumnAngleTilt = tableIn.GetColumn("rlnAngleTilt");
            //    string[] ColumnAnglePsi = tableIn.GetColumn("rlnAnglePsi");
            //    string[] ColumnSubset = tableIn.GetColumn("rlnRandomSubset");

            //    string[] ColumnPosX2 = tableIn.GetColumn("rlnOriginXPrior");
            //    string[] ColumnPosY2 = tableIn.GetColumn("rlnOriginYPrior");
            //    string[] ColumnPosZ2 = tableIn.GetColumn("rlnOriginZPrior");
            //    string[] ColumnAngleRot2 = tableIn.GetColumn("rlnAngleRotPrior");
            //    string[] ColumnAngleTilt2 = tableIn.GetColumn("rlnAngleTiltPrior");
            //    string[] ColumnAnglePsi2 = tableIn.GetColumn("rlnAnglePsiPrior");

            //    for (int i = 0; i < NParticles; i++)
            //    {
            //        float3 Pos = new float3(float.Parse(ColumnPosX[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                float.Parse(ColumnPosY[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                float.Parse(ColumnPosZ[RowIndices[i]], CultureInfo.InvariantCulture));
            //        float3 Pos2 = Pos;
            //        //if (ColumnPosX2 != null && ColumnPosY2 != null && ColumnPosZ2 != null)
            //        //    Pos2 = new float3(float.Parse(ColumnPosX2[RowIndices[i]], CultureInfo.InvariantCulture),
            //        //                      float.Parse(ColumnPosY2[RowIndices[i]], CultureInfo.InvariantCulture),
            //        //                      float.Parse(ColumnPosZ2[RowIndices[i]], CultureInfo.InvariantCulture));

            //        float3 Shift = new float3(float.Parse(ColumnOriginX[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnOriginY[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnOriginZ[RowIndices[i]], CultureInfo.InvariantCulture));

            //        ParticleOrigins[i] = Pos - Shift;
            //        ParticleOrigins2[i] = Pos2 - Shift;

            //        float3 Angle = new float3(float.Parse(ColumnAngleRot[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnAngleTilt[RowIndices[i]], CultureInfo.InvariantCulture),
            //                                  float.Parse(ColumnAnglePsi[RowIndices[i]], CultureInfo.InvariantCulture));
            //        float3 Angle2 = Angle;
            //        //if (ColumnAngleRot2 != null && ColumnAngleTilt2 != null && ColumnAnglePsi2 != null)
            //        //    Angle2 = new float3(float.Parse(ColumnAngleRot2[RowIndices[i]], CultureInfo.InvariantCulture),
            //        //                        float.Parse(ColumnAngleTilt2[RowIndices[i]], CultureInfo.InvariantCulture),
            //        //                        float.Parse(ColumnAnglePsi2[RowIndices[i]], CultureInfo.InvariantCulture));

            //        ParticleAngles[i] = Angle;
            //        ParticleAngles2[i] = Angle2;

            //        ParticleSubset[i] = (int)float.Parse(ColumnSubset[RowIndices[i]], CultureInfo.InvariantCulture);

            //        tableIn.SetRowValue(RowIndices[i], "rlnCoordinateX", ParticleOrigins[i].X.ToString(CultureInfo.InvariantCulture));
            //        tableIn.SetRowValue(RowIndices[i], "rlnCoordinateY", ParticleOrigins[i].Y.ToString(CultureInfo.InvariantCulture));
            //        tableIn.SetRowValue(RowIndices[i], "rlnCoordinateZ", ParticleOrigins[i].Z.ToString(CultureInfo.InvariantCulture));
            //        tableIn.SetRowValue(RowIndices[i], "rlnOriginX", "0.0");
            //        tableIn.SetRowValue(RowIndices[i], "rlnOriginY", "0.0");
            //        tableIn.SetRowValue(RowIndices[i], "rlnOriginZ", "0.0");
            //    }
            //}

            //#endregion

            //#region Deal with subsets

            //List<int> SubsetIDs = new List<int>();
            //foreach (var i in ParticleSubset)
            //    if (!SubsetIDs.Contains(i))
            //        SubsetIDs.Add(i);
            //SubsetIDs.Sort();

            //// For each subset, create a list of its particle IDs
            //Dictionary<int, List<int>> SubsetParticleIDs = SubsetIDs.ToDictionary(subsetID => subsetID, subsetID => new List<int>());
            //for (int i = 0; i < ParticleSubset.Length; i++)
            //    SubsetParticleIDs[ParticleSubset[i]].Add(i);
            //foreach (var list in SubsetParticleIDs.Values)
            //    list.Sort();

            //// Note where each subset starts and ends in a unified, sorted (by subset) particle ID list
            //Dictionary<int, Tuple<int, int>> SubsetRanges = new Dictionary<int, Tuple<int, int>>();
            //{
            //    int Start = 0;
            //    foreach (var pair in SubsetParticleIDs)
            //    {
            //        SubsetRanges.Add(pair.Key, new Tuple<int, int>(Start, Start + pair.Value.Count));
            //        Start += pair.Value.Count;
            //    }
            //}

            //List<int> SubsetContinuousIDs = new List<int>();
            //foreach (var pair in SubsetParticleIDs)
            //    SubsetContinuousIDs.AddRange(pair.Value);

            //// Reorder particle information to match the order of SubsetContinuousIDs
            //ParticleOrigins = SubsetContinuousIDs.Select(i => ParticleOrigins[i]).ToArray();
            //ParticleOrigins2 = SubsetContinuousIDs.Select(i => ParticleOrigins2[i]).ToArray();
            //ParticleAngles = SubsetContinuousIDs.Select(i => ParticleAngles[i]).ToArray();
            //ParticleAngles2 = SubsetContinuousIDs.Select(i => ParticleAngles2[i]).ToArray();
            //ParticleSubset = SubsetContinuousIDs.Select(i => ParticleSubset[i]).ToArray();

            //#endregion

            //float3[] OptimizedOrigins, OptimizedOrigins2, OptimizedAngles, OptimizedAngles2;

            //GPU.Normalize(tiltStack.GetDevice(Intent.Read),
            //              tiltStack.GetDevice(Intent.Write),
            //              (uint)tiltStack.ElementsSliceReal,
            //              (uint)tiltStack.Dims.Z);

            //#region Perform alignment of everything if desired

            //if (doAnyAlignment)
            //{
            //    #region Update tilt parameter grids if needed

            //    if (doImageAlignment)
            //    {
            //        if (GridMovementX.Dimensions.Elements() == 1)
            //        {
            //            int MaxSlice = SubsetRanges.Last().Value.Item2 > 100 ? 1 : 1;

            //            GridMovementX = new CubicGrid(new int3(MaxSlice, MaxSlice, NTilts));
            //            GridMovementY = new CubicGrid(new int3(MaxSlice, MaxSlice, NTilts));

            //            GridAngleX = new CubicGrid(new int3(1, 1, NTilts));
            //            GridAngleY = new CubicGrid(new int3(1, 1, NTilts));
            //            GridAngleZ = new CubicGrid(new int3(1, 1, NTilts));
            //        }

            //        int NSpace = 1, NTime = 2;
            //        while (NSpace * NSpace * NTime < NParticles / 4 && NTime < NTilts)
            //            NTime++;
            //        NTime = Math.Min(1, NTime);

            //        if (GridLocalX.Dimensions.Elements() != NSpace * NSpace * NTime)
            //        {
            //            GridLocalX = GridLocalX.Resize(new int3(NSpace, NSpace, NTime));
            //            GridLocalY = GridLocalY.Resize(new int3(NSpace, NSpace, NTime));
            //            GridLocalZ = GridLocalZ.Resize(new int3(NSpace, NSpace, NTime));
            //        }
            //    }

            //    #endregion

            //    Image FilteredTiltStack = new Image(tiltStack.GetDevice(Intent.Read), tiltStack.Dims);
            //    tiltStack.FreeDevice();
            //    {
            //        float HighpassFraction = (float)CTF.PixelSize * 2 / highpassAngstrom;
            //        for (int t = 0; t < NTilts; t++)
            //        {
            //            GPU.Bandpass(FilteredTiltStack.GetDeviceSlice(t, Intent.Read),
            //                         FilteredTiltStack.GetDeviceSlice(t, Intent.Write),
            //                         FilteredTiltStack.Dims.Slice(),
            //                         HighpassFraction,
            //                         1,
            //                         1);
            //        }
            //        FilteredTiltStack.FreeDevice();
            //        //if (GPU.GetDevice() == 0)
            //        //    FilteredTiltStack.WriteMRC("d_filteredstack.mrc");
            //    }

            //    int CoarseSize = (int)Math.Round(size * ((float)CTF.PixelSize * 2 / lowpassAngstrom)) / 2 * 2;
            //    float HighpassSize = size * (float)CTF.PixelSize * 2 / highpassAngstrom;
            //    int3 CoarseDims = new int3(CoarseSize, CoarseSize, 1);

            //    // Positions the particles were extracted at/shifted to, to calculate effectively needed shifts later
            //    float2[] ExtractedAt = new float2[NParticles * NTilts];

            //    // Extract images, mask and resize them, create CTFs
            //    Image ParticleImages = new Image(new int3(CoarseSize, CoarseSize, NParticles * NTilts), true, true);
            //    Image ParticleCTFs = new Image(new int3(CoarseSize, CoarseSize, NParticles * NTilts), true);
            //    Image ParticleWeights = null;
            //    Image ShiftFactors = null;

            //    #region Preflight

            //    float KeepBFac = GlobalBfactor;
            //    GlobalBfactor = 0;
            //    {
            //        Image CTFCoords = CTF.GetCTFCoords(CoarseSize, size);

            //        #region Precalculate vectors for shifts in Fourier space

            //        {
            //            float2[] ShiftFactorsData = new float2[(CoarseSize / 2 + 1) * CoarseSize];
            //            for (int y = 0; y < CoarseSize; y++)
            //                for (int x = 0; x < CoarseSize / 2 + 1; x++)
            //                {
            //                    int xx = x;
            //                    int yy = y < CoarseSize / 2 + 1 ? y : y - CoarseSize;

            //                    ShiftFactorsData[y * (CoarseSize / 2 + 1) + x] = new float2((float)-xx / size * 2f * (float)Math.PI,
            //                                                                                (float)-yy / size * 2f * (float)Math.PI);
            //                }

            //            ShiftFactors = new Image(ShiftFactorsData, new int3(CoarseSize, CoarseSize, 1), true);
            //        }

            //        #endregion

            //        #region Create mask with soft edge

            //        Image Mask;
            //        Image MaskSubt;
            //        {
            //            Image MaskBig = new Image(new int3(size, size, 1));
            //            float MaskRadius = 0;//MainWindow.Options.ExportParticleRadius / (float)CTF.PixelSize;         // FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            //            float SoftEdge = 16f;

            //            float[] MaskBigData = MaskBig.GetHost(Intent.Write)[0];
            //            for (int y = 0; y < size; y++)
            //            {
            //                int yy = y - size / 2;
            //                yy *= yy;
            //                for (int x = 0; x < size; x++)
            //                {
            //                    int xx = x - size / 2;
            //                    xx *= xx;
            //                    float R = (float)Math.Sqrt(xx + yy);

            //                    if (R <= MaskRadius)
            //                        MaskBigData[y * size + x] = 1;
            //                    else
            //                        MaskBigData[y * size + x] = (float)(Math.Cos(Math.Min(1, (R - MaskRadius) / SoftEdge) * Math.PI) * 0.5 + 0.5);
            //                }
            //            }
            //            //MaskBig.WriteMRC("d_maskbig.mrc");

            //            Mask = MaskBig.AsScaled(new int2(CoarseSize, CoarseSize));
            //            Mask.RemapToFT();

            //            MaskBigData = MaskBig.GetHost(Intent.Write)[0];
            //            for (int y = 0; y < size; y++)
            //            {
            //                int yy = y - size / 2;
            //                yy *= yy;
            //                for (int x = 0; x < size; x++)
            //                {
            //                    int xx = x - size / 2;
            //                    xx *= xx;
            //                    float R = (float)Math.Sqrt(xx + yy);

            //                    if (R <= 30)
            //                        MaskBigData[y * size + x] = 1;
            //                    else
            //                        MaskBigData[y * size + x] = 0;
            //                }
            //            }

            //            MaskSubt = MaskBig.AsScaled(new int2(CoarseSize, CoarseSize));
            //            MaskSubt.RemapToFT();

            //            MaskBig.Dispose();
            //        }
            //        //Mask.WriteMRC("d_masksmall.mrc");

            //        #endregion

            //        #region Create Fourier space mask

            //        Image FourierMask = new Image(CoarseDims, true);
            //        {
            //            float[] FourierMaskData = FourierMask.GetHost(Intent.Write)[0];
            //            float MinR = HighpassSize / 2;
            //            float MaxR = CoarseSize / 2;
            //            for (int y = 0; y < CoarseSize; y++)
            //            {
            //                int yy = y < CoarseSize / 2 + 1 ? y : y - CoarseSize;
            //                yy *= yy;

            //                for (int x = 0; x < CoarseSize / 2 + 1; x++)
            //                {
            //                    int xx = x * x;
            //                    float R = (float)Math.Sqrt(yy + xx);

            //                    float ValHigh = Math.Max(0, Math.Min(1, 1 - (MinR - R)));
            //                    float ValLow = Math.Max(0, Math.Min(1, 1 - (R - MaxR)));

            //                    FourierMaskData[y * (CoarseSize / 2 + 1) + x] = ValHigh * ValLow;
            //                }
            //            }
            //        }
            //        //if (GPU.GetDevice() == 0)
            //        //    FourierMask.WriteMRC("d_fouriermask.mrc");

            //        #endregion

            //        #region For each particle, create CTFs and extract & preprocess images for entire tilt series

            //        for (int p = 0; p < NParticles; p++)
            //        {
            //            float3 ParticleCoords = ParticleOrigins[p];
            //            float3[] Positions = GetPositionInImages(ParticleCoords);
            //            float3[] ProjAngles = GetParticleAngleInImages(ParticleCoords, ParticleAngles[p]);

            //            Image Extracted = new Image(new int3(size, size, NTilts));
            //            float[][] ExtractedData = Extracted.GetHost(Intent.Write);
            //            float3[] Residuals = new float3[NTilts];

            //            Image SubtrahendsCTF = new Image(new int3(CoarseSize, CoarseSize, NTilts), true);

            //            // Create CTFs
            //            {
            //                CTFStruct[] CTFParams = new CTFStruct[NTilts];

            //                float GridStep = 1f / (NTilts - 1);
            //                CTFStruct[] Params = new CTFStruct[NTilts];
            //                for (int t = 0; t < NTilts; t++)
            //                {
            //                    decimal Defocus = (decimal)Positions[t].Z;
            //                    decimal DefocusDelta = (decimal)GridCTFDefocusDelta.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));
            //                    decimal DefocusAngle = (decimal)GridCTFDefocusAngle.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));

            //                    CTF CurrCTF = CTF.GetCopy();
            //                    CurrCTF.Defocus = Defocus;
            //                    CurrCTF.DefocusDelta = DefocusDelta;
            //                    CurrCTF.DefocusAngle = DefocusAngle;
            //                    CurrCTF.Scale = (decimal)Math.Cos(Angles[t] * Helper.ToRad);
            //                    CurrCTF.Bfactor = (decimal)-Dose[t] * 8;

            //                    Params[t] = CurrCTF.ToStruct();
            //                }

            //                GPU.CreateCTF(ParticleCTFs.GetDeviceSlice(NTilts * p, Intent.Write),
            //                              CTFCoords.GetDevice(Intent.Read),
            //                              (uint)CoarseDims.ElementsFFT(),
            //                              Params,
            //                              false,
            //                              (uint)NTilts);
            //            }
            //            //{
            //            //    CTFStruct[] CTFParams = new CTFStruct[NTilts];

            //            //    float GridStep = 1f / (NTilts - 1);
            //            //    CTFStruct[] Params = new CTFStruct[NTilts];
            //            //    for (int t = 0; t < NTilts; t++)
            //            //    {
            //            //        decimal Defocus = (decimal)Positions[t].Z;
            //            //        decimal DefocusDelta = (decimal)GridCTFDefocusDelta.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));
            //            //        decimal DefocusAngle = (decimal)GridCTFDefocusAngle.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));

            //            //        CTF CurrCTF = CTF.GetCopy();
            //            //        CurrCTF.Defocus = Defocus;
            //            //        CurrCTF.DefocusDelta = DefocusDelta;
            //            //        CurrCTF.DefocusAngle = DefocusAngle;
            //            //        CurrCTF.Scale = 1;
            //            //        CurrCTF.Bfactor = 0;

            //            //        Params[t] = CurrCTF.ToStruct();
            //            //    }

            //            //    GPU.CreateCTF(SubtrahendsCTF.GetDevice(Intent.Write),
            //            //                  CTFCoords.GetDevice(Intent.Read),
            //            //                  (uint)CoarseDims.ElementsFFT(),
            //            //                  Params,
            //            //                  false,
            //            //                  (uint)NTilts);
            //            //}

            //            // Extract images
            //            {
            //                for (int t = 0; t < NTilts; t++)
            //                {
            //                    ExtractedAt[p * NTilts + t] = new float2(Positions[t].X, Positions[t].Y);

            //                    Positions[t] -= size / 2;
            //                    int2 IntPosition = new int2((int)Positions[t].X, (int)Positions[t].Y);
            //                    float2 Residual = new float2(-(Positions[t].X - IntPosition.X), -(Positions[t].Y - IntPosition.Y));
            //                    Residuals[t] = new float3(Residual / size * CoarseSize);

            //                    float[] OriginalData;
            //                    OriginalData = FilteredTiltStack.GetHost(Intent.Read)[t];

            //                    float[] ImageData = ExtractedData[t];
            //                    for (int y = 0; y < size; y++)
            //                    {
            //                        int PosY = (y + IntPosition.Y + FilteredTiltStack.Dims.Y) % FilteredTiltStack.Dims.Y;
            //                        for (int x = 0; x < size; x++)
            //                        {
            //                            int PosX = (x + IntPosition.X + FilteredTiltStack.Dims.X) % FilteredTiltStack.Dims.X;
            //                            ImageData[y * size + x] = OriginalData[PosY * FilteredTiltStack.Dims.X + PosX];
            //                        }
            //                    }
            //                }

            //                GPU.NormParticles(Extracted.GetDevice(Intent.Read),
            //                                  Extracted.GetDevice(Intent.Write),
            //                                  new int3(size, size, 1),
            //                                  (uint)(123 / CTF.PixelSize),         // FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            //                                  true,
            //                                  (uint)NTilts);

            //                Image Scaled = Extracted.AsScaled(new int2(CoarseSize, CoarseSize));
            //                //Scaled.WriteMRC("d_scaled.mrc");
            //                Extracted.Dispose();

            //                Scaled.ShiftSlices(Residuals);
            //                Scaled.RemapToFT();

            //                //GPU.NormalizeMasked(Scaled.GetDevice(Intent.Read),
            //                //              Scaled.GetDevice(Intent.Write),
            //                //              MaskSubt.GetDevice(Intent.Read),
            //                //              (uint)Scaled.ElementsSliceReal,
            //                //              (uint)NTilts);

            //                //{
            //                //    //Image SubtrahendsFT = subtrahendReference.Project(new int2(CoarseSize, CoarseSize), ProjAngles, CoarseSize / 2);
            //                //    //SubtrahendsFT.Multiply(SubtrahendsCTF);

            //                //    //Image Subtrahends = SubtrahendsFT.AsIFFT();
            //                //    //SubtrahendsFT.Dispose();

            //                //    ////GPU.NormalizeMasked(Subtrahends.GetDevice(Intent.Read),
            //                //    ////                    Subtrahends.GetDevice(Intent.Write),
            //                //    ////                    MaskSubt.GetDevice(Intent.Read),
            //                //    ////                    (uint)Subtrahends.ElementsSliceReal,
            //                //    ////                    (uint)NTilts);

            //                //    //Scaled.Subtract(Subtrahends);
            //                //    //Subtrahends.Dispose();

            //                //    Image FocusMaskFT = maskReference.Project(new int2(CoarseSize, CoarseSize), ProjAngles, CoarseSize / 2);
            //                //    Image FocusMask = FocusMaskFT.AsIFFT();
            //                //    FocusMaskFT.Dispose();

            //                //    Scaled.Multiply(FocusMask);
            //                //    FocusMask.Dispose();
            //                //}

            //                Scaled.MultiplySlices(Mask);

            //                GPU.FFT(Scaled.GetDevice(Intent.Read),
            //                        ParticleImages.GetDeviceSlice(p * NTilts, Intent.Write),
            //                        CoarseDims,
            //                        (uint)NTilts,
            //                        -1);

            //                Scaled.Dispose();
            //                SubtrahendsCTF.Dispose();
            //            }
            //        }

            //        #endregion

            //        ParticleCTFs.MultiplySlices(FourierMask);

            //        Mask.Dispose();
            //        FourierMask.Dispose();
            //        MaskSubt.Dispose();

            //        Image ParticleCTFsAbs = new Image(ParticleCTFs.GetDevice(Intent.Read), ParticleCTFs.Dims, true);
            //        ParticleCTFsAbs.Abs();
            //        ParticleWeights = ParticleCTFsAbs.AsSum2D();
            //        ParticleCTFsAbs.Dispose();
            //        {
            //            float[] ParticleWeightsData = ParticleWeights.GetHost(Intent.ReadWrite)[0];
            //            float Max = MathHelper.Max(ParticleWeightsData);
            //            for (int i = 0; i < ParticleWeightsData.Length; i++)
            //                ParticleWeightsData[i] /= Max;
            //        }

            //        CTFCoords.Dispose();

            //        //Image CheckImages = ParticleImages.AsIFFT();
            //        //CheckImages.WriteMRC("d_particleimages.mrc");
            //        //CheckImages.Dispose();

            //        //ParticleCTFs.WriteMRC("d_particlectfs.mrc");
            //    }
            //    GlobalBfactor = KeepBFac;

            //    #endregion

            //    #region BFGS evaluation and gradient

            //    double[] StartParams;

            //    Func<double[], Tuple<float2[], float3[]>> GetImageShiftsAndAngles;
            //    Func<double[], float2[]> GetImageShifts;
            //    Func<float3[], Image> GetProjections;
            //    Func<double[], double[]> EvalIndividual;
            //    Func<double[], double> Eval;
            //    Func<double[], double[]> Gradient;
            //    {
            //        List<double> StartParamsList = new List<double>();
            //        //StartParamsList.AddRange(CreateVectorFromGrids(Dimensions.X));         // FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            //        StartParamsList.AddRange(CreateVectorFromParameters(ParticleOrigins, ParticleOrigins2, ParticleAngles, ParticleAngles2, size));
            //        StartParams = StartParamsList.ToArray();

            //        // Remember where the values for each grid are stored in the optimized vector
            //        List<Tuple<int, int>> VectorGridRanges = new List<Tuple<int, int>>();
            //        List<int> GridElements = new List<int>();
            //        List<int> GridSliceElements = new List<int>();
            //        {
            //            int Start = 0;
            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridMovementX.Dimensions.Elements()));
            //            Start += (int)GridMovementX.Dimensions.Elements();
            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridMovementY.Dimensions.Elements()));
            //            Start += (int)GridMovementY.Dimensions.Elements();

            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridAngleX.Dimensions.Elements()));
            //            Start += (int)GridAngleX.Dimensions.Elements();
            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridAngleY.Dimensions.Elements()));
            //            Start += (int)GridAngleY.Dimensions.Elements();
            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridAngleZ.Dimensions.Elements()));
            //            Start += (int)GridAngleZ.Dimensions.Elements();

            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridLocalX.Dimensions.Elements()));
            //            Start += (int)GridLocalX.Dimensions.Elements();
            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridLocalY.Dimensions.Elements()));
            //            Start += (int)GridLocalY.Dimensions.Elements();
            //            VectorGridRanges.Add(new Tuple<int, int>(Start, Start + (int)GridLocalZ.Dimensions.Elements()));

            //            GridElements.Add((int)GridMovementX.Dimensions.Elements());
            //            GridElements.Add((int)GridMovementY.Dimensions.Elements());

            //            GridElements.Add((int)GridAngleX.Dimensions.Elements());
            //            GridElements.Add((int)GridAngleY.Dimensions.Elements());
            //            GridElements.Add((int)GridAngleZ.Dimensions.Elements());

            //            GridElements.Add((int)GridLocalX.Dimensions.Elements());
            //            GridElements.Add((int)GridLocalY.Dimensions.Elements());
            //            GridElements.Add((int)GridLocalZ.Dimensions.Elements());

            //            GridSliceElements.Add((int)GridMovementX.Dimensions.ElementsSlice());
            //            GridSliceElements.Add((int)GridMovementY.Dimensions.ElementsSlice());

            //            GridSliceElements.Add((int)GridAngleX.Dimensions.ElementsSlice());
            //            GridSliceElements.Add((int)GridAngleY.Dimensions.ElementsSlice());
            //            GridSliceElements.Add((int)GridAngleZ.Dimensions.ElementsSlice());

            //            GridSliceElements.Add((int)GridLocalX.Dimensions.ElementsSlice());
            //            GridSliceElements.Add((int)GridLocalY.Dimensions.ElementsSlice());
            //            GridSliceElements.Add((int)GridLocalZ.Dimensions.ElementsSlice());
            //        }
            //        int NVectorGridParams = VectorGridRanges.Last().Item2;
            //        int NVectorParticleParams = NParticles * 12;

            //        GetImageShiftsAndAngles = input =>
            //        {
            //            // Retrieve particle positions & angles, and grids from input vector
            //            float3[] NewPositions, NewPositions2, NewAngles, NewAngles2;
            //            GetParametersFromVector(input, NParticles, size, out NewPositions, out NewPositions2, out NewAngles, out NewAngles2);
            //            //SetGridsFromVector(input, Dimensions.X);         // FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            //            // Using current positions, angles and grids, get parameters for image shifts and reference projection angles
            //            float2[] ImageShifts = new float2[NParticles * NTilts];
            //            float3[] ImageAngles = new float3[NParticles * NTilts];
            //            float3[] PerTiltPositions = new float3[NParticles * NTilts];
            //            float3[] PerTiltAngles = new float3[NParticles * NTilts];
            //            int[] SortedDosePrecalc = IndicesSortedDose;
            //            for (int p = 0; p < NParticles; p++)
            //            {
            //                if (doPerParticleMotion)
            //                {
            //                    float3 CoordsDiff = NewPositions2[p] - NewPositions[p];
            //                    float3 AnglesDiff = NewAngles2[p] - NewAngles[p];
            //                    for (int t = 0; t < NTilts; t++)
            //                    {
            //                        float DoseID = SortedDosePrecalc[t] / (float)(NTilts - 1);
            //                        PerTiltPositions[p * NTilts + t] = NewPositions[p] + CoordsDiff * DoseID;
            //                        PerTiltAngles[p * NTilts + t] = NewAngles[p] + AnglesDiff * DoseID;
            //                    }
            //                }
            //                else
            //                {
            //                    for (int t = 0; t < NTilts; t++)
            //                    {
            //                        PerTiltPositions[p * NTilts + t] = NewPositions[p];
            //                        PerTiltAngles[p * NTilts + t] = NewAngles[p];
            //                    }
            //                }
            //            }

            //            float3[] CurrPositions = GetPositionInImages(PerTiltPositions);
            //            float3[] CurrAngles = GetParticleAngleInImages(PerTiltPositions, PerTiltAngles);
            //            for (int i = 0; i < ImageShifts.Length; i++)
            //            {
            //                ImageShifts[i] = new float2(ExtractedAt[i].X - CurrPositions[i].X,
            //                                            ExtractedAt[i].Y - CurrPositions[i].Y); // -diff because those are extraction positions, i. e. opposite direction of shifts
            //                ImageAngles[i] = CurrAngles[i];
            //            }

            //            return new Tuple<float2[], float3[]>(ImageShifts, ImageAngles);
            //        };

            //        GetImageShifts = input =>
            //        {
            //            // Retrieve particle positions & angles, and grids from input vector
            //            float3[] NewPositions, NewPositions2, NewAngles, NewAngles2;
            //            GetParametersFromVector(input, NParticles, size, out NewPositions, out NewPositions2, out NewAngles, out NewAngles2);
            //            //SetGridsFromVector(input, Dimensions.X);         // FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            //            // Using current positions, angles and grids, get parameters for image shifts and reference projection angles
            //            float2[] ImageShifts = new float2[NParticles * NTilts];
            //            float3[] PerTiltPositions = new float3[NParticles * NTilts];
            //            int[] SortedDosePrecalc = IndicesSortedDose;
            //            for (int p = 0; p < NParticles; p++)
            //            {
            //                if (doPerParticleMotion)
            //                {
            //                    float3 CoordsDiff = NewPositions2[p] - NewPositions[p];
            //                    float3 AnglesDiff = NewAngles2[p] - NewAngles[p];
            //                    for (int t = 0; t < NTilts; t++)
            //                    {
            //                        float DoseID = SortedDosePrecalc[t] / (float)(NTilts - 1);
            //                        PerTiltPositions[p * NTilts + t] = NewPositions[p] + CoordsDiff * DoseID;
            //                    }
            //                }
            //                else
            //                {
            //                    for (int t = 0; t < NTilts; t++)
            //                        PerTiltPositions[p * NTilts + t] = NewPositions[p];
            //                }
            //            }

            //            float3[] CurrPositions = GetPositionInImages(PerTiltPositions);
            //            for (int i = 0; i < ImageShifts.Length; i++)
            //                ImageShifts[i] = new float2(ExtractedAt[i].X - CurrPositions[i].X,
            //                                            ExtractedAt[i].Y - CurrPositions[i].Y); // -diff because those are extraction positions, i. e. opposite direction of shifts

            //            return ImageShifts;
            //        };

            //        GetProjections = imageAngles =>
            //        {
            //            Image Projections = new Image(IntPtr.Zero, new int3(CoarseSize, CoarseSize, NParticles * NTilts), true, true);
            //            foreach (var subset in SubsetRanges)
            //            {
            //                Projector Reference = references[subset.Key];
            //                int SubsetStart = subset.Value.Item1 * NTilts;
            //                int SubsetEnd = subset.Value.Item2 * NTilts;
            //                float3[] SubsetAngles = imageAngles.Skip(SubsetStart).Take(SubsetEnd - SubsetStart).ToArray();

            //                GPU.ProjectForward(Reference.Data.GetDevice(Intent.Read),
            //                                   Projections.GetDeviceSlice(SubsetStart, Intent.Write),
            //                                   Reference.Data.Dims,
            //                                   new int2(CoarseSize, CoarseSize),
            //                                   Helper.ToInterleaved(SubsetAngles),
            //                                   Reference.Oversampling,
            //                                   (uint)(SubsetEnd - SubsetStart));
            //            }

            //            /*Image CheckProjections = Projections.AsIFFT();
            //        //CheckProjections.RemapFromFT();
            //        CheckProjections.WriteMRC("d_projections.mrc");
            //        CheckProjections.Dispose();*/

            //            return Projections;
            //        };

            //        EvalIndividual = input =>
            //        {
            //            Tuple<float2[], float3[]> ShiftsAndAngles = GetImageShiftsAndAngles(input);

            //            Image Projections = GetProjections(ShiftsAndAngles.Item2);

            //            float[] Results = new float[NParticles * NTilts];

            //            GPU.TomoRefineGetDiff(ParticleImages.GetDevice(Intent.Read),
            //                                  Projections.GetDevice(Intent.Read),
            //                                  ShiftFactors.GetDevice(Intent.Read),
            //                                  ParticleCTFs.GetDevice(Intent.Read),
            //                                  ParticleWeights.GetDevice(Intent.Read),
            //                                  new int2(CoarseSize, CoarseSize),
            //                                  Helper.ToInterleaved(ShiftsAndAngles.Item1),
            //                                  Results,
            //                                  (uint)(NParticles * NTilts));

            //            Projections.Dispose();

            //            return Results.Select(i => (double)i).ToArray();
            //        };

            //        int OptimizationIterations = 0;
            //        bool GetOut = false;

            //        double Delta = 0.1;
            //        float Delta2 = 2 * (float)Delta;

            //        int[] WarpGridIDs = { 5, 6, 7 };
            //        Dictionary<int, float2[][]> WiggleWeightsWarp = new Dictionary<int, float2[][]>();
            //        if (doImageAlignment)
            //        {
            //            foreach (var gridID in WarpGridIDs)
            //            {
            //                int NElements = GridElements[gridID];
            //                WiggleWeightsWarp.Add(gridID, new float2[NElements][]);

            //                for (int ge = 0; ge < NElements; ge++)
            //                {
            //                    double[] InputMinus = new double[StartParams.Length], InputPlus = new double[StartParams.Length];
            //                    for (int i = 0; i < StartParams.Length; i++)
            //                    {
            //                        InputMinus[i] = StartParams[i];
            //                        InputPlus[i] = StartParams[i];
            //                    }

            //                    InputMinus[VectorGridRanges[gridID].Item1 + ge] -= Delta;
            //                    InputPlus[VectorGridRanges[gridID].Item1 + ge] += Delta;

            //                    float2[] ImageShiftsPlus = GetImageShifts(InputPlus);
            //                    float2[] ImageShiftsMinus = GetImageShifts(InputMinus);

            //                    float2[] Weights = new float2[ImageShiftsPlus.Length];

            //                    for (int i = 0; i < ImageShiftsPlus.Length; i++)
            //                        Weights[i] = (ImageShiftsPlus[i] - ImageShiftsMinus[i]) / Delta2;

            //                    WiggleWeightsWarp[gridID][ge] = Weights;
            //                }
            //            }
            //        }

            //        Eval = input =>
            //        {
            //            double Result = EvalIndividual(input).Sum();
            //            lock (tableIn)
            //                Debug.WriteLine(GPU.GetDevice() + ", " + RootName + ": " + Result);
            //            OptimizationIterations++;

            //            return Result;
            //        };

            //        Func<double[], double[], double, double[]> GradientParticles = (inputMinus, inputPlus, delta) =>
            //        {
            //            double[] EvalMinus = EvalIndividual(inputMinus);
            //            double[] EvalPlus = EvalIndividual(inputPlus);

            //            double[] Diff = new double[EvalMinus.Length];
            //            for (int i = 0; i < Diff.Length; i++)
            //                Diff[i] = (EvalPlus[i] - EvalMinus[i]) / (2 * delta);

            //            return Diff;
            //        };

            //        Gradient = input =>
            //        {
            //            double[] Result = new double[input.Length];

            //            if (OptimizationIterations > nIterations)
            //                return Result;

            //            float2[] ImageShiftGradients = new float2[NParticles * NTilts];

            //            #region Compute gradient for individual image shifts

            //            {
            //                Tuple<float2[], float3[]> ShiftsAndAngles = GetImageShiftsAndAngles(input);
            //                Image Projections = GetProjections(ShiftsAndAngles.Item2);

            //                float2[] ShiftsXPlus = new float2[NParticles * NTilts];
            //                float2[] ShiftsXMinus = new float2[NParticles * NTilts];
            //                float2[] ShiftsYPlus = new float2[NParticles * NTilts];
            //                float2[] ShiftsYMinus = new float2[NParticles * NTilts];

            //                float2 DeltaX = new float2((float)Delta, 0);
            //                float2 DeltaY = new float2(0, (float)Delta);

            //                for (int i = 0; i < ShiftsXPlus.Length; i++)
            //                {
            //                    ShiftsXPlus[i] = ShiftsAndAngles.Item1[i] + DeltaX;
            //                    ShiftsXMinus[i] = ShiftsAndAngles.Item1[i] - DeltaX;

            //                    ShiftsYPlus[i] = ShiftsAndAngles.Item1[i] + DeltaY;
            //                    ShiftsYMinus[i] = ShiftsAndAngles.Item1[i] - DeltaY;
            //                }

            //                float[] ScoresXPlus = new float[NParticles * NTilts];
            //                float[] ScoresXMinus = new float[NParticles * NTilts];
            //                float[] ScoresYPlus = new float[NParticles * NTilts];
            //                float[] ScoresYMinus = new float[NParticles * NTilts];

            //                GPU.TomoRefineGetDiff(ParticleImages.GetDevice(Intent.Read),
            //                                      Projections.GetDevice(Intent.Read),
            //                                      ShiftFactors.GetDevice(Intent.Read),
            //                                      ParticleCTFs.GetDevice(Intent.Read),
            //                                      ParticleWeights.GetDevice(Intent.Read),
            //                                      new int2(CoarseSize, CoarseSize),
            //                                      Helper.ToInterleaved(ShiftsXPlus),
            //                                      ScoresXPlus,
            //                                      (uint)(NParticles * NTilts));
            //                GPU.TomoRefineGetDiff(ParticleImages.GetDevice(Intent.Read),
            //                                      Projections.GetDevice(Intent.Read),
            //                                      ShiftFactors.GetDevice(Intent.Read),
            //                                      ParticleCTFs.GetDevice(Intent.Read),
            //                                      ParticleWeights.GetDevice(Intent.Read),
            //                                      new int2(CoarseSize, CoarseSize),
            //                                      Helper.ToInterleaved(ShiftsXMinus),
            //                                      ScoresXMinus,
            //                                      (uint)(NParticles * NTilts));
            //                GPU.TomoRefineGetDiff(ParticleImages.GetDevice(Intent.Read),
            //                                      Projections.GetDevice(Intent.Read),
            //                                      ShiftFactors.GetDevice(Intent.Read),
            //                                      ParticleCTFs.GetDevice(Intent.Read),
            //                                      ParticleWeights.GetDevice(Intent.Read),
            //                                      new int2(CoarseSize, CoarseSize),
            //                                      Helper.ToInterleaved(ShiftsYPlus),
            //                                      ScoresYPlus,
            //                                      (uint)(NParticles * NTilts));
            //                GPU.TomoRefineGetDiff(ParticleImages.GetDevice(Intent.Read),
            //                                      Projections.GetDevice(Intent.Read),
            //                                      ShiftFactors.GetDevice(Intent.Read),
            //                                      ParticleCTFs.GetDevice(Intent.Read),
            //                                      ParticleWeights.GetDevice(Intent.Read),
            //                                      new int2(CoarseSize, CoarseSize),
            //                                      Helper.ToInterleaved(ShiftsYMinus),
            //                                      ScoresYMinus,
            //                                      (uint)(NParticles * NTilts));

            //                Projections.Dispose();

            //                for (int i = 0; i < ImageShiftGradients.Length; i++)
            //                {
            //                    ImageShiftGradients[i] = new float2((ScoresXPlus[i] - ScoresXMinus[i]) / Delta2,
            //                                                        (ScoresYPlus[i] - ScoresYMinus[i]) / Delta2);
            //                }
            //            }

            //            #endregion

            //            // First, do particle parameters, i. e. 3D position within tomogram, rotation, across 2 points in time
            //            // Altering each particle's parameters results in a change in its NTilts images, but nothing else
            //            {
            //                int[] TranslationIDs = doPerParticleMotion ? new[] { 0, 1, 2, 3, 4, 5 } : new[] { 0, 1, 2 };
            //                int[] RotationIDs = doPerParticleMotion ? new[] { 6, 7, 8, 9, 10, 11 } : new[] { 6, 7, 8 };
            //                foreach (var paramID in RotationIDs)
            //                {
            //                    double[] InputMinus = new double[input.Length], InputPlus = new double[input.Length];
            //                    for (int i = 0; i < input.Length; i++)
            //                    {
            //                        InputMinus[i] = input[i];
            //                        InputPlus[i] = input[i];
            //                    }
            //                    for (int p = 0; p < NParticles; p++)
            //                    {
            //                        InputMinus[NVectorGridParams + p * 12 + paramID] -= Delta;
            //                        InputPlus[NVectorGridParams + p * 12 + paramID] += Delta;
            //                    }

            //                    double[] ResultParticles = GradientParticles(InputMinus, InputPlus, Delta);
            //                    for (int p = 0; p < NParticles; p++)
            //                    {
            //                        double ParticleSum = 0;
            //                        for (int t = 0; t < NTilts; t++)
            //                            ParticleSum += ResultParticles[p * NTilts + t];

            //                        Result[NVectorGridParams + p * 12 + paramID] = ParticleSum;
            //                    }
            //                }

            //                // Translation-related gradients can all be computed efficiently from previously retrieved per-image gradients
            //                foreach (var paramID in TranslationIDs)
            //                {
            //                    double[] InputMinus = new double[input.Length], InputPlus = new double[input.Length];
            //                    for (int i = 0; i < input.Length; i++)
            //                    {
            //                        InputMinus[i] = input[i];
            //                        InputPlus[i] = input[i];
            //                    }
            //                    for (int p = 0; p < NParticles; p++)
            //                    {
            //                        InputMinus[NVectorGridParams + p * 12 + paramID] -= Delta;
            //                        InputPlus[NVectorGridParams + p * 12 + paramID] += Delta;
            //                    }

            //                    float2[] ImageShiftsPlus = GetImageShifts(InputPlus);
            //                    float2[] ImageShiftsMinus = GetImageShifts(InputMinus);

            //                    for (int p = 0; p < NParticles; p++)
            //                    {
            //                        double ParticleSum = 0;
            //                        for (int t = 0; t < NTilts; t++)
            //                        {
            //                            int i = p * NTilts + t;
            //                            float2 ShiftDelta = (ImageShiftsPlus[i] - ImageShiftsMinus[i]) / Delta2;
            //                            float ShiftGradient = ShiftDelta.X * ImageShiftGradients[i].X + ShiftDelta.Y * ImageShiftGradients[i].Y;

            //                            ParticleSum += ShiftGradient;
            //                        }

            //                        Result[NVectorGridParams + p * 12 + paramID] = ParticleSum;
            //                    }
            //                }

            //                // If there is no per-particle motion, just copy the gradients for these parameters from parameterIDs 0-5
            //                if (!doPerParticleMotion)
            //                {
            //                    int[] RedundantIDs = { 3, 4, 5, 9, 10, 11 };
            //                    foreach (var paramID in RedundantIDs)
            //                        for (int p = 0; p < NParticles; p++)
            //                            Result[NVectorGridParams + p * 12 + paramID] = Result[NVectorGridParams + p * 12 + paramID - 3];
            //                }
            //            }

            //            // Now deal with grids. Each grid slice (i. e. temporal point) will correspond to one tilt only, thus the gradient
            //            // for each slice is the (weighted, in case of spatial resolution) sum of NParticles images in the corresponding tilt.
            //            if (doImageAlignment)
            //            {
            //                int[] RotationGridIDs = { 2, 3, 4 };
            //                foreach (var gridID in RotationGridIDs)
            //                {
            //                    int SliceElements = GridSliceElements[gridID];

            //                    for (int se = 0; se < SliceElements; se++)
            //                    {
            //                        double[] InputMinus = new double[input.Length], InputPlus = new double[input.Length];
            //                        for (int i = 0; i < input.Length; i++)
            //                        {
            //                            InputMinus[i] = input[i];
            //                            InputPlus[i] = input[i];
            //                        }
            //                        for (int gp = VectorGridRanges[gridID].Item1 + se; gp < VectorGridRanges[gridID].Item2; gp += SliceElements)
            //                        {
            //                            InputMinus[gp] -= Delta;
            //                            InputPlus[gp] += Delta;
            //                        }

            //                        double[] ResultParticles = GradientParticles(InputMinus, InputPlus, Delta);
            //                        for (int i = 0; i < ResultParticles.Length; i++)
            //                        {
            //                            int GridTime = i % NTilts;
            //                            Result[VectorGridRanges[gridID].Item1 + GridTime * SliceElements + se] += ResultParticles[i];
            //                        }
            //                    }
            //                }

            //                // Translation-related gradients can all be computed efficiently from previously retrieved per-image gradients
            //                int[] TranslationGridIDs = { 0, 1 };
            //                foreach (var gridID in TranslationGridIDs)
            //                {
            //                    int SliceElements = GridSliceElements[gridID];

            //                    for (int se = 0; se < SliceElements; se++)
            //                    {
            //                        double[] InputMinus = new double[input.Length], InputPlus = new double[input.Length];
            //                        for (int i = 0; i < input.Length; i++)
            //                        {
            //                            InputMinus[i] = input[i];
            //                            InputPlus[i] = input[i];
            //                        }
            //                        for (int gp = VectorGridRanges[gridID].Item1 + se; gp < VectorGridRanges[gridID].Item2; gp += SliceElements)
            //                        {
            //                            InputMinus[gp] -= Delta;
            //                            InputPlus[gp] += Delta;
            //                        }

            //                        float2[] ImageShiftsPlus = GetImageShifts(InputPlus);
            //                        float2[] ImageShiftsMinus = GetImageShifts(InputMinus);

            //                        for (int i = 0; i < ImageShiftsPlus.Length; i++)
            //                        {
            //                            float2 ShiftDelta = (ImageShiftsPlus[i] - ImageShiftsMinus[i]) / Delta2;
            //                            float ShiftGradient = ShiftDelta.X * ImageShiftGradients[i].X + ShiftDelta.Y * ImageShiftGradients[i].Y;

            //                            int GridSlice = i % NTilts;
            //                            Result[VectorGridRanges[gridID].Item1 + GridSlice * SliceElements + se] += ShiftGradient;
            //                        }
            //                    }
            //                }
            //                // Warp grids don't have any shortcuts for getting multiple gradients at once, so they use pre-calculated wiggle weights
            //                foreach (var gridID in WarpGridIDs)
            //                {
            //                    int NElements = GridElements[gridID];

            //                    for (int ge = 0; ge < NElements; ge++)
            //                    {
            //                        float2[] Weights = WiggleWeightsWarp[gridID][ge];

            //                        for (int i = 0; i < Weights.Length; i++)
            //                        {
            //                            float2 ShiftDelta = Weights[i];
            //                            float ShiftGradient = ShiftDelta.X * ImageShiftGradients[i].X + ShiftDelta.Y * ImageShiftGradients[i].Y;

            //                            Result[VectorGridRanges[gridID].Item1 + ge] += ShiftGradient;
            //                        }
            //                    }
            //                }
            //            }

            //            return Result;
            //        };
            //    }

            //    #endregion

            //    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient);
            //    Optimizer.Epsilon = 3e-7;

            //    Optimizer.Maximize(StartParams);

            //    GetParametersFromVector(StartParams, NParticles, size, out OptimizedOrigins, out OptimizedOrigins2, out OptimizedAngles, out OptimizedAngles2);
            //    //SetGridsFromVector(StartParams, Dimensions.X);         // FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            //    #region Calculate correlation scores, update table with new positions and angles

            //    {
            //        double[] ImageScores = EvalIndividual(StartParams);
            //        float[] ParticleScores = new float[NParticles];
            //        for (int i = 0; i < ImageScores.Length; i++)
            //            ParticleScores[i / NTilts] += (float)ImageScores[i];

            //        lock (tableIn)
            //            for (int p = 0; p < NParticles; p++)
            //            {
            //                int Row = RowIndices[SubsetContinuousIDs[p]];

            //                tableIn.SetRowValue(Row, "rlnCoordinateX", OptimizedOrigins[p].X.ToString(CultureInfo.InvariantCulture));
            //                tableIn.SetRowValue(Row, "rlnCoordinateY", OptimizedOrigins[p].Y.ToString(CultureInfo.InvariantCulture));
            //                tableIn.SetRowValue(Row, "rlnCoordinateZ", OptimizedOrigins[p].Z.ToString(CultureInfo.InvariantCulture));

            //                tableIn.SetRowValue(Row, "rlnAngleRot", OptimizedAngles[p].X.ToString(CultureInfo.InvariantCulture));
            //                tableIn.SetRowValue(Row, "rlnAngleTilt", OptimizedAngles[p].Y.ToString(CultureInfo.InvariantCulture));
            //                tableIn.SetRowValue(Row, "rlnAnglePsi", OptimizedAngles[p].Z.ToString(CultureInfo.InvariantCulture));

            //                tableIn.SetRowValue(Row, "rlnParticleSelectZScore", ParticleScores[p].ToString(CultureInfo.InvariantCulture));
            //            }
            //    }

            //    #endregion

            //    ParticleImages?.Dispose();
            //    ParticleCTFs?.Dispose();
            //    ParticleWeights?.Dispose();
            //    ShiftFactors?.Dispose();
            //    FilteredTiltStack.Dispose();
            //}
            //else
            //{
            //    OptimizedOrigins = ParticleOrigins;
            //    OptimizedOrigins2 = ParticleOrigins2;
            //    OptimizedAngles = ParticleAngles;
            //    OptimizedAngles2 = ParticleAngles2;
            //}

            //#endregion

            //#region Extract particles at full resolution and back-project them into the reconstruction volumes

            //if (updateReconstructions)
            //{
            //    GPU.SetDevice(0);

            //    Image CTFCoords = CTF.GetCTFCoords(size, size);
            //    int[] SortedDosePrecalc = IndicesSortedDose;

            //    foreach (var subsetRange in SubsetRanges)
            //    {
            //        lock (outReconstructions[subsetRange.Key])
            //        {
            //            for (int p = subsetRange.Value.Item1; p < subsetRange.Value.Item2; p++)
            //            {
            //                float3[] PerTiltPositions = new float3[NTilts];
            //                float3[] PerTiltAngles = new float3[NTilts];
            //                float3 CoordsDiff = OptimizedOrigins2[p] - OptimizedOrigins[p];
            //                float3 AnglesDiff = OptimizedAngles2[p] - OptimizedAngles[p];
            //                for (int t = 0; t < NTilts; t++)
            //                {
            //                    float DoseID = SortedDosePrecalc[t] / (float)(NTilts - 1);
            //                    PerTiltPositions[t] = OptimizedOrigins[p] + CoordsDiff * DoseID;
            //                    PerTiltAngles[t] = OptimizedAngles[p] + AnglesDiff * DoseID;
            //                }

            //                Image FullParticleImages = GetSubtomoImages(tiltStack, size, PerTiltPositions, false);
            //                Image FullParticleCTFs = GetSubtomoCTFs(PerTiltPositions, CTFCoords);

            //                FullParticleImages.Multiply(FullParticleCTFs);
            //                FullParticleCTFs.Abs();

            //                float3[] FullParticleAngles = GetParticleAngleInImages(PerTiltPositions, PerTiltAngles);

            //                outReconstructions[subsetRange.Key].BackProject(FullParticleImages, FullParticleCTFs, FullParticleAngles);

            //                FullParticleImages.Dispose();
            //                FullParticleCTFs.Dispose();
            //            }

            //            for (int p = subsetRange.Value.Item1; p < subsetRange.Value.Item2; p++)
            //            {
            //                float3[] PerTiltPositions = new float3[NTilts];
            //                float3[] PerTiltAngles = new float3[NTilts];
            //                float3 CoordsDiff = OptimizedOrigins2[p] - OptimizedOrigins[p];
            //                float3 AnglesDiff = OptimizedAngles2[p] - OptimizedAngles[p];
            //                for (int t = 0; t < NTilts; t++)
            //                {
            //                    float DoseID = SortedDosePrecalc[t] / (float)(NTilts - 1);
            //                    PerTiltPositions[t] = OptimizedOrigins[p] + CoordsDiff * DoseID;
            //                    PerTiltAngles[t] = OptimizedAngles[p] + AnglesDiff * DoseID;
            //                }

            //                float3[] FullParticleAngles = GetParticleAngleInImages(PerTiltPositions, PerTiltAngles);

            //                Image FullParticleCTFs = GetSubtomoCTFs(PerTiltPositions, CTFCoords, false);
            //                Image FullParticleCTFWeights = GetSubtomoCTFs(PerTiltPositions, CTFCoords, true);

            //                // CTF has to be converted to complex numbers with imag = 0
            //                float2[] CTFsComplexData = new float2[FullParticleCTFs.ElementsComplex];
            //                float[] CTFWeightsData = new float[FullParticleCTFs.ElementsComplex];
            //                float[] CTFsContinuousData = FullParticleCTFs.GetHostContinuousCopy();
            //                float[] CTFWeightsContinuousData = FullParticleCTFWeights.GetHostContinuousCopy();
            //                for (int i = 0; i < CTFsComplexData.Length; i++)
            //                {
            //                    CTFsComplexData[i] = new float2(Math.Abs(CTFsContinuousData[i] * CTFWeightsContinuousData[i]), 0);
            //                    CTFWeightsData[i] = Math.Abs(CTFWeightsContinuousData[i]);
            //                }

            //                Image CTFsComplex = new Image(CTFsComplexData, FullParticleCTFs.Dims, true);
            //                Image CTFWeights = new Image(CTFWeightsData, FullParticleCTFs.Dims, true);

            //                outCTFReconstructions[subsetRange.Key].BackProject(CTFsComplex, CTFWeights, FullParticleAngles);

            //                FullParticleCTFs.Dispose();
            //                FullParticleCTFWeights.Dispose();
            //                CTFsComplex.Dispose();
            //                CTFWeights.Dispose();
            //            }

            //            outReconstructions[subsetRange.Key].FreeDevice();
            //            outCTFReconstructions[subsetRange.Key].FreeDevice();
            //        }
            //    }

            //    CTFCoords.Dispose();
            //}

            //#endregion

            //if (doImageAlignment)
            //    SaveMeta();
        }

        private void SetGridsFromVector(double[] v, int sizeImage)
        {
            float AverageMoveImage = sizeImage / 180f;
            int Start = 0;

            GridMovementX = new CubicGrid(GridMovementX.Dimensions, v.Skip(Start).Take((int)GridMovementX.Dimensions.Elements()).Select(i => (float)i).ToArray());
            Start += (int)GridMovementX.Dimensions.Elements();

            GridMovementY = new CubicGrid(GridMovementY.Dimensions, v.Skip(Start).Take((int)GridMovementY.Dimensions.Elements()).Select(i => (float)i).ToArray());
            Start += (int)GridMovementY.Dimensions.Elements();

            GridAngleX = new CubicGrid(GridAngleX.Dimensions, v.Skip(Start).Take((int)GridAngleX.Dimensions.Elements()).Select(i => (float)i / AverageMoveImage).ToArray());
            Start += (int)GridAngleX.Dimensions.Elements();

            GridAngleY = new CubicGrid(GridAngleY.Dimensions, v.Skip(Start).Take((int)GridAngleY.Dimensions.Elements()).Select(i => (float)i / AverageMoveImage).ToArray());
            Start += (int)GridAngleY.Dimensions.Elements();

            GridAngleZ = new CubicGrid(GridAngleZ.Dimensions, v.Skip(Start).Take((int)GridAngleZ.Dimensions.Elements()).Select(i => (float)i / AverageMoveImage).ToArray());
            Start += (int)GridAngleZ.Dimensions.Elements();

            GridLocalX = new CubicGrid(GridLocalX.Dimensions, v.Skip(Start).Take((int)GridLocalX.Dimensions.Elements()).Select(i => (float)i).ToArray());
            Start += (int)GridLocalX.Dimensions.Elements();

            GridLocalY = new CubicGrid(GridLocalY.Dimensions, v.Skip(Start).Take((int)GridLocalY.Dimensions.Elements()).Select(i => (float)i).ToArray());
            Start += (int)GridLocalY.Dimensions.Elements();

            GridLocalZ = new CubicGrid(GridLocalZ.Dimensions, v.Skip(Start).Take((int)GridLocalZ.Dimensions.Elements()).Select(i => (float)i).ToArray());
            Start += (int)GridLocalZ.Dimensions.Elements();
        }

        private void GetParametersFromVector(double[] v, int nparticles, int sizeParticle, out float3[] particleShifts, out float3[] particleShifts2, out float3[] particleAngles, out float3[] particleAngles2)
        {
            float AverageMoveParticle = sizeParticle / 180f;
            int Start = 0;
            Start += (int)GridMovementX.Dimensions.Elements();
            Start += (int)GridMovementY.Dimensions.Elements();
            Start += (int)GridAngleX.Dimensions.Elements();
            Start += (int)GridAngleY.Dimensions.Elements();
            Start += (int)GridAngleZ.Dimensions.Elements();
            Start += (int)GridLocalX.Dimensions.Elements();
            Start += (int)GridLocalY.Dimensions.Elements();
            Start += (int)GridLocalZ.Dimensions.Elements();

            particleShifts = new float3[nparticles];
            particleShifts2 = new float3[nparticles];
            particleAngles = new float3[nparticles];
            particleAngles2 = new float3[nparticles];
            for (int n = 0; n < nparticles; n++)
            {
                particleShifts[n] = new float3((float)v[Start + n * 12 + 0], (float)v[Start + n * 12 + 1], (float)v[Start + n * 12 + 2]);
                particleShifts2[n] = new float3((float)v[Start + n * 12 + 3], (float)v[Start + n * 12 + 4], (float)v[Start + n * 12 + 5]);
                particleAngles[n] = new float3((float)v[Start + n * 12 + 6], (float)v[Start + n * 12 + 7], (float)v[Start + n * 12 + 8]) / AverageMoveParticle;
                particleAngles2[n] = new float3((float)v[Start + n * 12 + 9], (float)v[Start + n * 12 + 10], (float)v[Start + n * 12 + 11]) / AverageMoveParticle;
            }
        }

        private double[] CreateVectorFromGrids(int sizeImage)
        {
            float AverageMoveImage = sizeImage / 180f;
            List<float> V = new List<float>();

            V.AddRange(GridMovementX.FlatValues);
            V.AddRange(GridMovementY.FlatValues);

            V.AddRange(GridAngleX.FlatValues.Select(i => i * AverageMoveImage));
            V.AddRange(GridAngleY.FlatValues.Select(i => i * AverageMoveImage));
            V.AddRange(GridAngleZ.FlatValues.Select(i => i * AverageMoveImage));

            V.AddRange(GridLocalX.FlatValues);
            V.AddRange(GridLocalY.FlatValues);
            V.AddRange(GridLocalZ.FlatValues);

            return V.Select(i => (double)i).ToArray();
        }

        private double[] CreateVectorFromParameters(float3[] particleShifts, float3[] particleShifts2, float3[] particleAngles, float3[] particleAngles2, int sizeParticle)
        {
            float AverageMoveParticle = sizeParticle / 180f;
            double[] V = new double[particleShifts.Length * 12];

            for (int n = 0; n < particleShifts.Length; n++)
            {
                V[n * 12 + 0] = particleShifts[n].X;
                V[n * 12 + 1] = particleShifts[n].Y;
                V[n * 12 + 2] = particleShifts[n].Z;

                V[n * 12 + 3] = particleShifts2[n].X;
                V[n * 12 + 4] = particleShifts2[n].Y;
                V[n * 12 + 5] = particleShifts2[n].Z;

                V[n * 12 + 6] = particleAngles[n].X * AverageMoveParticle;
                V[n * 12 + 7] = particleAngles[n].Y * AverageMoveParticle;
                V[n * 12 + 8] = particleAngles[n].Z * AverageMoveParticle;

                V[n * 12 + 9] = particleAngles2[n].X * AverageMoveParticle;
                V[n * 12 + 10] = particleAngles2[n].Y * AverageMoveParticle;
                V[n * 12 + 11] = particleAngles2[n].Z * AverageMoveParticle;
            }

            return V;
        }

        #endregion

        #region Helper methods

        #region GetPosition methods

        public float3[] GetPositionInAllTilts(float3 coords)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetPositionInAllTilts(PerTiltCoords);
        }

        public float3[] GetPositionInAllTilts(float3[] coords)
        {
            float3[] Result = new float3[coords.Length];

            float3 VolumeCenter = VolumeDimensionsPhysical / 2;
            float2 ImageCenter = ImageDimensions[0] / 2;

            float GridStep = 1f / (NTilts - 1);
            float DoseStep = 1f / MaxDose;

            float3[] GridCoords = new float3[coords.Length];
            float3[] TemporalGridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                GridCoords[i] = new float3(coords[i].X / VolumeDimensionsPhysical.X, coords[i].Y / VolumeDimensionsPhysical.Y, t * GridStep);
                TemporalGridCoords[i] = new float3(GridCoords[i].X, GridCoords[i].Y, Dose[t] * GridStep);
            }

            float[] GridAngleXInterp = GridAngleX.GetInterpolatedNative(GridCoords);
            float[] GridAngleYInterp = GridAngleY.GetInterpolatedNative(GridCoords);
            float[] GridAngleZInterp = GridAngleZ.GetInterpolatedNative(GridCoords);

            float[] GridLocalXInterp = GridLocalX.GetInterpolatedNative(TemporalGridCoords);
            float[] GridLocalYInterp = GridLocalY.GetInterpolatedNative(TemporalGridCoords);
            float[] GridLocalZInterp = GridLocalZ.GetInterpolatedNative(TemporalGridCoords);

            float[] GridDefocusInterp = GridCTF.GetInterpolatedNative(GridCoords.Take(NTilts).ToArray());

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, AnglesCorrect[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);

            float3[] TransformedCoords = new float3[coords.Length];
            float3[] Normals = new float3[coords.Length];

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;
                float3 Centered = coords[i] - VolumeCenter;

                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleYInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleXInterp[i] * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrices[t];

                float3 SampleWarping = new float3(GridLocalXInterp[i],
                                                  GridLocalYInterp[i],
                                                  GridLocalZInterp[i]);
                Centered += SampleWarping;

                float3 Transformed = (Rotation * Centered);

                Transformed.X += TiltAxisOffsetX[t];   // Tilt axis offset is in image space
                Transformed.Y += TiltAxisOffsetY[t];

                Transformed.X += ImageCenter.X;
                Transformed.Y += ImageCenter.Y;

                TransformedCoords[i] = new float3(Transformed.X / ImageDimensions[t].X, Transformed.Y / ImageDimensions[t].Y, t * GridStep);

                Normals[i] = Rotation * float3.UnitZ;
                Result[i] = Transformed;
            }

            float[] GridMovementXInterp = GridMovementX.GetInterpolatedNative(TransformedCoords);
            float[] GridMovementYInterp = GridMovementY.GetInterpolatedNative(TransformedCoords);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                // Additional stage shift determined for this tilt
                Result[i].X -= GridMovementXInterp[i];
                Result[i].Y -= GridMovementYInterp[i];

                // Figure out what height (= defocus) this is on the tilted plane
                // Coordinates are in Angstrom, can be converted directly in um
                Result[i].Z = GridDefocusInterp[t] -
                              (AreAnglesInverted ? -1e-4f : 1e-4f) *
                              float3.Dot(Normals[i], new float3(Result[i].X - ImageCenter.X, Result[i].Y - ImageCenter.Y, 0)) / Normals[i].Z;
            }

            return Result;
        }

        public float3[] GetPositionInAllTiltsNoLocalWarp(float3[] coords)
        {
            float3[] Result = new float3[coords.Length * NTilts];

            float3 VolumeCenter = VolumeDimensionsPhysical / 2;
            float2 ImageCenter = ImageDimensions[0] / 2;

            float GridStep = 1f / (NTilts - 1);
            float DoseStep = 1f / MaxDose;

            float3[] GridCoords = new float3[NTilts];
            float3[] TemporalGridCoords = new float3[NTilts];
            for (int t = 0; t < NTilts; t++)
            {
                GridCoords[t] = new float3(0.5f, 0.5f, t * GridStep);
                TemporalGridCoords[t] = new float3(0.5f, 0.5f, Dose[t] * GridStep);
            }

            float[] GridAngleXInterp = GridAngleX.GetInterpolatedNative(GridCoords);
            float[] GridAngleYInterp = GridAngleY.GetInterpolatedNative(GridCoords);
            float[] GridAngleZInterp = GridAngleZ.GetInterpolatedNative(GridCoords);

            float[] GridLocalXInterp = GridLocalX.GetInterpolatedNative(TemporalGridCoords);
            float[] GridLocalYInterp = GridLocalY.GetInterpolatedNative(TemporalGridCoords);
            float[] GridLocalZInterp = GridLocalZ.GetInterpolatedNative(TemporalGridCoords);
            float3[] SampleWarpings = Helper.ArrayOfFunction(t => new float3(GridLocalXInterp[t],
                                                                             GridLocalYInterp[t],
                                                                             GridLocalZInterp[t]), NTilts);

            float[] GridMovementXInterp = GridMovementX.GetInterpolatedNative(GridCoords);
            float[] GridMovementYInterp = GridMovementY.GetInterpolatedNative(GridCoords);

            float[] GridDefocusInterp = Helper.ArrayOfFunction(t => GetTiltDefocus(t), NTilts);

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, AnglesCorrect[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);
            Matrix3[] CorrectionMatrices = Helper.ArrayOfFunction(t => Matrix3.RotateZ(GridAngleZInterp[t] * Helper.ToRad) *
                                                                       Matrix3.RotateY(GridAngleYInterp[t] * Helper.ToRad) *
                                                                       Matrix3.RotateX(GridAngleXInterp[t] * Helper.ToRad), NTilts);

            Matrix3[] OverallRotations = Helper.ArrayOfFunction(t => CorrectionMatrices[t] * TiltMatrices[t], NTilts);
            float3[] OverallOffsets = Helper.ArrayOfFunction(t => new float3(TiltAxisOffsetX[t] + ImageCenter.X - GridMovementXInterp[t],
                                                                             TiltAxisOffsetY[t] + ImageCenter.Y - GridMovementYInterp[t],
                                                                             GridDefocusInterp[t] * 1e4f), NTilts);

            for (int i = 0; i < coords.Length; i++)
            {
                float3 Centered = coords[i] - VolumeCenter;

                for (int t = 0; t < NTilts; t++)
                {
                    float3 Transformed = OverallRotations[t] * (Centered + SampleWarpings[t]) + OverallOffsets[t];
                    Transformed.Z *= 1e-4f;

                    Result[i * NTilts + t] = Transformed;
                }
            }

            return Result;
        }

        public float3[] GetPositionsInOneTilt(float3[] coords, int tiltID)
        {
            float3[] Result = new float3[coords.Length];

            float3 VolumeCenter = VolumeDimensionsPhysical / 2;
            float2 ImageCenter = ImageDimensions[tiltID] / 2;

            float GridStep = 1f / (NTilts - 1);
            float DoseStep = 1f / MaxDose;

            for (int p = 0; p < coords.Length; p++)
            {
                float3 GridCoords = new float3(coords[p].X / VolumeDimensionsPhysical.X, coords[p].Y / VolumeDimensionsPhysical.Y, tiltID * GridStep);
                float3 Centered = coords[p] - VolumeCenter;

                Matrix3 TiltMatrix = Matrix3.Euler(0, AnglesCorrect[tiltID] * Helper.ToRad, -TiltAxisAngles[tiltID] * Helper.ToRad);
                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZ.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleY.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleX.GetInterpolated(GridCoords) * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrix;

                float3 TemporalGridCoords = new float3(GridCoords.X, GridCoords.Y, Dose[tiltID] * DoseStep);
                float3 SampleWarping = new float3(GridLocalX.GetInterpolated(TemporalGridCoords),
                                                  GridLocalY.GetInterpolated(TemporalGridCoords),
                                                  GridLocalZ.GetInterpolated(TemporalGridCoords));
                Centered += SampleWarping;

                float3 Transformed = (Rotation * Centered);

                Transformed.X += TiltAxisOffsetX[tiltID];   // Tilt axis offset is in image space
                Transformed.Y += TiltAxisOffsetY[tiltID];

                Transformed.X += ImageCenter.X;
                Transformed.Y += ImageCenter.Y;

                float3 TransformedCoords = new float3(Transformed.X / ImageDimensions[tiltID].X, Transformed.Y / ImageDimensions[tiltID].Y, tiltID * GridStep);

                // Additional stage shift determined for this tilt
                Transformed.X -= GridMovementX.GetInterpolated(TransformedCoords);
                Transformed.Y -= GridMovementY.GetInterpolated(TransformedCoords);

                // Figure out what height (= defocus) this is on the tilted plane
                // Coordinates are in Angstrom, can be converted directly in um
                float3 Normal = Rotation * float3.UnitZ;
                Transformed.Z = GridCTF.GetInterpolated(GridCoords) -
                                (AreAnglesInverted ? -1e-4f : 1e-4f) *
                                float3.Dot(Normal, new float3(Transformed.X - ImageCenter.X, Transformed.Y - ImageCenter.Y, 0)) / Normal.Z;

                Result[p] = Transformed;
            }

            return Result;
        }

        #endregion

        #region GetAngle methods

        public float3[] GetAngleInAllTilts(float3 coords)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetAngleInAllTilts(PerTiltCoords);
        }

        public float3[] GetAngleInAllTilts(float3[] coords)
        {
            float3[] Result = new float3[coords.Length];

            float GridStep = 1f / (NTilts - 1);

            float3[] GridCoords = new float3[coords.Length];
            float3[] TemporalGridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;
                GridCoords[i] = new float3(coords[i].X / VolumeDimensionsPhysical.X, coords[i].Y / VolumeDimensionsPhysical.X, t * GridStep);
            }

            float[] GridAngleXInterp = GridAngleX.GetInterpolatedNative(GridCoords);
            float[] GridAngleYInterp = GridAngleY.GetInterpolatedNative(GridCoords);
            float[] GridAngleZInterp = GridAngleZ.GetInterpolatedNative(GridCoords);

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, AnglesCorrect[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleYInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleXInterp[i] * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrices[t];

                Result[i] = Matrix3.EulerFromMatrix(Rotation);
            }

            return Result;
        }

        public float3[] GetParticleAngleInAllTilts(float3 coords, float3 angle)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            float3[] PerTiltAngles = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
            {
                PerTiltCoords[i] = coords;
                PerTiltAngles[i] = angle;
            }

            return GetParticleAngleInAllTilts(PerTiltCoords, PerTiltAngles);
        }

        public float3[] GetParticleAngleInAllTilts(float3[] coords, float3[] angle)
        {
            float3[] Result = new float3[coords.Length];

            float GridStep = 1f / (NTilts - 1);

            float3[] GridCoords = new float3[coords.Length];
            float3[] TemporalGridCoords = new float3[coords.Length];
            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;
                GridCoords[i] = new float3(coords[i].X / VolumeDimensionsPhysical.X, coords[i].Y / VolumeDimensionsPhysical.X, t * GridStep);
            }

            float[] GridAngleXInterp = GridAngleX.GetInterpolatedNative(GridCoords);
            float[] GridAngleYInterp = GridAngleY.GetInterpolatedNative(GridCoords);
            float[] GridAngleZInterp = GridAngleZ.GetInterpolatedNative(GridCoords);

            Matrix3[] TiltMatrices = Helper.ArrayOfFunction(t => Matrix3.Euler(0, AnglesCorrect[t] * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad), NTilts);

            for (int i = 0; i < coords.Length; i++)
            {
                int t = i % NTilts;

                Matrix3 ParticleMatrix = Matrix3.Euler(angle[i].X * Helper.ToRad,
                                                       angle[i].Y * Helper.ToRad,
                                                       angle[i].Z * Helper.ToRad);


                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleYInterp[i] * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleXInterp[i] * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrices[t] * ParticleMatrix;

                Result[i] = Matrix3.EulerFromMatrix(Rotation);
            }

            return Result;
        }

        public float3[] GetAnglesInOneTilt(float3[] coords, float3[] particleAngles, int tiltID)
        {
            int NParticles = coords.Length;
            float3[] Result = new float3[NParticles];

            float GridStep = 1f / (NTilts - 1);

            for (int p = 0; p < NParticles; p++)
            {
                float3 GridCoords = new float3(coords[p].X / VolumeDimensionsPhysical.X, coords[p].Y / VolumeDimensionsPhysical.Y, tiltID * GridStep);

                Matrix3 ParticleMatrix = Matrix3.Euler(particleAngles[p].X * Helper.ToRad,
                                                       particleAngles[p].Y * Helper.ToRad,
                                                       particleAngles[p].Z * Helper.ToRad);

                Matrix3 TiltMatrix = Matrix3.Euler(0, AnglesCorrect[tiltID] * Helper.ToRad, -TiltAxisAngles[tiltID] * Helper.ToRad);

                Matrix3 CorrectionMatrix = Matrix3.RotateZ(GridAngleZ.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateY(GridAngleY.GetInterpolated(GridCoords) * Helper.ToRad) *
                                           Matrix3.RotateX(GridAngleX.GetInterpolated(GridCoords) * Helper.ToRad);

                Matrix3 Rotation = CorrectionMatrix * TiltMatrix * ParticleMatrix;

                Result[p] = Matrix3.EulerFromMatrix(Rotation);
            }

            return Result;
        }

        #endregion

        #region GetImages methods
        
        public Image GetImagesForOneParticle(TomoProcessingOptionsBase options, Movie[] tiltMovies, Image[] tiltData, int size, float3 coords, int planForw = 0)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetImagesForOneParticle(options, tiltMovies, tiltData, size, PerTiltCoords);
        }
        
        public Image GetImagesForOneParticle(TomoProcessingOptionsBase options, Movie[] tiltMovies, Image[] tiltData, int size, float3[] coordsMoving, int planForw = 0)
        {
            float3[] ImagePositions = GetPositionInAllTilts(coordsMoving);
            for (int t = 0; t < NTilts; t++)
                ImagePositions[t] /= (float)options.BinnedPixelSizeMean;

            Image Result = new Image(new int3(size, size, NTilts));
            float[][] ResultData = Result.GetHost(Intent.Write);
            float3[] Shifts = new float3[NTilts];

            //Parallel.For(0, NTilts, t =>
            for (int t = 0; t < NTilts; t++)
            {
                int3 DimsMovie = tiltData[t].Dims;
                
                ImagePositions[t] -= size / 2;

                int2 IntPosition = new int2((int)ImagePositions[t].X, (int)ImagePositions[t].Y);
                float2 Residual = new float2(-(ImagePositions[t].X - IntPosition.X), -(ImagePositions[t].Y - IntPosition.Y));
                IntPosition.X += DimsMovie.X;                                                                                   // In case it is negative, for the periodic boundaries modulo later
                IntPosition.Y += DimsMovie.Y;
                Shifts[t] = new float3(Residual.X + size / 2, Residual.Y + size / 2, 0);                                        // Include an fftshift() for Fourier-space rotations later

                float[] OriginalData = tiltData[t].GetHost(Intent.Read)[0];
                float[] ImageData = ResultData[t];

                unsafe
                {
                    fixed (float* OriginalDataPtr = OriginalData)
                    fixed (float* ImageDataPtr = ImageData)
                    {
                        for (int y = 0; y < size; y++)
                        {
                            int PosY = (y + IntPosition.Y) % DimsMovie.Y;
                            for (int x = 0; x < size; x++)
                            {
                                int PosX = (x + IntPosition.X) % DimsMovie.X;
                                ImageDataPtr[y * size + x] = OriginalDataPtr[PosY * DimsMovie.X + PosX];
                            }
                        }
                    }
                }
            }//);

            Image ResultFT = Result.AsFFT(false, planForw);
            ResultFT.ShiftSlices(Shifts);

            Result.Dispose();

            return ResultFT;
        }

        #endregion

        #region GetCTFs methods

        public Image GetCTFsForOneParticle(TomoProcessingOptionsBase options, float3 coords, Image ctfCoords, bool weighted = true, bool weightsonly = false, bool useglobalweights = true)
        {
            float3[] PerTiltCoords = new float3[NTilts];
            for (int i = 0; i < NTilts; i++)
                PerTiltCoords[i] = coords;

            return GetCTFsForOneParticle(options, PerTiltCoords, ctfCoords, weighted, weightsonly, useglobalweights);
        }

        public Image GetCTFsForOneParticle(TomoProcessingOptionsBase options, float3[] coordsMoving, Image ctfCoords, bool weighted = true, bool weightsonly = false, bool useglobalweights = true)
        {
            float3[] ImagePositions = GetPositionInAllTilts(coordsMoving);

            float GridStep = 1f / (NTilts - 1);
            CTFStruct[] Params = new CTFStruct[NTilts];
            for (int t = 0; t < NTilts; t++)
            {
                decimal Defocus = (decimal)ImagePositions[t].Z;
                decimal DefocusDelta = (decimal)GetTiltDefocusDelta(t);
                decimal DefocusAngle = (decimal)GetTiltDefocusAngle(t);

                CTF CurrCTF = CTF.GetCopy();
                CurrCTF.PixelSize = options.BinnedPixelSizeMean;
                if (!weightsonly)
                {
                    CurrCTF.Defocus = Defocus;
                    CurrCTF.DefocusDelta = DefocusDelta;
                    CurrCTF.DefocusAngle = DefocusAngle;
                }
                else
                {
                    CurrCTF.Defocus = 0;
                    CurrCTF.DefocusDelta = 0;
                    CurrCTF.Cs = 0;
                    CurrCTF.Amplitude = 1;
                }
                
                if (weighted)
                {
                    if (GridAngleWeights.Dimensions.Elements() <= 1)
                        CurrCTF.Scale = (decimal)Math.Cos(AnglesCorrect[t] * Helper.ToRad);
                    else
                        CurrCTF.Scale = (decimal)GridAngleWeights.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));

                    if (GridDoseBfacs.Dimensions.Elements() <= 1)
                        CurrCTF.Bfactor = (decimal)-Dose[t] * 8;
                    else
                        CurrCTF.Bfactor = (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, t * GridStep));

                    if (useglobalweights)
                    {
                        CurrCTF.Scale *= (decimal)GlobalWeight;
                        CurrCTF.Bfactor += (decimal)GlobalBfactor;
                    }
                }

                Params[t] = CurrCTF.ToStruct();
            }

            Image Result = new Image(IntPtr.Zero, new int3(ctfCoords.Dims.X, ctfCoords.Dims.Y, NTilts), true);
            GPU.CreateCTF(Result.GetDevice(Intent.Write), ctfCoords.GetDevice(Intent.Read), (uint)Result.ElementsSliceReal, Params, false, (uint)NTilts);

            return Result;
        }

        #endregion

        #region Many-particles GetImages and GetCTFs

        public Image GetParticleImagesFromOneTilt(Image tiltStack, int size, float3[] particleOrigins, int angleID, bool normalize)
        {
            int NParticles = particleOrigins.Length;

            float3[] ImagePositions = GetPositionsInOneTilt(particleOrigins, angleID);

            Image Result = new Image(new int3(size, size, NParticles));
            float[][] ResultData = Result.GetHost(Intent.Write);
            float3[] Shifts = new float3[NParticles];

            int3 DimsStack = tiltStack.Dims;

            Parallel.For(0, NParticles, p =>
            {
                ImagePositions[p] -= new float3(size / 2, size / 2, 0);
                int2 IntPosition = new int2((int)ImagePositions[p].X, (int)ImagePositions[p].Y);
                float2 Residual = new float2(-(ImagePositions[p].X - IntPosition.X), -(ImagePositions[p].Y - IntPosition.Y));
                Residual -= size / 2;
                Shifts[p] = new float3(Residual);

                float[] OriginalData;
                lock (tiltStack)
                    OriginalData = tiltStack.GetHost(Intent.Read)[angleID];

                float[] ImageData = ResultData[p];
                for (int y = 0; y < size; y++)
                {
                    int PosY = (y + IntPosition.Y + DimsStack.Y) % DimsStack.Y;
                    for (int x = 0; x < size; x++)
                    {
                        int PosX = (x + IntPosition.X + DimsStack.X) % DimsStack.X;
                        ImageData[y * size + x] = OriginalData[PosY * DimsStack.X + PosX];
                    }
                }
            });
            if (normalize)
                GPU.NormParticles(Result.GetDevice(Intent.Read),
                                  Result.GetDevice(Intent.Write),
                                  Result.Dims.Slice(),
                                  (uint)(123 / CTF.PixelSize),     // FIX THE PARTICLE RADIUS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                  true,
                                  (uint)NParticles);
            //Result.WriteMRC($"d_paticleimages_{angleID:D3}.mrc");

            Result.ShiftSlices(Shifts);

            Image ResultFT = Result.AsFFT();
            Result.Dispose();

            return ResultFT;
        }

        public Image GetParticleSeriesFromMovies(Movie[] movies, Image[] movieData, int size, float3[] particleOrigins, float pixelSize, int planForw = 0)
        {
            int NParticles = particleOrigins.Length;
            Image Result = new Image(new int3(size, size, NParticles * NTilts), true, true);
            float[][] ResultData = Result.GetHost(Intent.Write);

            int PlanForw = planForw > 0 ? planForw : GPU.CreateFFTPlan(new int3(size, size, 1), (uint)NParticles);

            Image ParticleExtracts = new Image(IntPtr.Zero, new int3(size, size, NParticles));
            Image ParticleExtractsFT = new Image(IntPtr.Zero, new int3(size, size, NParticles), true, true);

            for (int t = 0; t < NTilts; t++)
            {
                float3 Scaling = new float3(1 / pixelSize, 1 / pixelSize, 1);
                float3 DimsInv = new float3(1f / movieData[t].Dims.X,
                                            1f / movieData[t].Dims.Y,
                                            1f / Math.Max(1, movieData[t].Dims.Z));

                float3[] ImagePositions = GetPositionsInOneTilt(particleOrigins, t);
                for (int p = 0; p < NParticles; p++)
                    ImagePositions[p] *= Scaling;       // Tilt image positions are returned in Angstroms initially

                float3[] MovieGridPositions = new float3[NParticles];
                for (int p = 0; p < NParticles; p++)
                    MovieGridPositions[p] = new float3(ImagePositions[p].X * DimsInv.X,
                                                       ImagePositions[p].Y * DimsInv.Y,
                                                       0);

                int3[] ExtractPositions = new int3[NParticles];
                float3[] ResidualShifts = new float3[NParticles];

                Image ParticleSumsFT = new Image(IntPtr.Zero, new int3(size, size, NParticles), true, true);
                ParticleSumsFT.Fill(0);

                for (int z = 0; z < movieData[t].Dims.Z; z++)
                {
                    for (int p = 0; p < NParticles; p++)
                        MovieGridPositions[p].Z = z * DimsInv.Z;

                    float2[] FrameShifts = movies[t].GetShiftFromPyramid(MovieGridPositions);
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3 Shifted = new float3(ImagePositions[p].X - FrameShifts[p].X / pixelSize,     // Don't forget, shifts are stored in Angstroms
                                                    ImagePositions[p].Y - FrameShifts[p].Y / pixelSize,
                                                    0);
                        ExtractPositions[p] = new int3(Shifted);
                        ResidualShifts[p] = new float3(ExtractPositions[p].X - Shifted.X + size / 2,
                                                       ExtractPositions[p].Y - Shifted.Y + size / 2,
                                                       0);

                        GPU.Extract(movieData[t].GetDeviceSlice(z, Intent.Read),
                                    ParticleExtracts.GetDevice(Intent.Write),
                                    movieData[t].Dims.Slice(),
                                    new int3(size, size, 1),
                                    Helper.ToInterleaved(ExtractPositions),
                                    (uint)NParticles);

                        GPU.FFT(ParticleExtracts.GetDevice(Intent.Read),
                                ParticleExtractsFT.GetDevice(Intent.Write),
                                ParticleExtracts.Dims.Slice(),
                                (uint)NParticles,
                                PlanForw);

                        ParticleExtractsFT.ShiftSlices(ResidualShifts);

                        ParticleSumsFT.Add(ParticleExtracts);
                    }
                }
                
                ParticleSumsFT.Multiply(1f / size / size / movieData[t].Dims.Z);

                float[][] ParticleSumsFTData = ParticleSumsFT.GetHost(Intent.Read);
                for (int p = 0; p < NParticles; p++)
                    ResultData[p * NTilts + t] = ParticleSumsFTData[p];

                ParticleSumsFT.Dispose();
                movieData[t].FreeDevice();
            }

            ParticleExtracts.Dispose();
            ParticleExtractsFT.Dispose();
            if (planForw <= 0)
                GPU.DestroyFFTPlan(PlanForw);

            return Result;
        }

        public Image Get2DCTFsOneTilt(Image tiltStack, Image ctfCoords, float3[] particleOrigins, int tiltID, bool weighted = true, bool weightsonly = false, bool useglobalweights = true)
        {
            int NParticles = particleOrigins.Length;
            float3[] ImagePositions = GetPositionsInOneTilt(particleOrigins, tiltID);

            float GridStep = 1f / (NTilts - 1);
            CTFStruct[] Params = new CTFStruct[NParticles];
            for (int p = 0; p < NParticles; p++)
            {
                decimal Defocus = (decimal)ImagePositions[p].Z;
                decimal DefocusDelta = (decimal)GridCTFDefocusDelta.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));
                decimal DefocusAngle = (decimal)GridCTFDefocusAngle.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));

                CTF CurrCTF = CTF.GetCopy();
                if (!weightsonly)
                {
                    CurrCTF.Defocus = Defocus;
                    CurrCTF.DefocusDelta = DefocusDelta;
                    CurrCTF.DefocusAngle = DefocusAngle;
                }
                else
                {
                    CurrCTF.Defocus = 0;
                    CurrCTF.DefocusDelta = 0;
                    CurrCTF.Cs = 0;
                    CurrCTF.Amplitude = 1;
                }
                //CurrCTF.PixelSize *= 4;

                if (weighted)
                {
                    if (GridAngleWeights.Dimensions.Elements() <= 1)
                        CurrCTF.Scale = (decimal)Math.Cos(AnglesCorrect[tiltID] * Helper.ToRad);
                    else
                        CurrCTF.Scale = (decimal)GridAngleWeights.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));

                    if (GridDoseBfacs.Dimensions.Elements() <= 1)
                        CurrCTF.Bfactor = (decimal)-Dose[tiltID] * 8;
                    else
                        CurrCTF.Bfactor = (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, tiltID * GridStep));

                    if (useglobalweights)
                    {
                        CurrCTF.Scale *= (decimal)GlobalWeight;
                        CurrCTF.Bfactor += (decimal)GlobalBfactor;
                    }
                }

                Params[p] = CurrCTF.ToStruct();
            }

            Image Result = new Image(IntPtr.Zero, new int3(ctfCoords.Dims.X, ctfCoords.Dims.Y, NParticles), true);
            GPU.CreateCTF(Result.GetDevice(Intent.Write), ctfCoords.GetDevice(Intent.Read), (uint)Result.ElementsSliceReal, Params, false, (uint)NParticles);

            return Result;
        }

        #endregion

        public void GetSubtomoForOneParticle(TomoProcessingOptionsBase options, Movie[] tiltMovies, Image[] tiltData, float3 coords, float3 angles, Image ctfCoords, out Image subtomo, out Image subtomoCTF, int planForw = 0, int planBack = 0, int planForwCTF = 0, int planForwImages = 0)
        {
            int Size = ctfCoords.Dims.X;
            float3[] ImageAngles = GetAngleInAllTilts(coords);

            Image ImagesFT = null;//GetSubtomoImages(tiltStack, Size * downsample, coords, true);
            Image ImagesFTCropped = ImagesFT.AsPadded(new int2(Size, Size));
            ImagesFT.Dispose();

            Image CTFs = GetCTFsForOneParticle(options, coords, ctfCoords, true, false, false);
            //Image CTFWeights = GetSubtomoCTFs(coords, ctfCoords, true, true);

            ImagesFTCropped.Multiply(CTFs);    // Weight and phase-flip image FTs by CTF, which still has its sign here
            //ImagesFT.Multiply(CTFWeights);
            CTFs.Abs();                 // CTF has to be positive from here on since image FT phases are now flipped

            // CTF has to be converted to complex numbers with imag = 0, and weighted by itself
            float2[] CTFsComplexData = new float2[CTFs.ElementsComplex];
            float[] CTFsContinuousData = CTFs.GetHostContinuousCopy();
            for (int i = 0; i < CTFsComplexData.Length; i++)
                CTFsComplexData[i] = new float2(CTFsContinuousData[i] * CTFsContinuousData[i], 0);

            Image CTFsComplex = new Image(CTFsComplexData, CTFs.Dims, true);
            
            Projector ProjSubtomo = new Projector(new int3(Size, Size, Size), 2);
            lock (GPU.Sync)
                ProjSubtomo.BackProject(ImagesFTCropped, CTFs, ImageAngles);
            subtomo = ProjSubtomo.Reconstruct(false, planForw, planBack, planForwCTF);
            ProjSubtomo.Dispose();
            
            GPU.NormParticles(subtomo.GetDevice(Intent.Read),
                              subtomo.GetDevice(Intent.Write),
                              subtomo.Dims,
                              (uint)(123 / CTF.PixelSize),     // FIX THE PARTICLE RADIUS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                              false,
                              1);
            //subtomo = new Image(new int3(1, 1, 1));

            Projector ProjCTF = new Projector(new int3(Size, Size, Size), 2);
            lock (GPU.Sync)
                ProjCTF.BackProject(CTFsComplex, CTFs, ImageAngles);
            subtomoCTF = ProjCTF.Reconstruct(true, planForw, planBack, planForwCTF);
            ProjCTF.Dispose();
            //subtomoCTF = new Image(new int3(1, 1, 1));

            ImagesFTCropped.Dispose();
            CTFs.Dispose();
            //CTFWeights.Dispose();
            CTFsComplex.Dispose();
        }

        public void LoadMovieData(ProcessingOptionsBase options, bool doShift, bool flattenIfNoLocal, bool forceFlatten, out Movie[] movies, out Image[] movieData)
        {
            if (TiltMoviePaths.Length != NTilts)
                throw new Exception("A valid path is needed for each tilt.");

            movies = new Movie[NTilts];
            movieData = new Image[NTilts];

            Image Gain = null;
            if (!string.IsNullOrEmpty(options.GainPath))
                Gain = Image.FromFile(options.GainPath);

            for (int t = 0; t < NTilts; t++)
                movies[t] = new Movie(DirectoryName + TiltMoviePaths[t]);

            MapHeader Header = MapHeader.ReadFromFile(DirectoryName + TiltMoviePaths[0]);

            int2 DimsScaled = new int2((int)Math.Round(Header.Dimensions.X / (float)options.DownsampleFactor / 2) * 2,
                                        (int)Math.Round(Header.Dimensions.Y / (float)options.DownsampleFactor / 2) * 2);

            bool DoScale = DimsScaled != new int2(Header.Dimensions);
            bool DoShift = movies.All(m => m.HasGlobalMovement && !m.HasLocalMovement) && doShift;
            bool DoFlatten = forceFlatten || (movies.All(m => !m.HasLocalMovement) && flattenIfNoLocal);

            int PlanForw = 0, PlanBack = 0;
            if (DoScale || DoShift)
            {
                PlanForw = GPU.CreateFFTPlan(Header.Dimensions.Slice(), 1);
                PlanBack = GPU.CreateIFFTPlan(new int3(DimsScaled), 1);
            }

            ImageDimensions = new float2[NTilts];

            for (int t = 0; t < NTilts; t++)
            {
                Image RawData = Image.FromFile(DirectoryName + TiltMoviePaths[t]);
                int NFrames = RawData.Dims.Z;
                ImageDimensions[t] = new float2(RawData.Dims.X, RawData.Dims.Y) * (float)options.PixelSizeMean;

                if (Gain != null)
                    RawData.MultiplySlices(Gain);
                
                float3[] Shifts = new float3[NFrames];
                for (int z = 0; z < NFrames; z++)
                    Shifts[z] = new float3(movies[t].GetShiftFromPyramid(new float3(0.5f, 0.5f, z / (float)(NFrames - 1)))) / (float)options.BinnedPixelSizeMean;

                if (options.PixelSizeDelta != 0)
                    GPU.CorrectMagAnisotropy(RawData.GetDevice(Intent.Read),
                                             new int2(RawData.Dims),
                                             RawData.GetDevice(Intent.Write),
                                             new int2(RawData.Dims),
                                             (float)options.PixelSizeX,
                                             (float)options.PixelSizeY,
                                             (float)options.PixelSizeAngle * Helper.ToRad,
                                             1,
                                             (uint)NFrames);

                if (DoScale || DoShift)
                {
                    Image RawDataFT = new Image(IntPtr.Zero, RawData.Dims, true, true);

                    for (int z = 0; z < NFrames; z++)
                        GPU.FFT(RawData.GetDeviceSlice(z, Intent.Read),
                                RawDataFT.GetDeviceSlice(z, Intent.Write),
                                RawData.Dims.Slice(),
                                1,
                                PlanForw);
                    RawDataFT.Multiply(1f / RawDataFT.Dims.Slice().Elements()); // Normalize FT here to avoid branching later

                    if (DoScale)
                    {
                        Image ScaledFT = RawDataFT.AsPadded(DimsScaled);
                        RawDataFT.Dispose();
                        RawDataFT = ScaledFT;

                        RawData.Dispose();
                        RawData = new Image(IntPtr.Zero, RawDataFT.Dims);
                    }

                    if (DoShift && NFrames > 1)
                    {
                        RawDataFT.ShiftSlices(Shifts);
                    }

                    for (int z = 0; z < NFrames; z++)
                        GPU.IFFT(RawDataFT.GetDeviceSlice(z, Intent.Read),
                                 RawData.GetDeviceSlice(z, Intent.Write),
                                 RawDataFT.Dims.Slice(),
                                 1,
                                 PlanBack,
                                 false);
                    RawDataFT.Dispose();
                }

                if (DoFlatten)
                {
                    Image Flattened = RawData.AsReducedAlongZ();
                    RawData.Dispose();
                    RawData = Flattened;
                }

                RawData.FreeDevice();
                movieData[t] = RawData;
            }

            if (DoScale || DoShift)
            {
                GPU.DestroyFFTPlan(PlanForw);
                GPU.DestroyFFTPlan(PlanBack);
            }
        }

        public void LoadMovieSizes(ProcessingOptionsBase options)
        {
            if (TiltMoviePaths.Length != NTilts)
                throw new Exception("A valid path is needed for each tilt.");

            ImageDimensions = new float2[NTilts];

            for (int t = 0; t < NTilts; t++)
            {
                MapHeader Header = MapHeader.ReadFromFile(DirectoryName + TiltMoviePaths[t]);
                ImageDimensions[t] = new float2(Header.Dimensions.X, Header.Dimensions.Y) * (float)options.PixelSizeMean;
            }
        }

        public void LoadMovieMasks(ProcessingOptionsBase options, float extendEdge, float softEdge, out Image[] maskData)
        {
            MapHeader Header = MapHeader.ReadFromFile(DirectoryName + TiltMoviePaths[0]);
            
            int2 DimsScaled = new int2((int)Math.Round(Header.Dimensions.X / (float)options.DownsampleFactor / 2) * 2,
                                       (int)Math.Round(Header.Dimensions.Y / (float)options.DownsampleFactor / 2) * 2);

            maskData = new Image[NTilts];

            for (int t = 0; t < NTilts; t++)
            {
                Movie M = new Movie(DirectoryName + TiltMoviePaths[t]);
                string MaskPath = M.MaskPath;

                if (File.Exists(MaskPath))
                {
                    Image MaskOri = Image.FromFile(MaskPath);

                    #region Rescale

                    CubicGrid MaskGrid = new CubicGrid(MaskOri.Dims, MaskOri.GetHostContinuousCopy());
                    float3[] ScaledCoords = new float3[DimsScaled.Elements()];
                    for (int y = 0; y < DimsScaled.Y; y++)
                        for (int x = 0; x < DimsScaled.X; x++)
                            ScaledCoords[y * DimsScaled.X + x] = new float3(x / (DimsScaled.X - 1f),
                                                                            y / (DimsScaled.Y - 1f),
                                                                            0);
                    float[] ScaledValues = MaskGrid.GetInterpolated(ScaledCoords);

                    Image MaskScaled = new Image(ScaledValues, new int3(DimsScaled));
                    MaskScaled.Binarize(0.5f);

                    #endregion

                    Image BinaryExpanded;
                    if (extendEdge > 0)
                    {
                        BinaryExpanded = MaskScaled.AsDistanceMapExact((int)(extendEdge + 0.5f));
                        BinaryExpanded.Multiply(-1);
                        BinaryExpanded.Binarize(-extendEdge + 1e-6f);
                    }
                    else
                    {
                        BinaryExpanded = MaskScaled.GetCopyGPU();
                    }
                    MaskScaled.Dispose();

                    Image ExpandedSmooth;
                    if (softEdge > 0)
                    {
                        ExpandedSmooth = BinaryExpanded.AsDistanceMapExact((int)(softEdge + 0.5f));
                        ExpandedSmooth.Multiply((float)Math.PI / softEdge);
                        ExpandedSmooth.Cos();
                        ExpandedSmooth.Add(1);
                        ExpandedSmooth.Multiply(0.5f);
                    }
                    else
                    {
                        ExpandedSmooth = BinaryExpanded.GetCopyGPU();
                    }
                    BinaryExpanded.Dispose();

                    ExpandedSmooth.Add(-1);
                    ExpandedSmooth.Multiply(-1);

                    ExpandedSmooth.FreeDevice();
                    maskData[t] = ExpandedSmooth;
                }
                else
                {
                    maskData[t] = new Image(Helper.ArrayOfConstant(1f, (int)DimsScaled.Elements()), new int3(DimsScaled));
                }
            }
        }

        #endregion

        public override void LoadMeta()
        {
            if (!File.Exists(XMLPath))
                return;

            using (Stream SettingsStream = File.OpenRead(XMLPath))
            {
                XPathDocument Doc = new XPathDocument(SettingsStream);
                XPathNavigator Reader = Doc.CreateNavigator();
                Reader.MoveToRoot();
                Reader.MoveToFirstChild();

                #region Attributes

                AreAnglesInverted = XMLHelper.LoadAttribute(Reader, "AreAnglesInverted", AreAnglesInverted);
                GlobalWeight = XMLHelper.LoadAttribute(Reader, "Weight", GlobalWeight);
                GlobalBfactor = XMLHelper.LoadAttribute(Reader, "Bfactor", GlobalBfactor);
                PlaneNormal = XMLHelper.LoadAttribute(Reader, "PlaneNormal", PlaneNormal);

                //_UnselectFilter = XMLHelper.LoadAttribute(Reader, "UnselectFilter", _UnselectFilter);
                string UnselectManualString = XMLHelper.LoadAttribute(Reader, "UnselectManual", "null");
                if (UnselectManualString != "null")
                    _UnselectManual = bool.Parse(UnselectManualString);
                else
                    _UnselectManual = null;
                CTFResolutionEstimate = XMLHelper.LoadAttribute(Reader, "CTFResolutionEstimate", CTFResolutionEstimate);

                #endregion

                #region Per-tilt propertries

                {
                    XPathNavigator Nav = Reader.SelectSingleNode("//Angles");
                    if (Nav != null)
                        Angles = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                }

                {
                    XPathNavigator Nav = Reader.SelectSingleNode("//Dose");
                    if (Nav != null)
                        Dose = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    else
                        Dose = new float[Angles.Length];
                }

                {
                    XPathNavigator Nav = Reader.SelectSingleNode("//UseTilt");
                    if (Nav != null)
                        UseTilt = Nav.InnerXml.Split('\n').Select(v => bool.Parse(v)).ToArray();
                    else
                        UseTilt = Helper.ArrayOfConstant(true, Angles.Length);
                }

                {
                    XPathNavigator Nav = Reader.SelectSingleNode("//AxisAngle");
                    if (Nav != null)
                        TiltAxisAngles = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    else
                        TiltAxisAngles = new float[Angles.Length];
                }

                {
                    XPathNavigator Nav = Reader.SelectSingleNode("//AxisOffsetX");
                    if (Nav != null)
                        TiltAxisOffsetX = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    else
                        TiltAxisOffsetX = new float[Angles.Length];
                }

                {
                    XPathNavigator Nav = Reader.SelectSingleNode("//AxisOffsetY");
                    if (Nav != null)
                        TiltAxisOffsetY = Nav.InnerXml.Split('\n').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                    else
                        TiltAxisOffsetY = new float[Angles.Length];
                }

                {
                    XPathNavigator Nav = Reader.SelectSingleNode("//MoviePath");
                    if (Nav != null)
                        TiltMoviePaths = Nav.InnerXml.Split('\n').Select(v => v.Replace(" ", "").Replace("\r", "").Replace("\t", "")).ToArray();
                    else
                        TiltMoviePaths = new[] { "" };
                }

                #endregion

                #region CTF fitting-related

                {
                    TiltPS1D.Clear();
                    List<Tuple<int, float2[]>> TempPS1D = (from XPathNavigator NavPS1D in Reader.Select("//TiltPS1D")
                                                           let ID = int.Parse(NavPS1D.GetAttribute("ID", ""))
                                                           let NewPS1D = NavPS1D.InnerXml.Split(';').Select(v =>
                                                           {
                                                               string[] Pair = v.Split('|');
                                                               return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                                                           }).ToArray()
                                                           select new Tuple<int, float2[]>(ID, NewPS1D)).ToList();

                    TempPS1D.Sort((a, b) => a.Item1.CompareTo(b.Item1));
                    foreach (var ps1d in TempPS1D)
                        TiltPS1D.Add(ps1d.Item2);
                }

                {
                    TiltSimulatedScale.Clear();
                    List<Tuple<int, Cubic1D>> TempScale = (from XPathNavigator NavSimScale in Reader.Select("//TiltSimulatedScale")
                                                           let ID = int.Parse(NavSimScale.GetAttribute("ID", ""))
                                                           let NewScale = new Cubic1D(NavSimScale.InnerXml.Split(';').Select(v =>
                                                           {
                                                               string[] Pair = v.Split('|');
                                                               return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                                                           }).ToArray())
                                                           select new Tuple<int, Cubic1D>(ID, NewScale)).ToList();

                    TempScale.Sort((a, b) => a.Item1.CompareTo(b.Item1));
                    foreach (var scale in TempScale)
                        TiltSimulatedScale.Add(scale.Item2);
                }

                {
                    XPathNavigator NavPS1D = Reader.SelectSingleNode("//PS1D");
                    if (NavPS1D != null)
                        PS1D = NavPS1D.InnerXml.Split(';').Select(v =>
                        {
                            string[] Pair = v.Split('|');
                            return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                        }).ToArray();
                }

                {
                    XPathNavigator NavSimScale = Reader.SelectSingleNode("//SimulatedScale");
                    if (NavSimScale != null)
                        SimulatedScale = new Cubic1D(NavSimScale.InnerXml.Split(';').Select(v =>
                        {
                            string[] Pair = v.Split('|');
                            return new float2(float.Parse(Pair[0], CultureInfo.InvariantCulture), float.Parse(Pair[1], CultureInfo.InvariantCulture));
                        }).ToArray());
                }

                XPathNavigator NavCTF = Reader.SelectSingleNode("//CTF");
                if (NavCTF != null)
                    CTF.ReadFromXML(NavCTF);

                XPathNavigator NavOptionsCTF = Reader.SelectSingleNode("//OptionsCTF");
                if (NavOptionsCTF != null)
                {
                    ProcessingOptionsMovieCTF Temp = new ProcessingOptionsMovieCTF();
                    Temp.ReadFromXML(NavOptionsCTF);
                    OptionsCTF = Temp;
                }

                #endregion

                #region Grids

                XPathNavigator NavGridCTF = Reader.SelectSingleNode("//GridCTF");
                if (NavGridCTF != null)
                    GridCTF = CubicGrid.Load(NavGridCTF);

                XPathNavigator NavGridCTFDefocusDelta = Reader.SelectSingleNode("//GridCTFDefocusDelta");
                if (NavGridCTFDefocusDelta != null)
                    GridCTFDefocusDelta = CubicGrid.Load(NavGridCTFDefocusDelta);

                XPathNavigator NavGridCTFDefocusAngle = Reader.SelectSingleNode("//GridCTFDefocusAngle");
                if (NavGridCTFDefocusAngle != null)
                    GridCTFDefocusAngle = CubicGrid.Load(NavGridCTFDefocusAngle);

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

                XPathNavigator NavLocalZ = Reader.SelectSingleNode("//GridLocalMovementZ");
                if (NavLocalZ != null)
                    GridLocalZ = CubicGrid.Load(NavLocalZ);

                XPathNavigator NavAngleX = Reader.SelectSingleNode("//GridAngleX");
                if (NavAngleX != null)
                    GridAngleX = CubicGrid.Load(NavAngleX);

                XPathNavigator NavAngleY = Reader.SelectSingleNode("//GridAngleY");
                if (NavAngleY != null)
                    GridAngleY = CubicGrid.Load(NavAngleY);

                XPathNavigator NavAngleZ = Reader.SelectSingleNode("//GridAngleZ");
                if (NavAngleZ != null)
                    GridAngleZ = CubicGrid.Load(NavAngleZ);

                XPathNavigator NavAngleWeights = Reader.SelectSingleNode("//GridAngleWeights");
                if (NavAngleWeights != null)
                    GridAngleWeights = CubicGrid.Load(NavAngleWeights);

                XPathNavigator NavDoseBfacs = Reader.SelectSingleNode("//GridDoseBfacs");
                if (NavDoseBfacs != null)
                    GridDoseBfacs = CubicGrid.Load(NavDoseBfacs);

                #endregion
            }
        }

        public override void SaveMeta()
        {
            using (XmlTextWriter Writer = new XmlTextWriter(XMLPath, Encoding.Unicode))
            {
                Writer.Formatting = Formatting.Indented;
                Writer.IndentChar = '\t';
                Writer.Indentation = 1;
                Writer.WriteStartDocument();
                Writer.WriteStartElement("TiltSeries");

                #region Attributes

                Writer.WriteAttributeString("AreAnglesInverted", AreAnglesInverted.ToString());
                Writer.WriteAttributeString("Weight", GlobalWeight.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("Bfactor", GlobalBfactor.ToString(CultureInfo.InvariantCulture));
                Writer.WriteAttributeString("PlaneNormal", PlaneNormal.ToString());

                Writer.WriteAttributeString("UnselectFilter", UnselectFilter.ToString());
                Writer.WriteAttributeString("UnselectManual", UnselectManual.ToString());
                Writer.WriteAttributeString("CTFResolutionEstimate", CTFResolutionEstimate.ToString(CultureInfo.InvariantCulture));

                #endregion

                #region Per-tilt propertries

                Writer.WriteStartElement("Angles");
                Writer.WriteString(string.Join("\n", Angles.Select(v => v.ToString(CultureInfo.InvariantCulture))));
                Writer.WriteEndElement();

                Writer.WriteStartElement("Dose");
                Writer.WriteString(string.Join("\n", Dose.Select(v => v.ToString(CultureInfo.InvariantCulture))));
                Writer.WriteEndElement();

                Writer.WriteStartElement("UseTilt");
                Writer.WriteString(string.Join("\n", UseTilt.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("AxisAngle");
                Writer.WriteString(string.Join("\n", TiltAxisAngles.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("AxisOffsetX");
                Writer.WriteString(string.Join("\n", TiltAxisOffsetX.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("AxisOffsetY");
                Writer.WriteString(string.Join("\n", TiltAxisOffsetY.Select(v => v.ToString())));
                Writer.WriteEndElement();

                Writer.WriteStartElement("MoviePath");
                Writer.WriteString(string.Join("\n", TiltMoviePaths.Select(v => v.ToString())));
                Writer.WriteEndElement();

                #endregion

                #region CTF fitting-related

                foreach (float2[] ps1d in TiltPS1D)
                {
                    Writer.WriteStartElement("TiltPS1D");
                    XMLHelper.WriteAttribute(Writer, "ID", TiltPS1D.IndexOf(ps1d));
                    Writer.WriteString(string.Join(";", ps1d.Select(v => v.X.ToString(CultureInfo.InvariantCulture) + "|" + v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                foreach (Cubic1D simulatedScale in TiltSimulatedScale)
                {
                    Writer.WriteStartElement("TiltSimulatedScale");
                    XMLHelper.WriteAttribute(Writer, "ID", TiltSimulatedScale.IndexOf(simulatedScale));
                    Writer.WriteString(string.Join(";",
                                                   simulatedScale.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                   "|" +
                                                                                   v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (PS1D != null)
                {
                    Writer.WriteStartElement("PS1D");
                    Writer.WriteString(string.Join(";", PS1D.Select(v => v.X.ToString(CultureInfo.InvariantCulture) + "|" + v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (SimulatedScale != null)
                {
                    Writer.WriteStartElement("SimulatedScale");
                    Writer.WriteString(string.Join(";",
                                                   SimulatedScale.Data.Select(v => v.X.ToString(CultureInfo.InvariantCulture) +
                                                                                    "|" +
                                                                                    v.Y.ToString(CultureInfo.InvariantCulture))));
                    Writer.WriteEndElement();
                }

                if (OptionsCTF != null)
                {
                    Writer.WriteStartElement("OptionsCTF");
                    OptionsCTF.WriteToXML(Writer);
                    Writer.WriteEndElement();
                }

                Writer.WriteStartElement("CTF");
                CTF.WriteToXML(Writer);
                Writer.WriteEndElement();

                #endregion

                #region Grids

                Writer.WriteStartElement("GridCTF");
                GridCTF.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFDefocusDelta");
                GridCTFDefocusDelta.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFDefocusAngle");
                GridCTFDefocusAngle.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridCTFPhase");
                GridCTFPhase.Save(Writer);
                Writer.WriteEndElement();

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

                Writer.WriteStartElement("GridLocalMovementZ");
                GridLocalZ.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridAngleX");
                GridAngleX.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridAngleY");
                GridAngleY.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridAngleZ");
                GridAngleZ.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridAngleWeights");
                GridAngleWeights.Save(Writer);
                Writer.WriteEndElement();

                Writer.WriteStartElement("GridDoseBfacs");
                GridDoseBfacs.Save(Writer);
                Writer.WriteEndElement();

                #endregion

                Writer.WriteEndElement();
                Writer.WriteEndDocument();
            }
        }

        public override string GetDataHash()
        {
            FileInfo Info = new FileInfo(TiltMoviePaths[0]);
            byte[] DataBytes = new byte[Math.Min(1 << 19, Info.Length)];
            using (BinaryReader Reader = new BinaryReader(File.OpenRead(TiltMoviePaths[0])))
            {
                Reader.Read(DataBytes, 0, DataBytes.Length);
            }

            DataBytes = Helper.Combine(Helper.ToBytes(RootName.ToCharArray()), DataBytes);

            return MathHelper.GetSHA1(DataBytes);
        }

        public override string GetProcessingHash()
        {
            List<byte[]> Arrays = new List<byte[]>();

            if (CTF != null)
                Arrays.Add(Helper.ToBytes(new[]
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

            #region Grids

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

            if (GridLocalZ != null)
            {
                Arrays.Add(GridLocalZ.Dimensions);
                Arrays.Add(Helper.ToBytes(GridLocalZ.FlatValues));
            }

            if (GridAngleX != null)
            {
                Arrays.Add(GridAngleX.Dimensions);
                Arrays.Add(Helper.ToBytes(GridAngleX.FlatValues));
            }

            if (GridAngleY != null)
            {
                Arrays.Add(GridAngleY.Dimensions);
                Arrays.Add(Helper.ToBytes(GridAngleY.FlatValues));
            }

            if (GridAngleZ != null)
            {
                Arrays.Add(GridAngleZ.Dimensions);
                Arrays.Add(Helper.ToBytes(GridAngleZ.FlatValues));
            }

            if (GridCTFDefocusAngle != null)
            {
                Arrays.Add(GridCTFDefocusAngle.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTFDefocusAngle.FlatValues));
            }

            if (GridCTFDefocusDelta != null)
            {
                Arrays.Add(GridCTFDefocusDelta.Dimensions);
                Arrays.Add(Helper.ToBytes(GridCTFDefocusDelta.FlatValues));
            }

            if (GridAngleWeights != null)
            {
                Arrays.Add(GridAngleWeights.Dimensions);
                Arrays.Add(Helper.ToBytes(GridAngleWeights.FlatValues));
            }

            if (GridDoseBfacs != null)
            {
                Arrays.Add(GridDoseBfacs.Dimensions);
                Arrays.Add(Helper.ToBytes(GridDoseBfacs.FlatValues));
            }

            #endregion

            Arrays.Add(Helper.ToBytes(TiltAxisAngles));
            Arrays.Add(Helper.ToBytes(TiltAxisOffsetX));
            Arrays.Add(Helper.ToBytes(TiltAxisOffsetX));
            Arrays.Add(Helper.ToBytes(Angles));
            Arrays.Add(Helper.ToBytes(Dose));
            Arrays.Add(Helper.ToBytes(UseTilt));

            foreach (var moviePath in TiltMoviePaths)
            {
                Movie TiltMovie = new Movie(DirectoryName + moviePath);
                Arrays.Add(Helper.ToBytes(TiltMovie.GetProcessingHash().ToCharArray()));
            }

            byte[] ArraysCombined = Helper.Combine(Arrays);
            return MathHelper.GetSHA1(ArraysCombined);
        }
    }

    public abstract class TomoProcessingOptionsBase : ProcessingOptionsBase
    {
        public float3 DimensionsPhysical => Dimensions * (float)PixelSizeMean;

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((TomoProcessingOptionsBase)obj);
        }

        protected bool Equals(TomoProcessingOptionsBase other)
        {
            return base.Equals(other) &&
                   Dimensions == other.Dimensions;
        }

        public static bool operator ==(TomoProcessingOptionsBase left, TomoProcessingOptionsBase right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(TomoProcessingOptionsBase left, TomoProcessingOptionsBase right)
        {
            return !Equals(left, right);
        }
    }

    public class ProcessingOptionsTomoFullReconstruction : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public bool Invert { get; set; }

        [WarpSerializable]
        public bool Normalize { get; set; }

        [WarpSerializable]
        public bool DoDeconv { get; set; }

        [WarpSerializable]
        public decimal DeconvStrength { get; set; }

        [WarpSerializable]
        public decimal DeconvFalloff { get; set; }

        [WarpSerializable]
        public decimal DeconvHighpass { get; set; }

        [WarpSerializable]
        public int SubVolumeSize { get; set; }

        [WarpSerializable]
        public decimal SubVolumePadding { get; set; }

        [WarpSerializable]
        public bool KeepOnlyFullVoxels { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsTomoFullReconstruction)obj);
        }

        protected bool Equals(ProcessingOptionsTomoFullReconstruction other)
        {
            return base.Equals(other) &&
                   Invert == other.Invert &&
                   Normalize == other.Normalize &&
                   DoDeconv == other.DoDeconv &&
                   SubVolumeSize == other.SubVolumeSize &&
                   SubVolumePadding == other.SubVolumePadding;
        }

        public static bool operator ==(ProcessingOptionsTomoFullReconstruction left, ProcessingOptionsTomoFullReconstruction right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsTomoFullReconstruction left, ProcessingOptionsTomoFullReconstruction right)
        {
            return !Equals(left, right);
        }
    }

    public class ProcessingOptionsTomoFullMatch : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public bool OverwriteFiles { get; set; }

        [WarpSerializable]
        public int SubVolumeSize { get; set; }

        [WarpSerializable]
        public string TemplateName { get; set; }

        [WarpSerializable]
        public decimal TemplatePixel { get; set; }

        [WarpSerializable]
        public decimal TemplateDiameter { get; set; }

        [WarpSerializable]
        public decimal TemplateFraction { get; set; }

        [WarpSerializable]
        public bool KeepOnlyFullVoxels { get; set; }

        [WarpSerializable]
        public bool WhitenSpectrum { get; set; }

        [WarpSerializable]
        public string Symmetry { get; set; }

        [WarpSerializable]
        public int HealpixOrder { get; set; }

        [WarpSerializable]
        public int Supersample { get; set; }

        [WarpSerializable]
        public int NResults { get; set; }
    }

    public class ProcessingOptionsTomoSubReconstruction : TomoProcessingOptionsBase
    {
        [WarpSerializable]
        public int BoxSize { get; set; }
        [WarpSerializable]
        public int ParticleDiameter { get; set; }
        [WarpSerializable]
        public bool Invert { get; set; }
        [WarpSerializable]
        public bool NormalizeInput { get; set; }
        [WarpSerializable]
        public bool NormalizeOutput { get; set; }
        [WarpSerializable]
        public bool PrerotateParticles { get; set; }
    }
}
