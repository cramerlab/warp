using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp
{
    public class CTF : WarpBase
    {
        private decimal _PixelSize = 1.0M;
        /// <summary>
        /// Pixel size in Angstrom
        /// </summary>
        [WarpSerializable]
        public decimal PixelSize
        {
            get { return _PixelSize; }
            set { if (value != _PixelSize) { _PixelSize = value; OnPropertyChanged(); } }
        }

        private decimal _PixelSizeDelta = 0M;
        /// <summary>
        /// Pixel size anisotropy delta in Angstrom
        /// </summary>
        [WarpSerializable]
        public decimal PixelSizeDelta
        {
            get { return _PixelSizeDelta; }
            set { if (value != _PixelSizeDelta) { _PixelSizeDelta = value; OnPropertyChanged(); } }
        }

        private decimal _PixelSizeAngle = 0M;
        /// <summary>
        /// Pixel size anisotropy angle in radians
        /// </summary>
        [WarpSerializable]
        public decimal PixelSizeAngle
        {
            get { return _PixelSizeAngle; }
            set
            {
                value = (decimal)Accord.Math.Tools.Mod((double)value + 180.0, 180.0);
                if (value != _PixelSizeAngle)
                {
                    _PixelSizeAngle = value;
                    OnPropertyChanged();
                }
            }
        }

        private decimal _Cs = 2.7M;
        /// <summary>
        /// Spherical aberration in mm
        /// </summary>
        [WarpSerializable]
        public decimal Cs
        {
            get { return _Cs; }
            set { if (value != _Cs) { _Cs = value; OnPropertyChanged(); } }
        }

        private decimal _Cc = 2.7M;
        /// <summary>
        /// Chromatic aberration in mm
        /// </summary>
        [WarpSerializable]
        public decimal Cc
        {
            get { return _Cc; }
            set { if (value != _Cc) { _Cc = value; OnPropertyChanged(); } }
        }

        private decimal _Voltage = 300.0M;
        /// <summary>
        /// Voltage in kV
        /// </summary>
        [WarpSerializable]
        public decimal Voltage
        {
            get { return _Voltage; }
            set { if (value != _Voltage) { _Voltage = value; OnPropertyChanged(); } }
        }

        private decimal _Defocus = 1.0M;
        /// <summary>
        /// Defocus in um, underfocus (first peak positive) is positive
        /// </summary>
        [WarpSerializable]
        public decimal Defocus
        {
            get { return _Defocus; }
            set { if (value != _Defocus) { _Defocus = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusDelta = 0M;
        /// <summary>
        /// Astigmatism delta defocus in um
        /// </summary>
        [WarpSerializable]
        public decimal DefocusDelta
        {
            get { return _DefocusDelta; }
            set { if (value != _DefocusDelta) { _DefocusDelta = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusAngle = 0M;
        /// <summary>
        /// Astigmatism angle in radians
        /// </summary>
        [WarpSerializable]
        public decimal DefocusAngle
        {
            get { return _DefocusAngle; }
            set
            {
                //value = (decimal) Accord.Math.Tools.Mod((double)value + 180.0, 180.0);
                if (value != _DefocusAngle)
                {
                    _DefocusAngle = value; OnPropertyChanged();
                }
            }
        }

        private decimal _Amplitude = 0.07M;
        /// <summary>
        /// Amplitude contrast
        /// </summary>
        [WarpSerializable]
        public decimal Amplitude
        {
            get { return _Amplitude; }
            set { if (value != _Amplitude) { _Amplitude = value; OnPropertyChanged(); } }
        }

        private decimal _Bfactor = 0M;
        /// <summary>
        /// B factor in Angstrom^2
        /// </summary>
        [WarpSerializable]
        public decimal Bfactor
        {
            get { return _Bfactor; }
            set { if (value != _Bfactor) { _Bfactor = value; OnPropertyChanged(); } }
        }

        private decimal _Scale = 1.0M;
        /// <summary>
        /// Scale, i. e. CTF oscillates within [-Scale; +Scale]
        /// </summary>
        [WarpSerializable]
        public decimal Scale
        {
            get { return _Scale; }
            set { if (value != _Scale) { _Scale = value; OnPropertyChanged(); } }
        }

        private decimal _PhaseShift = 0M;
        /// <summary>
        /// Phase shift in Pi
        /// </summary>
        [WarpSerializable]
        public decimal PhaseShift
        {
            get { return _PhaseShift; }
            set { if (value != _PhaseShift) { _PhaseShift = value; OnPropertyChanged(); } }
        }

        private float2 _BeamTilt = new float2(0, 0);
        /// <summary>
        /// Beam tilt in millirad
        /// </summary>
        [WarpSerializable]
        public float2 BeamTilt
        {
            get { return _BeamTilt; }
            set { if (value != _BeamTilt) { _BeamTilt = value; OnPropertyChanged(); } }
        }

        private float3 _BeamTilt2 = new float3(1, 0, 1);
        /// <summary>
        /// Higher-order beam tilt in millirad
        /// </summary>
        [WarpSerializable]
        public float3 BeamTilt2
        {
            get { return _BeamTilt2; }
            set { if (value != _BeamTilt2) { _BeamTilt2 = value; OnPropertyChanged(); } }
        }

        private decimal _IllumAngle = 30M;
        /// <summary>
        /// Illumination angle in microradians
        /// </summary>
        [WarpSerializable]
        public decimal IllumAngle
        {
            get { return _IllumAngle; }
            set { if (value != _IllumAngle) { _IllumAngle = value; OnPropertyChanged(); } }
        }

        private decimal _EnergySpread = 0.7M;
        /// <summary>
        /// Energy spread in eV
        /// </summary>
        [WarpSerializable]
        public decimal EnergySpread
        {
            get { return _EnergySpread; }
            set { if (value != _EnergySpread) { _EnergySpread = value; OnPropertyChanged(); } }
        }

        private decimal _Thickness = 0;
        /// <summary>
        /// Thickness in nm
        /// </summary>
        [WarpSerializable]
        public decimal Thickness
        {
            get { return _Thickness; }
            set { if (value != _Thickness) { _Thickness = value; OnPropertyChanged(); } }
        }

        private decimal _IceOffset = 0;
        [WarpSerializable]
        public decimal IceOffset
        {
            get { return _IceOffset; }
            set { if (value != _IceOffset) { _IceOffset = value; OnPropertyChanged(); } }
        }

        private decimal _IceIntensity = 0;
        [WarpSerializable]
        public decimal IceIntensity
        {
            get { return _IceIntensity; }
            set { if (value != _IceIntensity) { _IceIntensity = value; OnPropertyChanged(); } }
        }

        private float2 _IceStd = new float2(0.5f);
        [WarpSerializable]
        public float2 IceStd
        {
            get { return _IceStd; }
            set { if (value != _IceStd) { _IceStd = value; OnPropertyChanged(); } }
        }

        public void FromStruct(CTFStruct s)
        {
            _PixelSize = (decimal) s.PixelSize * 1e10M;
            _PixelSizeDelta = (decimal) s.PixelSizeDelta * 1e10M;
            _PixelSizeAngle = (decimal) s.PixelSizeAngle * (180M / (decimal) Math.PI);

            _Cs = (decimal) s.Cs * 1e3M;
            _Voltage = (decimal) s.Voltage * 1e-3M;
            _Amplitude = (decimal) s.Amplitude;

            _Defocus = (decimal) -s.Defocus * 1e6M;
            _DefocusDelta = (decimal) -s.DefocusDelta * 1e6M;
            _DefocusAngle = (decimal) s.AstigmatismAngle * (180M / (decimal)Math.PI);

            //_Bfactor = (decimal) s.Bfactor * 1e20M;
            _Scale = (decimal) s.Scale;

            _PhaseShift = (decimal) s.PhaseShift / (decimal) Math.PI;

            OnPropertyChanged("");
        }

        public CTFStruct ToStruct()
        {
            CTFStruct Result = new CTFStruct();

            Result.PixelSize = (float)(PixelSize * 1e-10M);
            Result.PixelSizeDelta = (float)(PixelSizeDelta * 1e-10M);
            Result.PixelSizeAngle = (float)PixelSizeAngle / (float)(180.0 / Math.PI);

            Result.Cs = (float)(Cs * 1e-3M);
            Result.Voltage = (float)(Voltage * 1e3M);
            Result.Amplitude = (float)Amplitude;

            Result.Defocus = (float)(-Defocus * 1e-6M);
            Result.DefocusDelta = (float)(-DefocusDelta * 1e-6M);
            Result.AstigmatismAngle = (float)DefocusAngle / (float)(180.0 / Math.PI);

            Result.Bfactor = (float)(Bfactor * 1e-20M);
            Result.Scale = (float)Scale;

            Result.PhaseShift = (float)(PhaseShift * (decimal)Math.PI);

            return Result;
        }

        public float[] Get1D(int width, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float[] Output = new float[width];

            double ny = 0.5 / (double)PixelSize / width;

            for (int i = 0; i < width; i++)
                Output[i] = Get1D(i * (float)ny, ampsquared, ignorebfactor, ignorescale);

            return Output;
        }

        public float Get1D(float freq, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            double voltage = (double) Voltage * 1e3;
            double lambda = 12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6));
            double defocus = -(double) Defocus * 1e4;
            double cs = (double) Cs * 1e7;
            double amplitude = (double) Amplitude;
            double scale = (double) Scale;
            double phaseshift = (double) PhaseShift * Math.PI;
            double K1 = Math.PI * lambda;
            double K2 = Math.PI * 0.5f * cs * lambda * lambda * lambda;
            double K3 = Math.Sqrt(1f - amplitude * amplitude);
            double K4 = (double)Bfactor * 0.25f;

            double r2 = freq * freq;
            double r4 = r2 * r2;

            double deltaf = defocus;
            double argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;
            double retval = amplitude * Math.Cos(argument) - K3 * Math.Sin(argument);

            if (K4 != 0)
                retval *= Math.Exp(K4 * r2);

            if (ampsquared)
                retval = Math.Abs(retval);

            return (float)(scale * retval);
        }

        public double GetPhaseAtFrequency(double freq)
        {
            double voltage = (double)Voltage * 1e3;
            double lambda = 12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6));
            double defocus = -(double)Defocus * 1e4;
            double cs = (double)Cs * 1e7;
            double amplitude = (double)Amplitude;
            double scale = (double)Scale;
            double phaseshift = (double)PhaseShift * Math.PI;
            double K1 = Math.PI * lambda;
            double K2 = Math.PI * 0.5f * cs * lambda * lambda * lambda;

            double r2 = freq * freq;
            double r4 = r2 * r2;

            double deltaf = defocus;
            double argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;

            return argument;
        }

        public double GetFrequencyAtPhase(double phase)
        {
            double voltage = (double)Voltage * 1e3;
            double lambda = 12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6));
            double z = -(double)Defocus * 1e4;
            double cs = (double)Cs * 1e7;
            double amplitude = (double)Amplitude;
            double scale = (double)Scale;
            double phaseshift = (double)PhaseShift * Math.PI;
            double a = Math.PI * lambda;
            double b = Math.PI * 0.5f * cs * lambda * lambda * lambda;

            double Root1 = Math.Sqrt(a * a * z * z + 4 * b * (phase + phaseshift));
            double Frac1 = -(Root1 + a * z) / b;
            double Frac2 = Math.Sqrt(Frac1) / Math.Sqrt(2.0);

            return Frac2;
        }

        public float[] Get1DWithIce(int width, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float[] Output = new float[width];

            float[] CTFProtein = Get1D(width, true, ignorebfactor, ignorescale);
            CTF Ice = GetCopy();
            Ice.Defocus += IceOffset;
            float[] CTFIce = Ice.Get1D(width, true, ignorebfactor, ignorescale);
            float[] MaskIce = GetIceMask(width);

            for (int i = 0; i < Output.Length; i++)
                Output[i] = CTFIce[i] * MaskIce[i] + CTFProtein[i] * (1f - MaskIce[i]);

            return Output;
        }

        public double[] Get1DDouble(int width, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            double[] Output = new double[width];

            double ny = 0.5 / (double)PixelSize / width;

            for (int i = 0; i < width; i++)
                Output[i] = Get1DDouble(i * ny, ampsquared, ignorebfactor, ignorescale);

            return Output;
        }

        public double Get1DDouble(double freq, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            double voltage = (double)Voltage * 1e3;
            double lambda = 12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6));
            double defocus = -(double)Defocus * 1e4;
            double cs = (double)Cs * 1e7;
            double amplitude = (double)Amplitude;
            double scale = (double)Scale;
            double phaseshift = (double)PhaseShift * Math.PI;
            double K1 = Math.PI * lambda;
            double K2 = Math.PI * 0.5f * cs * lambda * lambda * lambda;
            double K3 = Math.Sqrt(1f - amplitude * amplitude);
            double K4 = (double)Bfactor * 0.25f;

            double r2 = freq * freq;
            double r4 = r2 * r2;

            double deltaf = defocus;
            double argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;
            double retval = amplitude * Math.Cos(argument) - K3 * Math.Sin(argument);

            if (K4 != 0)
                retval *= Math.Exp(K4 * r2);

            if (ampsquared)
                retval = Math.Abs(retval);

            return scale * retval;
        }

        public float[] Get2D(float2[] coordinates, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float[] Output = new float[coordinates.Length];
            
            float pixelsize = (float) PixelSize;
            float pixeldelta = (float) PixelSizeDelta;
            float pixelangle = (float) PixelSizeAngle / (float)(180.0 / Math.PI);
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float) DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float scale = (float)Scale;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Sqrt(1f - amplitude * amplitude);
            float K4 = (float)Bfactor * 0.25f;

            Parallel.For(0, coordinates.Length, i =>
            {
                float angle = coordinates[i].Y;
                float r = coordinates[i].X / (pixelsize + pixeldelta * (float) Math.Cos(2.0 * (angle - pixelangle)));
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float) Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;
                float retval = amplitude * (float) Math.Cos(argument) - K3 * (float) Math.Sin(argument);

                if (!ignorebfactor && K4 != 0)
                    retval *= (float) Math.Exp(K4 * r2);

                if (ampsquared)
                    retval = Math.Abs(retval);// * retval);

                if (!ignorescale)
                    Output[i] = scale * retval;
                else
                    Output[i] = retval;
            });

            return Output;
        }

        public float[] Get2DFromScaledCoords(float2[] coordinates, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float[] Output = new float[coordinates.Length];
            
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float)DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float scale = (float)Scale;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Sqrt(1f - amplitude * amplitude);
            float K4 = (float)Bfactor * 0.25f;

            Parallel.For(0, coordinates.Length, i =>
            {
                float angle = coordinates[i].Y;
                float r = coordinates[i].X;
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float)Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;
                float retval = amplitude * (float)Math.Cos(argument) - K3 * (float)Math.Sin(argument);

                if (!ignorebfactor && K4 != 0)
                    retval *= (float)Math.Exp(K4 * r2);

                if (ampsquared)
                    retval = Math.Abs(retval);// * retval);

                if (!ignorescale)
                    Output[i] = scale * retval;
                else
                    Output[i] = retval;
            });

            return Output;
        }

        public float[] GetPeaks(bool considerIce = false)
        {
            List<float> Result = new List<float>();

            float[] Values = considerIce ? Get1DWithIce(1 << 12, true) : Get1D(1 << 12, true);
            float[] dValues = MathHelper.Diff(Values);

            for (int i = 0; i < dValues.Length - 1; i++)
                if (Math.Sign(dValues[i]) > 0 && Math.Sign(dValues[i + 1]) < 0)
                    Result.Add(0.5f * i / Values.Length);
            
            return Result.ToArray();
        }

        public float[] GetZeros(bool considerIce = false)
        {
            List<float> Result = new List<float>();

            float[] Values = considerIce ? Get1DWithIce(1 << 12, true) : Get1D(1 << 12, true);
            float[] dValues = MathHelper.Diff(Values);

            for (int i = 0; i < dValues.Length - 1; i++)
                if (Math.Sign(dValues[i]) < 0 && Math.Sign(dValues[i + 1]) > 0)
                    Result.Add(0.5f * i / Values.Length);

            return Result.ToArray();
        }

        public Image GetBeamTilt(int size, int originalSize)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float2[] Data = new float2[new int2(size).ElementsFFT()];

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                float delta_phase = factor * q * (yy * _BeamTilt.Y + xx * _BeamTilt.X);
                delta_phase = delta_phase * Helper.ToRad;
                Data[y * (size / 2 + 1) + x] = new float2((float)Math.Cos(delta_phase), (float)Math.Sin(delta_phase));
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public Image GetBeamTilt(int size, int originalSize, float2[] beamTilts)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float2[][] Data = Helper.ArrayOfFunction(i => new float2[new int2(size).ElementsFFT()], beamTilts.Length);

            for (int z = 0; z < beamTilts.Length; z++)
            {
                float2 t = beamTilts[z];
                Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
                {
                    float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                    float delta_phase = factor * q * (yy * t.Y + xx * t.X);
                    delta_phase = delta_phase * Helper.ToRad;
                    Data[z][y * (size / 2 + 1) + x] = new float2((float)Math.Cos(delta_phase), (float)Math.Sin(delta_phase));
                });
            }

            return new Image(Data, new int3(size, size, beamTilts.Length), true);
        }

        public Image GetBeamTiltCoords(int size, int originalSize)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float2[] Data = new float2[new int2(size).ElementsFFT()];

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                Data[y * (size / 2 + 1) + x] = new float2(factor * q * xx, factor * q * yy) * Helper.ToRad;
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public Image GetBeamTiltPhase(int size, int originalSize)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float[] Data = new float[new int2(size).ElementsFFT()];

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                float delta_phase = factor * q * (yy * _BeamTilt.Y + xx * _BeamTilt.X);
                delta_phase = delta_phase * Helper.ToRad;
                Data[y * (size / 2 + 1) + x] = delta_phase;
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public int GetAliasingFreeSize(float maxResolution)
        {
            double Freq0 = 1.0 / maxResolution;
            float[] P = Helper.ArrayOfFunction(i => i > 0 ? (float)(GetPhaseAtFrequency(i / 1000.0 * Freq0) / Math.PI) : 0, 1000);
            float[] dP = MathHelper.Diff(P).Select(v => Math.Abs(v)).ToArray();
            float dPMax = MathHelper.Max(dP);

            return (int)Math.Ceiling(dPMax / 0.5f * 1000) * 2;
        }

        public static Image GetCTFCoords(int size, int originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[(size / 2 + 1) * size];
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size / 2 + 1; x++)
                    {
                        int xx = x;
                        int yy = y < size / 2 + 1 ? y : y - size;

                        float xs = xx / (float)originalSize;
                        float ys = yy / (float)originalSize;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * (size / 2 + 1) + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size, size, 1), true);
            }

            return CTFCoords;
        }

        public static Image GetCTFCoords(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[(size.X / 2 + 1) * size.Y];
                for (int y = 0; y < size.Y; y++)
                for (int x = 0; x < size.X / 2 + 1; x++)
                {
                    int xx = x;
                    int yy = y < size.Y / 2 + 1 ? y : y - size.Y;

                    float xs = xx / (float)originalSize.X;
                    float ys = yy / (float)originalSize.Y;
                    float r = (float)Math.Sqrt(xs * xs + ys * ys);
                    float angle = (float)Math.Atan2(yy, xx);

                    if (pixelSize != 1 || pixelSizeDelta != 0)
                        r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                    CTFCoordsData[y * (size.X / 2 + 1) + x] = new float2(r, angle);
                }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1), true);
            }

            return CTFCoords;
        }

        public static Image GetCTFCoordsParallel(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[(size.X / 2 + 1) * size.Y];
                Helper.ForCPU(0, size.Y, 4, null, (y, threadID) =>
                {
                    for (int x = 0; x < size.X / 2 + 1; x++)
                    {
                        int xx = x;
                        int yy = y < size.Y / 2 + 1 ? y : y - size.Y;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * (size.X / 2 + 1) + x] = new float2(r, angle);
                    }
                }, null);

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1), true);
            }

            return CTFCoords;
        }

        public static Image GetCTFCoordsFull(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[size.X * size.Y];
                for (int y = 0; y < size.Y; y++)
                    for (int x = 0; x < size.X; x++)
                    {
                        int xx = x < size.X / 2 + 1 ? x : x - size.X;
                        int yy = y < size.Y / 2 + 1 ? y : y - size.Y;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * size.X + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1));
            }

            return CTFCoords;
        }

        public static Image GetCTFCoordsFullCentered(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[size.X * size.Y];
                for (int y = 0; y < size.Y; y++)
                    for (int x = 0; x < size.X; x++)
                    {
                        int xx = x - size.X / 2;
                        int yy = y - size.Y / 2;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * size.X + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1));
            }

            return CTFCoords;
        }

        public CTF GetCopy()
        {
            return new CTF
            {
                _Amplitude = Amplitude,
                _BeamTilt = BeamTilt,
                _BeamTilt2 = BeamTilt2,
                _Bfactor = Bfactor,
                _Cs = Cs,
                _Defocus = Defocus,
                _DefocusAngle = DefocusAngle,
                _DefocusDelta = DefocusDelta,
                _PhaseShift = PhaseShift,
                _PixelSize = PixelSize,
                _PixelSizeAngle = PixelSizeAngle,
                _PixelSizeDelta = PixelSizeDelta,
                _Scale = Scale,
                _Voltage = Voltage
            };
        }

        public float[] GetIceMask(int length)
        {
            float IceFreq = (float)PixelSize / 3.7f * 2 * length;
            float2 Std = new float2((float)PixelSize / 3.7f - (float)PixelSize / (3.7f + _IceStd.X),
                                       (float)PixelSize / (3.7f + _IceStd.Y) - (float)PixelSize / 3.7f) * 2 * length;
            Std *= Std;

            return Helper.ArrayOfFunction(i =>
            {
                float d = IceFreq - i;
                return (float)Math.Exp(-(d * d) / (2 * (d > 0 ? Std.X : Std.Y))) * (float)_IceIntensity;
            }, length);
        }

        public float[] EstimateQuality(float[] experimental, float[] experimentalScale, float minFreq, int minSamples, bool considerIce = false)
        {
            try
            {
                int N = experimental.Length;
                int MinN = (int)(minFreq * N);

                float[] Quality = Helper.ArrayOfConstant(float.NaN, N);
                float[] Zeros = GetZeros(considerIce).Select(v => v * 2 * N).ToArray();
                if (Zeros.Length < 2)
                    return Quality;

                // Calculate indices and widths of all peaks
                int[] Peaks = new int[Zeros.Length - 1];
                int[] PeakWidths = new int[Peaks.Length];
                for (int i = 0; i < Zeros.Length - 1; i++)
                {
                    Peaks[i] = (int)((Zeros[i] + Zeros[i + 1]) * 0.5f);
                    PeakWidths[i] = (int)(Zeros[i + 1] - Zeros[i]);
                }

                // For each index, find the closest peak
                int[] ClosestPeak = new int[N];
                int[] WindowLength = new int[N];
                for (int i = 0, peak = 0; i < N; i++)
                {
                    int ClosestDist = Math.Abs(i - Peaks[peak]);
                    if (Math.Abs(i - Peaks[peak + 1]) < ClosestDist)
                        peak = Math.Min(Peaks.Length - 2, peak + 1);

                    ClosestPeak[i] = peak;
                    WindowLength[i] = Math.Max(minSamples, PeakWidths[peak]);
                }

                // Calculate simulated CTF and multiply it by the scale curve if needed
                float[] Simulated = considerIce ? Get1DWithIce(N, true) : Get1D(N, true);
                if (experimentalScale != null)
                {
                    if (experimental.Length != experimentalScale.Length)
                        throw new Exception("Experimental values and scale arrays should have equal length.");
                    for (int i = 0; i < Simulated.Length; i++)
                        Simulated[i] *= experimentalScale[i];
                }

                for (int i = MinN; i < N - minSamples / 2; i++)
                {
                    int WindowStart = Math.Max(0, i - WindowLength[i] / 2);
                    int WindowEnd = Math.Min(N, i + WindowLength[i] / 2);
                    float[] WindowExperimental = Helper.Subset(experimental, WindowStart, WindowEnd);
                    float[] WindowSimulated = Helper.Subset(Simulated, WindowStart, WindowEnd);

                    Quality[i] = MathHelper.Mult(MathHelper.Normalize(WindowExperimental), MathHelper.Normalize(WindowSimulated)).Sum() / WindowExperimental.Length;
                }

                return Quality;
            }
            catch
            {
                return Helper.ArrayOfConstant(1f, experimental.Length);
            }
        }
    }

    /// <summary>
    /// Everything is in SI units
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CTFStruct
    {
        public float PixelSize;
        public float PixelSizeDelta;
        public float PixelSizeAngle;
        public float Cs;
        public float Voltage;
        public float Defocus;
        public float AstigmatismAngle;
        public float DefocusDelta;
        public float Amplitude;
        public float Bfactor;
        public float Scale;
        public float PhaseShift;
    }

    /// <summary>
    /// Everything is in SI units
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CTFFitStruct
    {
        public float3 Pixelsize;
        public float3 Pixeldelta;
        public float3 Pixelangle;
        public float3 Cs;
        public float3 Voltage;
        public float3 Defocus;
        public float3 Astigmatismangle;
        public float3 Defocusdelta;
        public float3 Amplitude;
        public float3 Bfactor;
        public float3 Scale;
        public float3 Phaseshift;

        public int2 DimsPeriodogram;
        public int MaskInnerRadius;
        public int MaskOuterRadius;
    }
}