using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Accord;
using BitMiracle.LibTiff.Classic;
using Warp.Headers;
using Warp.Tools;

namespace Warp
{
    public class Image : IDisposable
    {
        private readonly object Sync = new object();
        public static object FFT_CPU_Sync = new object();
        
        public int3 Dims;
        public int3 DimsFT => new int3(Dims.X / 2 + 1, Dims.Y, Dims.Z);
        public int2 DimsSlice => new int2(Dims.X, Dims.Y);
        public int2 DimsFTSlice => new int2(DimsFT.X, DimsFT.Y);
        public int3 DimsEffective => IsFT ? DimsFT : Dims;

        public float PixelSize = 1;

        public bool IsFT;
        public readonly bool IsComplex;
        public readonly bool IsHalf;

        public long ElementsComplex => IsFT ? DimsFT.Elements() : Dims.Elements();
        public long ElementsReal => IsComplex ? ElementsComplex * 2 : ElementsComplex;

        public long ElementsSliceComplex => IsFT ? DimsFTSlice.Elements() : DimsSlice.Elements();
        public long ElementsSliceReal => IsComplex ? ElementsSliceComplex * 2 : ElementsSliceComplex;

        public long ElementsLineComplex => IsFT ? DimsFTSlice.X : DimsSlice.X;
        public long ElementsLineReal => IsComplex ? ElementsLineComplex * 2 : ElementsLineComplex;

        private bool IsDeviceDirty = false;
        private IntPtr _DeviceData = IntPtr.Zero;

        private IntPtr DeviceData
        {
            get
            {
                if (_DeviceData == IntPtr.Zero)
                {
                    _DeviceData = !IsHalf ? GPU.MallocDevice(ElementsReal) : GPU.MallocDeviceHalf(ElementsReal);
                    GPU.OnMemoryChanged();
                }

                return _DeviceData;
            }
        }

        private bool IsHostDirty = false;
        private float[][] _HostData = null;

        private float[][] HostData
        {
            get
            {
                if (_HostData == null)
                {
                    _HostData = new float[Dims.Z][];
                    for (int i = 0; i < Dims.Z; i++)
                        _HostData[i] = new float[ElementsSliceReal];
                }

                return _HostData;
            }
        }

        private bool IsHostPinnedDirty = false;
        private IntPtr _HostPinnedData = IntPtr.Zero;

        private IntPtr HostPinnedData
        {
            get
            {
                if (_HostPinnedData == IntPtr.Zero)
                {
                    _HostPinnedData = GPU.MallocHostPinned(ElementsReal);
                }

                return _HostPinnedData;
            }
        }

        public Image(float[][] data, int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            if (data.Length != dims.Z || data[0].Length != ElementsSliceComplex)
                throw new DimensionMismatchException();

            _HostData = data.Select(v => v.ToArray()).ToArray();
            IsHostDirty = true;
        }

        public Image(float2[][] data, int3 dims, bool isft = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = true;
            IsHalf = ishalf;

            if (data.Length != dims.Z || data[0].Length != ElementsSliceComplex)
                throw new DimensionMismatchException();

            UpdateHostWithComplex(data);
            IsHostDirty = true;
        }

        public Image(float[] data, int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            if (data.Length != ElementsReal)
                throw new DimensionMismatchException();

            float[][] Slices = new float[dims.Z][];

            for (int z = 0, i = 0; z < dims.Z; z++)
            {
                Slices[z] = new float[ElementsSliceReal];
                for (int j = 0; j < Slices[z].Length; j++)
                    Slices[z][j] = data[i++];
            }

            _HostData = Slices;
            IsHostDirty = true;
        }

        public Image(float2[] data, int3 dims, bool isft = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = true;
            IsHalf = ishalf;

            if (data.Length != ElementsComplex)
                throw new DimensionMismatchException();

            float[][] Slices = new float[dims.Z][];
            int i = 0;
            for (int z = 0; z < dims.Z; z++)
            {
                Slices[z] = new float[ElementsSliceReal];
                for (int j = 0; j < Slices[z].Length / 2; j++)
                {
                    Slices[z][j * 2] = data[i].X;
                    Slices[z][j * 2 + 1] = data[i].Y;
                    i++;
                }
            }

            _HostData = Slices;
            IsHostDirty = true;
        }

        public Image(float[] data, bool isft = false, bool iscomplex = false, bool ishalf = false) : 
            this(data, new int3(data.Length, 1, 1), isft, iscomplex, ishalf) { }

        public Image(float2[] data, bool isft = false, bool ishalf = false) : 
            this(data, new int3(data.Length, 1, 1), isft, ishalf) { }

        public Image(int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            _HostData = HostData; // Initializes new array since _HostData is null
            IsHostDirty = true;
        }

        public Image(IntPtr deviceData, int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false, bool fromPinned = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            if (!fromPinned)
            {
                _DeviceData = !IsHalf ? GPU.MallocDevice(ElementsReal) : GPU.MallocDeviceHalf(ElementsReal);
                GPU.OnMemoryChanged();
                if (deviceData != IntPtr.Zero)
                {
                    if (!IsHalf)
                        GPU.CopyDeviceToDevice(deviceData, _DeviceData, ElementsReal);
                    else
                        GPU.CopyDeviceHalfToDeviceHalf(deviceData, _DeviceData, ElementsReal);
                }

                IsDeviceDirty = true;
            }
            else
            {
                _HostPinnedData = GPU.MallocHostPinned(ElementsReal);
                IsHostPinnedDirty = true;
            }
        }

        ~Image()
        {
            Dispose();
        }

        public static Image FromFile(string path, int2 headerlessSliceDims, int headerlessOffset, Type headerlessType, int layer = -1, Stream stream = null)
        {
            MapHeader Header = MapHeader.ReadFromFile(path, headerlessSliceDims, headerlessOffset, headerlessType);
            float[][] Data = IOHelper.ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, layer, stream);
            if (layer >= 0)
                Header.Dimensions.Z = 1;

            return new Image(Data, Header.Dimensions) { PixelSize = Header.PixelSize.X };
        }

        public static Image FromFile(string path, int layer = -1, Stream stream = null)
        {
            return FromFile(path, new int2(1, 1), 0, typeof(float), layer, stream);
        }

        public static Image FromFilePatient(int attempts, int mswait, string path, int2 headerlessSliceDims, int headerlessOffset, Type headerlessType, int layer = -1, Stream stream = null)
        {
            Image Result = null;
            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    Result = FromFile(path, headerlessSliceDims, headerlessOffset, headerlessType, layer, stream);
                    break;
                }
                catch
                {
                    Thread.Sleep(mswait);
                }
            }

            if (Result == null)
                throw new Exception("Could not successfully read file within the specified number of attempts.");

            return Result;
        }

        public static Image FromFilePatient(int attempts, int mswait, string path, int layer = -1, Stream stream = null)
        {
            return FromFilePatient(attempts, mswait, path, new int2(1, 1), 0, typeof(float), layer, stream);
        }

        public IntPtr GetDevice(Intent intent)
        {
            lock (Sync)
            {
                if ((intent & Intent.Read) > 0 && IsHostDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        if (!IsHalf)
                            GPU.CopyHostToDevice(HostData[z], new IntPtr((long) DeviceData + ElementsSliceReal * z * sizeof (float)), ElementsSliceReal);
                        else
                            GPU.CopyHostToDeviceHalf(HostData[z], new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(short)), ElementsSliceReal);

                    IsHostDirty = false;
                }
                else if ((intent & Intent.Read) > 0 && IsHostPinnedDirty)
                {
                    GPU.CopyDeviceToHostPinned(HostPinnedData, DeviceData, ElementsReal);

                    IsHostPinnedDirty = false;
                }

                if ((intent & Intent.Write) > 0)
                {
                    IsDeviceDirty = true;
                    IsHostDirty = false;
                    IsHostPinnedDirty = false;
                }

                return DeviceData;
            }
        }

        public void CopyToDevicePointer(IntPtr pointer)
        {
            if (IsDeviceDirty)
                GetHost(Intent.Read);

            lock (Sync)
            {
                for (int z = 0; z < Dims.Z; z++)
                    GPU.CopyHostToDevice(HostData[z], new IntPtr((long)pointer + ElementsSliceReal * z * sizeof(float)), ElementsSliceReal);
            }
        }

        public IntPtr GetDeviceSlice(int slice, Intent intent)
        {
            IntPtr Start = GetDevice(intent);
            Start = new IntPtr((long)Start + slice * ElementsSliceReal * (IsHalf ? sizeof(short) : sizeof (float)));

            return Start;
        }

        public float[][] GetHost(Intent intent)
        {
            lock (Sync)
            {
                if ((intent & Intent.Read) > 0 && IsDeviceDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        if (!IsHalf)
                            GPU.CopyDeviceToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(float)), HostData[z], ElementsSliceReal);
                        else
                            GPU.CopyDeviceHalfToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(short)), HostData[z], ElementsSliceReal);

                    IsDeviceDirty = false;
                }
                else if ((intent & Intent.Read) > 0 && IsHostPinnedDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        GPU.CopyHostToHost(new IntPtr((long)HostPinnedData + ElementsSliceReal * z * sizeof(float)), HostData[z], ElementsSliceReal);

                    IsHostPinnedDirty = false;
                }

                if ((intent & Intent.Write) > 0)
                {
                    IsHostDirty = true;
                    IsDeviceDirty = false;
                    IsHostPinnedDirty = false;
                }

                return HostData;
            }
        }

        public IntPtr GetHostPinned(Intent intent)
        {
            lock (Sync)
            {
                if ((intent & Intent.Read) > 0 && IsHostDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        GPU.CopyHostToHost(HostData[z], new IntPtr((long)HostPinnedData + ElementsSliceReal * z * sizeof(float)), ElementsSliceReal);

                    IsHostDirty = false;
                }
                else if ((intent & Intent.Read) > 0 && IsDeviceDirty)
                {
                    GPU.CopyDeviceToHostPinned(DeviceData, HostPinnedData, ElementsReal);

                    IsDeviceDirty = false;
                }

                if ((intent & Intent.Write) > 0)
                {
                    IsHostDirty = false;
                    IsDeviceDirty = false;
                    IsHostPinnedDirty = true;
                }

                return HostPinnedData;
            }
        }

        public IntPtr GetHostPinnedSlice(int slice, Intent intent)
        {
            IntPtr Start = GetHostPinned(intent);
            Start = new IntPtr((long)Start + slice * ElementsSliceReal * sizeof(float));

            return Start;
        }

        public float2[][] GetHostComplexCopy()
        {
            if (!IsComplex)
                throw new Exception("Data must be of complex type.");

            float[][] Data = GetHost(Intent.Read);
            float2[][] ComplexData = new float2[Dims.Z][];

            for (int z = 0; z < Dims.Z; z++)
            {
                float[] Slice = Data[z];
                float2[] ComplexSlice = new float2[DimsEffective.ElementsSlice()];
                for (int i = 0; i < ComplexSlice.Length; i++)
                    ComplexSlice[i] = new float2(Slice[i * 2], Slice[i * 2 + 1]);

                ComplexData[z] = ComplexSlice;
            }

            return ComplexData;
        }

        public void UpdateHostWithComplex(float2[][] complexData)
        {
            if (complexData.Length != Dims.Z ||
                complexData[0].Length != DimsEffective.ElementsSlice())
                throw new DimensionMismatchException();

            float[][] Data = GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                float[] Slice = Data[z];
                float2[] ComplexSlice = complexData[z];

                for (int i = 0; i < ComplexSlice.Length; i++)
                {
                    Slice[i * 2] = ComplexSlice[i].X;
                    Slice[i * 2 + 1] = ComplexSlice[i].Y;
                }
            }
        }

        public float[] GetHostContinuousCopy()
        {
            float[] Continuous = new float[ElementsReal];
            float[][] Data = GetHost(Intent.Read);
            unsafe
            {
                fixed (float* ContinuousPtr = Continuous)
                {
                    float* ContinuousP = ContinuousPtr;
                    for (int i = 0; i < Data.Length; i++)
                    {
                        fixed (float* DataPtr = Data[i])
                        {
                            float* DataP = DataPtr;
                            for (int j = 0; j < Data[i].Length; j++)
                                *ContinuousP++ = *DataP++;
                        }
                    }
                }
            }

            return Continuous;
        }

        public void FreeDevice()
        {
            lock (Sync)
            {
                if (_DeviceData != IntPtr.Zero)
                {
                    if (IsDeviceDirty)
                        for (int z = 0; z < Dims.Z; z++)
                            if (!IsHalf)
                                GPU.CopyDeviceToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(float)), HostData[z], ElementsSliceReal);
                            else
                                GPU.CopyDeviceHalfToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(short)), HostData[z], ElementsSliceReal);
                    GPU.FreeDevice(DeviceData);
                    GPU.OnMemoryChanged();
                    _DeviceData = IntPtr.Zero;
                    IsDeviceDirty = false;
                }

                IsHostDirty = true;
            }
        }

        public void WriteMRC(string path, float pixelSize, bool doStatistics = false, HeaderMRC header = null)
        {
            if (header == null)
            {
                header = new HeaderMRC();
                header.PixelSize = new float3(pixelSize);
            }
            header.Dimensions = IsFT ? DimsFT : Dims;
            header.Dimensions.X *= IsComplex ? 2 : 1;

            if (doStatistics)
            {
                float[][] Data = GetHost(Intent.Read);
                float Min = float.MaxValue, Max = -float.MaxValue;
                Parallel.For(0, Data.Length, z =>
                {
                    float LocalMin = MathHelper.Min(Data[z]);
                    float LocalMax = MathHelper.Max(Data[z]);
                    lock (Data)
                    {
                        Min = Math.Min(LocalMin, Min);
                        Max = Math.Max(LocalMax, Max);
                    }
                });
                header.MinValue = Min;
                header.MaxValue = Max;
            }

            IOHelper.WriteMapFloat(path, header, GetHost(Intent.Read));
        }

        public void WriteMRC(string path, bool doStatistics = false, HeaderMRC header = null)
        {
            WriteMRC(path, PixelSize, doStatistics, header);
        }

        public void WriteTIFF(string path, float pixelSize, Type dataType)
        {
            int Width = Dims.X;
            int Height = Dims.Y;
            int SamplesPerPixel = 1;
            int BitsPerSample = 8;

            if (dataType == typeof(byte))
                BitsPerSample = 8;
            else if (dataType == typeof(short))
                BitsPerSample = 16;
            else if (dataType == typeof(int))
                BitsPerSample = 32;
            else if (dataType == typeof(long))
                BitsPerSample = 64;
            else if (dataType == typeof(float))
                BitsPerSample = 32;
            else if (dataType == typeof(double))
                BitsPerSample = 64;
            else
                throw new Exception("Unsupported data type.");

            SampleFormat Format = SampleFormat.INT;
            if (dataType == typeof(byte))
                Format = SampleFormat.UINT;
            else if (dataType == typeof(float) || dataType == typeof(double))
                Format = SampleFormat.IEEEFP;

            int BytesPerSample = BitsPerSample / 8;

            byte[][] BytesData = Helper.ArrayOfFunction(i => new byte[Dims.ElementsSlice() * BytesPerSample], Dims.Z);
            float[][] Data = GetHost(Intent.Read);
            int PageLength = Data[0].Length;
            unsafe
            {
                for (int z = 0; z < Dims.Z; z++)
                {
                    fixed (byte* BytesPtr = BytesData[z])
                    fixed (float* DataPtr = Data[z])
                    {
                        if (dataType == typeof(byte))
                        {
                            for (int i = 0; i < PageLength; i++)
                                BytesPtr[i] = (byte)Math.Max(0, Math.Min(byte.MaxValue, (int)DataPtr[i]));
                        }
                        else if (dataType == typeof(short))
                        {
                            short* ConvPtr = (short*)BytesPtr;
                            for (int i = 0; i < PageLength; i++)
                                ConvPtr[i] = (short)Math.Max(short.MinValue, Math.Min(short.MaxValue, (int)DataPtr[i]));
                        }
                        else if (dataType == typeof(int))
                        {
                            int* ConvPtr = (int*)BytesPtr;
                            for (int i = 0; i < PageLength; i++)
                                ConvPtr[i] = (int)Math.Max(int.MinValue, Math.Min(int.MaxValue, (int)DataPtr[i]));
                        }
                        else if (dataType == typeof(long))
                        {
                            long* ConvPtr = (long*)BytesPtr;
                            for (int i = 0; i < PageLength; i++)
                                ConvPtr[i] = (long)Math.Max(long.MinValue, Math.Min(long.MaxValue, (long)DataPtr[i]));
                        }
                        else if (dataType == typeof(float))
                        {
                            float* ConvPtr = (float*)BytesPtr;
                            for (int i = 0; i < PageLength; i++)
                                ConvPtr[i] = DataPtr[i];
                        }
                        else if (dataType == typeof(double))
                        {
                            double* ConvPtr = (double*)BytesPtr;
                            for (int i = 0; i < PageLength; i++)
                                ConvPtr[i] = DataPtr[i];
                        }
                    }
                }
            }

            using (Tiff output = Tiff.Open(path, "w"))
            {
                for (int page = 0; page < Dims.Z; page++)
                {
                    output.SetField(TiffTag.IMAGEWIDTH, Width / SamplesPerPixel);
                    output.SetField(TiffTag.IMAGELENGTH, Height);
                    output.SetField(TiffTag.SAMPLESPERPIXEL, SamplesPerPixel);
                    output.SetField(TiffTag.SAMPLEFORMAT, Format);
                    output.SetField(TiffTag.BITSPERSAMPLE, BitsPerSample);
                    output.SetField(TiffTag.ORIENTATION, Orientation.BOTLEFT);
                    output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

                    output.SetField(TiffTag.COMPRESSION, Compression.LZW);

                    output.SetField(TiffTag.ROWSPERSTRIP, output.DefaultStripSize(0));
                    output.SetField(TiffTag.XRESOLUTION, 100.0);
                    output.SetField(TiffTag.YRESOLUTION, 100.0);
                    output.SetField(TiffTag.RESOLUTIONUNIT, ResUnit.INCH);

                    // specify that it's a page within the multipage file
                    output.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
                    // specify the page number
                    output.SetField(TiffTag.PAGENUMBER, page, Dims.Z);

                    for (int j = 0; j < Height; j++)
                        output.WriteScanline(Helper.Subset(BytesData[page], j * Width * BytesPerSample, (j + 1) * Width * BytesPerSample), j);

                    output.WriteDirectory();
                    output.FlushData();
                }
            }
        }

        public void WritePNG(string path)
        {
            if (Dims.Z > 1)
                throw new DimensionMismatchException("Image cannot have more than 1 layer for PNG.");

            Bitmap Image = new Bitmap(Dims.X, Dims.Y);
            Image.SetResolution(96, 96);
            BitmapData ImageData = Image.LockBits(new Rectangle(0, 0, Dims.X, Dims.Y), ImageLockMode.ReadWrite, Image.PixelFormat);
            IntPtr ImageDataPtr = ImageData.Scan0;

            unsafe
            {
                byte* ImageDataP = (byte*)ImageDataPtr;
                float[] Data = GetHost(Intent.Read)[0];

                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int i = y * Dims.X + x;
                        byte PixelValue = (byte)Math.Max(0, Math.Min(255, (long)Data[(Dims.Y - 1 - y) * Dims.X + x]));
                        ImageDataP[i * 4 + 3] = 255;
                        ImageDataP[i * 4 + 2] = PixelValue;
                        ImageDataP[i * 4 + 1] = PixelValue;
                        ImageDataP[i * 4 + 0] = PixelValue;
                    }
                }
            }

            Image.UnlockBits(ImageData);
            Image.Save(path, ImageFormat.Png);
        }

        public void Dispose()
        {
            lock (Sync)
            {
                if (_DeviceData != IntPtr.Zero)
                {
                    GPU.FreeDevice(_DeviceData);
                    GPU.OnMemoryChanged();
                    _DeviceData = IntPtr.Zero;
                    IsDeviceDirty = false;
                }

                if (_HostPinnedData != IntPtr.Zero)
                {
                    GPU.FreeHostPinned(_HostPinnedData);
                    _HostPinnedData = IntPtr.Zero;
                    IsHostPinnedDirty = false;
                }

                _HostData = null;
                IsHostDirty = false;
            }
        }

        public Image GetCopy()
        {
            return new Image(GetHostContinuousCopy(), Dims, IsFT, IsComplex, IsHalf) { PixelSize = PixelSize };
        }

        public Image GetCopyGPU()
        {
            return new Image(GetDevice(Intent.Read), Dims, IsFT, IsComplex, IsHalf) { PixelSize = PixelSize };
        }

        public void TransformValues(Func<float, float> f)
        {
            float[][] Data = GetHost(Intent.ReadWrite);
            foreach (var slice in Data)
                for (int i = 0; i < slice.Length; i++)
                    slice[i] = f(slice[i]);
        }

        public void TransformValues(Func<int, int, int, float, float> f)
        {
            float[][] Data = GetHost(Intent.ReadWrite);
            for (int z = 0; z < Dims.Z; z++)
                for (int y = 0; y < Dims.Y; y++)
                    for (int x = 0; x < Dims.X; x++)
                        Data[z][y * Dims.X + x] = f(x, y, z, Data[z][y * Dims.X + x]);
        }

        public Image AsHalf()
        {
            Image Result;

            if (!IsHalf)
            {
                Result = new Image(IntPtr.Zero, Dims, IsFT, IsComplex, true);
                GPU.SingleToHalf(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), ElementsReal);
            }
            else
            {
                Result = new Image(GetDevice(Intent.Read), Dims, IsFT, IsComplex, true);
            }

            return Result;
        }

        public Image AsSingle()
        {
            Image Result;

            if (IsHalf)
            {
                IntPtr Temp = GPU.MallocDevice(ElementsReal);
                GPU.OnMemoryChanged();
                GPU.HalfToSingle(GetDevice(Intent.Read), Temp, ElementsReal);

                Result = new Image(Temp, Dims, IsFT, IsComplex, false);
                GPU.FreeDevice(Temp);
                GPU.OnMemoryChanged();
            }
            else
            {
                Result = new Image(GetDevice(Intent.Read), Dims, IsFT, IsComplex, false);
            }

            return Result;
        }

        public Image AsSum3D()
        {
            if (IsComplex || IsHalf)
                throw new Exception("Data type not supported.");

            Image Result = new Image(IntPtr.Zero, new int3(1, 1, 1));
            GPU.Sum(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), (uint)ElementsReal, 1);

            return Result;
        }

        public Image AsSum2D()
        {
            if (IsComplex || IsHalf)
                throw new Exception("Data type not supported.");

            Image Result = new Image(IntPtr.Zero, new int3(Dims.Z, 1, 1));
            GPU.Sum(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), (uint)ElementsSliceReal, (uint)Dims.Z);

            return Result;
        }

        public Image AsSum1D()
        {
            if (IsComplex || IsHalf)
                throw new Exception("Data type not supported.");

            Image Result = new Image(IntPtr.Zero, new int3(Dims.Y * Dims.Z, 1, 1));
            GPU.Sum(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), (uint)ElementsLineReal, (uint)(Dims.Y * Dims.Z));

            return Result;
        }

        public Image AsRegion(int3 origin, int3 dimensions)
        {
            if (origin.X + dimensions.X >= Dims.X || 
                origin.Y + dimensions.Y >= Dims.Y || 
                origin.Z + dimensions.Z >= Dims.Z)
                throw new IndexOutOfRangeException();

            float[][] Source = GetHost(Intent.Read);
            float[][] Region = new float[dimensions.Z][];

            int3 RealSourceDimensions = DimsEffective;
            if (IsComplex)
                RealSourceDimensions.X *= 2;
            int3 RealDimensions = new int3((IsFT ? dimensions.X / 2 + 1 : dimensions.X) * (IsComplex ? 2 : 1),
                                           dimensions.Y,
                                           dimensions.Z);

            for (int z = 0; z < RealDimensions.Z; z++)
            {
                float[] SourceSlice = Source[z + origin.Z];
                float[] Slice = new float[RealDimensions.ElementsSlice()];

                unsafe
                {
                    fixed (float* SourceSlicePtr = SourceSlice)
                    fixed (float* SlicePtr = Slice)
                        for (int y = 0; y < RealDimensions.Y; y++)
                        {
                            int YOffset = y + origin.Y;
                            for (int x = 0; x < RealDimensions.X; x++)
                                SlicePtr[y * RealDimensions.X + x] = SourceSlicePtr[YOffset * RealSourceDimensions.X + x + origin.X];
                        }
                }

                Region[z] = Slice;
            }

            return new Image(Region, dimensions, IsFT, IsComplex, IsHalf);
        }

        public Image AsPadded(int2 dimensions)
        {
            if (IsHalf)
                throw new Exception("Half precision not supported for padding.");

            if (IsComplex != IsFT)
                throw new Exception("FT format can only have complex data for padding purposes.");

            if (IsFT && (new int2(Dims) < dimensions) == (new int2(Dims) > dimensions))
                throw new Exception("For FT padding/cropping, both dimensions must be either smaller, or bigger.");

            Image Padded = null;

            if (!IsComplex && !IsFT)
            {
                Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, Dims.Z), false, false, false);
                GPU.Pad(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
            }
            else if (IsComplex && IsFT)
            {
                Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, Dims.Z), true, true, false);
                if (dimensions > new int2(Dims))
                    GPU.PadFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                else
                    GPU.CropFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
            }

            return Padded;
        }

        public Image AsPadded(int3 dimensions)
        {
            if (IsHalf)
                throw new Exception("Half precision not supported for padding.");

            if (IsComplex != IsFT)
                throw new Exception("FT format can only have complex data for padding purposes.");

            if (IsFT && Dims < dimensions == Dims > dimensions)
                throw new Exception("For FT padding/cropping, both dimensions must be either smaller, or bigger.");

            Image Padded = null;

            if (!IsComplex && !IsFT)
            {
                Padded = new Image(IntPtr.Zero, dimensions, false, false, false);
                GPU.Pad(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
            }
            else if (IsComplex && IsFT)
            {
                Padded = new Image(IntPtr.Zero, dimensions, true, true, false);
                if (dimensions > Dims)
                    GPU.PadFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
                else
                    GPU.CropFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
            }

            return Padded;
        }

        public Image AsFFT(bool isvolume = false, int plan = 0)
        {
            if (IsHalf || IsComplex || IsFT)
                throw new Exception("Data format not supported.");

            Image FFT = new Image(IntPtr.Zero, Dims, true, true, false);
            GPU.FFT(GetDevice(Intent.Read), FFT.GetDevice(Intent.Write), isvolume ? Dims : Dims.Slice(), isvolume ? 1 : (uint)Dims.Z, plan);

            return FFT;
        }

        public Image AsFFT_CPU()
        {
            float[] Continuous = GetHostContinuousCopy();
            float[] Transformed = new float[Dims.ElementsFFT() * 2];

            lock (FFT_CPU_Sync)
                CPU.FFT_CPU(Continuous, Transformed, Dims, 16);

            return new Image(Helper.FromInterleaved2(Transformed), Dims, true);
        }

        public Image AsIFFT(bool isvolume = false, int plan = 0, bool normalize = false)
        {
            if (IsHalf || !IsComplex || !IsFT)
                throw new Exception("Data format not supported.");

            Image IFFT = new Image(IntPtr.Zero, Dims, false, false, false);
            GPU.IFFT(GetDevice(Intent.Read), IFFT.GetDevice(Intent.Write), isvolume ? Dims : Dims.Slice(), isvolume ? 1 : (uint)Dims.Z, plan, normalize);

            return IFFT;
        }

        public Image AsIFFT_CPU()
        {
            float[] Continuous = GetHostContinuousCopy();
            float[] Transformed = new float[Dims.Elements()];

            lock (FFT_CPU_Sync)
                CPU.IFFT_CPU(Continuous, Transformed, Dims, 16);

            return new Image(Transformed, Dims);
        }

        public Image AsMultipleRegions(int3[] origins, int2 dimensions)
        {
            Image Extracted = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, origins.Length), false, IsComplex, IsHalf);

            if (IsHalf)
                GPU.ExtractHalf(GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                Dims, new int3(dimensions),
                                Helper.ToInterleaved(origins),
                                (uint) origins.Length);
            else
                GPU.Extract(GetDevice(Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            Dims, new int3(dimensions),
                            Helper.ToInterleaved(origins),
                            (uint) origins.Length);

            return Extracted;
        }

        public Image AsReducedAlongZ()
        {
            Image Reduced = new Image(IntPtr.Zero, new int3(Dims.X, Dims.Y, 1), IsFT, IsComplex, IsHalf);

            if (IsHalf)
                GPU.ReduceMeanHalf(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)ElementsSliceReal, (uint)Dims.Z, 1);
            else
                GPU.ReduceMean(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)ElementsSliceReal, (uint)Dims.Z, 1);

            return Reduced;
        }

        public Image AsReducedAlongY()
        {
            Image Reduced = new Image(IntPtr.Zero, new int3(Dims.X, 1, Dims.Z), IsFT, IsComplex, IsHalf);

            if (IsHalf)
                GPU.ReduceMeanHalf(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)(DimsEffective.X * (IsComplex ? 2 : 1)), (uint)Dims.Y, (uint)Dims.Z);
            else
                GPU.ReduceMean(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)(DimsEffective.X * (IsComplex ? 2 : 1)), (uint)Dims.Y, (uint)Dims.Z);

            return Reduced;
        }

        public Image AsPolar(uint innerradius = 0, uint exclusiveouterradius = 0)
        {
            if (IsHalf || IsComplex)
                throw new Exception("Cannot transform fp16 or complex image.");

            if (exclusiveouterradius == 0)
                exclusiveouterradius = (uint)Dims.X / 2;
            exclusiveouterradius = (uint)Math.Min(Dims.X / 2, (int)exclusiveouterradius);
            uint R = exclusiveouterradius - innerradius;

            if (IsFT)
            {
                Image Result = new Image(IntPtr.Zero, new int3((int)R, Dims.Y, Dims.Z));
                GPU.Cart2PolarFFT(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), DimsSlice, innerradius, exclusiveouterradius, (uint) Dims.Z);
                return Result;
            }
            else
            {
                Image Result = new Image(IntPtr.Zero, new int3((int)R, Dims.Y * 2, Dims.Z));
                GPU.Cart2Polar(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), DimsSlice, innerradius, exclusiveouterradius, (uint)Dims.Z);
                return Result;
            }
        }

        public Image AsAmplitudes()
        {
            if (IsHalf || !IsComplex)
                throw new Exception("Data type not supported.");

            Image Amplitudes = new Image(IntPtr.Zero, Dims, IsFT, false, false);
            GPU.Amplitudes(GetDevice(Intent.Read), Amplitudes.GetDevice(Intent.Write), ElementsComplex);

            return Amplitudes;
        }

        public Image AsReal()
        {
            if (!IsComplex)
                throw new Exception("Data must be complex.");

            float[][] Real = new float[Dims.Z][];
            float[][] Complex = GetHost(Intent.Read);
            for (int z = 0; z < Real.Length; z++)
            {
                float[] ComplexSlice = Complex[z];
                float[] RealSlice = new float[ComplexSlice.Length / 2];
                for (int i = 0; i < RealSlice.Length; i++)
                    RealSlice[i] = ComplexSlice[i * 2];

                Real[z] = RealSlice;
            }

            return new Image(Real, Dims, IsFT, false, IsHalf);
        }

        public Image AsImaginary()
        {
            if (!IsComplex)
                throw new Exception("Data must be complex.");

            float[][] Imaginary = new float[Dims.Z][];
            float[][] Complex = GetHost(Intent.Read);
            for (int z = 0; z < Imaginary.Length; z++)
            {
                float[] ComplexSlice = Complex[z];
                float[] ImaginarySlice = new float[ComplexSlice.Length / 2];
                for (int i = 0; i < ImaginarySlice.Length; i++)
                    ImaginarySlice[i] = ComplexSlice[i * 2 + 1];

                Imaginary[z] = ImaginarySlice;
            }

            return new Image(Imaginary, Dims, IsFT, false, IsHalf);
        }

        public Image AsScaledMassive(int2 newSliceDims, int planForw = 0, int planBack = 0)
        {
            int3 Scaled = new int3(newSliceDims.X, newSliceDims.Y, Dims.Z);
            Image Output = new Image(Scaled);
            
            for (int z = 0; z < Dims.Z; z++)
            {
                GPU.Scale(GetDeviceSlice(z, Intent.Read),
                          Output.GetDeviceSlice(z, Intent.Write),
                          Dims.Slice(),
                          new int3(newSliceDims),
                          1,
                          planForw,
                          planBack);
            }

            return Output;
        }

        public Image AsScaled(int2 newSliceDims, int planForw = 0, int planBack = 0)
        {
            int3 Scaled = new int3(newSliceDims.X, newSliceDims.Y, Dims.Z);
            Image Output = new Image(IntPtr.Zero, Scaled);

            GPU.Scale(GetDevice(Intent.Read),
                      Output.GetDevice(Intent.Write),
                      new int3(DimsSlice),
                      new int3(newSliceDims),
                      (uint)Dims.Z,
                      planForw,
                      planBack);

            return Output;
        }

        public Image AsScaled(int3 newDims, int planForw = 0, int planBack = 0)
        {
            Image Output = new Image(IntPtr.Zero, newDims);

            GPU.Scale(GetDevice(Intent.Read),
                      Output.GetDevice(Intent.Write),
                      new int3(Dims),
                      new int3(newDims),
                      1,
                      planForw,
                      planBack);

            return Output;
        }

        public Image AsShiftedVolume(float3 shift)
        {
            if (IsComplex)
            {
                if (IsHalf)
                    throw new Exception("Cannot shift complex fp16 volume.");
                if (!IsFT)
                    throw new Exception("Volume must be in FFTW format");

                Image Result = new Image(IntPtr.Zero, Dims, true, true);

                GPU.ShiftStackFT(GetDevice(Intent.Read),
                                 Result.GetDevice(Intent.Write),
                                 Dims,
                                 Helper.ToInterleaved(new[] { shift }),
                                 1);

                return Result;
            }
            else
            {
                if (IsHalf)
                    throw new Exception("Cannot shift fp16 volume.");

                Image Result = new Image(IntPtr.Zero, Dims);

                GPU.ShiftStack(GetDevice(Intent.Read),
                               Result.GetDevice(Intent.Write),
                               DimsEffective,
                               Helper.ToInterleaved(new[] { shift }),
                               1);

                return Result;
            }
        }

        public Image AsProjections(float3[] angles, int2 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, angles.Length), true, true);

            GPU.ProjectForward(GetDevice(Intent.Read),
                               Projections.GetDevice(Intent.Write),
                               Dims,
                               dimsprojection,
                               Helper.ToInterleaved(angles),
                               supersample,
                               (uint)angles.Length);

            return Projections;
        }

        public Image AsProjections(float3[] angles, float3[] shifts, float[] globalweights, int2 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, angles.Length), true, true);

            GPU.ProjectForwardShifted(GetDevice(Intent.Read),
                                      Projections.GetDevice(Intent.Write),
                                      Dims,
                                      dimsprojection,
                                      Helper.ToInterleaved(angles),
                                      Helper.ToInterleaved(shifts),
                                      globalweights,
                                      supersample,
                                      (uint)angles.Length);

            return Projections;
        }

        public Image AsProjections3D(float3[] angles, int3 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, dimsprojection.Z * angles.Length), true, true);

            GPU.ProjectForward3D(GetDevice(Intent.Read),
                                 Projections.GetDevice(Intent.Write),
                                 Dims,
                                 dimsprojection,
                                 Helper.ToInterleaved(angles),
                                 supersample,
                                 (uint)angles.Length);

            return Projections;
        }

        public Image AsProjections3D(float3[] angles, float3[] shifts, float[] globalweights, int3 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, dimsprojection.Z * angles.Length), true, true);

            GPU.ProjectForward3DShifted(GetDevice(Intent.Read),
                                        Projections.GetDevice(Intent.Write),
                                        Dims,
                                        dimsprojection,
                                        Helper.ToInterleaved(angles),
                                        Helper.ToInterleaved(shifts),
                                        globalweights,
                                        supersample,
                                        (uint)angles.Length);

            return Projections;
        }

        public Image AsAnisotropyCorrected(int2 dimsscaled, float majorpixel, float minorpixel, float majorangle, uint supersample)
        {
            Image Corrected = new Image(IntPtr.Zero, new int3(dimsscaled.X, dimsscaled.Y, Dims.Z));

            GPU.CorrectMagAnisotropy(GetDevice(Intent.Read),
                                     DimsSlice,
                                     Corrected.GetDevice(Intent.Write),
                                     dimsscaled,
                                     majorpixel,
                                     minorpixel,
                                     majorangle,
                                     supersample,
                                     (uint)Dims.Z);

            return Corrected;
        }

        public Image AsDistanceMap(int maxDistance = -1)
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("No other formats than fp32 non-FT realspace supported.");

            Image Distance = new Image(IntPtr.Zero, Dims);

            GPU.DistanceMap(GetDevice(Intent.Read), Distance.GetDevice(Intent.Write), Dims, maxDistance <= 0 ? Dims.X : maxDistance);

            return Distance;

            /*float[] Distance = new float[ElementsReal];
            float[] Intensity = GetHostContinuousCopy();

            if (maxDistance < 0)
                maxDistance = Dims.X;

            for (int i = 0; i < Distance.Length; i++)
                Distance[i] = Intensity[i] == 1 ? 0 : maxDistance;

            float[] Steps = new float[27];
            int[] IDOffset = new int[27];
            for (int dz = -1, i = 0; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++, i++)
                    {
                        Steps[i] = (float)Math.Sqrt(dz * dz + dy * dy + dx * dx);
                        IDOffset[i] = (dz * Dims.Y + dy) * Dims.X + dx;
                    }

            unsafe
            {
                for (int d = 0; d < maxDistance; d++)
                {
                    bool Updated = false;
                    float[] NewDistance = new float[Distance.Length];
                    Array.Copy(Distance, NewDistance, Distance.Length);

                    // For every voxel
                    Parallel.For(0, 16, p =>
                    {
                        bool ThreadUpdated = false;

                        fixed (float* DistancePtr = Distance)
                        fixed (float* NewDistancePtr = NewDistance)
                        {
                            for (int z = 1 + p; z < Dims.Z - 1; z += 16)
                                for (int y = 1; y < Dims.Y - 1; y++)
                                    for (int x = 1; x < Dims.X - 1; x++)
                                    {
                                        int CenterID = (z * Dims.Y + y) * Dims.X + x;
                                        float NearestNeighbor = DistancePtr[CenterID];

                                        // Check neighbors
                                        for (int i = 0; i < 27; i++)
                                        {
                                            float Neighbor = DistancePtr[CenterID + IDOffset[i]];
                                            if (DistancePtr[CenterID + IDOffset[i]] + Steps[i] < NearestNeighbor)
                                                NearestNeighbor = Neighbor + Steps[i];
                                        }

                                        if (NearestNeighbor < DistancePtr[CenterID])
                                        {
                                            NewDistancePtr[CenterID] = NearestNeighbor;
                                            ThreadUpdated = true;
                                        }
                                    }
                        }

                        lock (NewDistance)
                            Updated |= ThreadUpdated;
                    });

                    Distance = NewDistance;

                    if (!Updated)
                        break;
                }
            }

            return new Image(Distance, Dims);*/
        }

        public Image AsDistanceMapExact(int maxDistance)
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("No other formats than fp32 non-FT realspace supported.");

            Image Result = new Image(Dims);
            GPU.DistanceMapExact(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), Dims, maxDistance);

            return Result;
        }

        public Image AsDilatedMask(float distance)
        {
            Image DistanceMap = AsDistanceMap((int)(distance * 2));
            float[] DistanceMapData = DistanceMap.GetHostContinuousCopy();

            DistanceMapData = DistanceMapData.Select(v => v <= distance + 1e-4f ? 1f : 0f).ToArray();
            DistanceMap.Dispose();

            return new Image(DistanceMapData, Dims);
        }

        public float3 AsCenterOfMass()
        {
            double VX = 0, VY = 0, VZ = 0;
            double Samples = 0;
            float[] ContData = GetHostContinuousCopy();

            for (int z = 0, i = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++, i++)
                    {
                        float Val = ContData[i];
                        VX += x * Val;
                        VY += y * Val;
                        VZ += z * Val;
                        Samples += Val;
                    }
                }
            }

            return new float3((float)(VX / Samples), (float)(VY / Samples), (float)(VZ / Samples));
        }

        public Image AsSymmetrized(string symmetry, int paddingFactor = 2)
        {
            if (Dims.Z <= 1)
                throw new Exception("Must be a volume.");

            Image Padded = AsPadded(Dims * paddingFactor);
            Padded.RemapToFT(true);
            Image PaddedFT = Padded.AsFFT(true);
            Padded.Dispose();

            GPU.SymmetrizeFT(PaddedFT.GetDevice(Intent.ReadWrite), PaddedFT.Dims, symmetry);

            Padded = PaddedFT.AsIFFT(true, -1, true);
            PaddedFT.Dispose();
            Padded.RemapFromFT(true);

            Image Unpadded = Padded.AsPadded(Dims);
            Padded.Dispose();

            return Unpadded;
        }

        public Image AsSpectrumFlattened(bool isVolume = true, float nyquistLowpass = 1f, int spectrumLength = -1)
        {
            Image FT = isVolume ? AsFFT_CPU() : AsFFT(false);
            Image FTAmp = FT.AsAmplitudes();

            int SpectrumLength = Math.Min(Dims.X, Dims.Z > 1 ? Math.Min(Dims.Y, Dims.Z) : Dims.Y) / 2;
            if (spectrumLength > 0)
                SpectrumLength = Math.Min(spectrumLength, SpectrumLength);

            float[] Spectrum = new float[SpectrumLength];
            float[] Samples = new float[SpectrumLength];

            float[][] FTAmpData = FTAmp.GetHost(Intent.ReadWrite);
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        if (r > nyquistLowpass)
                            continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        Val *= Val;

                        Spectrum[(int)r] += WeightLow * Val;
                        Samples[(int)r] += WeightLow;

                        if ((int)r < SpectrumLength - 1)
                        {
                            Spectrum[(int)r + 1] += WeightHigh * Val;
                            Samples[(int)r + 1] += WeightHigh;
                        }
                    }
                }
            }

            for (int i = 0; i < Spectrum.Length; i++)
                Spectrum[i] = (float)Math.Sqrt(Spectrum[i]) / Math.Max(1e-5f, Samples[i]);

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz) * SpectrumLength;
                        r = Math.Min(SpectrumLength - 2, r);

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = Spectrum[(int)r] * WeightLow + Spectrum[(int)r + 1] * WeightHigh;

                        FTAmpData[z][y * (Dims.X / 2 + 1) + x] = Val > 1e-10f ? 1f / (float)(Val) : 0;
                    }
                }
            }

            FT.Multiply(FTAmp);
            FTAmp.Dispose();

            Image IFT = isVolume ? FT.AsIFFT_CPU() : FT.AsIFFT(false);
            FT.Dispose();

            return IFT;
        }
        public float[] AsAmplitudes1D(bool isVolume = true, float nyquistLowpass = 1f, int spectrumLength = -1)
        {
            if (IsComplex || IsHalf)
                throw new Exception("Not implemented for complex or half data.");
            if (IsFT)
                throw new DimensionMismatchException();

            Image FT = isVolume ? AsFFT_CPU() : AsFFT(false);
            Image FTAmp = FT.AsAmplitudes();
            FT.Dispose();

            int SpectrumLength = Math.Min(Dims.X, isVolume ? Math.Min(Dims.Y, Dims.Z) : Dims.Y) / 2;
            if (spectrumLength > 0)
                SpectrumLength = Math.Min(spectrumLength, SpectrumLength);

            float[] Spectrum = new float[SpectrumLength];
            float[] Samples = new float[SpectrumLength];

            float[][] FTAmpData = FTAmp.GetHost(Intent.ReadWrite);
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        if (r > nyquistLowpass)
                            continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        Val *= Val;

                        Spectrum[(int)r] += WeightLow * Val;
                        Samples[(int)r] += WeightLow;

                        if ((int)r < SpectrumLength - 1)
                        {
                            Spectrum[(int)r + 1] += WeightHigh * Val;
                            Samples[(int)r + 1] += WeightHigh;
                        }
                    }
                }
            }

            for (int i = 0; i < Spectrum.Length; i++)
                Spectrum[i] = (float)Math.Sqrt(Spectrum[i]) / Math.Max(1e-5f, Samples[i]);

            FTAmp.Dispose();

            return Spectrum;
        }

        public Image AsConvolvedGaussian(float sigma, bool normalize = true)
        {
            sigma = -1f / (sigma * sigma * 2);

            Image Gaussian = new Image(Dims);
            float[][] GaussianData = Gaussian.GetHost(Intent.Write);
            double GaussianSum = 0;
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z - Dims.Z / 2;
                zz *= zz;
                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y - Dims.Y / 2;
                    yy *= yy;
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int xx = x - Dims.X / 2;
                        xx *= xx;

                        float R2 = xx + yy + zz;
                        double G = Math.Exp(R2 * sigma);
                        if (G < 1e-4)
                            continue;

                        GaussianSum += G;
                        GaussianData[z][y * Dims.X + x] = (float)G;
                    }
                }
            }

            Gaussian.RemapToFT(true);
            Image GaussianFT = Gaussian.AsFFT(true);
            Gaussian.Dispose();

            if (normalize)
                GaussianFT.Multiply(1f / (float)GaussianSum);

            Image ThisFT = AsFFT(true);

            ThisFT.MultiplyConj(GaussianFT);
            GaussianFT.Dispose();

            Image Convolved = ThisFT.AsIFFT(true, 0, true);
            ThisFT.Dispose();

            return Convolved;
        }

        public Image AsConvolvedRaisedCosine(float innerRadius, float falloff, bool normalize = true)
        {
            Image Cosine = new Image(Dims);
            float[][] CosineData = Cosine.GetHost(Intent.Write);
            double CosineSum = 0;
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z - Dims.Z / 2;
                zz *= zz;
                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y - Dims.Y / 2;
                    yy *= yy;
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int xx = x - Dims.X / 2;
                        xx *= xx;

                        float R = (float)Math.Sqrt(xx + yy + zz);
                        double C = Math.Cos(Math.Max(0, Math.Min(falloff, R - innerRadius)) / falloff * Math.PI) * 0.5 + 0.5;
                        
                        CosineSum += C;
                        CosineData[z][y * Dims.X + x] = (float)C;
                    }
                }
            }

            Cosine.RemapToFT(true);
            Image CosineFT = Cosine.AsFFT(true);
            Cosine.Dispose();

            if (normalize)
                CosineFT.Multiply(1f / (float)CosineSum);

            Image ThisFT = AsFFT(true);

            ThisFT.MultiplyConj(CosineFT);
            CosineFT.Dispose();

            Image Convolved = ThisFT.AsIFFT(true, 0, true);
            ThisFT.Dispose();

            return Convolved;
        }

        public Image AsFlippedX()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Flipped = new Image(Dims);

            float[][] Data = GetHost(Intent.Read);
            float[][] FlippedData = Flipped.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int xx = Dims.X - 1 - x;

                        FlippedData[z][y * Dims.X + x] = Data[z][y * Dims.X + xx];
                    }
                }
            }

            return Flipped;
        }

        public Image AsFlippedY()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Flipped = new Image(Dims);

            float[][] Data = GetHost(Intent.Read);
            float[][] FlippedData = Flipped.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = Dims.Y - 1 - y;
                    for (int x = 0; x < Dims.X; x++)
                    {
                        FlippedData[z][y * Dims.X + x] = Data[z][yy * Dims.X + x];
                    }
                }
            }

            return Flipped;
        }

        public Image AsTransposed()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Transposed = new Image(new int3(Dims.Y, Dims.X, Dims.Z));

            float[][] Data = GetHost(Intent.Read);
            float[][] TransposedData = Transposed.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        TransposedData[z][x * Dims.Y + y] = Data[z][y * Dims.X + x];
                    }
                }
            }

            return Transposed;
        }

        public void RemapToFT(bool isvolume = false)
        {
            if (!IsFT && IsComplex)
                throw new Exception("Complex remap only supported for FT layout.");

            int3 WorkDims = isvolume ? Dims : Dims.Slice();
            uint WorkBatch = isvolume ? 1 : (uint)Dims.Z;

            if (IsComplex)
                GPU.RemapToFTComplex(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            else
            {
                if (IsFT)
                    GPU.RemapToFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
                else
                    GPU.RemapFullToFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            }
        }

        public void RemapFromFT(bool isvolume = false)
        {
            if (!IsFT && IsComplex)
                throw new Exception("Complex remap only supported for FT layout.");

            int3 WorkDims = isvolume ? Dims : Dims.Slice();
            uint WorkBatch = isvolume ? 1 : (uint)Dims.Z;

            if (IsComplex)
                GPU.RemapFromFTComplex(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            else
            {
                if (IsFT)
                    GPU.RemapFromFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
                else
                    GPU.RemapFullFromFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            }
        }

        public void Min(float value)
        {
            GPU.MinScalar(GetDevice(Intent.Read), GetDevice(Intent.Write), value, (uint)ElementsReal);
        }

        public void Max(float value)
        {
            GPU.MaxScalar(GetDevice(Intent.Read), GetDevice(Intent.Write), value, (uint)ElementsReal);
        }

        public void Xray(float ndevs)
        {
            if (IsComplex || IsHalf)
                throw new Exception("Complex and half are not supported.");

            for (int i = 0; i < Dims.Z; i++)
                GPU.Xray(new IntPtr((long)GetDevice(Intent.Read) + DimsEffective.ElementsSlice() * i * sizeof (float)),
                         new IntPtr((long)GetDevice(Intent.Write) + DimsEffective.ElementsSlice() * i * sizeof(float)),
                         ndevs,
                         new int2(DimsEffective),
                         1);
        }

        public void Fill(float val)
        {
            GPU.ValueFill(GetDevice(Intent.Write), ElementsReal, val);
        }

        public void Sign()
        {
            if (IsHalf)
                throw new Exception("Does not work for fp16.");

            GPU.Sign(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Cos()
        {
            if (IsHalf || IsComplex)
                throw new Exception("Does not work for fp16 or complex.");

            GPU.Cos(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Sin()
        {
            if (IsHalf || IsComplex)
                throw new Exception("Does not work for fp16 or complex.");

            GPU.Sin(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Abs()
        {
            if (IsHalf)
                throw new Exception("Does not work for fp16.");

            GPU.Abs(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        private void Add(Image summands, uint elements, uint batch)
        {
            if (ElementsReal != elements * batch ||
                summands.ElementsReal != elements ||
                //IsFT != summands.IsFT ||
                IsComplex != summands.IsComplex)
                throw new DimensionMismatchException();

            if (IsHalf && summands.IsHalf)
            {
                GPU.AddToSlicesHalf(GetDevice(Intent.Read), summands.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else if (!IsHalf && !summands.IsHalf)
            {
                GPU.AddToSlices(GetDevice(Intent.Read), summands.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else
            {
                Image ThisSingle = AsSingle();
                Image SummandsSingle = summands.AsSingle();

                GPU.AddToSlices(ThisSingle.GetDevice(Intent.Read), SummandsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                if (IsHalf)
                    GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                else
                    GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                ThisSingle.Dispose();
                SummandsSingle.Dispose();
            }
        }

        public void Add(Image summands)
        {
            Add(summands, (uint) ElementsReal, 1);
        }

        public void AddToSlices(Image summands)
        {
            Add(summands, (uint) ElementsSliceReal, (uint) Dims.Z);
        }

        public void AddToLines(Image summands)
        {
            Add(summands, (uint) ElementsLineReal, (uint) (Dims.Y * Dims.Z));
        }

        public void Add(float scalar)
        {
            GPU.AddScalar(GetDevice(Intent.Read), scalar, GetDevice(Intent.Write), ElementsReal);
        }

        private void Subtract(Image subtrahends, uint elements, uint batch)
        {
            if (ElementsReal != elements * batch ||
                subtrahends.ElementsReal != elements ||
                IsFT != subtrahends.IsFT ||
                IsComplex != subtrahends.IsComplex)
                throw new DimensionMismatchException();

            if (IsHalf && subtrahends.IsHalf)
            {
                GPU.SubtractFromSlicesHalf(GetDevice(Intent.Read), subtrahends.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else if (!IsHalf && !subtrahends.IsHalf)
            {
                GPU.SubtractFromSlices(GetDevice(Intent.Read), subtrahends.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else
            {
                Image ThisSingle = AsSingle();
                Image SubtrahendsSingle = subtrahends.AsSingle();

                GPU.SubtractFromSlices(ThisSingle.GetDevice(Intent.Read), SubtrahendsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                if (IsHalf)
                    GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                else
                    GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                ThisSingle.Dispose();
                SubtrahendsSingle.Dispose();
            }
        }

        public void Subtract(Image subtrahends)
        {
            Subtract(subtrahends, (uint) ElementsReal, 1);
        }

        public void SubtractFromSlices(Image subtrahends)
        {
            Subtract(subtrahends, (uint) ElementsSliceReal, (uint) Dims.Z);
        }

        public void SubtractFromLines(Image subtrahends)
        {
            Subtract(subtrahends, (uint) ElementsLineReal, (uint) (Dims.Y * Dims.Z));
        }

        public void Multiply(float multiplicator)
        {
            GPU.MultiplyByScalar(GetDevice(Intent.Read),
                                 GetDevice(Intent.Write),
                                 multiplicator,
                                 ElementsReal);
        }

        public void Multiply(float[] scalarMultiplicators)
        {
            if (scalarMultiplicators.Length != Dims.Z)
                throw new DimensionMismatchException("Number of scalar multiplicators must equal number of slices.");

            IntPtr d_multiplicators = GPU.MallocDeviceFromHost(scalarMultiplicators, scalarMultiplicators.Length);

            GPU.MultiplyByScalars(GetDevice(Intent.Read),
                                  GetDevice(Intent.Write),
                                  d_multiplicators,
                                  ElementsSliceReal,
                                  (uint)Dims.Z);

            GPU.FreeDevice(d_multiplicators);
        }

        private void Multiply(Image multiplicators, uint elements, uint batch)
        {
            if (ElementsComplex != elements * batch ||
                multiplicators.ElementsComplex != elements ||
                //IsFT != multiplicators.IsFT ||
                (multiplicators.IsComplex && !IsComplex))
                throw new DimensionMismatchException();

            if (!IsComplex)
            {
                if (IsHalf && multiplicators.IsHalf)
                {
                    GPU.MultiplySlicesHalf(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                }
                else if (!IsHalf && !multiplicators.IsHalf)
                {
                    GPU.MultiplySlices(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                }
                else
                {
                    Image ThisSingle = AsSingle();
                    Image MultiplicatorsSingle = multiplicators.AsSingle();

                    GPU.MultiplySlices(ThisSingle.GetDevice(Intent.Read), MultiplicatorsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                    if (IsHalf)
                        GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                    else
                        GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                    ThisSingle.Dispose();
                    MultiplicatorsSingle.Dispose();
                }
            }
            else
            {
                if (IsHalf)
                    throw new Exception("Complex multiplication not supported for fp16.");

                if (!multiplicators.IsComplex)
                    GPU.MultiplyComplexSlicesByScalar(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                else
                    GPU.MultiplyComplexSlicesByComplex(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
        }

        public void Multiply(Image multiplicators)
        {
            Multiply(multiplicators, (uint) ElementsComplex, 1);
        }

        public void MultiplySlices(Image multiplicators)
        {
            Multiply(multiplicators, (uint) ElementsSliceComplex, (uint) Dims.Z);
        }

        public void MultiplyLines(Image multiplicators)
        {
            Multiply(multiplicators, (uint) ElementsLineComplex, (uint) (Dims.Y * Dims.Z));
        }
        
        private void MultiplyConj(Image multiplicators, uint elements, uint batch)
        {
            if (ElementsComplex != elements * batch ||
                multiplicators.ElementsComplex != elements ||
                !multiplicators.IsComplex || 
                !IsComplex)
                throw new DimensionMismatchException();
            
            if (IsHalf)
                throw new Exception("Complex multiplication not supported for fp16.");

            GPU.MultiplyComplexSlicesByComplexConj(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
        }

        public void MultiplyConj(Image multiplicators)
        {
            MultiplyConj(multiplicators, (uint)ElementsComplex, 1);
        }

        public void MultiplyConjSlices(Image multiplicators)
        {
            MultiplyConj(multiplicators, (uint)ElementsSliceComplex, (uint)Dims.Z);
        }

        public void MultiplyConjLines(Image multiplicators)
        {
            MultiplyConj(multiplicators, (uint)ElementsLineComplex, (uint)(Dims.Y * Dims.Z));
        }

        private void Divide(Image divisors, uint elements, uint batch)
        {
            if (ElementsComplex != elements * batch ||
                divisors.ElementsComplex != elements ||
                //IsFT != divisors.IsFT ||
                divisors.IsComplex)
                throw new DimensionMismatchException();

            if (!IsComplex)
            {
                if (!IsHalf && !divisors.IsHalf)
                {
                    GPU.DivideSlices(GetDevice(Intent.Read), divisors.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                }
                else
                {
                    Image ThisSingle = AsSingle();
                    Image DivisorsSingle = divisors.AsSingle();

                    GPU.DivideSlices(ThisSingle.GetDevice(Intent.Read), DivisorsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                    if (IsHalf)
                        GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                    else
                        GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                    ThisSingle.Dispose();
                    DivisorsSingle.Dispose();
                }
            }
            else
            {
                if (IsHalf)
                    throw new Exception("Complex division not supported for fp16.");
                GPU.DivideComplexSlicesByScalar(GetDevice(Intent.Read), divisors.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
        }

        public void Divide(Image divisors)
        {
            Divide(divisors, (uint)ElementsComplex, 1);
        }

        public void DivideSlices(Image divisors)
        {
            Divide(divisors, (uint)ElementsSliceComplex, (uint)Dims.Z);
        }

        public void DivideLines(Image divisors)
        {
            Divide(divisors, (uint)ElementsLineComplex, (uint)(Dims.Y * Dims.Z));
        }

        public void ShiftSlices(float3[] shifts)
        {
            if (IsComplex)
            {
                if (IsHalf)
                    throw new Exception("Cannot shift complex half image.");
                if (!IsFT)
                    throw new Exception("Image must be in FFTW format");

                GPU.ShiftStackFT(GetDevice(Intent.Read),
                                 GetDevice(Intent.Write),
                                 Dims.Slice(),
                                 Helper.ToInterleaved(shifts),
                                 (uint)Dims.Z);
            }
            else
            {
                IntPtr Data;
                if (!IsHalf)
                    Data = GetDevice(Intent.ReadWrite);
                else
                {
                    Data = GPU.MallocDevice(ElementsReal);
                    GPU.OnMemoryChanged();
                    GPU.HalfToSingle(GetDevice(Intent.Read), Data, ElementsReal);
                }

                GPU.ShiftStack(Data,
                               Data,
                               DimsEffective.Slice(),
                               Helper.ToInterleaved(shifts),
                               (uint)Dims.Z);

                if (IsHalf)
                {
                    GPU.SingleToHalf(Data, GetDevice(Intent.Write), ElementsReal);
                    GPU.FreeDevice(Data);
                    GPU.OnMemoryChanged();
                }
            }
        }

        public void ShiftSlicesMassive(float3[] shifts)
        {
            if (IsComplex)
                throw new Exception("Cannot shift complex image.");

            IntPtr Data;
            if (!IsHalf)
                Data = GetDevice(Intent.ReadWrite);
            else
            {
                Data = GPU.MallocDevice(ElementsReal);
                GPU.OnMemoryChanged();
                GPU.HalfToSingle(GetDevice(Intent.Read), Data, ElementsReal);
            }

            GPU.ShiftStackMassive(Data,
                                  Data,
                                  DimsEffective.Slice(),
                                  Helper.ToInterleaved(shifts),
                                  (uint)Dims.Z);

            if (IsHalf)
            {
                GPU.SingleToHalf(Data, GetDevice(Intent.Write), ElementsReal);
                GPU.FreeDevice(Data);
                GPU.OnMemoryChanged();
            }
        }

        public void Bandpass(float nyquistLow, float nyquistHigh, bool isVolume, float nyquistsoftedge = 0)
        {
            if (IsComplex || IsHalf || IsFT)
                throw new Exception("Bandpass only works on single precision, real data");

            GPU.Bandpass(GetDevice(Intent.Read), GetDevice(Intent.Write), isVolume ? Dims : Dims.Slice(), nyquistLow, nyquistHigh, nyquistsoftedge, isVolume ? 1 : (uint)Dims.Z);
        }

        public int3[] GetLocalPeaks(int localExtent, float threshold)
        {
            int[] NPeaks = new int[1];

            IntPtr PeaksPtr = GPU.LocalPeaks(GetDevice(Intent.Read), NPeaks, Dims, localExtent, threshold);

            if (NPeaks[0] > 0)
            {
                int[] Peaks = new int[NPeaks[0] * 3];
                Marshal.Copy(PeaksPtr, Peaks, 0, Peaks.Length);

                CPU.HostFree(PeaksPtr);

                return Helper.FromInterleaved3(Peaks);
            }
            else
            {
                return new int3[0];
            }
        }

        public void Binarize(float threshold)
        {
            foreach (var slice in GetHost(Intent.ReadWrite))
                for (int i = 0; i < slice.Length; i++)
                    slice[i] = slice[i] >= threshold ? 1 : 0;
        }

        public void SubtractMeanPlane()
        {
            if (Dims.Z > 1)
                throw new Exception("Does not work for volumes or stacks.");

            float[] MicData = GetHost(Intent.ReadWrite)[0];
            float[] MicPlane = MathHelper.FitAndGeneratePlane(MicData, DimsSlice);
            for (int i = 0; i < MicData.Length; i++)
                MicData[i] -= MicPlane[i];
        }

        public bool IsSameAs(Image other, float error = 0.001f)
        {
            float[] ThisMemory = GetHostContinuousCopy();
            float[] OtherMemory = other.GetHostContinuousCopy();

            if (ThisMemory.Length != OtherMemory.Length)
                return false;

            for (int i = 0; i < ThisMemory.Length; i++)
            {
                float ThisVal = ThisMemory[i];
                float OtherVal = OtherMemory[i];
                if (ThisVal != OtherVal)
                {
                    if (OtherVal == 0)
                        return false;

                    float Diff = Math.Abs((ThisVal - OtherVal) / OtherVal);
                    if (Diff > error)
                        return false;
                }
            }

            return true;
        }

        public override string ToString()
        {
            return Dims.ToString() + ", " + PixelSize + " A/px";
        }
    }
    
    [Flags]
    public enum Intent
    {
        Read = 1 << 0,
        Write = 1 << 1,
        ReadWrite = Read | Write
    }
}
