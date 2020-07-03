using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        private static readonly object GlobalSync = new object();
        public static object FFT_CPU_Sync = new object();

        private readonly int ObjectID = -1;
        private readonly StackTrace ObjectCreationLocation = null;
        private static int LifetimeObjectCounter = 0;
        private static readonly List<int> LifetimeObjectIDs = new List<int>();
        private static readonly List<Image> LifetimeObjects = new List<Image>();
        private static readonly bool EnableObjectLogging = false;

        private static readonly HashSet<Image> OnDeviceObjects = new HashSet<Image>();
        public static void FreeDeviceAll()
        {
            Image[] Objects = OnDeviceObjects.ToArray();
            foreach (var item in Objects)
                item.FreeDevice();
        }
        
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

                    lock (GlobalSync)
                        OnDeviceObjects.Add(this);

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

            if (data.Length != dims.Z || data[0].Length != ElementsSliceReal)
                throw new DimensionMismatchException();

            _HostData = data.ToArray();
            IsHostDirty = true;

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
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

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
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

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
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

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
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

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
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

                lock (GlobalSync)
                    OnDeviceObjects.Add(this);
            }
            else
            {
                _HostPinnedData = GPU.MallocHostPinned(ElementsReal);
                IsHostPinnedDirty = true;
            }

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
        }

        ~Image()
        {
            Dispose();
        }

        public static Image FromFile(string path, int2 headerlessSliceDims, int headerlessOffset, Type headerlessType, int layer = -1, Stream stream = null)
        {
            MapHeader Header = MapHeader.ReadFromFile(path, headerlessSliceDims, headerlessOffset, headerlessType);
            float[][] Data = IOHelper.ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, layer < 0 ? null : new[] { layer }, stream);
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

                    lock (GlobalSync)
                        OnDeviceObjects.Remove(this);
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
                Parallel.For(0, Dims.Z, z =>
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
            string FlipYEnvVar = Environment.GetEnvironmentVariable("WARP_DONT_FLIPY");
            bool DoFlipY = string.IsNullOrEmpty(FlipYEnvVar);

            int Width = (int)ElementsLineReal;
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

            float[][] Data = GetHost(Intent.Read);
            int PageLength = Data[0].Length;
            unsafe
            {
                using (Tiff output = Tiff.Open(path, "w"))
                {
                    float[] DataFlipped = DoFlipY ? new float[Data[0].Length] : null;
                    byte[] BytesData = new byte[ElementsSliceReal * BytesPerSample];

                    for (int z = 0; z < Dims.Z; z++)
                    {
                        if (DoFlipY)
                        {
                            // Annoyingly, flip Y axis to adhere to MRC convention

                            fixed (float* DataFlippedPtr = DataFlipped)
                            fixed (float* DataPtr = Data[z])
                            {
                                for (int y = 0; y < Height; y++)
                                {
                                    int YOffset = y * Width;
                                    int YOffsetFlipped = (Height - 1 - y) * Width;

                                    for (int x = 0; x < Width; x++)
                                        DataFlippedPtr[YOffset + x] = DataPtr[YOffsetFlipped + x];
                                }
                            }
                        }
                        else
                        {
                            DataFlipped = Data[z];
                        }

                        fixed (byte* BytesPtr = BytesData)
                        fixed (float* DataPtr = DataFlipped)
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

                        int page = z;
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
                                output.WriteScanline(Helper.Subset(BytesData, j * Width * BytesPerSample, (j + 1) * Width * BytesPerSample), j);

                            output.WriteDirectory();
                            output.FlushData();
                        }
                    }
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

                    lock (GlobalSync)
                        OnDeviceObjects.Remove(this);
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

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    if (LifetimeObjectIDs.Contains(ObjectID))
                        LifetimeObjectIDs.Remove(ObjectID);
                    if (LifetimeObjects.Contains(this))
                        LifetimeObjects.Remove(this);
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
            int Width = IsFT ? Dims.X / 2 + 1 : Dims.X;
            if (IsComplex)
                Width *= 2;

            for (int z = 0; z < Dims.Z; z++)
                for (int y = 0; y < Dims.Y; y++)
                    for (int x = 0; x < Width; x++)
                        Data[z][y * Width + x] = f(x, y, z, Data[z][y * Width + x]);
        }

        public float GetInterpolatedValue(float3 pos)
        {
            float3 Weights = new float3(pos.X - (float)Math.Floor(pos.X),
                                        pos.Y - (float)Math.Floor(pos.Y),
                                        pos.Z - (float)Math.Floor(pos.Z));

            float[][] Data = GetHost(Intent.Read);

            int3 Pos0 = new int3(Math.Max(0, Math.Min(Dims.X - 1, (int)pos.X)),
                                 Math.Max(0, Math.Min(Dims.Y - 1, (int)pos.Y)),
                                 Math.Max(0, Math.Min(Dims.Z - 1, (int)pos.Z)));
            int3 Pos1 = new int3(Math.Min(Dims.X - 1, Pos0.X + 1),
                                 Math.Min(Dims.Y - 1, Pos0.Y + 1),
                                 Math.Min(Dims.Z - 1, Pos0.Z + 1));

            if (Dims.Z == 1)
            {
                float v00 = Data[0][Pos0.Y * Dims.X + Pos0.X];
                float v01 = Data[0][Pos0.Y * Dims.X + Pos1.X];
                float v10 = Data[0][Pos1.Y * Dims.X + Pos0.X];
                float v11 = Data[0][Pos1.Y * Dims.X + Pos1.X];

                float v0 = MathHelper.Lerp(v00, v01, Weights.X);
                float v1 = MathHelper.Lerp(v10, v11, Weights.X);

                return MathHelper.Lerp(v0, v1, Weights.Y);
            }
            else
            {
                float v000 = Data[Pos0.Z][Pos0.Y * Dims.X + Pos0.X];
                float v001 = Data[Pos0.Z][Pos0.Y * Dims.X + Pos1.X];
                float v010 = Data[Pos0.Z][Pos1.Y * Dims.X + Pos0.X];
                float v011 = Data[Pos0.Z][Pos1.Y * Dims.X + Pos1.X];

                float v100 = Data[Pos1.Z][Pos0.Y * Dims.X + Pos0.X];
                float v101 = Data[Pos1.Z][Pos0.Y * Dims.X + Pos1.X];
                float v110 = Data[Pos1.Z][Pos1.Y * Dims.X + Pos0.X];
                float v111 = Data[Pos1.Z][Pos1.Y * Dims.X + Pos1.X];

                float v00 = MathHelper.Lerp(v000, v001, Weights.X);
                float v01 = MathHelper.Lerp(v010, v011, Weights.X);
                float v10 = MathHelper.Lerp(v100, v101, Weights.X);
                float v11 = MathHelper.Lerp(v110, v111, Weights.X);

                float v0 = MathHelper.Lerp(v00, v01, Weights.Y);
                float v1 = MathHelper.Lerp(v10, v11, Weights.Y);

                return MathHelper.Lerp(v0, v1, Weights.Z);
            }
        }

        #region As...

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
            if (origin.X + dimensions.X > Dims.X || 
                origin.Y + dimensions.Y > Dims.Y || 
                origin.Z + dimensions.Z > Dims.Z)
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

        public Image AsPadded(int2 dimensions, bool isDecentered = false)
        {
            if (IsHalf)
                throw new Exception("Half precision not supported for padding.");

            if (IsComplex != IsFT)
                throw new Exception("FT format can only have complex data for padding purposes.");

            if (dimensions == new int2(Dims))
                return GetCopy();

            if (IsFT && (new int2(Dims) < dimensions) == (new int2(Dims) > dimensions))
                throw new Exception("For FT padding/cropping, both dimensions must be either smaller, or bigger.");

            Image Padded = null;

            if (!IsComplex && !IsFT)
            {
                Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, Dims.Z), false, false, false);
                if (isDecentered)
                {
                    if (dimensions.X > Dims.X && dimensions.Y > Dims.Y)
                        GPU.PadFTFull(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                    else if (dimensions.X < Dims.X && dimensions.Y < Dims.Y)
                        GPU.CropFTFull(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                    else
                        throw new Exception("All new dimensions must be either bigger or smaller than old ones.");
                }
                else
                {
                    GPU.Pad(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                }
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

        public Image AsPaddedClamped(int2 dimensions)
        {
            if (IsHalf || IsComplex || IsFT)
                throw new Exception("Wrong data format, only real-valued non-FT supported.");

            Image Padded = null;

            Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, Dims.Z), false, false, false);
            GPU.PadClamped(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);

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
                          planBack,
                          IntPtr.Zero,
                          IntPtr.Zero);
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
                      planBack,
                      IntPtr.Zero,
                      IntPtr.Zero);

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
                      planBack,
                      IntPtr.Zero,
                      IntPtr.Zero);

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
            float[][] Values = GetHost(Intent.Read);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0, i = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++, i++)
                    {
                        float Val = Values[z][i];
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
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        //Val *= Val;

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
                Spectrum[i] = Spectrum[i] / Math.Max(1e-5f, Samples[i]);

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
                        {
                            FTAmpData[z][y * (Dims.X / 2 + 1) + x] = 0;
                            continue;
                        }

                        r *= SpectrumLength;
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
            if (IsHalf)
                throw new Exception("Not implemented for half data.");
            //if (IsFT)
            //    throw new DimensionMismatchException();

            Image FT = IsFT ? this : (isVolume ? AsFFT_CPU() : AsFFT(false));
            Image FTAmp = (IsFT && !IsComplex) ? this : FT.AsAmplitudes();
            FTAmp.FreeDevice();
            if (!IsFT)
                FT.Dispose();

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
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x]; ;
                        if (Math.Abs(Val) < 1e-10f)
                            continue;

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
                Spectrum[i] = Spectrum[i] / Math.Max(1e-5f, Samples[i]);

            return Spectrum;
        }

        public float[] AsAmplitudeVariance1D(bool isVolume = true, float nyquistLowpass = 1f, int spectrumLength = -1)
        {
            if (IsHalf)
                throw new Exception("Not implemented for half data.");
            //if (IsFT)
            //    throw new DimensionMismatchException();

            Image FT = IsFT ? this : (isVolume ? AsFFT_CPU() : AsFFT(false));
            Image FTAmp = (IsFT && !IsComplex) ? this : FT.AsAmplitudes();
            FTAmp.FreeDevice();
            if (!IsFT)
                FT.Dispose();

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
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        if (Math.Abs(Val) < 1e-10f)
                            continue;
                        //Val *= Val;

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
                Spectrum[i] = Spectrum[i] / Math.Max(1e-5f, Samples[i]);

            float[] Variance = new float[SpectrumLength];
            float[] VarianceSamples = new float[SpectrumLength];

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
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        if (Math.Abs(Val) < 1e-10f)
                            continue;

                        float Mean = Spectrum[(int)r] * WeightLow + Spectrum[Math.Min(Spectrum.Length - 1, (int)r + 1)] * WeightHigh;
                        float Diff = Val - Mean;
                        Diff *= Diff;

                        Variance[(int)r] += WeightLow * Diff;
                        VarianceSamples[(int)r] += WeightLow;

                        if ((int)r < SpectrumLength - 1)
                        {
                            Variance[(int)r + 1] += WeightHigh * Diff;
                            VarianceSamples[(int)r + 1] += WeightHigh;
                        }
                    }
                }
            }

            for (int i = 0; i < Spectrum.Length; i++)
                Variance[i] = Variance[i] / Math.Max(1e-5f, VarianceSamples[i]);

            return Variance;
        }

        public Image AsSpectrumMultiplied(bool isVolume, float[] spectrumMultiplicators)
        {
            Image FT = AsFFT(isVolume);
            Image FTAmp = FT.AsAmplitudes();
            float[][] FTAmpData = FTAmp.GetHost(Intent.ReadWrite);

            int SpectrumLength = spectrumMultiplicators.Length;

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
                        if (r > 1)
                        {
                            FTAmpData[z][y * (Dims.X / 2 + 1) + x] = 0;
                            continue;
                        }

                        r *= SpectrumLength;
                        r = Math.Min(SpectrumLength - 2, r);

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = spectrumMultiplicators[(int)r] * WeightLow + spectrumMultiplicators[(int)r + 1] * WeightHigh;

                        FTAmpData[z][y * (Dims.X / 2 + 1) + x] = Val;
                    }
                }
            }

            FT.Multiply(FTAmp);
            FTAmp.Dispose();

            Image IFT = FT.AsIFFT(isVolume, 0, true);
            FT.Dispose();

            return IFT;
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

        public Image AsFlippedZ()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Flipped = new Image(Dims);

            float[][] Data = GetHost(Intent.Read);
            float[][] FlippedData = Flipped.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = Dims.Z - 1 - z;
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        FlippedData[z][y * Dims.X + x] = Data[zz][y * Dims.X + x];
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

        public Image AsExpandedBinary(int expandDistance)
        {
            Image BinaryExpanded = AsDistanceMapExact(expandDistance);
            BinaryExpanded.Multiply(-1);
            BinaryExpanded.Binarize(-expandDistance + 1e-6f);

            return BinaryExpanded;
        }

        public Image AsExpandedSmooth(int expandDistance)
        {
            Image ExpandedSmooth = AsDistanceMapExact(expandDistance);
            ExpandedSmooth.Multiply((float)Math.PI / expandDistance);
            ExpandedSmooth.Cos();
            ExpandedSmooth.Add(1);
            ExpandedSmooth.Multiply(0.5f);

            return ExpandedSmooth;
        }

        public Image AsComplex()
        {
            Image Result = new Image(Dims, IsFT, true);
            Result.Fill(new float2(1, 0));
            Result.Multiply(this);

            return Result;
        }

        public Image AsSliceXY(int z)
        {
            Image Slice = new Image(GetHost(Intent.Read)[z], new int3(Dims.X, Dims.Y, 1), IsFT, IsComplex);
            return Slice;
        }

        public Image AsSliceXZ(int y)
        {
            int Width = IsFT ? Dims.X / 2 + 1 : Dims.X;

            Image Slice = new Image(new int3(Width, Dims.Z, 1));
            float[] SliceData = Slice.GetHost(Intent.Write)[0];
            float[][] Data = GetHost(Intent.Read);
            for (int z = 0; z < Dims.Z; z++)
                for (int x = 0; x < Width; x++)
                    SliceData[z * Width + x] = Data[z][y * Width + x];

            return Slice;
        }

        public Image AsSliceYZ(int x)
        {
            int Width = IsFT ? Dims.X / 2 + 1 : Dims.X;

            Image Slice = new Image(new int3(Dims.Y, Dims.Z, 1));
            float[] SliceData = Slice.GetHost(Intent.Write)[0];
            float[][] Data = GetHost(Intent.Read);
            for (int z = 0; z < Dims.Z; z++)
                for (int y = 0; y < Dims.Y; y++)
                    SliceData[z * Dims.Y + y] = Data[z][y * Width + x];

            return Slice;
        }

        #endregion

        #region In-place

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

        public void Fill(float2 val)
        {
            GPU.ValueFillComplex(GetDevice(Intent.Write), ElementsComplex, val);
        }

        public void Sign()
        {
            if (IsHalf)
                throw new Exception("Does not work for fp16.");

            GPU.Sign(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Sqrt()
        {
            if (IsHalf)
                throw new Exception("Does not work for fp16.");
            if (IsComplex)
                throw new Exception("Does not work for complex data.");

            GPU.Sqrt(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
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

        public void MaskSpherically(float diameter, float softEdge, bool isVolume)
        {
            GPU.SphereMask(GetDevice(Intent.Read),
                           GetDevice(Intent.Write),
                           isVolume ? Dims : Dims.Slice(),
                           diameter / 2,
                           softEdge,
                           false,
                           isVolume ? 1 : (uint)Dims.Z);
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

        public void Binarize(float threshold)
        {
            foreach (var slice in GetHost(Intent.ReadWrite))
                for (int i = 0; i < slice.Length; i++)
                    slice[i] = slice[i] >= threshold ? 1 : 0;
        }

        public void SubtractMeanGrid(int2 gridDims)
        {
            //if (Dims.Z > 1)
            //    throw new Exception("Does not work for volumes or stacks.");

            foreach (var MicData in GetHost(Intent.ReadWrite))
                if (gridDims.Elements() <= 1)
                    MathHelper.FitAndSubtractPlane(MicData, DimsSlice);
                else
                    MathHelper.FitAndSubtractGrid(MicData, DimsSlice, gridDims);
        }

        public void Taper(float distance, bool isVolume = false)
        {
            if (!isVolume)
            {
                Image MaskTaper = new Image(Dims.Slice());
                MaskTaper.TransformValues((x, y, z, v) =>
                {
                    float dx = 0, dy = 0;

                    if (x < distance)
                        dx = distance - x;
                    else if (x > Dims.X - 1 - distance)
                        dx = x - (Dims.X - 1 - distance);

                    if (y < distance)
                        dy = distance - y;
                    else if (y > Dims.Y - 1 - distance)
                        dy = y - (Dims.Y - 1 - distance);

                    float R = (float)Math.Sqrt(dx * dx + dy * dy) / distance;

                    return Math.Max(0.01f, (float)Math.Cos(Math.Max(0, Math.Min(1, R)) * Math.PI) * 0.5f + 0.5f);
                });

                MultiplySlices(MaskTaper);
                MaskTaper.Dispose();
            }
            else
            {
                Image MaskTaper = new Image(Dims);
                MaskTaper.TransformValues((x, y, z, v) =>
                {
                    float dx = 0, dy = 0, dz = 0;

                    if (x < distance)
                        dx = distance - x;
                    else if (x > Dims.X - 1 - distance)
                        dx = x - (Dims.X - 1 - distance);

                    if (y < distance)
                        dy = distance - y;
                    else if (y > Dims.Y - 1 - distance)
                        dy = y - (Dims.Y - 1 - distance);

                    if (z < distance)
                        dz = distance - z;
                    else if (z > Dims.Z - 1 - distance)
                        dz = z - (Dims.Z - 1 - distance);

                    float R = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz) / distance;

                    return Math.Max(0, (float)Math.Cos(Math.Max(0, Math.Min(1, R)) * Math.PI) * 0.5f + 0.5f);
                });

                Multiply(MaskTaper);
                MaskTaper.Dispose();
            }
        }

        public void Normalize()
        {
            if (IsHalf || IsComplex)
                throw new Exception("Wrong format, only real-valued input supported.");

            GPU.Normalize(GetDevice(Intent.Read),
                          GetDevice(Intent.Write),
                          (uint)ElementsSliceReal,
                          (uint)Dims.Z);
        }

        public void Symmetrize(string sym)
        {
            Symmetry Sym = new Symmetry(sym);
            Matrix3[] Rotations = Sym.GetRotationMatrices();

            RemapToFT(true);

            Image FT = AsFFT(true);
            FT.FreeDevice();
            float[][] FTData = FT.GetHost(Intent.Read);
            Image FTSym = new Image(Dims, true, true);
            float[][] FTSymData = FTSym.GetHost(Intent.Write);

            int DimX = Dims.X / 2 + 1;
            float R2 = Dims.X * Dims.X / 4f;

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z <= Dims.Z / 2 ? z : z - Dims.Z;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y <= Dims.Y / 2 ? y : y - Dims.Y;

                    for (int x = 0; x < DimX; x++)
                    {
                        int xx = x;
                        float3 PosCentered = new float3(xx, yy, zz);
                        if (PosCentered.LengthSq() >= R2)
                            continue;

                        float2 VSum = new float2(0, 0);

                        foreach (var rotation in Rotations)
                        {
                            float3 PosRotated = rotation * PosCentered;
                            bool IsFlipped = false;
                            if (PosRotated.X < 0)
                            {
                                PosRotated *= -1;
                                IsFlipped = true;
                            }

                            int X0 = (int)Math.Floor(PosRotated.X);
                            int X1 = X0 + 1;
                            float XInterp = PosRotated.X - X0;
                            int Y0 = (int)Math.Floor(PosRotated.Y);
                            int Y1 = Y0 + 1;
                            float YInterp = PosRotated.Y - Y0;
                            int Z0 = (int)Math.Floor(PosRotated.Z);
                            int Z1 = Z0 + 1;
                            float ZInterp = PosRotated.Z - Z0;

                            X0 = Math.Max(0, Math.Min(DimX - 1, X0));
                            X1 = Math.Max(0, Math.Min(DimX - 1, X1));
                            Y0 = Math.Max(0, Math.Min(Dims.Y - 1, Y0 >= 0 ? Y0 : Y0 + Dims.Y));
                            Y1 = Math.Max(0, Math.Min(Dims.Y - 1, Y1 >= 0 ? Y1 : Y1 + Dims.Y));
                            Z0 = Math.Max(0, Math.Min(Dims.Z - 1, Z0 >= 0 ? Z0 : Z0 + Dims.Z));
                            Z1 = Math.Max(0, Math.Min(Dims.Z - 1, Z1 >= 0 ? Z1 : Z1 + Dims.Z));

                            {
                                float v000 = FTData[Z0][(Y0 * DimX + X0) * 2 + 0];
                                float v001 = FTData[Z0][(Y0 * DimX + X1) * 2 + 0];
                                float v010 = FTData[Z0][(Y1 * DimX + X0) * 2 + 0];
                                float v011 = FTData[Z0][(Y1 * DimX + X1) * 2 + 0];

                                float v100 = FTData[Z1][(Y0 * DimX + X0) * 2 + 0];
                                float v101 = FTData[Z1][(Y0 * DimX + X1) * 2 + 0];
                                float v110 = FTData[Z1][(Y1 * DimX + X0) * 2 + 0];
                                float v111 = FTData[Z1][(Y1 * DimX + X1) * 2 + 0];

                                float v00 = MathHelper.Lerp(v000, v001, XInterp);
                                float v01 = MathHelper.Lerp(v010, v011, XInterp);
                                float v10 = MathHelper.Lerp(v100, v101, XInterp);
                                float v11 = MathHelper.Lerp(v110, v111, XInterp);

                                float v0 = MathHelper.Lerp(v00, v01, YInterp);
                                float v1 = MathHelper.Lerp(v10, v11, YInterp);

                                float v = MathHelper.Lerp(v0, v1, ZInterp);

                                VSum.X += v;
                            }

                            {
                                float v000 = FTData[Z0][(Y0 * DimX + X0) * 2 + 1];
                                float v001 = FTData[Z0][(Y0 * DimX + X1) * 2 + 1];
                                float v010 = FTData[Z0][(Y1 * DimX + X0) * 2 + 1];
                                float v011 = FTData[Z0][(Y1 * DimX + X1) * 2 + 1];

                                float v100 = FTData[Z1][(Y0 * DimX + X0) * 2 + 1];
                                float v101 = FTData[Z1][(Y0 * DimX + X1) * 2 + 1];
                                float v110 = FTData[Z1][(Y1 * DimX + X0) * 2 + 1];
                                float v111 = FTData[Z1][(Y1 * DimX + X1) * 2 + 1];

                                float v00 = MathHelper.Lerp(v000, v001, XInterp);
                                float v01 = MathHelper.Lerp(v010, v011, XInterp);
                                float v10 = MathHelper.Lerp(v100, v101, XInterp);
                                float v11 = MathHelper.Lerp(v110, v111, XInterp);

                                float v0 = MathHelper.Lerp(v00, v01, YInterp);
                                float v1 = MathHelper.Lerp(v10, v11, YInterp);

                                float v = MathHelper.Lerp(v0, v1, ZInterp);

                                VSum.Y += IsFlipped ? -v : v;
                            }
                        }

                        FTSymData[z][(y * DimX + x) * 2 + 0] = VSum.X / Rotations.Length;
                        FTSymData[z][(y * DimX + x) * 2 + 1] = VSum.Y / Rotations.Length;
                    }
                }
            }

            //FT.IsSameAs(FTSym, 1e-5f);

            FT.Dispose();

            GPU.IFFT(FTSym.GetDevice(Intent.Read),
                     this.GetDevice(Intent.Write),
                     Dims,
                     1,
                     -1,
                     true);
            FTSym.Dispose();

            RemapFromFT(true);
            FreeDevice();
        }

        #endregion

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

        public void RealspaceProject(float3[] angles, Image result, int supersample)
        {
            GPU.RealspaceProjectForward(GetDevice(Intent.Read),
                                        Dims,
                                        result.GetDevice(Intent.ReadWrite),
                                        new int2(result.Dims),
                                        supersample,
                                        Helper.ToInterleaved(angles),
                                        angles.Length);
        }

        public void RealspaceBackproject(Image projections, float3[] angles, int supersample, bool normalizesamples = true)
        {
            GPU.RealspaceProjectBackward(GetDevice(Intent.Write),
                                         Dims,
                                         projections.GetDevice(Intent.Read),
                                         new int2(projections.Dims),
                                         supersample,
                                         Helper.ToInterleaved(angles),
                                         normalizesamples,
                                         angles.Length);
        }

        public (int[] ComponentIndices, int[] NeighborhoodIndices)[] GetConnectedComponents(int neighborhoodExtent = 0, int[] labelsBuffer = null)
        {
            if (Dims.Z > 1)
                throw new Exception("No volumetric data supported!");

            float[] PixelData = GetHost(Intent.Read)[0];

            List<List<int>> Components = new List<List<int>>();
            List<List<int>> Neighborhoods = new List<List<int>>();
            if (labelsBuffer == null)
                labelsBuffer = new int[PixelData.Length];

            for (int i = 0; i < labelsBuffer.Length; i++)
                labelsBuffer[i] = -1;

            List<int> Peaks = new List<int>();
            for (int i = 0; i < PixelData.Length; i++)
                if (PixelData[i] != 0)
                    Peaks.Add(i);

            Queue<int> Expansion = new Queue<int>(100);


            foreach (var peak in Peaks)
            {
                if (labelsBuffer[peak] >= 0)
                    continue;

                #region Connected component

                List<int> Component = new List<int>() { peak };
                int CN = Components.Count;

                labelsBuffer[peak] = CN;
                Expansion.Clear();
                Expansion.Enqueue(peak);

                while (Expansion.Count > 0)
                {
                    int PosElement = Expansion.Dequeue();
                    int2 pos = new int2(PosElement % Dims.X, PosElement / Dims.X);

                    if (pos.X > 0 && PixelData[PosElement - 1] != 0 && labelsBuffer[PosElement - 1] < 0)
                    {
                        labelsBuffer[PosElement - 1] = CN;
                        Component.Add(PosElement + (-1));
                        Expansion.Enqueue(PosElement + (-1));
                    }
                    if (pos.X < Dims.X - 1 && PixelData[PosElement + 1] > 0 && labelsBuffer[PosElement + 1] < 0)
                    {
                        labelsBuffer[PosElement + 1] = CN;
                        Component.Add(PosElement + (1));
                        Expansion.Enqueue(PosElement + (1));
                    }

                    if (pos.Y > 0 && PixelData[PosElement - Dims.X] > 0 && labelsBuffer[PosElement - Dims.X] < 0)
                    {
                        labelsBuffer[PosElement - Dims.X] = CN;
                        Component.Add(PosElement + (-Dims.X));
                        Expansion.Enqueue(PosElement + (-Dims.X));
                    }
                    if (pos.Y < Dims.Y - 1 && PixelData[PosElement + Dims.X] > 0 && labelsBuffer[PosElement + Dims.X] < 0)
                    {
                        labelsBuffer[PosElement + Dims.X] = CN;
                        Component.Add(PosElement + (Dims.X));
                        Expansion.Enqueue(PosElement + (Dims.X));
                    }
                }

                Components.Add(Component);

                #endregion

                #region Optional neighborhood around component

                List<int> CurrentFrontier = new List<int>(Component);
                List<int> NextFrontier = new List<int>();
                List<int> Neighborhood = new List<int>();
                int NN = -(CN + 2);

                for (int iexpansion = 0; iexpansion < neighborhoodExtent; iexpansion++)
                {
                    foreach (int PosElement in CurrentFrontier)
                    {
                        int2 pos = new int2(PosElement % Dims.X, PosElement / Dims.X);

                        if (pos.X > 0 && PixelData[PosElement - 1] == 0 && labelsBuffer[PosElement - 1] != NN)
                        {
                            labelsBuffer[PosElement - 1] = NN;
                            Neighborhood.Add(PosElement + (-1));
                            NextFrontier.Add(PosElement + (-1));
                        }
                        if (pos.X < Dims.X - 1 && PixelData[PosElement + 1] == 0 && labelsBuffer[PosElement + 1] != NN)
                        {
                            labelsBuffer[PosElement + 1] = NN;
                            Neighborhood.Add(PosElement + (1));
                            NextFrontier.Add(PosElement + (1));
                        }

                        if (pos.Y > 0 && PixelData[PosElement - Dims.X] == 0 && labelsBuffer[PosElement - Dims.X] != NN)
                        {
                            labelsBuffer[PosElement - Dims.X] = NN;
                            Neighborhood.Add(PosElement + (-Dims.X));
                            NextFrontier.Add(PosElement + (-Dims.X));
                        }
                        if (pos.Y < Dims.Y - 1 && PixelData[PosElement + Dims.X] == 0 && labelsBuffer[PosElement + Dims.X] != NN)
                        {
                            labelsBuffer[PosElement + Dims.X] = NN;
                            Neighborhood.Add(PosElement + (Dims.X));
                            NextFrontier.Add(PosElement + (Dims.X));
                        }
                    }

                    CurrentFrontier = NextFrontier;
                    NextFrontier = new List<int>();
                }

                Neighborhoods.Add(Neighborhood);

                #endregion
            }

            return Helper.ArrayOfFunction(i => (Components[i].ToArray(), Neighborhoods[i].ToArray()), Components.Count);
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
            return Dims.ToString() + ", " + 
                   (IsFT ? "FT, " : "normal, ") + 
                   (IsComplex ? "complex, " : "real, ") + 
                   PixelSize + " A/px, " + 
                   "ID = " + ObjectID +
                   (_DeviceData == IntPtr.Zero ? "" : ", on device");
        }

        public static Image Stack(Image[] images)
        {
            int SumZ = images.Sum(i => i.Dims.Z);

            Image Stacked = new Image(new int3(images[0].Dims.X, images[0].Dims.Y, SumZ), images[0].IsFT, images[0].IsComplex);
            float[][] StackedData = Stacked.GetHost(Intent.Write);

            int OffsetZ = 0;
            foreach (var image in images)
            {
                float[][] ImageData = image.GetHost(Intent.Read);
                for (int i = 0; i < ImageData.Length; i++)
                    Array.Copy(ImageData[i], 0, StackedData[i + OffsetZ], 0, ImageData[i].Length);
                OffsetZ += ImageData.Length;
            }

            return Stacked;
        }

        public static Image ReconstructSIRT(Image data, float3[] angles, int3 dimsrec, int supersample, int niterations, Image residuals = null)
        {
            int2 DimsProj = new int2(data.Dims);
            int2 DimsProjSuper = DimsProj * supersample;

            int PlanForwUp = GPU.CreateFFTPlan(new int3(DimsProj), (uint)angles.Length);
            int PlanBackUp = GPU.CreateIFFTPlan(new int3(DimsProjSuper), (uint)angles.Length);

            int PlanForwDown = GPU.CreateFFTPlan(new int3(DimsProjSuper), (uint)angles.Length);
            int PlanBackDown = GPU.CreateIFFTPlan(new int3(DimsProj), (uint)angles.Length);

            Image Projections = new Image(new int3(DimsProj.X, DimsProj.Y, angles.Length));
            Image ProjectionsSuper = new Image(new int3(DimsProjSuper.X, DimsProjSuper.Y, angles.Length));

            Image ProjectionSamples = new Image(Projections.Dims);
            
            Image VolReconstruction = new Image(dimsrec);
            Image VolCorrection = new Image(dimsrec);
            VolCorrection.Fill(1f);

            // Figure out number of samples per projection pixel
            {
                VolCorrection.RealspaceProject(angles, ProjectionsSuper, supersample);

                GPU.Scale(ProjectionsSuper.GetDevice(Intent.Read),
                            ProjectionSamples.GetDevice(Intent.Write),
                            new int3(DimsProjSuper),
                            new int3(DimsProj),
                            (uint)angles.Length,
                            PlanForwDown,
                            PlanBackDown,
                            IntPtr.Zero,
                            IntPtr.Zero);

                ProjectionSamples.Max(1f);
                //ProjectionSamples.WriteMRC("d_samples.mrc", true);
            }

            // Supersample data and backproject to initialize volume
            {
                GPU.Scale(data.GetDevice(Intent.Read),
                          ProjectionsSuper.GetDevice(Intent.Write),
                          new int3(DimsProj),
                          new int3(DimsProjSuper),
                          (uint)angles.Length,
                          PlanForwUp,
                          PlanBackUp,
                          IntPtr.Zero,
                          IntPtr.Zero);

                //ProjectionsSuper.WriteMRC("d_datasuper.mrc", true);

                VolReconstruction.RealspaceBackproject(ProjectionsSuper, angles, supersample);
            }

            for (int i = 0; i < niterations; i++)
            {
                VolReconstruction.RealspaceProject(angles, ProjectionsSuper, supersample);
                
                GPU.Scale(ProjectionsSuper.GetDevice(Intent.Read),
                            Projections.GetDevice(Intent.Write),
                            new int3(DimsProjSuper),
                            new int3(DimsProj),
                            (uint)angles.Length,
                            PlanForwDown,
                            PlanBackDown,
                            IntPtr.Zero,
                            IntPtr.Zero);
                //Projections.WriteMRC("d_projections.mrc", true);

                GPU.SubtractFromSlices(data.GetDevice(Intent.Read),
                                       Projections.GetDevice(Intent.Read),
                                       Projections.GetDevice(Intent.Write),
                                       Projections.ElementsReal,
                                       1);

                if (i == niterations - 1 && residuals != null)
                    GPU.CopyDeviceToDevice(Projections.GetDevice(Intent.Read),
                                           residuals.GetDevice(Intent.Write),
                                           Projections.ElementsReal);

                Projections.Divide(ProjectionSamples);
                //Projections.WriteMRC("d_correction2d.mrc", true);
                Projections.Taper(8);

                GPU.Scale(Projections.GetDevice(Intent.Read),
                          ProjectionsSuper.GetDevice(Intent.Write),
                          new int3(DimsProj),
                          new int3(DimsProjSuper),
                          (uint)angles.Length,
                          PlanForwUp,
                          PlanBackUp,
                          IntPtr.Zero,
                          IntPtr.Zero);

                VolCorrection.RealspaceBackproject(ProjectionsSuper, angles, supersample);
                //VolCorrection.WriteMRC("d_correction3d.mrc", true);

                VolReconstruction.Add(VolCorrection);
                //VolReconstruction.WriteMRC($"d_reconstruction_{i:D3}.mrc", true);
            }

            //VolReconstruction.WriteMRC($"d_reconstruction_final.mrc", true);

            VolCorrection.Dispose();
            ProjectionSamples.Dispose();
            ProjectionsSuper.Dispose();
            Projections.Dispose();

            GPU.DestroyFFTPlan(PlanForwUp);
            GPU.DestroyFFTPlan(PlanForwDown);
            GPU.DestroyFFTPlan(PlanBackUp);
            GPU.DestroyFFTPlan(PlanBackDown);

            return VolReconstruction;
        }

        public static void PrintObjectIDs()
        {
            lock (GlobalSync)
            {
                for (int i = 0; i < LifetimeObjects.Count; i++)
                {
                    Debug.WriteLine(LifetimeObjects[i].ToString());
                    Debug.WriteLine(LifetimeObjects[i].ObjectCreationLocation.ToString() + "\n");
                }
            }
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
