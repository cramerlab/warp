using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;
using Warp.Headers;

namespace Warp.Tools
{
    public static class IOHelper
    {
        public static Stream OpenWithBigBuffer(string path)
        {
            return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 22, FileOptions.SequentialScan);
        }

        public static Stream CreateWithBigBuffer(string path)
        {
            return new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write, 1 << 22);
        }

        public static int3 GetMapDimensions(string path)
        {
            int3 Dims = new int3(1, 1, 1);
            FileInfo Info = new FileInfo(path);

            using (BinaryReader Reader = new BinaryReader(OpenWithBigBuffer(path)))
            {
                if (Info.Extension.ToLower() == ".mrc" || Info.Extension.ToLower() == ".mrcs")
                {
                    HeaderMRC Header = new HeaderMRC(Reader);
                    Dims = Header.Dimensions;
                }
                else if (Info.Extension.ToLower() == ".em")
                {
                    HeaderEM Header = new HeaderEM(Reader);
                    Dims = Header.Dimensions;
                }
                else
                    throw new Exception("Format not supported.");
            }

            return Dims;
        }

        public static float[][] ReadMapFloat(string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, int layer = -1, Stream stream = null)
        {
            try
            {
                return ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, false, layer, stream);
            }
            catch
            {
                return ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, true, layer, stream);
            }
        }

        public static float[][] ReadMapFloat(string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, bool isBigEndian, int layer = -1, Stream stream = null)
        {
            MapHeader Header = null;
            Type ValueType = null;
            float[][] Data;

            if (MapHeader.GetHeaderType(path) != typeof(HeaderTiff))
                using (BinaryReader Reader = isBigEndian ? new BinaryReaderBE(OpenWithBigBuffer(path)) : new BinaryReader(OpenWithBigBuffer(path)))
                {
                    Header = MapHeader.ReadFromFile(Reader, path, headerlessSliceDims, headerlessOffset, headerlessType);
                    ValueType = Header.GetValueType();
                    Data = new float[layer < 0 ? Header.Dimensions.Z : 1][];

                    for (int z = 0; z < Data.Length; z++)
                    {
                        if (layer >= 0)
                            Reader.BaseStream.Seek((long)Header.Dimensions.ElementsSlice() * (long)ImageFormatsHelper.SizeOf(ValueType) * layer, SeekOrigin.Current);

                        byte[] Bytes = Reader.ReadBytes((int)Header.Dimensions.ElementsSlice() * (int)ImageFormatsHelper.SizeOf(ValueType));
                        Data[z] = new float[(int)Header.Dimensions.ElementsSlice()];

                        if (isBigEndian)
                        {
                            if (ValueType == typeof(short) || ValueType == typeof(ushort))
                            {
                                for (int i = 0; i < Bytes.Length / 2; i++)
                                    Array.Reverse(Bytes, i * 2, 2);
                            }
                            else if (ValueType == typeof(int) || ValueType == typeof(float))
                            {
                                for (int i = 0; i < Bytes.Length / 4; i++)
                                    Array.Reverse(Bytes, i * 4, 4);
                            }
                            else if (ValueType == typeof(double))
                            {
                                for (int i = 0; i < Bytes.Length / 8; i++)
                                    Array.Reverse(Bytes, i * 8, 8);
                            }
                        }

                        unsafe
                        {
                            int Elements = (int)Header.Dimensions.ElementsSlice();

                            fixed (byte* BytesPtr = Bytes)
                            fixed (float* DataPtr = Data[z])
                            {
                                float* DataP = DataPtr;

                                if (ValueType == typeof(byte))
                                {
                                    byte* BytesP = BytesPtr;
                                    for (int i = 0; i < Elements; i++)
                                        *DataP++ = (float)*BytesP++;
                                }
                                else if (ValueType == typeof(short))
                                {
                                    short* BytesP = (short*)BytesPtr;
                                    for (int i = 0; i < Elements; i++)
                                        *DataP++ = (float)*BytesP++;
                                }
                                else if (ValueType == typeof(ushort))
                                {
                                    ushort* BytesP = (ushort*)BytesPtr;
                                    for (int i = 0; i < Elements; i++)
                                        *DataP++ = (float)*BytesP++;
                                }
                                else if (ValueType == typeof(int))
                                {
                                    int* BytesP = (int*)BytesPtr;
                                    for (int i = 0; i < Elements; i++)
                                        *DataP++ = (float)*BytesP++;
                                }
                                else if (ValueType == typeof(float))
                                {
                                    float* BytesP = (float*)BytesPtr;
                                    for (int i = 0; i < Elements; i++)
                                        *DataP++ = *BytesP++;
                                }
                                else if (ValueType == typeof(double))
                                {
                                    double* BytesP = (double*)BytesPtr;
                                    for (int i = 0; i < Elements; i++)
                                        *DataP++ = (float)*BytesP++;
                                }
                            }
                        }
                    }
                }
            else
            {
                Header = MapHeader.ReadFromFile(null, path, headerlessSliceDims, headerlessOffset, headerlessType, stream);
                Data = ((HeaderTiff)Header).ReadData(stream, layer);
            }

            return Data;
        }

        public static float[] ReadSmallMapFloat(string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType)
        {
            float[] Data;

            using (BinaryReader Reader = new BinaryReader(OpenWithBigBuffer(path)))
            {
                MapHeader Header = MapHeader.ReadFromFile(Reader, path, headerlessSliceDims, headerlessOffset, headerlessType);
                Type ValueType = Header.GetValueType();
                Data = new float[Header.Dimensions.Elements()];
                
                byte[] Bytes = Reader.ReadBytes((int)Header.Dimensions.Elements() * (int)ImageFormatsHelper.SizeOf(ValueType));

                unsafe
                {
                    int Elements = (int)Header.Dimensions.Elements();

                    fixed (byte* BytesPtr = Bytes)
                    fixed (float* DataPtr = Data)
                    {
                        float* DataP = DataPtr;

                        if (ValueType == typeof(byte))
                        {
                            byte* BytesP = BytesPtr;
                            for (int i = 0; i < Elements; i++)
                                *DataP++ = (float)*BytesP++;
                        }
                        else if (ValueType == typeof(short))
                        {
                            short* BytesP = (short*)BytesPtr;
                            for (int i = 0; i < Elements; i++)
                                *DataP++ = (float)*BytesP++;
                        }
                        else if (ValueType == typeof(ushort))
                        {
                            ushort* BytesP = (ushort*)BytesPtr;
                            for (int i = 0; i < Elements; i++)
                                *DataP++ = (float)*BytesP++;
                        }
                        else if (ValueType == typeof(int))
                        {
                            int* BytesP = (int*)BytesPtr;
                            for (int i = 0; i < Elements; i++)
                                *DataP++ = (float)*BytesP++;
                        }
                        else if (ValueType == typeof(float))
                        {
                            float* BytesP = (float*)BytesPtr;
                            for (int i = 0; i < Elements; i++)
                                *DataP++ = *BytesP++;
                        }
                        else if (ValueType == typeof(double))
                        {
                            double* BytesP = (double*)BytesPtr;
                            for (int i = 0; i < Elements; i++)
                                *DataP++ = (float)*BytesP++;
                        }
                    }
                }
            }

            return Data;
        }

        public static void ReadMapFloatIntoMemory(string path, long fileOffsetBytes, int elements, byte[] data, int destinationOffset)
        {
            using (BinaryReader Reader = new BinaryReader(OpenWithBigBuffer(path)))
            {
                Reader.BaseStream.Seek(fileOffsetBytes, SeekOrigin.Current);
                Reader.BaseStream.Read(data, destinationOffset * sizeof(float), elements * sizeof(float));
            }
        }

        public static void WriteMapFloat(string path, MapHeader header, float[] data)
        {
            Type ValueType = header.GetValueType();
            long Elements = header.Dimensions.Elements();

            using (BinaryWriter Writer = new BinaryWriter(CreateWithBigBuffer(path)))
            {
                header.Write(Writer);
                byte[] Bytes = null;
                
                Bytes = new byte[Elements * ImageFormatsHelper.SizeOf(ValueType)];

                unsafe
                {
                    fixed (float* DataPtr = data)
                    fixed (byte* BytesPtr = Bytes)
                    {
                        float* DataP = DataPtr;

                        if (ValueType == typeof(byte))
                        {
                            byte* BytesP = BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (byte)*DataP++;
                        }
                        else if (ValueType == typeof(short))
                        {
                            short* BytesP = (short*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (short)*DataP++;
                        }
                        else if (ValueType == typeof(ushort))
                        {
                            ushort* BytesP = (ushort*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (ushort)Math.Min(Math.Max(0f, *DataP++ * 16f), 65536f);
                        }
                        else if (ValueType == typeof(int))
                        {
                            int* BytesP = (int*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (int)*DataP++;
                        }
                        else if (ValueType == typeof(float))
                        {
                            float* BytesP = (float*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = *DataP++;
                        }
                        else if (ValueType == typeof(double))
                        {
                            double* BytesP = (double*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (double)*DataP++;
                        }
                    }
                }

                Writer.Write(Bytes);
            }
        }

        public static void WriteMapFloat(string path, MapHeader header, float[][] data)
        {
            Type ValueType = header.GetValueType();

            using (BinaryWriter Writer = new BinaryWriter(CreateWithBigBuffer(path)))
            {
                header.Write(Writer);
                long Elements = header.Dimensions.ElementsSlice();

                for (int z = 0; z < data.Length; z++)
                {
                    byte[] Bytes = new byte[Elements * ImageFormatsHelper.SizeOf(ValueType)];

                    unsafe
                    {
                        fixed (float* DataPtr = data[z])
                        fixed (byte* BytesPtr = Bytes)
                        {
                            float* DataP = DataPtr;

                            if (ValueType == typeof(byte))
                            {
                                byte* BytesP = BytesPtr;
                                for (long i = 0; i < Elements; i++)
                                    *BytesP++ = (byte)*DataP++;
                            }
                            else if (ValueType == typeof(short))
                            {
                                short* BytesP = (short*)BytesPtr;
                                for (long i = 0; i < Elements; i++)
                                    *BytesP++ = (short)*DataP++;
                            }
                            else if (ValueType == typeof(ushort))
                            {
                                ushort* BytesP = (ushort*)BytesPtr;
                                for (long i = 0; i < Elements; i++)
                                    *BytesP++ = (ushort)Math.Min(Math.Max(0f, *DataP++ * 16f), 65536f);
                            }
                            else if (ValueType == typeof(int))
                            {
                                int* BytesP = (int*)BytesPtr;
                                for (long i = 0; i < Elements; i++)
                                    *BytesP++ = (int)*DataP++;
                            }
                            else if (ValueType == typeof(float))
                            {
                                float* BytesP = (float*)BytesPtr;
                                for (long i = 0; i < Elements; i++)
                                    *BytesP++ = *DataP++;
                            }
                            else if (ValueType == typeof(double))
                            {
                                double* BytesP = (double*)BytesPtr;
                                for (long i = 0; i < Elements; i++)
                                    *BytesP++ = (double)*DataP++;
                            }
                        }
                    }

                    Writer.Write(Bytes);
                }
            }
        }

        public static bool CheckFolderPermission(string path)
        {
            if (!Directory.Exists(path))
                return false;

            try
            {
                // Attempt to get a list of security permissions from the folder. 
                // This will raise an exception if the path is read only, or we do not have access to view the permissions. 
                DirectorySecurity ds = Directory.GetAccessControl(path);
                return true;
            }
            catch (UnauthorizedAccessException)
            {
                return false;
            }
        }
    }
}
