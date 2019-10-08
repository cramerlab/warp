using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    public struct int4
    {
        public int X, Y, Z, W;

        public int4(int x, int y, int z, int w)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
        }

        public int4(int v)
        {
            X = v;
            Y = v;
            Z = v;
            W = v;
        }

        public int4(byte[] value)
        {
            X = BitConverter.ToInt32(value, 0);
            Y = BitConverter.ToInt32(value, sizeof(int));
            Z = BitConverter.ToInt32(value, 2 * sizeof(int));
            W = BitConverter.ToInt32(value, 3 * sizeof(int));
        }

        public int4(float4 value)
        {
            X = (int)value.X;
            Y = (int)value.Y;
            Z = (int)value.Z;
            W = (int)value.W;
        }

        public int Dimension(int d)
        {
            if (d == 0)
                return X;
            else if (d == 1)
                return Y;
            else if (d == 2)
                return Z;
            else if (d == 3)
                return W;
            else
                return -1;
        }

        public long Elements()
        {
            return (long)X * (long)Y * (long)Z * (long)W;
        }

        public long ElementsSlice()
        {
            return (long)X * (long)Y;
        }

        public long ElementsFFT()
        {
            return (long)(X / 2 + 1) * Y * Z * W;
        }

        public uint ElementFromPosition(int4 position)
        {
            return (((uint)position.W * (uint)Z + (uint)position.Z) * (uint)Y + (uint)position.Y) * (uint)X + (uint)position.X;
        }

        public uint ElementFromPosition(int x, int y, int z, int w)
        {
            return (((uint)w * (uint)Z + (uint)z) * (uint)Y + (uint)y) * (uint)X + (uint)x;
        }

        public long ElementFromPositionLong(int4 position)
        {
            return (((long)position.W * (long)Z + (long)position.Z) * (long)Y + (long)position.Y) * (long)X + (long)position.X;
        }

        public float Length()
        {
            return (float)Math.Sqrt(X * X + Y * Y + Z * Z + W * W);
        }

        public static readonly int4 UnitX = new int4(1, 0, 0, 0);
        public static readonly int4 UnitY = new int4(0, 1, 0, 0);
        public static readonly int4 UnitZ = new int4(0, 0, 1, 0);
        public static readonly int4 UnitW = new int4(0, 0, 0, 1);

        public static implicit operator byte[] (int4 value)
        {
            return Helper.Combine(new[]
            {
                BitConverter.GetBytes(value.X),
                BitConverter.GetBytes(value.Y),
                BitConverter.GetBytes(value.Z),
                BitConverter.GetBytes(value.W)
            });
        }

        public static bool operator <(int4 v1, int4 v2)
        {
            return v1.X < v2.X && v1.Y < v2.Y && v1.Z < v2.Z && v1.W < v2.W;
        }

        public static bool operator >(int4 v1, int4 v2)
        {
            return v1.X > v2.X && v1.Y > v2.Y && v1.Z > v2.Z && v1.W < v2.W;
        }

        public override bool Equals(Object obj)
        {
            return obj is int4 && this == (int4)obj;
        }

        public static bool operator ==(int4 o1, int4 o2)
        {
            return o1.X == o2.X && o1.Y == o2.Y && o1.Z == o2.Z && o1.W == o2.W;
        }

        public static bool operator !=(int4 o1, int4 o2)
        {
            return !(o1 == o2);
        }

        public override string ToString()
        {
            return X + ", " + Y + ", " + Z + ", " + W;
        }

        public static int4 operator *(int4 o1, int o2)
        {
            return new int4(o1.X * o2, o1.Y * o2, o1.Z * o2, o1.W * o2);
        }

        public static int4 operator /(int4 o1, int o2)
        {
            return new int4(o1.X / o2, o1.Y / o2, o1.Z / o2, o1.W / o2);
        }

        public static int4 operator *(int4 o1, float o2)
        {
            return new int4((int)(o1.X * o2), (int)(o1.Y * o2), (int)(o1.Z * o2), (int)(o1.W * o2));
        }

        public static int4 operator /(int4 o1, float o2)
        {
            return new int4((int)(o1.X / o2), (int)(o1.Y / o2), (int)(o1.Z / o2), (int)(o1.W / o2));
        }

        public static int4 operator +(int4 o1, float o2)
        {
            return new int4((int)(o1.X + o2), (int)(o1.Y + o2), (int)(o1.Z + o2), (int)(o1.W + o2));
        }

        public static int4 operator -(int4 o1, float o2)
        {
            return new int4((int)(o1.X - o2), (int)(o1.Y - o2), (int)(o1.Z - o2), (int)(o1.W - o2));
        }

        public static int4 operator +(int4 o1, int4 o2)
        {
            return new int4((int)(o1.X + o2.X), (int)(o1.Y + o2.Y), (int)(o1.Z + o2.Z), (int)(o1.W + o2.W));
        }

        public static int4 operator -(int4 o1, int4 o2)
        {
            return new int4((int)(o1.X - o2.X), (int)(o1.Y - o2.Y), (int)(o1.Z - o2.Z), (int)(o1.W - o2.W));
        }

        public static int4 operator /(int4 o1, int4 o2)
        {
            return new int4((int)(o1.X / o2.X), (int)(o1.Y / o2.Y), (int)(o1.Z / o2.Z), (int)(o1.W / o2.W));
        }

        public static int4 operator *(int4 o1, int4 o2)
        {
            return new int4((int)(o1.X * o2.X), (int)(o1.Y * o2.Y), (int)(o1.Z * o2.Z), (int)(o1.W * o2.W));
        }

        public static int4 Max(int4 o1, int o2)
        {
            return new int4(Math.Max(o1.X, o2), Math.Max(o1.Y, o2), Math.Max(o1.Z, o2), Math.Max(o1.W, o2));
        }

        public static int4 Max(int4 o1, int4 o2)
        {
            return new int4(Math.Max(o1.X, o2.X), Math.Max(o1.Y, o2.Y), Math.Max(o1.Z, o2.Z), Math.Max(o1.W, o2.W));
        }

        public static int4 Min(int4 o1, int o2)
        {
            return new int4(Math.Min(o1.X, o2), Math.Min(o1.Y, o2), Math.Min(o1.Z, o2), Math.Min(o1.W, o2));
        }

        public static int4 Min(int4 o1, int4 o2)
        {
            return new int4(Math.Min(o1.X, o2.X), Math.Min(o1.Y, o2.Y), Math.Min(o1.Z, o2.Z), Math.Min(o1.W, o2.W));
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    public struct int3
    {
        public int X, Y, Z;

        public int3(int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public int3(int2 v)
        {
            X = v.X;
            Y = v.Y;
            Z = 1;
        }

        public int3(int v)
        {
            X = v;
            Y = v;
            Z = v;
        }

        public int3(byte[] value)
        {
            X = BitConverter.ToInt32(value, 0);
            Y = BitConverter.ToInt32(value, sizeof(int));
            Z = BitConverter.ToInt32(value, 2 * sizeof(int));
        }

        public int3(float3 value)
        {
            X = (int)value.X;
            Y = (int)value.Y;
            Z = (int)value.Z;
        }

        public int Dimension(int d)
        {
            if (d == 0)
                return X;
            else if (d == 1)
                return Y;
            else if (d == 2)
                return Z;
            else
                return -1;
        }

        public long Elements()
        {
            return (long)X * (long)Y * (long)Z;
        }

        public long ElementsSlice()
        {
            return (long)X * (long)Y;
        }

        public long ElementsFFT()
        {
            return (long)(X / 2 + 1) * Y * Z;
        }

        public uint ElementFromPosition(int3 position)
        {
            return ((uint)position.Z * (uint)Y + (uint)position.Y) * (uint)X + (uint)position.X;
        }

        public uint ElementFromPosition(int x, int y, int z)
        {
            return ((uint)z * (uint)Y + (uint)y) * (uint)X + (uint)x;
        }

        public long ElementFromPositionLong(int3 position)
        {
            return ((long)position.Z * (long)Y + (long)position.Y) * (long)X + (long)position.X;
        }

        public int3 Slice()
        {
            return new int3(X, Y, 1);
        }

        public int3 MultXY(int mult)
        {
            return new int3(X * mult, Y * mult, Z);
        }

        public float3 MultXY(float mult)
        {
            return new float3(X * mult, Y * mult, Z);
        }

        public float Length()
        {
            return (float)Math.Sqrt(X * X + Y * Y + Z * Z);
        }

        public static readonly int3 UnitX = new int3(1, 0, 0);
        public static readonly int3 UnitY = new int3(0, 1, 0);
        public static readonly int3 UnitZ = new int3(0, 0, 1);

        public static implicit operator byte[] (int3 value)
        {
            return Helper.Combine(new[]
            {
                BitConverter.GetBytes(value.X),
                BitConverter.GetBytes(value.Y),
                BitConverter.GetBytes(value.Z)
            });
        }

        public static bool operator <(int3 v1, int3 v2)
        {
            return v1.X < v2.X && v1.Y < v2.Y && v1.Z < v2.Z;
        }

        public static bool operator >(int3 v1, int3 v2)
        {
            return v1.X > v2.X && v1.Y > v2.Y && v1.Z > v2.Z;
        }

        public override bool Equals(Object obj)
        {
            return obj is int3 && this == (int3)obj;
        }

        public static bool operator ==(int3 o1, int3 o2)
        {
            return o1.X == o2.X && o1.Y == o2.Y && o1.Z == o2.Z;
        }

        public static bool operator !=(int3 o1, int3 o2)
        {
            return !(o1 == o2);
        }

        public override string ToString()
        {
            return X + ", " + Y + ", " + Z;
        }

        public static int3 operator *(int3 o1, int o2)
        {
            return new int3(o1.X * o2, o1.Y * o2, o1.Z * o2);
        }

        public static int3 operator /(int3 o1, int o2)
        {
            return new int3(o1.X / o2, o1.Y / o2, o1.Z / o2);
        }

        public static int3 operator *(int3 o1, float o2)
        {
            return new int3((int)(o1.X * o2), (int)(o1.Y * o2), (int)(o1.Z * o2));
        }

        public static int3 operator /(int3 o1, float o2)
        {
            return new int3((int)(o1.X / o2), (int)(o1.Y / o2), (int)(o1.Z / o2));
        }

        public static int3 operator +(int3 o1, float o2)
        {
            return new int3((int)(o1.X + o2), (int)(o1.Y + o2), (int)(o1.Z + o2));
        }

        public static int3 operator -(int3 o1, float o2)
        {
            return new int3((int)(o1.X - o2), (int)(o1.Y - o2), (int)(o1.Z - o2));
        }

        public static int3 operator +(int3 o1, int3 o2)
        {
            return new int3((int)(o1.X + o2.X), (int)(o1.Y + o2.Y), (int)(o1.Z + o2.Z));
        }

        public static int3 operator -(int3 o1, int3 o2)
        {
            return new int3((int)(o1.X - o2.X), (int)(o1.Y - o2.Y), (int)(o1.Z - o2.Z));
        }

        public static int3 operator /(int3 o1, int3 o2)
        {
            return new int3((int)(o1.X / o2.X), (int)(o1.Y / o2.Y), (int)(o1.Z / o2.Z));
        }

        public static int3 operator *(int3 o1, int3 o2)
        {
            return new int3((int)(o1.X * o2.X), (int)(o1.Y * o2.Y), (int)(o1.Z * o2.Z));
        }

        public static int3 Max(int3 o1, int o2)
        {
            return new int3(Math.Max(o1.X, o2), Math.Max(o1.Y, o2), Math.Max(o1.Z, o2));
        }

        public static int3 Max(int3 o1, int3 o2)
        {
            return new int3(Math.Max(o1.X, o2.X), Math.Max(o1.Y, o2.Y), Math.Max(o1.Z, o2.Z));
        }

        public static int3 Min(int3 o1, int o2)
        {
            return new int3(Math.Min(o1.X, o2), Math.Min(o1.Y, o2), Math.Min(o1.Z, o2));
        }

        public static int3 Min(int3 o1, int3 o2)
        {
            return new int3(Math.Min(o1.X, o2.X), Math.Min(o1.Y, o2.Y), Math.Min(o1.Z, o2.Z));
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    public struct int2
    {
        public int X, Y;

        public int2(int x, int y)
        {
            X = x;
            Y = y;
        }

        public int2(int3 v)
        {
            X = v.X;
            Y = v.Y;
        }

        public int2(int v)
        {
            X = v;
            Y = v;
        }

        public int2(byte[] value)
        {
            X = BitConverter.ToInt32(value, 0);
            Y = BitConverter.ToInt32(value, sizeof(float));
        }

        public int2(float2 v)
        {
            X = (int)v.X;
            Y = (int)v.Y;
        }

        public int Dimension(int d)
        {
            if (d == 0)
                return X;
            else if (d == 1)
                return Y;
            else
                return -1;
        }

        public long Elements()
        {
            return (long)X * (long)Y;
        }

        public long ElementsFFT()
        {
            return (long)(X / 2 + 1) * Y;
        }

        public int ElementFromPosition(int2 position)
        {
            return position.Y * X + position.X;
        }

        public long ElementFromPositionLong(int2 position)
        {
            return (long)position.Y * (long)X + (long)position.X;
        }

        public float Length()
        {
            return (float)Math.Sqrt(X * X + Y * Y);
        }

        public static implicit operator byte[] (int2 value)
        {
            return Helper.Combine(new[]
            {
                BitConverter.GetBytes(value.X),
                BitConverter.GetBytes(value.Y)
            });
        }

        public static bool operator <(int2 v1, int2 v2)
        {
            return v1.X < v2.X && v1.Y < v2.Y;
        }

        public static bool operator >(int2 v1, int2 v2)
        {
            return v1.X > v2.X && v1.Y > v2.Y;
        }

        public override bool Equals(Object obj)
        {
            return obj is int2 && this == (int2)obj;
        }

        public static bool operator ==(int2 o1, int2 o2)
        {
            return o1.X == o2.X && o1.Y == o2.Y;
        }

        public static bool operator !=(int2 o1, int2 o2)
        {
            return !(o1 == o2);
        }

        public static int2 operator *(int2 o1, int o2)
        {
            return new int2(o1.X * o2, o1.Y * o2);
        }

        public static int2 operator /(int2 o1, int o2)
        {
            return new int2(o1.X / o2, o1.Y / o2);
        }

        public static int2 operator +(int2 o1, int o2)
        {
            return new int2(o1.X + o2, o1.Y + o2);
        }

        public static int2 operator -(int2 o1, int o2)
        {
            return new int2(o1.X - o2, o1.Y - o2);
        }

        public static int2 operator *(int2 o1, float o2)
        {
            return new int2((int)(o1.X * o2), (int)(o1.Y * o2));
        }

        public static int2 operator /(int2 o1, float o2)
        {
            return new int2((int)(o1.X / o2), (int)(o1.Y / o2));
        }

        public static int2 operator *(int2 o1, int2 o2)
        {
            return new int2((int)(o1.X * o2.X), (int)(o1.Y * o2.Y));
        }

        public static int2 operator /(int2 o1, int2 o2)
        {
            return new int2((int)(o1.X / o2.X), (int)(o1.Y / o2.Y));
        }

        public static int2 operator +(int2 o1, int2 o2)
        {
            return new int2((int)(o1.X + o2.X), (int)(o1.Y + o2.Y));
        }

        public static int2 operator -(int2 o1, int2 o2)
        {
            return new int2((int)(o1.X - o2.X), (int)(o1.Y - o2.Y));
        }

        public override string ToString()
        {
            return X + ", " + Y;
        }

        public static int2 Max(int2 o1, int o2)
        {
            return new int2(Math.Max(o1.X, o2), Math.Max(o1.Y, o2));
        }

        public static int2 Max(int2 o1, int2 o2)
        {
            return new int2(Math.Max(o1.X, o2.X), Math.Max(o1.Y, o2.Y));
        }

        public static int2 Min(int2 o1, int o2)
        {
            return new int2(Math.Min(o1.X, o2), Math.Min(o1.Y, o2));
        }

        public static int2 Min(int2 o1, int2 o2)
        {
            return new int2(Math.Min(o1.X, o2.X), Math.Min(o1.Y, o2.Y));
        }
    }
}
