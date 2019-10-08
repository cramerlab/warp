using System;
using System.Runtime.InteropServices;
using System.Security;
using System.Threading;
using Warp.Tools;

namespace Warp
{
    [SuppressUnmanagedCodeSecurity]
    public static class TiffNative
    {
        [DllImport("GPUAcceleration.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "ReadTIFF")]
        public static extern void ReadTIFF(string path, int layer, bool flipy, float[] h_result);

        public static void ReadTIFFPatient(int attempts, int mswait, string path, int layer, bool flipy, float[] h_result)
        {
            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    ReadTIFF(path, layer, flipy, h_result);
                    return;
                }
                catch
                {
                    Thread.Sleep(mswait);
                }
            }

            throw new Exception("Could not successfully read file within the specified number of attempts.");
        }
    }
}
