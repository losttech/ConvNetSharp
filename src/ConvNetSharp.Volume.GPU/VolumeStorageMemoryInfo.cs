using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNetSharp.Volume.GPU
{
    public static class VolumeStorageMemoryInfo
    {
        internal static long gpuMemoryUsage;
        internal static long notDisposedDueToOwnership;
        public static long TotalGpuMemoryUsage => gpuMemoryUsage;
        public static long NotDisposedDueToOwnership => notDisposedDueToOwnership;
    }
}
