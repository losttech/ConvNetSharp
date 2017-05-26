using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNetSharp.Volume.GPU
{
    using System.Threading;
    using ManagedCuda;
    using ManagedCuda.BasicTypes;

    class CudaHostMemoryRegion: IDisposable
    {
        readonly IntPtr startPointer;
        bool releasedCudaMemory;

        public CudaHostMemoryRegion(long byteCount)
        {
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));

            this.releasedCudaMemory = true;
            this.ByteCount = byteCount;
            var result = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref this.startPointer, byteCount);
            if (result != CUResult.Success)
                throw new CudaException(result);
            this.releasedCudaMemory = false;
            Interlocked.Add(ref VolumeStorageMemoryInfo.gpuMemoryUsage, this.ByteCount);
        }

        public IntPtr Start => this.startPointer;
        public long ByteCount { get; }

        void ReleaseUnmanagedResources()
        {
            if (this.releasedCudaMemory)
                return;

            var result = DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(this.startPointer);
            if (result != CUResult.Success)
                throw new CudaException(result);

            Interlocked.Add(ref VolumeStorageMemoryInfo.gpuMemoryUsage, -this.ByteCount);

            this.releasedCudaMemory = true;
        }

        public void Dispose()
        {
            this.ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }

        ~CudaHostMemoryRegion()
        {
            this.ReleaseUnmanagedResources();
        }
    }
}
