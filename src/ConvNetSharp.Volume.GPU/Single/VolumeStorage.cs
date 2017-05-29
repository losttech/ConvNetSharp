using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ConvNetSharp.Volume.GPU.Single
{
    public unsafe class VolumeStorage : VolumeStorage<float>, IDisposable
    {
        private readonly CudaHostMemoryRegion _hostPointer;
        private readonly bool _isOwner;
        private bool _allocatedOnDevice;
        private bool disposed;

        public VolumeStorage(Shape shape, GpuContext context, long length = -1) : base(shape)
        {
            this.Context = context;

            this.InitializeUnknownDimension(length);

            this._hostPointer = InitializeSharedMemory(this.Shape.TotalLength);

            this._isOwner = true;
        }

        void InitializeUnknownDimension(long length)
        {
            if (length != -1)
            {
                this.Shape.GuessUnkownDimension(length);
            }
        }

        [DllImport("Kernel32.dll", EntryPoint = "RtlZeroMemory", SetLastError = false)]
        static extern void ZeroMemory(IntPtr dest, UIntPtr size);

        static CudaHostMemoryRegion InitializeSharedMemory(long elementCount)
        {
            var sharedMemory = new CudaHostMemoryRegion(byteCount: elementCount*sizeof(float));

            // Zero out
            FillWithZeroes(sharedMemory.Start, sharedMemory.ByteCount);
            return sharedMemory;
        }

        static void FillWithZeroes(IntPtr memoryStart, long size)
        {
            switch (Environment.OSVersion.Platform)
            {
            case PlatformID.Win32NT:
            case PlatformID.Win32Windows:
            case PlatformID.WinCE:
                ZeroMemory(memoryStart, (UIntPtr)size);
                break;
            default:
                byte* buffer = (byte*)memoryStart;
                for (var i = 0; i < size; i++) {
                    buffer[i] = 0;
                }
                break;
            }
        }

        public VolumeStorage(float[] array, Shape shape, GpuContext context) : this(shape, context, array.Length)
        {
            this.Context = context;

            if (this.Shape.TotalLength != array.Length)
            {
                throw new ArgumentException("Wrong dimensions");
            }

            // Fill host buffer
            for (var i = 0; i < array.Length; i++)
            {
                this.HostBuffer[i] = array[i];
            }

            this.Location = DataLocation.Host;
        }

        public VolumeStorage(VolumeStorage storage, Shape shape)
            : base(shape)
        {
            if (storage == null)
                throw new ArgumentNullException(nameof(storage));
            if (storage._hostPointer == null)
                throw new ArgumentException();

            this.Context = storage.Context;
            this.InitializeUnknownDimension(storage.Shape.TotalLength);
            this._hostPointer = storage._hostPointer;
            this._isOwner = false;
            this.Location = storage.Location;
            this._allocatedOnDevice = storage._allocatedOnDevice;

            storage.CopyToDevice();
            this.DeviceBuffer = new CudaDeviceVariable<float>(storage.DeviceBuffer.DevicePointer);

            this.Location = DataLocation.Device;
        }
        public long GpuMemory => this.Shape.TotalLength * sizeof(double);
        public CudaDeviceVariable<byte> ConvolutionBackwardFilterStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionBackwardStorage { get; set; }

        public CudaDeviceVariable<byte> ConvolutionStorage { get; set; }

        public DataLocation Location { get; set; }

        public float* HostBuffer => (float*)this._hostPointer.Start;

        public CudaDeviceVariable<float> DeviceBuffer { get; private set; }

        public GpuContext Context { get; }

        public void Dispose()
        {
            Dispose(true);
        }

        public override void Clear()
        {
            switch (this.Location)
            {
                case DataLocation.Host:
                    {
                        for (var i = 0; i < this.Shape.TotalLength; i++)
                        {
                            this.HostBuffer[i] = 0.0f;
                        }
                    }
                    break;
                case DataLocation.Device:
                    {
                        var res = DriverAPINativeMethods.Memset.cuMemsetD16_v2(this.DeviceBuffer.DevicePointer, 0, this.DeviceBuffer.Size * 2);
                        if (res != CUResult.Success)
                        {
                            throw new CudaException(res);
                        }
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public void CopyToDevice()
        {
            if (this.Location == DataLocation.Host)
            {
                // Device 
                if (!this._allocatedOnDevice)
                {
                    this.DeviceBuffer = new CudaDeviceVariable<float>(this.Shape.TotalLength);
                    this._allocatedOnDevice = true;
                }

                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(
                    this.DeviceBuffer.DevicePointer, this._hostPointer.Start, this.DeviceBuffer.SizeInBytes,
                    this.Context.DefaultStream.Stream);
                if (res != CUResult.Success)
                {
                    throw new CudaException(res);
                }

                // Synchro
                this.Context.DefaultStream.Synchronize();

                this.Location = DataLocation.Device;
            }
        }

        public void CopyToHost()
        {
            if (this.Location == DataLocation.Device)
            {
                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(
                    new IntPtr(this.HostBuffer),
                    this.DeviceBuffer.DevicePointer, this.DeviceBuffer.SizeInBytes, this.Context.DefaultStream.Stream);

                if (res != CUResult.Success)
                {
                    throw new CudaException(res);
                }

                // Synchro
                this.Context.DefaultStream.Synchronize();

                this.Location = DataLocation.Host;
            }
        }

        public virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }

            if (this.disposed)
                return;

            this.disposed = true;

            if (this._hostPointer != null && this.HostBuffer != default(float*))
            {
                if (this._isOwner)
                {
                    this._hostPointer.Dispose();
                }
                else
                {
                    Interlocked.Add(ref VolumeStorageMemoryInfo.notDisposedDueToOwnership, this.GpuMemory);
                }
            }

            if (this._isOwner)
            {
                this.DeviceBuffer?.Dispose();
            }

            this.ConvolutionBackwardFilterStorage?.Dispose();
            this.ConvolutionBackwardStorage?.Dispose();
            this.ConvolutionStorage?.Dispose();
        }

        public override bool Equals(VolumeStorage<float> other)
        {
            throw new NotImplementedException();
        }

        ~VolumeStorage()
        {
            Dispose(false);
        }

        public override float Get(int[] coordinates)
        {
            CopyToHost();

            var length = coordinates.Length;
            return Get(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0);
        }

        public override float Get(int w, int h, int c, int n)
        {
            CopyToHost();

            return this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)];
        }

        public override float Get(int w, int h, int c)
        {
            CopyToHost();
            return
                this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)];
        }

        public override float Get(int w, int h)
        {
            CopyToHost();
            return this.HostBuffer[w + h * this.Shape.GetDimension(0)];
        }

        public override float Get(int i)
        {
            CopyToHost();
            return this.HostBuffer[i];
        }

        public override void Set(int[] coordinates, float value)
        {
            CopyToHost();

            var length = coordinates.Length;
            Set(coordinates[0], length > 1 ? coordinates[1] : 0, length > 2 ? coordinates[2] : 0, length > 3 ? coordinates[3] : 0, value);
        }

        public override void Set(int w, int h, int c, int n, float value)
        {
            CopyToHost();
            this.HostBuffer[
                w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) +
                n * this.Shape.GetDimension(0) * this.Shape.GetDimension(1) * this.Shape.GetDimension(2)] = value;
        }

        public override void Set(int w, int h, int c, float value)
        {
            CopyToHost();
            this.HostBuffer[
                    w + h * this.Shape.GetDimension(0) + c * this.Shape.GetDimension(0) * this.Shape.GetDimension(1)] =
                value;
        }

        public override void Set(int w, int h, float value)
        {
            CopyToHost();
            this.HostBuffer[w + h * this.Shape.GetDimension(0)] = value;
        }

        public override void Set(int i, float value)
        {
            CopyToHost();
            this.HostBuffer[i] = value;
        }

        public override float[] ToArray()
        {
            CopyToHost();

            var array = new float[this.Shape.TotalLength];
            Marshal.Copy(new IntPtr(this.HostBuffer), array, 0, (int)this.Shape.TotalLength);
            return array;
        }
    }
}