using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNetSharp.Training
{
    using System.Diagnostics;
    using ConvNetSharp.Core;
    using ConvNetSharp.Core.Fluent;
    using ConvNetSharp.Core.Training;
    using ConvNetSharp.Volume;
    using ConvNetSharp.Volume.GPU;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    class TrainerBaseTests<T> where T : struct, IEquatable<T>, IFormattable
    {
        const int TestInputWidth = 31;
        const int TestInputHeight = 51;
        const int TestInputSize = TestInputWidth * TestInputHeight;
        const int TestOutputSize = 55;
        const int TestMinibatchSize = 64;
        readonly FluentNet<T> net =
            FluentNet<T>.Create(TestInputWidth, TestInputHeight, inputDepth: 1)
                .Conv(5, 5, 8).Stride(1).Pad(2)
                .Relu()
                .Pool(2, 2).Stride(2)
                .Conv(5, 5, 16).Stride(1).Pad(2)
                .Relu()
                .Pool(3, 3).Stride(3)
                .FullyConn(TestOutputSize)
                .Softmax(TestOutputSize)
                .Build();

        internal void DoesNotLeakGpuMemory(Func<INet<T>, TrainerBase<T>> trainerFactory, VolumeBuilder<T> volumeBuilder)
        {
            if (trainerFactory == null)
                throw new ArgumentNullException(nameof(trainerFactory));
            if (volumeBuilder == null)
                throw new ArgumentNullException(nameof(volumeBuilder));

            long initialGpuMemory = VolumeStorageMemoryInfo.TotalGpuMemoryUsage;
            TrainerBase<T> trainer = trainerFactory(this.net);
            if (trainer == null)
                throw new ArgumentException("Factory returned null", paramName: nameof(trainerFactory));

            long gpuMemoryWithTrainer = VolumeStorageMemoryInfo.TotalGpuMemoryUsage;
            T[] inputs = new T[TestMinibatchSize * TestInputSize];
            T[] outputs = new T[TestMinibatchSize * TestOutputSize];
            var inputShape = new Shape(TestInputWidth, TestInputHeight, 1, TestMinibatchSize);
            var outputShape = new Shape(1, 1, TestOutputSize, TestMinibatchSize);

            void Iteration()
            {
                Volume<T> input = volumeBuilder.SameAs(inputs, inputShape);
                Volume<T> expectedOutput = volumeBuilder.SameAs(outputs, outputShape);
                trainer.Train(input, expectedOutput);
                Dispose(input);
                Dispose(expectedOutput);
            }

            Iteration();
            long memoryAfterFirstIteration = VolumeStorageMemoryInfo.TotalGpuMemoryUsage;
            Iteration();
            long memoryAfterSecondIteration = VolumeStorageMemoryInfo.TotalGpuMemoryUsage;

            Dispose(trainer);
            long memoryAfterEverythingDone = VolumeStorageMemoryInfo.TotalGpuMemoryUsage;

            Assert.AreEqual(expected: memoryAfterFirstIteration, actual: memoryAfterSecondIteration, message: "Iteration does not preserve memory");
            Assert.AreEqual(expected: initialGpuMemory, actual: memoryAfterEverythingDone, message: "Training does not preserve memory");
        }

        static void Dispose(object obj)
        {
            if (obj is IDisposable disposable)
                disposable.Dispose();
        }
    }
}
