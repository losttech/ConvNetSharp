using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.VisualStudio.TestTools.UnitTesting;

using ConvNetSharp.Core.Training;
using GpuVolume = ConvNetSharp.Volume.GPU;

namespace ConvNetSharp.Training
{
    using ConvNetSharp.Volume;

    [TestClass]
    public class SgdTrainerTests
    {
        [TestMethod]
        public void FloatDoesNotLeakGpuMemory()
        {
            BuilderInstance<float>.Volume = new GpuVolume.Single.VolumeBuilder();
            new TrainerBaseTests<float>().DoesNotLeakGpuMemory(net => new SgdTrainer<float>(net), new GpuVolume.Single.VolumeBuilder());
        }
    }
}
