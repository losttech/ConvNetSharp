using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class SoftmaxLayer<T> : LastLayerBase<T>, IClassificationLayer where T : struct, IEquatable<T>, IFormattable
    {
        public SoftmaxLayer(Dictionary<string, object> data) : base(data)
        {
            this.ClassCount = Convert.ToInt32(data["ClassCount"]);
        }

        public SoftmaxLayer(int classCount)
        {
            this.ClassCount = classCount;
        }

        public int ClassCount { get; set; }

        public override void Backward(Volume<T> y, out T loss)
        {
            this.OutputActivation.DoSoftMaxGradient(this.OutputActivation - y, this.InputActivationGradients);

            //loss is the class negative log likelihood
            loss = Loss(y, this.OutputActivation);
        }

        public static T Loss(Volume<T> y, Volume<T> actualSet)
        {
            //loss is the class negative log likelihood
            var loss = Ops<T>.Zero;
            var expected = y.ToArray();
            var actual = actualSet.ToArray();
            
            for (int i = 0; i < expected.Length; i++)
            {
                actual[i] = Ops<T>.Max(actual[i], Ops<T>.Epsilon);
                var current = Ops<T>.Multiply(expected[i], Ops<T>.Log(actual[i]));
                loss = Ops<T>.Add(loss, current);
            }

            loss = Ops<T>.Negate(loss);

            if (Ops<T>.IsInvalid(loss))
                throw new ArgumentException("Error during calculation!");

            return loss;
        }

        public override void Backward(Volume<T> outputGradient)
        {
            throw new NotImplementedException();
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.DoSoftMax(this.OutputActivation);
            return this.OutputActivation;
        }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();
            dico["ClassCount"] = this.ClassCount;
            return dico;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
            this.OutputDepth = inputCount;
        }
    }
}