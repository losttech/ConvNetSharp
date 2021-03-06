﻿using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Flow.Ops
{
    internal class ConvolutionFilterGradient<T> : Op<T> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly Convolution<T> _convolution;

        public ConvolutionFilterGradient(Convolution<T> convolution, Op<T> derivate)
        {
            this._convolution = convolution;

            this.AddParent(convolution);
            this.AddParent(derivate);
        }

        public override string Representation => "ConvolutionFilterGradient";

        public override void Differentiate()
        {
            throw new NotImplementedException();
        }

        public override Volume<T> Evaluate(Session<T> session)
        {
            this._convolution.EvaluateGradient(session);
            return this._convolution.FilterGradient;
        }
    }
}