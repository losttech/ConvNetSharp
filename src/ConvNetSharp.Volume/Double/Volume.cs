﻿using System;

namespace ConvNetSharp.Volume.Double
{
    public class Volume : Volume<double>
    {
        public Volume(double[] array, Shape shape) : this(new NcwhVolumeStorage<double>(array, shape))
        {
        }

        public Volume(VolumeStorage<double> storage) : base(storage)
        {
        }

        public override void DoActivation(Volume<double> volume, ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    this.Storage.Map(x => 1.0 / (1.0 + Math.Exp(-x)), volume.Storage);
                    return;
                case ActivationType.Relu:
                    throw new NotImplementedException();
                case ActivationType.Tanh:
                    this.Storage.Map(Math.Tanh, volume.Storage);
                    break;
                case ActivationType.ClippedRelu:
                    throw new NotImplementedException();
            }
        }

        public override void DoActivationGradient(Volume<double> input, Volume<double> outputGradient,
            Volume<double> result, ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    this.Storage.Map((output, outGradient) => output * (1.0 - output) * outGradient, outputGradient.Storage,
                        result.Storage);
                    return;
                case ActivationType.Relu:
                    throw new NotImplementedException();
                case ActivationType.Tanh:
                    this.Storage.Map((output, outGradient) => (1.0 - output * output) * outGradient, outputGradient.Storage,
                        result.Storage);
                    return;
                case ActivationType.ClippedRelu:
                    throw new NotImplementedException();
            }
        }

        public override void DoAdd(Volume<double> other, Volume<double> result)
        {
            this.Storage.MapEx((x, y) => x + y, other.Storage, result.Storage);
        }

        protected override void DoBiasGradient(Volume<double> biasGradient)
        {
            var batchSize = this.Shape.GetDimension(3);

            var outputWidth = this.Shape.GetDimension(0);
            var outputHeight = this.Shape.GetDimension(1);
            var outputDepth = this.Shape.GetDimension(2);

            for (var n = 0; n < batchSize; n++)
            {
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            var chainGradient = Get(ax, ay, depth, n);

                            biasGradient.Storage.Set(0, 0, depth,
                                biasGradient.Storage.Get(0, 0, depth) + chainGradient);
                        }
                    }
                }
            }
        }

        public override void DoConvolution(Volume<double> filters, int pad, int stride, Volume<double> result)
        {
            var batchSize = this.Shape.GetDimension(3);

            var inputWidth = this.Shape.GetDimension(0);
            var inputHeight = this.Shape.GetDimension(1);

            var outputWidth = result.Shape.GetDimension(0);
            var outputHeight = result.Shape.GetDimension(1);
            var outputDepth = result.Shape.GetDimension(2);

            var filterWidth = filters.Shape.GetDimension(0);
            var filterHeight = filters.Shape.GetDimension(1);
            var filterDepth = filters.Shape.GetDimension(2);

            for (var n = 0; n < batchSize; n++)
            {
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    var y = -pad;
                    for (var ay = 0; ay < outputHeight; y += stride, ay++)
                    {
                        var x = -pad;
                        for (var ax = 0; ax < outputWidth; x += stride, ax++)
                        {
                            // convolve centered at this particular location
                            var a = 0.0;
                            for (var fy = 0; fy < filterHeight; fy++)
                            {
                                var oy = y + fy; // coordinates in the original input array coordinates
                                for (var fx = 0; fx < filterWidth; fx++)
                                {
                                    var ox = x + fx;
                                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                    {
                                        for (var fd = 0; fd < filterDepth; fd++)
                                        {
                                            a += filters.Storage.Get(fx, fy, fd, depth) *
                                                 this.Storage.Get(ox, oy, fd, n);
                                        }
                                    }
                                }
                            }

                            result.Storage.Set(ax, ay, depth, n, a);
                        }
                    }
                }
            }
        }

        public override void DoConvolutionGradient(Volume<double> filters, Volume<double> outputGradients,
            Volume<double> inputGradient, Volume<double> filterGradient, int pad,
            int stride)
        {
            inputGradient.Clear(); // zero out gradient wrt bottom data, we're about to fill it

            var batchSize = this.Shape.GetDimension(3);

            var inputWidth = this.Shape.GetDimension(0);
            var inputHeight = this.Shape.GetDimension(1);

            var outputWidth = outputGradients.Shape.GetDimension(0);
            var outputHeight = outputGradients.Shape.GetDimension(1);
            var outputDepth = outputGradients.Shape.GetDimension(2);

            var filterWidth = filters.Shape.GetDimension(0);
            var filterHeight = filters.Shape.GetDimension(1);
            var filterDepth = filters.Shape.GetDimension(2);

            for (var n = 0; n < batchSize; n++)
            {
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    var y = -pad;
                    for (var ay = 0; ay < outputHeight; y += stride, ay++)
                    {
                        var x = -pad;
                        for (var ax = 0; ax < outputWidth; x += stride, ax++)
                        {
                            // convolve centered at this particular location
                            var chainGradient = outputGradients.Get(ax, ay, depth, n);

                            // gradient from above, from chain rule
                            for (var fy = 0; fy < filterHeight; fy++)
                            {
                                var oy = y + fy; // coordinates in the original input array coordinates
                                for (var fx = 0; fx < filterWidth; fx++)
                                {
                                    var ox = x + fx;
                                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                    {
                                        for (var fd = 0; fd < filterDepth; fd++)
                                        {
                                            filterGradient.Set(fx, fy, fd, depth,
                                                filterGradient.Get(fx, fy, fd, depth) +
                                                Get(ox, oy, fd, n) * chainGradient);
                                            inputGradient.Set(ox, oy, fd, n,
                                                inputGradient.Get(ox, oy, fd, n) +
                                                filters.Get(fx, fy, fd, depth) * chainGradient);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        public override void DoDivide(Volume<double> other, Volume<double> result)
        {
            if (this.Shape.Equals(other.Shape))
            {
                this.Storage.Map((left, right) => left / right, other.Storage, result.Storage);
            }
            else
            {
                //Todo: broadcast
                throw new NotImplementedException();
            }
        }

        public override void DoExp(Volume<double> result)
        {
            this.Storage.Map(Math.Exp, result.Storage);
        }

        public override void DoLog(Volume<double> result)
        {
            this.Storage.Map(Math.Log, result.Storage);
        }

        public override void DoMax(Volume<double> result)
        {
            var batchSize = this.Shape.DimensionCount > 1 ? this.Shape.GetDimension(-1) : 1;
            var reshape = ReShape(-1, batchSize);

            var n = reshape.Shape.GetDimension(0);

            for (var i = 0; i < batchSize; i++)
            {
                var max = double.MinValue;

                for (var j = 0; j < n; j++)
                {
                    var d = reshape.Get(j, i);
                    if (d > max)
                    {
                        max = d;
                    }
                }

                result.Set(new[] { i }, max);
            }
        }

        public override void DoMin(Volume<double> result)
        {
            var batchSize = this.Shape.DimensionCount > 1 ? this.Shape.GetDimension(-1) : 1;
            var reshape = ReShape(-1, batchSize);

            var n = reshape.Shape.GetDimension(0);

            for (var i = 0; i < batchSize; i++)
            {
                var min = double.MaxValue;

                for (var j = 0; j < n; j++)
                {
                    var d = reshape.Get(j, i);
                    if (d < min)
                    {
                        min = d;
                    }
                }

                result.Set(new[] { i }, min);
            }
        }

        public override void DoMultiply(Volume<double> result, double factor)
        {
            this.Storage.Map(x => x * factor, result.Storage);
        }

        public override void DoMultiply(Volume<double> right, Volume<double> result)
        {
            this.Storage.MapEx((x, y) => x * y, right.Storage, result.Storage);
        }

        public override void DoNegate(Volume<double> volume)
        {
            DoMultiply(volume, -1.0);
        }

        public override void DoPool(Volume<double> result, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            var inputWidth = this.Shape.GetDimension(0);
            var inputHeight = this.Shape.GetDimension(1);

            var outputWidth = result.Shape.GetDimension(0);
            var outputHeight = result.Shape.GetDimension(1);
            var outputDepth = result.Shape.GetDimension(2);
            var batchSize = result.Shape.GetDimension(3);

            for (var n = 0; n < batchSize; n++)
            {
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    var x = -horizontalPad;
                    for (var ax = 0; ax < outputWidth; x += verticalStride, ax++)
                    {
                        var y = -verticalPad;
                        for (var ay = 0; ay < outputHeight; y += horizontalStride, ay++)
                        {
                            var a = double.MinValue;

                            for (var fx = 0; fx < windowWidth; fx++)
                            {
                                for (var fy = 0; fy < windowHeight; fy++)
                                {
                                    var oy = y + fy;
                                    var ox = x + fx;
                                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                    {
                                        var v = Get(ox, oy, depth, n);
                                        // perform max pooling and store pointers to where
                                        // the max came from. This will speed up backprop 
                                        // and can help make nice visualizations in future
                                        if (v > a)
                                        {
                                            a = v;
                                        }
                                    }
                                }
                            }

                            result.Storage.Set(ax, ay, depth, n, a);
                        }
                    }
                }
            }
        }

        public override void DoPoolGradient(Volume<double> input, Volume<double> outputGradient,
            Volume<double> inputGradient, int windowWidth, int windowHeight,
            int horizontalPad, int verticalPad, int horizontalStride, int verticalStride)
        {
            var inputWidth = input.Shape.GetDimension(0);
            var inputHeight = input.Shape.GetDimension(1);

            var outputWidth = outputGradient.Shape.GetDimension(0);
            var outputHeight = outputGradient.Shape.GetDimension(1);
            var outputDepth = outputGradient.Shape.GetDimension(2);
            var batchSize = outputGradient.Shape.GetDimension(3);

            for (var n = 0; n < batchSize; n++)
            {
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    var x = -horizontalPad;
                    for (var ax = 0; ax < outputWidth; x += verticalStride, ax++)
                    {
                        var y = -verticalPad;
                        for (var ay = 0; ay < outputHeight; y += horizontalStride, ay++)
                        {
                            var a = double.MinValue;
                            int winx = -1, winy = -1;

                            for (var fx = 0; fx < windowWidth; fx++)
                            {
                                for (var fy = 0; fy < windowHeight; fy++)
                                {
                                    var oy = y + fy;
                                    var ox = x + fx;
                                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                                    {
                                        var v = input.Get(ox, oy, depth, n);
                                        // perform max pooling and store pointers to where
                                        // the max came from. This will speed up backprop 
                                        // and can help make nice visualizations in future
                                        if (v > a)
                                        {
                                            a = v;
                                            winx = ox;
                                            winy = oy;
                                        }
                                    }
                                }
                            }

                            var chainGradient = outputGradient.Get(ax, ay, depth, n);
                            inputGradient.Storage.Set(winx, winy, depth, n, chainGradient);
                        }
                    }
                }
            }
        }

        public override void DoReduce(Volume<double> result, TensorReduceOp op)
        {
            switch (op)
            {
                case TensorReduceOp.Add:
                    DoSum(result);
                    break;
                case TensorReduceOp.Mul:
                    throw new NotImplementedException();
                case TensorReduceOp.Min:
                    throw new NotImplementedException();
                case TensorReduceOp.Max:
                    DoMax(result);
                    break;
                case TensorReduceOp.AMax:
                    throw new NotImplementedException();
                case TensorReduceOp.Avg:
                    throw new NotImplementedException();
                case TensorReduceOp.Norm1:
                    DoNorm1(result);
                    break;
                case TensorReduceOp.Norm2:
                    throw new NotImplementedException();
                default:
                    throw new ArgumentOutOfRangeException(nameof(op), op, null);
            }
        }

        public override void DoRelu(Volume<double> volume)
        {
            this.Storage.Map(x => x <= 0 ? 0 : x, volume.Storage);
        }

        public override void DoReluGradient(Volume<double> input, Volume<double> outputGradient,
            Volume<double> inputGradient)
        {
            this.Storage.Map((x, y) => x > 0 ? y : 0, outputGradient.Storage, inputGradient.Storage);
        }

        public override void DoSigmoid(Volume<double> volume)
        {
            this.Storage.Map(x => 1.0 / (1.0 + Math.Exp(-x)), volume.Storage);
        }

        public override void DoSigmoidGradient(Volume<double> input, Volume<double> outputGradient,
            Volume<double> inputGradient)
        {
            this.Storage.Map((output, outGradient) => output * (1.0 - output) * outGradient, outputGradient.Storage,
                inputGradient.Storage);
        }

        public override void DoSoftmax(Volume<double> result)
        {
            var batchSize = this.Shape.GetDimension(3);

            var outputWidth = this.Shape.GetDimension(0);
            var outputHeight = this.Shape.GetDimension(1);
            var outputDepth = this.Shape.GetDimension(2);

            for (var n = 0; n < batchSize; n++)
            {
                // compute max activation
                var amax = double.MinValue;
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            var v = Get(ax, ay, depth, n);
                            if (v > amax)
                            {
                                amax = v;
                            }
                        }
                    }
                }

                // compute exponentials (carefully to not blow up)
                var es = new double[outputDepth * outputHeight * outputWidth];
                var esum = 0.0;

                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            var e = Math.Exp(Get(ax, ay, depth, n) - amax);
                            esum += e;
                            es[ax + ay * outputWidth + depth * outputWidth * outputHeight] = e;
                        }
                    }
                }

                // normalize and output to sum to one
                for (var depth = 0; depth < outputDepth; depth++)
                {
                    for (var ay = 0; ay < outputHeight; ay++)
                    {
                        for (var ax = 0; ax < outputWidth; ax++)
                        {
                            es[ax + ay * outputWidth + depth * outputWidth * outputHeight] /= esum;

                            result.Storage.Set(ax, ay, depth, n,
                                es[ax + ay * outputWidth + depth * outputWidth * outputHeight]);
                        }
                    }
                }
            }
        }

        public override void DoSoftmaxGradient(Volume<double> output, Volume<double> outputGradient, Volume<double> inputGradient)
        {
            var batchSize = this.Shape.TotalLength == 1 ? 1 : this.Shape.GetDimension(-1);

            var outputReshape = output.ReShape(-1, batchSize);
            var outputGradientReshape = outputGradient.ReShape(-1, batchSize);
            var inputGradientReshape = inputGradient.ReShape(-1, batchSize);

            var firstDim = outputReshape.Shape.GetDimension(0);

            for (var b = 0; b < batchSize; b++)
            {
                var classIndex = -1;

                for (var i = 0; i < firstDim; i++)
                {
                    var yi = outputGradientReshape.Get(i, b);

                    if (yi == 1.0)
                    {
                        classIndex = i;
                    }
                }

                var pj = outputReshape.Get(classIndex, b);

                // input gradient:
                // pi(1 - pi) if i = class index
                // -pipj if i != class index
                for (var i = 0; i < firstDim; i++)
                {
                    var pi = outputReshape.Get(i, b);

                    if (i == classIndex)
                    {
                        inputGradientReshape.Set(i, b, pj * (1.0 - pj));
                    }
                    else
                    {
                        inputGradientReshape.Set(i, b, -pj * pi);
                    }
                }
            }
        }

        public override void DoSubtractFrom(Volume<double> other, Volume<double> result)
        {
            this.Storage.MapEx((x, y) => y - x, other.Storage, result.Storage);
        }

        public override void DoSum(Volume<double> result)
        {
            var batchSize = this.Shape.DimensionCount > 1 ? this.Shape.GetDimension(-1) : 1;
            var inputReshape = ReShape(-1, batchSize);

            var n = inputReshape.Shape.GetDimension(0);

            if (batchSize > 1 && result.Shape.DimensionCount > 1 && result.Shape.GetDimension(3) == 1)
            {
                var resultReshape = result.ReShape(-1, 1);

                for (var j = 0; j < n; j++)
                {
                    var sum = 0.0;

                    // Sum over batch
                    for (var i = 0; i < batchSize; i++)
                    {
                        var d = inputReshape.Get(j, i);
                        sum += d;
                    }

                    resultReshape.Set(j, 0, sum);
                }

            }
            else
            {
                for (var i = 0; i < batchSize; i++)
                {
                    var sum = 0.0;

                    for (var j = 0; j < n; j++)
                    {
                        var d = inputReshape.Get(j, i);
                        sum += d;
                    }

                    result.Set(new[] { i }, sum);
                }
            }
        }

        public override void DoNorm1(Volume<double> result)
        {
            var batchSize = this.Shape.DimensionCount > 1 ? this.Shape.GetDimension(-1) : 1;
            var reshape = ReShape(-1, batchSize);

            var n = reshape.Shape.GetDimension(0);

            for (var i = 0; i < batchSize; i++)
            {
                var sum = 0.0;

                for (var j = 0; j < n; j++)
                {
                    var d = reshape.Get(j, i);
                    sum += Math.Abs(d);
                }

                result.Set(new[] { i }, sum);
            }
        }

        public override void DoTanh(Volume<double> volume)
        {
            this.Storage.Map(Math.Tanh, volume.Storage);
        }

        public override void DoTanhGradient(Volume<double> input, Volume<double> outputGradient,
            Volume<double> inputGradient)
        {
            this.Storage.Map((output, outGradient) => (1.0 - output * output) * outGradient, outputGradient.Storage,
                inputGradient.Storage);
        }
    }
}