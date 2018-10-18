using DeepLearnCS;
using System;

namespace SupportVectorMachine
{
    public enum KernelType
    {
        POLYNOMIAL = 0,
        GAUSSIAN = 1,
        RADIAL = 2,
        SIGMOID = 3,
        LINEAR = 4,
        FOURIER = 5,
        UNKNOWN = -1
    };

    public static class KernelFunction
    {
        static int Rows(ManagedArray x)
        {
            return x.y;
        }

        static int Cols(ManagedArray x)
        {
            return x.x;
        }

        static void Vectorize(ManagedArray x1, ManagedArray x2)
        {
            // Reshape into column vectors
            ManagedMatrix.Vector(x1);
            ManagedMatrix.Vector(x2);
        }

        static double Multiply(ManagedArray x1, ManagedArray x2)
        {
            Vectorize(x1, x2);

            var tx = ManagedMatrix.Transpose(x1);
            var xx = ManagedMatrix.Multiply(tx, x2);

            var x = xx[0];

            ManagedOps.Free(tx, xx);

            return x;
        }

        static double SquaredDiff(ManagedArray x1, ManagedArray x2)
        {
            Vectorize(x1, x2);

            double x = 0;

            for (var i = 0; i < x1.Length(); i++)
            {
                var d = x1[i] - x2[i];

                x += d * d;
            }

            return x;
        }

        public static double Linear(ManagedArray x1, ManagedArray x2, ManagedArray k)
        {
            var x = Multiply(x1, x2);

            double m = k.Length() > 0 ? k[0] : 1;
            double b = k.Length() > 1 ? k[1] : 0;

            return x * m + b;
        }

        public static double Polynomial(ManagedArray x1, ManagedArray x2, ManagedArray k)
        {
            double b = k.Length() > 0 ? k[0] : 0;
            double a = k.Length() > 1 ? k[1] : 1;

            return Math.Pow(Multiply(x1, x2) + b, a);
        }

        public static double Gaussian(ManagedArray x1, ManagedArray x2, ManagedArray k)
        {
            var x = SquaredDiff(x1, x2);

            double sigma = k.Length() > 0 ? k[0] : 1;

            double denum = 2 * sigma * sigma;

            return Math.Abs(denum) > 0 ? Math.Exp(-x / denum) : 0;
        }

        public static double Radial(ManagedArray x1, ManagedArray x2, ManagedArray k)
        {
            double sigma = k.Length() > 0 ? k[0] : 1;

            double denum = 2 * sigma * sigma;

            return Math.Abs(denum) > 0 ? Math.Exp(-Math.Sqrt(SquaredDiff(x1, x2)) / denum) : 0;
        }

        public static double Sigmoid(ManagedArray x1, ManagedArray x2, ManagedArray k)
        {
            double m = k.Length() > 0 ? k[0] : 1;
            double b = k.Length() > 1 ? k[1] : 0;

            return Math.Tanh(m * Multiply(x1, x2) / x1.Length() + b);
        }

        public static double Fourier(ManagedArray x1, ManagedArray x2, ManagedArray k)
        {
            Vectorize(x1, x2);

            var z = new ManagedArray(x1);

            double prod = 0;

            double m = k.Length() > 0 ? k[0] : 1;

            for (var i = 0; i < x1.Length(); i++)
            {
                z[i] = Math.Sin(m + (double)1 / 2) * 2;

                var d = x1[i] - x2[i];

                z[i] = Math.Abs(d) > 0 ? Math.Sin(m + (double)1 / 2) * d / Math.Sin(d * (double)1 / 2) : z[i];

                prod = (i == 0) ? z[i] : prod * z[i];
            }

            ManagedOps.Free(z);

            return prod;
        }

        public static double Run(KernelType type, ManagedArray x1, ManagedArray x2, ManagedArray k)
        {
            double result = 0;

            if (type == KernelType.LINEAR)
            {
                return Linear(x1, x2, k);
            }

            if (type == KernelType.GAUSSIAN)
            {
                return Gaussian(x1, x2, k);
            }

            if (type == KernelType.FOURIER)
            {
                return Fourier(x1, x2, k);
            }

            if (type == KernelType.SIGMOID)
            {
                return Sigmoid(x1, x2, k);
            }

            if (type == KernelType.RADIAL)
            {
                return Radial(x1, x2, k);
            }

            if (type == KernelType.POLYNOMIAL)
            {
                return Polynomial(x1, x2, k);
            }

            return result;
        }
    }
}
