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

			var x = 0.0;

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

			var m = k.Length() > 0 ? k[0] : 1.0;
			var b = k.Length() > 1 ? k[1] : 0.0;

			return x * m + b;
		}

		public static double Polynomial(ManagedArray x1, ManagedArray x2, ManagedArray k)
		{
			var b = k.Length() > 0 ? k[0] : 0.0;
			var a = k.Length() > 1 ? k[1] : 1.0;

			return Math.Pow(Multiply(x1, x2), a) + b;
		}

		public static double Gaussian(ManagedArray x1, ManagedArray x2, ManagedArray k)
		{
			var x = SquaredDiff(x1, x2);

			var sigma = k.Length() > 0 ? k[0] : 1.0;

			var denum = 2.0 * sigma * sigma;

			return Math.Abs(denum) > 0.0 ? Math.Exp(-x / denum) : 0.0;
		}

		public static double Radial(ManagedArray x1, ManagedArray x2, ManagedArray k)
		{
			var sigma = k.Length() > 0 ? k[0] : 1.0;

			var denum = 2.0 * sigma * sigma;

			return Math.Abs(denum) > 0.0 ? Math.Exp(-Math.Sqrt(SquaredDiff(x1, x2)) / denum) : 0.0;
		}

		public static double Sigmoid(ManagedArray x1, ManagedArray x2, ManagedArray k)
		{
			var m = k.Length() > 0 ? k[0] : 1.0;
			var b = k.Length() > 1 ? k[1] : 0.0;

			return Math.Tanh(m * Multiply(x1, x2) / x1.Length() + b);
		}

		public static double Fourier(ManagedArray x1, ManagedArray x2, ManagedArray k)
		{
			Vectorize(x1, x2);

			var z = new ManagedArray(x1);

			var prod = 0.0;

			var m = k.Length() > 0 ? k[0] : 1.0;

			for (var i = 0; i < x1.Length(); i++)
			{
				z[i] = Math.Sin(m + 0.5) * 2;

				var d = x1[i] - x2[i];

				z[i] = Math.Abs(d) > 0.0 ? Math.Sin(m + 0.5) * d / Math.Sin(d * 0.5) : z[i];

				prod = (i == 0) ? z[i] : prod * z[i];
			}

			ManagedOps.Free(z);

			return prod;
		}

		public static double Run(KernelType type, ManagedArray x1, ManagedArray x2, ManagedArray k)
		{
			var result = 0.0;

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
