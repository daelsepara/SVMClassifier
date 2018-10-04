using DeepLearnCS;
using System;

namespace SupportVectorMachine
{
	public enum ModelParameters
	{
		X = 0,
		Y = 1,
		ALPHA = 2,
		W = 3,
		B = 4
	};

	public class Model
	{
		public ManagedArray ModelX;
		public ManagedArray ModelY;
		public KernelType Type;
		public ManagedArray KernelParam;
		public ManagedArray Alpha;
		public ManagedArray W;
		public double B;
		public double C;
		public double Tolerance;
		public int Category;
		public int Passes;
		public int Iterations;
		public int MaxIterations;
		public bool Trained;

		Random random = new Random(Guid.NewGuid().GetHashCode());

		readonly double Mepsilon = Math.Pow(2, -52);

		// Internal variables
		ManagedArray K;
		ManagedArray E;
		ManagedArray alpha;
		ManagedArray dx;
		ManagedArray dy;
		ManagedArray kparam;
		double b;
		double eta;
		double H;
		double L;
		KernelType ktype;

		public Model()
		{

		}

		public Model(ManagedArray x, ManagedArray y, KernelType type, ManagedArray kernelParam, ManagedArray alpha, double b, ManagedArray w, int passes)
		{
			ModelX = x;
			ModelY = y;
			Type = type;
			KernelParam = kernelParam;
			Alpha = alpha;
			B = b;
			W = w;
			Passes = passes;
			Trained = true;
		}

		int Rows(ManagedArray x)
		{
			return x.y;
		}

		int Cols(ManagedArray x)
		{
			return x.x;
		}

		public void Setup(ManagedArray x, ManagedArray y, double c, KernelType kernel, ManagedArray param, double tolerance = 0.001, int maxpasses = 5, int category = 1)
		{
			ManagedOps.Free(dx, dy);
			dx = new ManagedArray(x);
			dy = new ManagedArray(y);

			ManagedOps.Copy2D(dx, x, 0, 0);
			ManagedOps.Copy2D(dy, y, 0, 0);

			ktype = kernel;

			// Data parameters
			var m = Rows(dx);

			Category = category;
			MaxIterations = maxpasses;
			Tolerance = tolerance;
			C = c;

			// Reset internal variables
			ManagedOps.Free(K, kparam, E, alpha);

			kparam = new ManagedArray(param);
			ManagedOps.Copy2D(kparam, param, 0, 0);

			// Variables
			alpha = new ManagedArray(1, m);
			E = new ManagedArray(1, m);
			b = 0.0;
			Iterations = 0;

			// Pre-compute the Kernel Matrix since our dataset is small
			// (In practice, optimized SVM packages that handle large datasets
			// gracefully will *not* do this)
			if (kernel == KernelType.LINEAR)
			{
				// Computation for the Linear Kernel
				// This is equivalent to computing the kernel on every pair of examples
				var tinput = ManagedMatrix.Transpose(dx);

				K = ManagedMatrix.Multiply(dx, tinput);

				var slope = kparam.Length() > 0 ? kparam[0] : 1.0;
				var inter = kparam.Length() > 1 ? kparam[1] : 0.0;

				ManagedMatrix.Multiply(K, slope);
				ManagedMatrix.Add(K, inter);

				ManagedOps.Free(tinput);
			}
			else if (kernel == KernelType.GAUSSIAN)
			{
				// RBF Kernel
				// This is equivalent to computing the kernel on every pair of examples
				var pX2 = ManagedMatrix.Pow(dx, 2.0);
				var rX2 = ManagedMatrix.RowSums(pX2);
				var tX2 = ManagedMatrix.Transpose(rX2);
				var trX = ManagedMatrix.Transpose(dx);

				var tempK = new ManagedArray(m, m);
				var temp1 = new ManagedArray(m, m);
				var temp2 = ManagedMatrix.Multiply(dx, trX);

				ManagedMatrix.Expand(rX2, m, 1, tempK);
				ManagedMatrix.Expand(tX2, 1, m, temp1);
				ManagedMatrix.Multiply(temp2, -2.0);

				ManagedMatrix.Add(tempK, temp1);
				ManagedMatrix.Add(tempK, temp2);

				var sigma = kparam.Length() > 0 ? kparam[0] : 1.0;

				var g = Math.Abs(sigma) > 0.0 ? Math.Exp(-1.0 / (2.0 * sigma * sigma)) : 0.0;

				K = ManagedMatrix.Pow(g, tempK);

				ManagedOps.Free(pX2, rX2, tX2, trX, tempK, temp1, temp2);
			}
			else
			{
				// Pre-compute the Kernel Matrix
				// The following can be slow due to the lack of vectorization
				K = new ManagedArray(m, m);

				var Xi = new ManagedArray(Cols(dx), 1);
				var Xj = new ManagedArray(Cols(dx), 1);

				for (var i = 0; i < m; i++)
				{
					for (var j = 0; j < m; j++)
					{
						ManagedOps.Copy2D(Xi, dx, 0, i);
						ManagedOps.Copy2D(Xj, dx, 0, j);

						K[j, i] = KernelFunction.Run(kernel, Xi, Xj, kparam);

						// the matrix is symmetric
						K[i, j] = K[j, i];
					}
				}

				ManagedOps.Free(Xi, Xj);
			}

			eta = 0.0;
			L = 0.0;
			H = 0.0;

			// Map 0 (or other categories) to -1
			for (var i = 0; i < Rows(dy); i++)
			{
				dy[i] = (int)dy[i] != Category ? -1.0 : 1.0;
			}
		}

		public bool Step()
		{
			if (Iterations >= MaxIterations)
				return true;

			// Data parameters
			var m = Rows(dy);

			var num_changed_alphas = 0;

			for (var i = 0; i < m; i++)
			{
				// Calculate Ei = f(x(i)) - y(i) using (2).
				E[i] = b;

				for (var yy = 0; yy < m; yy++)
				{
					E[i] += alpha[yy] * dy[yy] * K[i, yy];
				}

				E[i] -= dy[i];

				if ((dy[i] * E[i] < -Tolerance && alpha[i] < C) || (dy[i] * E[i] > Tolerance && alpha[i] > 0.0))
				{
					// In practice, there are many heuristics one can use to select
					// the i and j. In this simplified code, we select them randomly.
					var j = i;

					while (j == i)
					{
						// Make sure i != j
						j = (int)Math.Floor(m * random.NextDouble());
					}

					// Calculate Ej = f(x(j)) - y(j) using (2).
					E[j] = b;

					for (var yy = 0; yy < m; yy++)
					{
						E[j] += alpha[yy] * dy[yy] * K[j, yy];
					}

					E[j] -= dy[j];

					// Save old alphas
					var alpha_i_old = alpha[i];
					var alpha_j_old = alpha[j];

					// Compute L and H by (10) or (11). 
					if ((int)dy[i] == (int)dy[j])
					{
						L = Math.Max(0, alpha[j] + alpha[i] - C);
						H = Math.Min(C, alpha[j] + alpha[i]);
					}
					else
					{
						L = Math.Max(0, alpha[j] - alpha[i]);
						H = Math.Min(C, C + alpha[j] - alpha[i]);
					}

					if (Math.Abs(L - H) <= Mepsilon)
					{
						// continue to next i 
						continue;
					}

					// Compute eta by (14).
					eta = 2.0 * K[j, i] - K[i, i] - K[j, j];

					if (eta >= 0.0)
					{
						// continue to next i. 
						continue;
					}

					// Compute and clip new value for alpha j using (12) and (15).
					alpha[j] = alpha[j] - (dy[j] * (E[i] - E[j])) / eta;

					// Clip
					alpha[j] = Math.Min(H, alpha[j]);
					alpha[j] = Math.Max(L, alpha[j]);

					// Check if change in alpha is significant
					if (Math.Abs(alpha[j] - alpha_j_old) < Tolerance)
					{
						// continue to next i. 
						// replace anyway
						alpha[j] = alpha_j_old;

						continue;
					}

					// Determine value for alpha i using (16). 
					alpha[i] = alpha[i] + dy[i] * dy[j] * (alpha_j_old - alpha[j]);

					// Compute b1 and b2 using (17) and (18) respectively. 
					var b1 = b - E[i] - dy[i] * (alpha[i] - alpha_i_old) * K[j, i] - dy[j] * (alpha[j] - alpha_j_old) * K[j, i];
					var b2 = b - E[j] - dy[i] * (alpha[i] - alpha_i_old) * K[j, i] - dy[j] * (alpha[j] - alpha_j_old) * K[j, j];

					// Compute b by (19). 
					if (0.0 < alpha[i] && alpha[i] < C)
					{
						b = b1;
					}
					else if (0.0 < alpha[j] && alpha[j] < C)
					{
						b = b2;
					}
					else
					{
						b = (b1 + b2) / 2.0;
					}

					num_changed_alphas++;
				}
			}

			if (num_changed_alphas == 0)
			{
				Iterations++;
			}

			return Iterations >= MaxIterations;
		}

		public void Generate()
		{
			var m = Rows(dx);
			var n = Cols(dx);

			var idx = 0;

			for (var i = 0; i < m; i++)
			{
				if (Math.Abs(alpha[i]) > 0.0)
				{
					idx++;
				}
			}

			ManagedOps.Free(ModelX, ModelY, Alpha, W, KernelParam);

			ModelX = new ManagedArray(Cols(dx), idx);
			ModelY = new ManagedArray(1, idx);
			Alpha = new ManagedArray(1, idx);
			KernelParam = new ManagedArray(kparam);

			var ii = 0;

			for (var i = 0; i < m; i++)
			{
				if (Math.Abs(alpha[i]) > 0.0)
				{
					for (int j = 0; j < n; j++)
					{
						ModelX[j, ii] = dx[j, i];
					}

					ModelY[ii] = dy[i];

					Alpha[ii] = alpha[i];

					ii++;
				}
			}

			B = b;
			Passes = Iterations;
			ManagedOps.Copy2D(KernelParam, kparam, 0, 0);
			Type = ktype;

			var axy = ManagedMatrix.BSXMUL(alpha, dy);
			var tay = ManagedMatrix.Transpose(axy);
			var txx = ManagedMatrix.Multiply(tay, dx);

			W = ManagedMatrix.Transpose(txx);

			Trained = true;

			ManagedOps.Free(dx, dy, K, kparam, E, alpha, axy, tay, txx);
		}

		// SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
		// algorithm.
		//
		// [model] = svm_train(X, Y, C, kernelFunction, kernelParam, tol, max_passes) trains an
		// SVM classifier and returns trained model. X is the matrix of training 
		// examples.  Each row is a training example, and the jth column holds the 
		// jth feature.  Y is a column matrix containing 1 for positive examples 
		// and 0 for negative examples.  C is the standard SVM regularization 
		// parameter.  tol is a tolerance value used for determining equality of 
		// floating point numbers. max_passes controls the number of iterations
		// over the dataset (without changes to alpha) before the algorithm quits.
		//
		// Note: This is a simplified version of the SMO algorithm for training
		// SVMs. In practice, if you want to train an SVM classifier, we
		// recommend using an optimized package such as:  
		//
		// LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
		// SVMLight (http://svmlight.joachims.org/)
		//
		// Converted to R by: SD Separa (2016/03/18)
		// Converted to C# by: SD Separa (2018/09/29)
		//
		public void Train(ManagedArray x, ManagedArray y, double c, KernelType kernel, ManagedArray param, double tolerance = 0.001, int maxpasses = 5, int category = 1)
		{
			Setup(x, y, c, kernel, param, tolerance, maxpasses, category);

			// Train
			while (!Step()) { }

			Generate();
		}

		// SVMPREDICT returns a vector of predictions using a trained SVM model
		//(svm_train). 
		//
		// pred = SVMPREDICT(model, X) returns a vector of predictions using a 
		// trained SVM model (svm_train). X is a mxn matrix where there each 
		// example is a row. model is a svm model returned from svm_train.
		// predictions pred is a m x 1 column of predictions of {0, 1} values.
		//
		// Converted to R by: SD Separa (2016/03/18)
		// Converted to C# by: SD Separa (2018/09/29)
		public ManagedArray Predict(ManagedArray input)
		{
			var predictions = new ManagedArray(1, Rows(input));

			if (Trained)
			{
				var x = new ManagedArray(input);

				if (Cols(x) == 1)
				{
					ManagedMatrix.Transpose(x, input);
				}
				else
				{
					ManagedOps.Copy2D(x, input, 0, 0);
				}

				var m = Rows(x);

				predictions.Resize(1, m);

				if (Type == KernelType.LINEAR)
				{
					ManagedMatrix.Multiply(predictions, x, W);
					ManagedMatrix.Add(predictions, B);
				}
				else if (Type == KernelType.GAUSSIAN)
				{
					// RBF Kernel
					// This is equivalent to computing the kernel on every pair of examples
					var pX1 = ManagedMatrix.Pow(x, 2.0);
					var pX2 = ManagedMatrix.Pow(ModelX, 2.0);
					var rX2 = ManagedMatrix.RowSums(pX2);

					var X1 = ManagedMatrix.RowSums(pX1);
					var X2 = ManagedMatrix.Transpose(rX2);
					var tX = ManagedMatrix.Transpose(ModelX);
					var tY = ManagedMatrix.Transpose(ModelY);
					var tA = ManagedMatrix.Transpose(Alpha);

					var rows = Rows(X1);
					var cols = Cols(X2);

					var tempK = new ManagedArray(rows, cols);
					var temp1 = new ManagedArray(cols, rows);
					var temp2 = ManagedMatrix.Multiply(x, tX);

					ManagedMatrix.Multiply(temp2, -2.0);

					ManagedMatrix.Expand(X1, cols, 1, tempK);
					ManagedMatrix.Expand(X2, 1, rows, temp1);

					ManagedMatrix.Add(tempK, temp1);
					ManagedMatrix.Add(tempK, temp2);

					var sigma = KernelParam.Length() > 0 ? KernelParam[0] : 1.0;

					var g = Math.Abs(sigma) > 0.0 ? Math.Exp(-1.0 / (2.0 * sigma * sigma)) : 0.0;
					var Kernel = ManagedMatrix.Pow(g, tempK);

					var tempY = new ManagedArray(Cols(tY), rows);
					var tempA = new ManagedArray(Cols(tA), rows);

					ManagedMatrix.Expand(tY, 1, rows, tempY);
					ManagedMatrix.Expand(tA, 1, rows, tempA);

					ManagedMatrix.Product(Kernel, tempY);
					ManagedMatrix.Product(Kernel, tempA);

					var p = ManagedMatrix.RowSums(Kernel);

					ManagedOps.Copy2D(predictions, p, 0, 0);
					ManagedMatrix.Add(predictions, B);

					ManagedOps.Free(pX1, pX2, rX2, X1, X2, tempK, temp1, temp2, tX, tY, tA, tempY, tempA, Kernel, p);
				}
				else
				{
					var Xi = new ManagedArray(Cols(x), 1);
					var Xj = new ManagedArray(Cols(ModelX), 1);

					for (var i = 0; i < m; i++)
					{
						var prediction = 0.0;

						for (var j = 0; j < Rows(ModelX); j++)
						{

							ManagedOps.Copy2D(Xi, x, 0, i);
							ManagedOps.Copy2D(Xj, ModelX, 0, j);

							prediction += Alpha[j] * ModelY[j] * KernelFunction.Run(Type, Xi, Xj, KernelParam);
						}

						predictions[i] = prediction + B;
					}

					ManagedOps.Free(Xi, Xj);
				}

				ManagedOps.Free(x);
			}

			return predictions;
		}

		public ManagedIntList Classify(ManagedArray input, double threshold = 0.0)
		{
			var classification = new ManagedIntList(Rows(input));

			var predictions = Predict(input);

			for (var i = 0; i < predictions.Length(); i++)
			{
				classification[i] = predictions[i] > threshold ? Category : 0;
			}

			ManagedOps.Free(predictions);

			return classification;
		}

		public int Test(ManagedArray output, ManagedIntList classification, int category = 1)
		{
			var errors = 0;

			for (var i = 0; i < classification.Length(); i++)
			{
				var correct = (int)output[i] != category ? 0 : category;

				errors += correct != classification[i] ? 1 : 0;
			}

			return errors;
		}

		public void Free()
		{
			// public variables
			ManagedOps.Free(ModelX, ModelY, Alpha, W, KernelParam);

			// internal variables
			ManagedOps.Free(K, E, alpha, kparam, dx, dy);
		}
	}
}
