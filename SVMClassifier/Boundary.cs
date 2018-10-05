using DeepLearnCS;
using Gdk;
using OxyPlot;
using System;
using System.Collections.Generic;

namespace SupportVectorMachine
{
	public static class Boundary
	{
		static double deltax;
		static double deltay;
		static double minx, maxx;
		static double miny, maxy;

		static int Rows(ManagedArray x)
		{
			return x.y;
		}

		static int Cols(ManagedArray x)
		{
			return x.x;
		}

		public static void Points(Pixbuf pixbuf, ManagedArray x, ManagedIntList c, int f1 = 0, int f2 = 0)
		{
			f1 = f1 >= 0 && f1 < Cols(x) ? f1 : 0;
			f2 = f2 >= 0 && f2 < Cols(x) ? f2 : 0;

			if (pixbuf != null)
			{
				for (var i = 0; i < Rows(x); i++)
				{
					if (Math.Abs(deltax) > 0.0 && Math.Abs(deltay) > 0.0)
					{
						var xp = (int)((x[f1, i] - minx) / deltax);
						var yp = (int)((x[f2, i] - miny) / deltay);

						Common.Circle(pixbuf, xp, yp, 2, c[i] != 0 ? new Color(255, 0, 0) : new Color(0, 0, 255));
					}
				}
			}
		}

		public static Pixbuf Plot(ManagedArray x, Model model, int width, int height, int f1 = 0, int f2 = 1)
		{
			var pixbuf = Common.Pixbuf(width, height, new Color(255, 255, 255));

			var m = Rows(x);

			var xplot = new double[width];
			var yplot = new double[height];

			minx = Double.MaxValue;
			maxx = Double.MinValue;

			miny = Double.MaxValue;
			maxy = Double.MinValue;

			f1 = f1 >= 0 && f1 < Cols(x) ? f1 : 0;
			f2 = f2 >= 0 && f2 < Cols(x) ? f2 : 0;

			for (var j = 0; j < m; j++)
			{
				minx = Math.Min(x[f1, j], minx);
				maxx = Math.Max(x[f1, j], maxx);

				miny = Math.Min(x[f2, j], miny);
				maxy = Math.Max(x[f2, j], maxy);
			}

			deltax = (maxx - minx) / width;
			deltay = (maxy - miny) / height;

			minx = minx - 8 * deltax;
			maxx = maxx + 8 * deltax;
			miny = miny - 8 * deltay;
			maxy = maxy + 8 * deltay;

			deltax = (maxx - minx) / width;
			deltay = (maxy - miny) / height;

			var classification = model.Classify(x);

			Points(pixbuf, x, classification, f1, f2);

			ManagedOps.Free(classification);

			return pixbuf;
		}

		public static void Plot(Pixbuf pixbuf, ManagedArray x, Model model, int f1 = 0, int f2 = 1)
		{
			var classification = model.Classify(x);

			Points(pixbuf, x, classification, f1, f2);

			ManagedOps.Free(classification);
		}

		static Pixbuf ContourGraph;
		static List<Color> Colors = new List<Color>();

		public static void InitializeContour(int zlevels, int width, int height)
		{
			if (ContourGraph != null)
				Common.Free(ContourGraph);

			ContourGraph = Common.Pixbuf(width, height, new Color(255, 255, 255));

			Colors.Clear();

			if (zlevels > 0)
				Colors.Add(new Color(0, 0, 255));

			if (zlevels > 1)
				Colors.Add(new Color(0, 255, 0));

			if (zlevels > 2)
				Colors.Add(new Color(255, 0, 0));
		}

		public static void ContourLine(double x1, double y1, double x2, double y2, double z)
		{
			if (ContourGraph != null)
			{
				if (Math.Abs(deltax) > 0.0 && Math.Abs(deltay) > 0.0)
				{
					var xs = (int)((x1 - minx) / deltax);
					var ys = (int)((y1 - miny) / deltay);
					var xe = (int)((x2 - minx) / deltax);
					var ye = (int)((y2 - miny) / deltay);

					var c = (int)z + 1;

					if (c >= 0 && c < Colors.Count)
						Common.Line(ContourGraph, xs, ys, xe, ye, Colors[c]);
				}
			}
		}

		public static Pixbuf Contour(ManagedArray x, Model model, int width, int height, int f1 = 0, int f2 = 1)
		{
			InitializeContour(3, width, height);

			var m = Rows(x);

			var xplot = new double[width];
			var yplot = new double[height];
			var data = new double[height, width];

			minx = Double.MaxValue;
			maxx = Double.MinValue;

			miny = Double.MaxValue;
			maxy = Double.MinValue;

			f1 = f1 >= 0 && f1 < Cols(x) ? f1 : 0;
			f2 = f2 >= 0 && f2 < Cols(x) ? f2 : 0;

			for (var j = 0; j < m; j++)
			{
				minx = Math.Min(x[f1, j], minx);
				maxx = Math.Max(x[f1, j], maxx);

				miny = Math.Min(x[f2, j], miny);
				maxy = Math.Max(x[f2, j], maxy);
			}

			deltax = (maxx - minx) / width;
			deltay = (maxy - miny) / height;

			minx = minx - 8 * deltax;
			maxx = maxx + 8 * deltax;
			miny = miny - 8 * deltay;
			maxy = maxy + 8 * deltay;

			deltax = (maxx - minx) / width;
			deltay = (maxy - miny) / height;

			// For predict
			for (var i = 0; i < width; i++)
			{
				xplot[i] = minx + i * deltax;
			}

			for (var i = 0; i < height; i++)
			{
				yplot[i] = miny + i * deltay;
			}

			var xx = new ManagedArray(2, height);

			for (var i = 0; i < width; i++)
			{
				for (var j = 0; j < height; j++)
				{
					xx[f1, j] = xplot[i];
					xx[f2, j] = yplot[j];
				}

				var p = model.Predict(xx);

				for (var j = 0; j < height; j++)
				{
					data[i, j] = p[j];
				}

				ManagedOps.Free(p);
			}

			var z = new double[] { -1.0, 0.0, 1.0 };

			Conrec.Contour(data, xplot, yplot, z, ContourLine);

			Plot(ContourGraph, x, model, f1, f2);

			ManagedOps.Free(xx);

			var border = new Color(128, 128, 128);

			var cw = ContourGraph.Width - 1;
			var ch = ContourGraph.Height - 1;

			Common.Line(ContourGraph, 0, 0, cw, 0, border);
			Common.Line(ContourGraph, cw, 0, cw, ch, border);
			Common.Line(ContourGraph, 0, ch, cw, ch, border);
			Common.Line(ContourGraph, 0, 0, 0, ch, border);

			return ContourGraph;
		}

		public static void Free()
		{
			Common.Free(ContourGraph);
		}
	}
}
