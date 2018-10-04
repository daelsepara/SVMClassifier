using Gdk;
using System;
using System.Runtime.InteropServices;

public static class Common
{
	public static uint C2I(Color color)
	{
		return (uint)((color.Red << 16) | (color.Green << 8) | color.Blue);
	}

	public static Color I2C(uint color)
	{
		byte a = (byte)(color >> 24);
		byte r = (byte)(color >> 16);
		byte g = (byte)(color >> 8);
		byte b = (byte)(color >> 0);

		return new Color(r, g, b);
	}

	public static Pixbuf Pixbuf(int width, int height, Color c)
	{
		var pixbuf = new Pixbuf(Colorspace.Rgb, false, 8, width, height);

		pixbuf.Fill(C2I(c));

		return pixbuf;
	}

	public static Pixbuf Pixbuf(int width, int height)
	{
		var pixbuf = Pixbuf(width, height, new Color(0, 0, 0));

		return pixbuf;
	}

	public static void Point(Pixbuf pixbuf, int xp, int yp, Color c)
	{
		if (pixbuf == null)
			return;

		var yr = pixbuf.Height - yp;

		if (xp >= 0 && xp < pixbuf.Width && yr >= 0 && yr < pixbuf.Height)
		{
			var ptr = pixbuf.Pixels + yr * pixbuf.Rowstride + xp * pixbuf.NChannels;

			Marshal.WriteByte(ptr, 0, (byte)c.Red);
			Marshal.WriteByte(ptr, 1, (byte)c.Green);
			Marshal.WriteByte(ptr, 2, (byte)c.Blue);
		}
	}

	public static void Circle(Pixbuf pixbuf, int xc, int yc, int x, int y, Color color)
	{
		for (var i = xc - x; i <= xc + x; i++)
			Common.Point(pixbuf, i, yc + y, color);

		for (var i = xc - x; i <= xc + x; i++)
			Common.Point(pixbuf, i, yc - y, color);

		for (var i = xc - y; i <= xc + y; i++)
			Common.Point(pixbuf, i, yc + x, color);

		for (var i = xc - y; i <= xc + y; i++)
			Common.Point(pixbuf, i, yc - x, color);
	}

	public static void Circle(Pixbuf pixbuf, int xc, int yc, int r, Color c)
	{
		int x = 0, y = r;
		int d = 3 - 2 * r;

		while (y >= x)
		{
			// for each pixel we will 
			// draw all eight pixels 
			Circle(pixbuf, xc, yc, x, y, c);

			x++;

			// check for decision parameter 
			// and correspondingly  
			// update d, x, y 
			if (d > 0)
			{
				y--;
				d = d + 4 * (x - y) + 10;
			}
			else
				d = d + 4 * x + 6;

			Circle(pixbuf, xc, yc, x, y, c);
		}
	}

	public static void Line(Pixbuf pixbuf, int x0, int y0, int x1, int y1, Color color)
	{
		int dx = Math.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;

		int dy = Math.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;

		int err = (dx > dy ? dx : -dy) / 2, e2;

		for (; ; )
		{
			Point(pixbuf, x0, y0, color);

			if (x0 == x1 && y0 == y1)
				break;

			e2 = err;

			if (e2 > -dx)
			{
				err -= dy;
				x0 += sx;
			}

			if (e2 < dy)
			{
				err += dx;
				y0 += sy;
			}
		}
	}

	public static void Free(params IDisposable[] trash)
	{
		foreach (var item in trash)
		{
			if (item != null)
			{
				item.Dispose();
			}
		}
	}
}
