using SupportVectorMachine;
using System;
using System.Collections.Generic;

public class KernelClass
{
    public String Name;
    public KernelType Type;
    public List<double> Parameters = new List<double>();
    public List<string> ParameterNames = new List<string>();
    public int FreeParameters;

    public KernelClass(string name, KernelType type, List<double> parameters, List<string> names)
    {
        Name = name;
        Type = type;

        if (parameters != null && parameters.Count > 0)
        {
            Parameters.Clear();
            Parameters.AddRange(parameters);

            FreeParameters = parameters.Count;
        }

        if (names != null && names.Count > 0)
        {
            ParameterNames.Clear();
            ParameterNames.AddRange(names);
        }
    }
}

public class ModelKernel
{
    public KernelClass Kernel;
    public int Category;
    public double C;
    public int MaxPasses;
    public double Tolerance;

    public ModelKernel(KernelClass kernel, int category, double c, double tolerance, int maxpasses)
    {
        Kernel = kernel;
        Category = category;
        C = c;
        Tolerance = tolerance;
        MaxPasses = maxpasses;
    }
}
