using System.Collections.Generic;

namespace SupportVectorMachine
{
    public class Classifier
    {
        public List<ModelJSON> Models;
        public double[,] Normalization;

        public Classifier()
        {

        }

        public Classifier(List<ModelJSON> models, double[,] normalization)
        {
            Models = models;
            Normalization = normalization;
        }

        public Classifier(ModelJSON model, double[,] normalization)
        {
            Models = new List<ModelJSON>
            {
                model
            };

            Normalization = normalization;
        }
    }

    public class ModelJSON
    {
        public double[,] ModelX;
        public double[] ModelY;
        public int Type;
        public double[] KernelParam;
        public double[] Alpha;
        public double[] W;
        public double B;
        public double C;
        public double Tolerance;
        public int Category;
        public int Passes;
        public int Iterations;
        public int MaxIterations;
        public bool Trained;
    }
}
