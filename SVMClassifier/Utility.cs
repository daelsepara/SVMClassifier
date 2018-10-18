using DeepLearnCS;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;
using SupportVectorMachine;

public static class Utility
{
    public static string Serialize(List<Model> models, ManagedArray normalizationData)
    {
        var modelData = new List<ModelJSON>();
        var normalization = normalizationData.Length() > 0 ? new double[normalizationData.y, normalizationData.x] : new double[0, 0];

        if (models.Count > 0)
        {
            foreach (var model in models)
            {
                var item = Convert(model);

                modelData.Add(item);
            }
        }

        if (normalizationData.Length() > 0 && normalization.GetLength(0) > 0)
        {
            for (var j = 0; j < normalizationData.y; j++)
                for (var i = 0; i < normalizationData.x; i++)
                    normalization[j, i] = normalizationData[i, j];
        }

        var classifier = new Classifier(modelData, normalization);

        string output = JsonConvert.SerializeObject(classifier);

        return output;
    }

    public static string Serialize(Model model, double[,] normalization)
    {
        var classifier = new Classifier(Convert(model), normalization);

        string output = JsonConvert.SerializeObject(classifier);

        return output;
    }

    public static ModelJSON Convert(Model model)
    {
        var json = new ModelJSON
        {
            ModelX = new double[model.ModelX.y, model.ModelX.x],
            ModelY = new double[model.ModelY.Length()],
            Type = (int)model.Type,
            KernelParam = new double[model.KernelParam.Length()],
            Alpha = new double[model.Alpha.Length()],
            W = new double[model.W.Length()],
            B = model.B,
            C = model.C,
            Tolerance = model.Tolerance,
            Category = model.Category,
            Passes = model.Passes,
            Iterations = model.Iterations,
            MaxIterations = model.MaxIterations,
            Trained = model.Trained
        };

        for (var j = 0; j < model.ModelX.y; j++)
            for (var i = 0; i < model.ModelX.x; i++)
                json.ModelX[j, i] = model.ModelX[i, j];

        for (var j = 0; j < model.ModelY.Length(); j++)
            json.ModelY[j] = model.ModelY[j];

        for (var j = 0; j < model.KernelParam.Length(); j++)
            json.KernelParam[j] = model.KernelParam[j];

        for (var j = 0; j < model.W.Length(); j++)
            json.W[j] = model.W[j];

        for (var j = 0; j < model.Alpha.Length(); j++)
            json.Alpha[j] = model.Alpha[j];

        return json;
    }

    public static string LoadJson(string FileName)
    {
        var json = "";

        if (File.Exists(FileName))
        {
            using (TextReader reader = File.OpenText(FileName))
            {
                string line = "";

                do
                {
                    line = reader.ReadLine();

                    if (!string.IsNullOrEmpty(line))
                        json += line;
                }
                while (!string.IsNullOrEmpty(line));
            }
        }

        return json;
    }

    public static List<Model> Deserialize(string json, ManagedArray normalization)
    {
        var models = new List<Model>();

        Classifier classifier = JsonConvert.DeserializeObject<Classifier>(json);

        if (classifier != null && classifier.Normalization != null)
        {
            var data = classifier.Normalization;

            normalization.Resize(data.GetLength(1), data.GetLength(0));

            for (var j = 0; j < data.GetLength(0); j++)
            {
                for (var i = 0; i < data.GetLength(1); i++)
                {
                    normalization[i, j] = data[j, i];
                }
            }
        }

        if (classifier.Models.Count > 0)
        {
            foreach (var model in classifier.Models)
            {
                var itemx = model.ModelX;
                var itemy = model.ModelY;
                var itemw = model.W;
                var itemk = model.KernelParam;
                var itema = model.Alpha;

                var ModelX = new ManagedArray(itemx.GetLength(1), itemx.GetLength(0));
                var ModelY = new ManagedArray(1, itemy.GetLength(0));
                var KernelParam = new ManagedArray(itemk.GetLength(0));
                var W = new ManagedArray(1, itemw.GetLength(0));
                var Alpha = new ManagedArray(1, itema.GetLength(0));

                for (var j = 0; j < itemx.GetLength(0); j++)
                    for (var i = 0; i < itemx.GetLength(1); i++)
                        ModelX[i, j] = itemx[j, i];

                for (var j = 0; j < itemy.GetLength(0); j++)
                    ModelY[j] = itemy[j];

                for (var j = 0; j < itemk.GetLength(0); j++)
                    KernelParam[j] = itemk[j];

                for (var j = 0; j < itemw.GetLength(0); j++)
                    W[j] = itemw[j];

                for (var j = 0; j < itema.GetLength(0); j++)
                    Alpha[j] = itema[j];

                var Type = (KernelType)model.Type;

                var svmmodel = new Model(ModelX, ModelY, Type, KernelParam, Alpha, model.B, W, model.Passes)
                {
                    C = model.C,
                    Tolerance = model.Tolerance,
                    Category = model.Category,
                    Iterations = model.Iterations,
                    MaxIterations = model.MaxIterations,
                    Trained = model.Trained
                };

                models.Add(svmmodel);
            }
        }

        return models;
    }
}
