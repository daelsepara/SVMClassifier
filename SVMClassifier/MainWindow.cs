using DeepLearnCS;
using Gdk;
using GLib;
using Gtk;
using SupportVectorMachine;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;

public partial class MainWindow : Gtk.Window
{
	Dialog Confirm;

	// User-supplied data
	ManagedArray InputData = new ManagedArray();
	ManagedArray NormalizationData = new ManagedArray();
	ManagedArray OutputData = new ManagedArray();
	ManagedArray TestData = new ManagedArray();

	// Models for multi-classification
	List<Model> Models = new List<Model>();

	List<Delimiter> Delimiters = new List<Delimiter>();

	String FileTraining, FileTest, FileModels;

	FileChooserDialog TextLoader, JsonLoader, JsonSaver;

	CultureInfo ci = new CultureInfo("en-us");

	List<KernelClass> Kernels = new List<KernelClass>();
	List<ModelKernel> ModelKernels = new List<ModelKernel>();

	bool ControlsEnabled = true;
	bool TrainingDone;

	bool ClassifierInitialized;

	enum Pages
	{
		DATA = 0,
		TRAINING = 1,
		MODELS = 2,
		ABOUT = 3
	};

	bool Paused = true;

	Mutex Processing = new Mutex();

	public MainWindow() : base(Gtk.WindowType.Toplevel)
	{
		Build();

		CultureInfo.DefaultThreadCurrentCulture = ci;
		CultureInfo.DefaultThreadCurrentUICulture = ci;

		InitializeUserInterface();
	}

	protected void EnableControls()
	{
		ControlsEnabled = true;
	}

	protected void DisableControls()
	{
		ControlsEnabled = false;
	}

	protected FileFilter AddFilter(string name, params string[] patterns)
	{
		var filter = new FileFilter { Name = name };

		foreach (var pattern in patterns)
			filter.AddPattern(pattern);

		return filter;
	}

	protected void UpdateDelimiterBox(ComboBox combo, List<Delimiter> delimeters)
	{
		combo.Clear();

		var cell = new CellRendererText();
		combo.PackStart(cell, false);
		combo.AddAttribute(cell, "text", 0);
		var store = new ListStore(typeof(string));
		combo.Model = store;

		foreach (var delimeter in delimeters)
		{
			store.AppendValues(delimeter.Name);
		}

		combo.Active = delimeters.Count > 0 ? 0 : -1;
	}

	protected void UpdateModelsBox(ComboBox combo, List<ModelKernel> kernels)
	{
		combo.Clear();

		var cell = new CellRendererText();
		combo.PackStart(cell, false);
		combo.AddAttribute(cell, "text", 0);
		var store = new ListStore(typeof(string));
		combo.Model = store;

		int index = 0;

		foreach (var kernel in kernels)
		{
			index++;

			store.AppendValues(String.Format("Model {0}:{1}", index, kernel.Category));
		}

		combo.Active = -1;
	}

	protected void UpdateTrainedModels(ComboBox combo, List<Model> models)
	{
		combo.Clear();

		var cell = new CellRendererText();
		combo.PackStart(cell, false);
		combo.AddAttribute(cell, "text", 0);
		var store = new ListStore(typeof(string));
		combo.Model = store;

		int index = 0;

		foreach (var model in models)
		{
			index++;

			store.AppendValues(String.Format("Model {0}:{1}", index, model.Category));
		}

		combo.Active = -1;
	}

	protected void UpdateTrainedParameters(ComboBox combo, List<string> parameters)
	{
		combo.Clear();

		var cell = new CellRendererText();
		combo.PackStart(cell, false);
		combo.AddAttribute(cell, "text", 0);
		var store = new ListStore(typeof(string));
		combo.Model = store;

		foreach (var parameter in parameters)
		{
			store.AppendValues(parameter);
		}

		combo.Active = -1;
	}

	protected void UpdateKernelBox(ComboBox combo, List<KernelClass> kernels, int select = 0)
	{
		combo.Clear();

		var cell = new CellRendererText();
		combo.PackStart(cell, false);
		combo.AddAttribute(cell, "text", 0);
		var store = new ListStore(typeof(string));
		combo.Model = store;

		foreach (var kernel in kernels)
		{
			store.AppendValues(kernel.Name);
		}

		combo.Active = select > 0 && kernels.Count > 0 ? select - 1 : -1;
	}

	protected void UpdateKernelParameters(KernelClass kernel, Label label1, SpinButton param1, Label label2, SpinButton param2)
	{
		if (kernel.FreeParameters > 0)
		{
			label1.Visible = true;
			param1.Visible = true;

			if (kernel.Parameters.Count > 0 && kernel.ParameterNames.Count > 0)
			{
				label1.LabelProp = "<b>" + kernel.ParameterNames[0] + "</b>";
				param1.Value = Convert.ToDouble(kernel.Parameters[0], ci);
			}
		}
		else
		{
			label1.Visible = false;
			param1.Visible = false;
		}

		if (kernel.FreeParameters > 1)
		{
			label2.Visible = true;
			param2.Visible = true;

			if (kernel.Parameters.Count > 1 && kernel.ParameterNames.Count > 1)
			{
				label2.LabelProp = "<b>" + kernel.ParameterNames[1] + "</b>";
				param2.Value = Convert.ToDouble(kernel.Parameters[1], ci);
			}
		}
		else
		{
			label2.Visible = false;
			param2.Visible = false;
		}
	}

	protected void UpdateKernelParameters(KernelClass kernel, Label label1, Entry param1, Label label2, Entry param2)
	{
		if (kernel.FreeParameters > 0)
		{
			label1.Visible = true;
			param1.Visible = true;

			if (kernel.Parameters.Count > 0 && kernel.ParameterNames.Count > 0)
			{
				label1.LabelProp = "<b>" + kernel.ParameterNames[0] + "</b>";
				param1.Text = kernel.Parameters[0].ToString("0.00000", ci);
			}
		}
		else
		{
			label1.Visible = false;
			param1.Visible = false;
		}

		if (kernel.FreeParameters > 1)
		{
			label2.Visible = true;
			param2.Visible = true;

			if (kernel.Parameters.Count > 1 && kernel.ParameterNames.Count > 1)
			{
				label2.LabelProp = "<b>" + kernel.ParameterNames[1] + "</b>";
				param2.Text = kernel.Parameters[1].ToString("0.00000", ci);
			}
		}
		else
		{
			label2.Visible = false;
			param2.Visible = false;
		}
	}

	protected List<double> SetKernelParameters(int current)
	{
		var kernelParams = new List<double>();

		switch (current)
		{
			case (int)KernelType.POLYNOMIAL:

				kernelParams.Add(Convert.ToDouble(Parameter1.Value, ci));
				kernelParams.Add(Convert.ToDouble(Parameter2.Value, ci));

				break;

			case (int)KernelType.GAUSSIAN:

				kernelParams.Add(Convert.ToDouble(Parameter1.Value, ci));

				break;

			case (int)KernelType.RADIAL:

				kernelParams.Add(Convert.ToDouble(Parameter1.Value, ci));

				break;

			case (int)KernelType.SIGMOID:

				kernelParams.Add(Convert.ToDouble(Parameter1.Value, ci));
				kernelParams.Add(Convert.ToDouble(Parameter2.Value, ci));

				break;

			case (int)KernelType.LINEAR:

				kernelParams.Add(Convert.ToDouble(Parameter1.Value, ci));
				kernelParams.Add(Convert.ToDouble(Parameter2.Value, ci));

				break;

			case (int)KernelType.FOURIER:

				kernelParams.Add(Convert.ToDouble(Parameter1.Value, ci));

				break;
		}

		return kernelParams;
	}

	protected void ToggleUserControls(bool toggle)
	{
		// Data Page - Training Set
		TrainingFilename.Sensitive = toggle;
		OpenTrainingButton.Sensitive = toggle;
		ReloadTrainingButton.Sensitive = toggle;
		TrainingView.Sensitive = toggle;
		Features.Sensitive = toggle;
		Categories.Sensitive = toggle;
		Examples.Sensitive = toggle;

		// Data Page - Test Set
		TestFilename.Sensitive = toggle;
		OpenTestButton.Sensitive = toggle;
		ReloadTestButton.Sensitive = toggle;
		TestView.Sensitive = toggle;
		Samples.Sensitive = toggle;

		// Data Page - Delimeter Box
		DelimiterBox.Sensitive = toggle;

		// Training Page - Kernel
		KernelBox.Sensitive = toggle;
		Category.Sensitive = toggle;
		KernelBox.Sensitive = toggle;
		Parameter1.Sensitive = toggle;
		Parameter2.Sensitive = toggle;
		Tolerance.Sensitive = toggle;
		Regularization.Sensitive = toggle;
		MaxPasses.Sensitive = toggle;

		// Training Page - Model
		ModelBox.Sensitive = toggle;
		AddModelButton.Sensitive = toggle;
		SaveModelButton.Sensitive = toggle;
		RemoveModelButton.Sensitive = toggle;
		ClearModelsButton.Sensitive = toggle;

		// Training Page - Training toolbar
		RunButton.Sensitive = toggle;
		PauseButton.Sensitive = !toggle;
		ResetButton.Sensitive = toggle;
		Normalize.Sensitive = toggle;

		// Training Page - Classification
		ClassificationView.Sensitive = toggle;
		ClassifyButton.Sensitive = toggle;
		ClassifyAllButton.Sensitive = toggle;

		// Models Page - Normalization
		NormalizationView.Sensitive = toggle;

		// Models Page - Parameters
		TrainedModelBox.Sensitive = toggle;
		TrainedModelKernel.Sensitive = toggle;
		TrainedParameter1.Sensitive = toggle;
		TrainedParameter2.Sensitive = toggle;
		TrainedModelCategory.Sensitive = toggle;
		TrainedRegularization.Sensitive = toggle;
		TrainedPasses.Sensitive = toggle;
		TrainedTolerance.Sensitive = toggle;

		// Models Page - Parameters display
		TrainedParametersBox.Sensitive = toggle;
		ModelFilename.Sensitive = toggle;
		SaveTrainedModelButton.Sensitive = toggle;
		OpenTrainedModelButton.Sensitive = toggle;
		ParametersView.Sensitive = toggle;
	}

	protected void InitializeUserInterface()
	{
		Title = "Support Vector Machine (SVM) Classifier";

		Confirm = new Dialog(
			"Are you sure?",
			this,
			DialogFlags.Modal,
			"Yes", ResponseType.Accept,
			"No", ResponseType.Cancel
		)
		{
			Resizable = false,
			KeepAbove = true,
			TypeHint = WindowTypeHint.Dialog,
			WidthRequest = 250
		};

		Confirm.ActionArea.LayoutStyle = ButtonBoxStyle.Center;
		Confirm.WindowStateEvent += OnWindowStateEvent;

		Delimiters.Add(new Delimiter("Tab \\t", '\t'));
		Delimiters.Add(new Delimiter("Comma ,", ','));
		Delimiters.Add(new Delimiter("Space \\s", ' '));
		Delimiters.Add(new Delimiter("Vertical Pipe |", '|'));
		Delimiters.Add(new Delimiter("Colon :", ':'));
		Delimiters.Add(new Delimiter("Semi-Colon ;", ';'));
		Delimiters.Add(new Delimiter("Forward Slash /", '/'));
		Delimiters.Add(new Delimiter("Backward Slash \\", '\\'));

		TextLoader = new FileChooserDialog(
			"Load Text File",
			this,
			FileChooserAction.Open,
			"Cancel", ResponseType.Cancel,
			"Load", ResponseType.Accept
		);

		JsonLoader = new FileChooserDialog(
			"Load trained models",
			this,
			FileChooserAction.Open,
			"Cancel", ResponseType.Cancel,
			"Load", ResponseType.Accept
		);

		JsonSaver = new FileChooserDialog(
			"Save trained models",
			this,
			FileChooserAction.Save,
			"Cancel", ResponseType.Cancel,
			"Save", ResponseType.Accept
		);

		TextLoader.AddFilter(AddFilter("Text files (csv/txt)", "*.txt", "*.csv"));
		JsonLoader.AddFilter(AddFilter("json", "*.json"));
		JsonSaver.AddFilter(AddFilter("json", "*.json"));

		TextLoader.Filter = TextLoader.Filters[0];
		JsonLoader.Filter = JsonLoader.Filters[0];
		JsonSaver.Filter = JsonSaver.Filters[0];

		ToggleUserControls(Paused);

		Kernels.Add(new KernelClass("Polynomial", KernelType.POLYNOMIAL, new List<double> { 0.0, 2.0 }, new List<string> { "bias", "exponent" }));
		Kernels.Add(new KernelClass("Gaussian", KernelType.GAUSSIAN, new List<double> { 0.01 }, new List<string> { "sigma" }));
		Kernels.Add(new KernelClass("Radial", KernelType.RADIAL, new List<double> { 0.01 }, new List<string> { "sigma" }));
		Kernels.Add(new KernelClass("Sigmoid", KernelType.SIGMOID, new List<double> { 1.0, 0.0 }, new List<string> { "slope", "intercept" }));
		Kernels.Add(new KernelClass("Linear", KernelType.LINEAR, new List<double> { 1.0, 0.0 }, new List<string> { "slope", "intercept" }));
		Kernels.Add(new KernelClass("Fourier", KernelType.FOURIER, new List<double> { 1.0 }, new List<string> { "scaling factor" }));

		DisableControls();

		UpdateDelimiterBox(DelimiterBox, Delimiters);
		UpdateKernelBox(KernelBox, Kernels);
		UpdateTrainedParameters(TrainedParametersBox, new List<string> { "X", "Y", "Alpha", "W", "B" });

		HideKernelParameters();

		TrainedParameter1.Visible = false;
		TrainedParameter2.Visible = false;
		LabelTrainedParameter1.Visible = false;
		LabelTrainedParameter2.Visible = false;

		EnableControls();

		Idle.Add(new IdleHandler(OnIdle));
	}

	protected void Pause()
	{
		Paused = true;

		ToggleUserControls(Paused);
	}

	protected void Run()
	{
		Paused = false;

		ToggleUserControls(Paused);
	}

	protected string GetBaseFileName(string fullpath)
	{
		return System.IO.Path.GetFileNameWithoutExtension(fullpath);
	}

	protected string GetDirectory(string fullpath)
	{
		return System.IO.Path.GetDirectoryName(fullpath);
	}

	protected void ReloadTextFile(string FileName, TextView view, bool isTraining = false, SpinButton counter = null)
	{
		try
		{
			var current = DelimiterBox.Active;
			var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';

			var categories = new List<int>();

			if (File.Exists(FileName) && view != null)
			{
				var text = "";
				var count = 0;
				var features = 0;

				using (TextReader reader = File.OpenText(FileName))
				{
					string line;

					while ((line = reader.ReadLine()) != null)
					{
						line = line.Trim();

						if (!string.IsNullOrEmpty(line))
						{
							if (isTraining)
							{
								var tokens = line.Split(delimiter);

								if (tokens.Length > 1)
								{
									features = tokens.Length - 1;

									var last = SafeConvert.ToInt32(tokens[tokens.Length - 1]);

									if (!categories.Contains(last) && last > 0)
									{
										categories.Add(last);
									}
								}
							}

							text += count > 0 ? "\n" + line : line;

							count++;
						}
					}
				}

				if (counter != null)
				{
					counter.Value = count;
				}

				if (isTraining)
				{
					Features.Value = features;
					Categories.Value = categories.Count;
				}

				view.Buffer.Clear();
				view.Buffer.Text = text.Trim();
			}
		}
		catch (Exception ex)
		{
			Console.WriteLine("Error: {0}", ex.Message);
		}
	}

	protected void HideKernelParameters()
	{
		LabelParam1.Visible = false;
		LabelParam2.Visible = false;
		Parameter1.Visible = false;
		Parameter2.Visible = false;
	}

	protected void LoadTextFile(ref string FileName, string title, TextView view, Entry entry, bool isTraining = false, SpinButton counter = null)
	{
		TextLoader.Title = title;

		// Add most recent directory
		if (!string.IsNullOrEmpty(TextLoader.Filename))
		{
			var directory = System.IO.Path.GetDirectoryName(TextLoader.Filename);

			if (Directory.Exists(directory))
			{
				TextLoader.SetCurrentFolder(directory);
			}
		}

		if (TextLoader.Run() == (int)ResponseType.Accept)
		{
			if (!string.IsNullOrEmpty(TextLoader.Filename))
			{
				FileName = TextLoader.Filename;

				ReloadTextFile(FileName, view, isTraining, counter);

				if (entry != null)
				{
					entry.Text = FileName;
				}
			}
		}

		TextLoader.Hide();
	}

	protected void LoadClassifier(string FileName)
	{
		var json = Utility.LoadJson(FileName);

		var models = Utility.Deserialize(json, NormalizationData);

		if (models.Count > 0)
		{
			ResetModels();

			Models.AddRange(models);

			DisableControls();

			var categories = 0;
			var features = 0;

			foreach (var model in models)
			{
				categories = model.Category > categories ? model.Category : categories;
				features = model.ModelX.x > features ? model.ModelX.x : features;
			}

			if (Models.Count > 0)
			{
				UpdateTrainedModels(TrainedModelBox, Models);
				UpdateTrainedModels(ClassificationModelsBox, Models);

				ResetModelKernels();

				foreach (var model in Models)
				{
					var kclass = (int)model.Type;
					var kparams = new List<double>();

					for (var i = 0; i < model.KernelParam.Length(); i++)
						kparams.Add(Convert.ToDouble(model.KernelParam[i], ci));

					var kernel = new KernelClass(Kernels[kclass].Name, model.Type, kparams, Kernels[kclass].ParameterNames);

					ModelKernels.Add(new ModelKernel(kernel, model.Category, model.C, model.Tolerance, model.MaxIterations));
				}

				UpdateModelsBox(ModelBox, ModelKernels);
			}

			if (DelimiterBox.Active < 0)
				DelimiterBox.Active = 0;

			UpdateTextView(NormalizationView, NormalizationData);

			Features.Value = features;
			Categories.Value = categories;

			ClassifierInitialized = true;
			TrainingDone = true;

			TrainingProgress.Text = "Pre-trained models";
			TrainingProgress.Fraction = 1;

			EnableControls();
		}
	}

	protected void LoadJson(ref string FileName, string title, Entry entry)
	{
		JsonLoader.Title = title;

		// Add most recent directory
		if (!string.IsNullOrEmpty(JsonLoader.Filename))
		{
			var directory = System.IO.Path.GetDirectoryName(JsonLoader.Filename);

			if (Directory.Exists(directory))
			{
				JsonLoader.SetCurrentFolder(directory);
			}
		}

		if (JsonLoader.Run() == (int)ResponseType.Accept)
		{
			if (!string.IsNullOrEmpty(JsonLoader.Filename))
			{
				FileName = JsonLoader.Filename;

				if (entry != null)
				{
					entry.Text = FileName;
				}
			}
		}

		JsonLoader.Hide();
	}

	protected void SaveJson(ref string FileName, string title, Entry entry, string data)
	{
		JsonSaver.Title = title;

		JsonSaver.SelectFilename(FileName);

		string directory;

		// Add most recent directory
		if (!string.IsNullOrEmpty(JsonSaver.Filename))
		{
			directory = System.IO.Path.GetDirectoryName(JsonSaver.Filename);

			if (Directory.Exists(directory))
			{
				JsonSaver.SetCurrentFolder(directory);
			}
		}

		if (JsonSaver.Run() == (int)ResponseType.Accept)
		{
			if (!string.IsNullOrEmpty(JsonSaver.Filename))
			{
				FileName = JsonSaver.Filename;

				directory = GetDirectory(FileName);

				var ext = JsonSaver.Filter.Name;

				FileName = String.Format("{0}.{1}", GetBaseFileName(FileName), ext);

				if (!string.IsNullOrEmpty(data))
				{
					var fullpath = String.Format("{0}/{1}", directory, FileName);

					try
					{
						using (var file = new StreamWriter(fullpath, false))
						{
							file.Write(data);
						}

						FileName = fullpath;

						entry.Text = FileName;
					}
					catch (Exception ex)
					{
						Console.WriteLine("Error saving {0}: {1}", FileName, ex.Message);
					}
				}
			}
		}

		JsonSaver.Hide();
	}

	protected void ReparentTextView(Fixed parent, ScrolledWindow window, int x, int y)
	{
		var source = (Fixed)window.Parent;
		source.Remove(window);

		parent.Add(window);

		Fixed.FixedChild child = ((Fixed.FixedChild)(parent[window]));

		child.X = x;
		child.Y = y;
	}

	protected void ReparentLabel(Fixed parent, Label label, int x, int y)
	{
		label.Reparent(parent);

		parent.Move(label, x, y);
	}

	protected void UpdateTextView(TextView view, ManagedArray data)
	{
		if (data != null)
		{
			var current = DelimiterBox.Active;
			var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';

			view.Buffer.Clear();

			var text = "";

			for (int y = 0; y < data.y; y++)
			{
				if (y > 0)
					text += "\n";

				for (int x = 0; x < data.x; x++)
				{
					if (x > 0)
						text += delimiter;

					text += data[x, y].ToString(ci);
				}
			}

			view.Buffer.Text = text;
		}
	}

	protected void NormalizeData(ManagedArray input, ManagedArray normalization)
	{
		for (int y = 0; y < input.y; y++)
		{
			for (int x = 0; x < input.x; x++)
			{
				var min = normalization[x, 0];
				var max = normalization[x, 1];

				input[x, y] = (input[x, y] - min) / (max - min);
			}
		}
	}

	protected bool SetupInputData(string training)
	{
		var text = training.Trim();

		if (string.IsNullOrEmpty(text))
			return false;

		var TrainingBuffer = new TextBuffer(new TextTagTable())
		{
			Text = text
		};

		Examples.Value = Convert.ToDouble(TrainingBuffer.LineCount, ci);

		var inpx = Convert.ToInt32(Features.Value, ci);
		var inpy = Convert.ToInt32(Examples.Value, ci);

		ManagedOps.Free(InputData, OutputData, NormalizationData);

		InputData = new ManagedArray(inpx, inpy);
		NormalizationData = new ManagedArray(inpx, 2);
		OutputData = new ManagedArray(1, inpy);

		int min = 0;
		int max = 1;

		for (int x = 0; x < inpx; x++)
		{
			NormalizationData[x, min] = double.MaxValue;
			NormalizationData[x, max] = double.MinValue;
		}

		var current = DelimiterBox.Active;
		var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';
		var inputs = inpx;

		using (var reader = new StringReader(TrainingBuffer.Text))
		{
			for (int y = 0; y < inpy; y++)
			{
				var line = reader.ReadLine();

				if (!string.IsNullOrEmpty(line))
				{
					var tokens = line.Split(delimiter);

					if (inputs > 0 && tokens.Length > inputs)
					{
						OutputData[0, y] = SafeConvert.ToDouble(tokens[inputs]);

						for (int x = 0; x < inpx; x++)
						{
							var data = SafeConvert.ToDouble(tokens[x]);

							NormalizationData[x, min] = data < NormalizationData[x, min] ? data : NormalizationData[x, min];
							NormalizationData[x, max] = data > NormalizationData[x, max] ? data : NormalizationData[x, max];

							InputData[x, y] = data;
						}
					}
				}
			}
		}

		if (Normalize.Active)
			NormalizeData(InputData, NormalizationData);

		UpdateTextView(NormalizationView, NormalizationData);

		return true;
	}

	protected bool SetupTestData(string test)
	{
		var text = test.Trim();

		if (string.IsNullOrEmpty(text))
			return false;

		var TestBuffer = new TextBuffer(new TextTagTable())
		{
			Text = text
		};

		Samples.Value = Convert.ToDouble(TestBuffer.LineCount, ci);

		var inpx = Convert.ToInt32(Features.Value, ci);
		var tsty = Convert.ToInt32(Samples.Value, ci);

		ManagedOps.Free(TestData);

		TestData = new ManagedArray(inpx, tsty);

		var current = DelimiterBox.Active;
		var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';
		var inputs = inpx;

		using (var reader = new StringReader(TestBuffer.Text))
		{
			for (int y = 0; y < tsty; y++)
			{
				var line = reader.ReadLine();

				if (!string.IsNullOrEmpty(line))
				{
					var tokens = line.Split(delimiter);

					if (inputs > 0 && tokens.Length >= inpx)
					{
						for (int x = 0; x < inpx; x++)
						{
							TestData[x, y] = SafeConvert.ToDouble(tokens[x]);
						}
					}
				}
			}
		}

		if (Normalize.Active)
			NormalizeData(TestData, NormalizationData);

		return true;
	}

	protected void Classify(Model model)
	{
		var test = TestView.Buffer.Text.Trim();

		if (string.IsNullOrEmpty(test))
			return;

		if (ClassifierInitialized && SetupTestData(test))
		{
			var classification = model.Classify(TestData);

			ClassificationView.Buffer.Clear();

			string text = "";

			for (var i = 0; i < classification.x; i++)
			{
				if (i > 0)
					text += "\n";

				text += Convert.ToString(classification[i], ci);
			}

			ClassificationView.Buffer.Text = text;

			ManagedOps.Free(classification);
		}

		TestView.Buffer.Text = test;
	}

	protected void ClassifyAll()
	{
		if (Models.Count < 1)
			return;

		var test = TestView.Buffer.Text.Trim();

		if (string.IsNullOrEmpty(test))
			return;

		ToggleUserControls(false);

		if (ClassifierInitialized && SetupTestData(test))
		{
			ClassificationView.Buffer.Clear();

			string text = "";

			var input = new ManagedArray(TestData.x, 1);

			for (var i = 0; i < TestData.y; i++)
			{
				var category = 0;
				var prediction = 0.0;
				var m = 0;

				ManagedOps.Copy2D(input, TestData, 0, i);

				foreach (var model in Models)
				{
					if (model.Trained)
					{
						var p = model.Predict(input);
						var c = model.Classify(input);

						if (m > 0)
						{
							if (p[0] > prediction)
							{
								prediction = p[0];
								category = c[0];
							}
						}
						else
						{
							prediction = p[0];
							category = c[0];
						}

						ManagedOps.Free(p);
						ManagedOps.Free(c);
					}

					m++;
				}

				if (i > 0)
					text += "\n";

				text += Convert.ToString(category, ci);
			}

			ManagedOps.Free(input);

			ClassificationView.Buffer.Text = text;
		}

		TestView.Buffer.Text = test;

		ToggleUserControls(true);
	}

	protected List<double> ToDouble(ManagedArray parameters)
	{
		var kparam = new List<double>();

		for (var i = 0; i < parameters.Length(); i++)
			kparam.Add(Convert.ToDouble(parameters[i], ci));

		return kparam;
	}

	protected ManagedArray CopyParameters(KernelClass kernel)
	{
		var kparam = new ManagedArray(kernel.Parameters.Count);

		var parameters = kernel.Parameters.Count;

		for (var i = 0; i < parameters; i++)
			kparam[i] = Convert.ToDouble(kernel.Parameters[i], ci);

		return kparam;
	}

	void SetupClassifiers()
	{
		ClassifierInitialized = false;

		var training = TrainingView.Buffer.Text.Trim();

		ClassifierInitialized = SetupInputData(training);

		if (string.IsNullOrEmpty(training))
			return;

		ClearModels();

		ClassifierInitialized &= ModelKernels.Count > 0;

		foreach (var kernel in ModelKernels)
		{
			var model = new Model();

			var kparam = CopyParameters(kernel.Kernel);

			model.Setup(InputData, OutputData, kernel.C, kernel.Kernel.Type, kparam, kernel.Tolerance, kernel.MaxPasses, kernel.Category);

			Models.Add(model);

			ManagedOps.Free(kparam);
		}
	}

	protected void UpdateProgress(int count, int max, bool done = false)
	{
		if (max > 0)
		{
			TrainingProgress.Fraction = Math.Round((double)count / max, 2);

			TrainingProgress.Text = done ? "Done" : String.Format("Training ({0}%)...", Convert.ToInt32(TrainingProgress.Fraction * 100, ci));
		}
	}

	protected void SetTrainedModelParameters(Model model)
	{
		if (model.Trained)
		{
			var current = (int)model.Type;
			var kparam = ToDouble(model.KernelParam);
			var kernel = new KernelClass(Kernels[current].Name, model.Type, kparam, Kernels[current].ParameterNames);

			UpdateKernelBox(TrainedModelKernel, new List<KernelClass> { kernel }, 1);
			UpdateKernelParameters(kernel, LabelTrainedParameter1, TrainedParameter1, LabelTrainedParameter2, TrainedParameter2);
			TrainedModelCategory.Text = Convert.ToInt32(model.Category).ToString(ci);
			TrainedRegularization.Text = model.C.ToString(ci);
			TrainedPasses.Text = Convert.ToInt32(model.Passes).ToString(ci);
			TrainedTolerance.Text = model.Tolerance.ToString(ci);
		}
	}

	protected void UpdateTrainedModelView(Model model, int parameter)
	{
		if (model.Trained)
		{
			switch (parameter)
			{
				case (int)ModelParameters.X:

					UpdateTextView(ParametersView, model.ModelX);

					break;

				case (int)ModelParameters.Y:

					UpdateTextView(ParametersView, model.ModelY);

					break;

				case (int)ModelParameters.ALPHA:

					UpdateTextView(ParametersView, model.Alpha);

					break;

				case (int)ModelParameters.W:

					UpdateTextView(ParametersView, model.W);

					break;

				case (int)ModelParameters.B:

					ParametersView.Buffer.Clear();
					ParametersView.Buffer.Text = model.B.ToString(ci);

					break;
			}
		}
	}

	protected bool GetConfirmation()
	{
		var confirm = Confirm.Run() == (int)ResponseType.Accept;

		Confirm.Hide();

		return confirm;
	}

	protected void ResetModelKernels()
	{
		DisableControls();

		ModelKernels.Clear();

		UpdateModelsBox(ModelBox, ModelKernels);

		LabelModel.LabelProp = ModelKernels.Count > 0 ? String.Format(ci, "<b>Model(s): {0}</b>", ModelKernels.Count) : "<b>Model</b>";

		KernelBox.Active = -1;

		HideKernelParameters();

		EnableControls();
	}

	protected void ResetModels()
	{
		DisableControls();

		ClearModels();

		UpdateTrainedModels(TrainedModelBox, Models);
		UpdateTrainedModels(ClassificationModelsBox, Models);
		UpdateKernelBox(TrainedModelKernel, new List<KernelClass>());

		TrainedParametersBox.Active = -1;

		TrainedModelCategory.Text = "";
		TrainedPasses.Text = "";
		TrainedTolerance.Text = "";
		TrainedRegularization.Text = "";

		TrainedParameter1.Text = "";
		TrainedParameter2.Text = "";
		TrainedParameter1.Visible = false;
		TrainedParameter2.Visible = false;
		LabelTrainedParameter1.Visible = false;
		LabelTrainedParameter2.Visible = false;

		NormalizationView.Buffer.Clear();
		ParametersView.Buffer.Clear();

		TrainingDone = false;
		ClassifierInitialized = false;

		TrainingProgress.Text = "";
		TrainingProgress.Fraction = 0.0;

		ClassificationModelsBox.Clear();
		ClassificationModelsBox.Active = -1;
		ClassificationView.Buffer.Clear();

		EnableControls();
	}

	protected void ClearModels()
	{
		// Clean-Up Models
		if (Models.Count > 0)
		{
			foreach (var model in Models)
				model.Free();

			Models.Clear();
		}
	}

	protected void CleanShutdown()
	{
		// Clean-Up Routines Here
		ManagedOps.Free(InputData, NormalizationData, OutputData, TestData);

		// Clean-Up Models
		ClearModels();
	}

	protected void Quit()
	{
		CleanShutdown();

		Application.Quit();
	}

	bool OnIdle()
	{
		var available = Processing.WaitOne();

		if (available && !Paused && !TrainingDone)
		{
			if (Models.Count > 0 && ClassifierInitialized)
			{
				int count = 0;
				int max = 0;

				bool done = true;

				foreach (var model in Models)
				{
					var result = model.Step();

					max += model.MaxIterations;
					count += model.Iterations;

					if (result && !model.Trained)
					{
						model.Generate();
					}

					done &= result;
				}

				UpdateProgress(count, max, done);

				TrainingDone = done;

				if (TrainingDone)
				{
					Pause();

					DisableControls();

					if (Models.Count > 0)
					{
						UpdateTrainedModels(TrainedModelBox, Models);
						UpdateTrainedModels(ClassificationModelsBox, Models);
					}

					EnableControls();
				}
			}

			Processing.ReleaseMutex();
		}

		return true;
	}

	protected void OnWindowStateEvent(object sender, WindowStateEventArgs args)
	{
		var state = args.Event.NewWindowState;

		if (state == WindowState.Iconified)
		{
			Confirm.Hide();
		}

		args.RetVal = true;
	}

	void OnQuitButtonClicked(object sender, EventArgs args)
	{
		OnDeleteEvent(sender, new DeleteEventArgs());
	}

	protected void OnDeleteEvent(object sender, DeleteEventArgs a)
	{
		if (GetConfirmation())
		{
			Quit();
		}

		a.RetVal = true;
	}

	protected void OnAboutButtonClicked(object sender, EventArgs e)
	{
		MainNotebook.Page = (int)Pages.ABOUT;
	}

	protected void OnMainNotebookSwitchPage(object sender, SwitchPageArgs args)
	{
		switch (args.PageNum)
		{
			case (int)Pages.DATA:

				ReparentTextView(LayoutPageData, TestWindow, 20, 290);
				ReparentLabel(LayoutPageData, LabelTestData, 20, 230);

				break;

			case (int)Pages.TRAINING:

				ReparentTextView(LayoutPageTraining, TestWindow, 20, 320);
				ReparentLabel(LayoutPageTraining, LabelTestData, 20, 300);

				break;

			default:

				ReparentTextView(LayoutPageData, TestWindow, 20, 290);
				ReparentLabel(LayoutPageData, LabelTestData, 20, 230);

				break;
		}
	}

	protected void OnOpenTrainingButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		LoadTextFile(ref FileTraining, "Load training set", TrainingView, TrainingFilename, true, Examples);
	}

	protected void OnReloadTrainingButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		ReloadTextFile(FileTraining, TrainingView, true, Examples);
	}

	protected void OnOpenTestButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		LoadTextFile(ref FileTest, "Load test set", TestView, TestFilename, false, Samples);
	}

	protected void OnReloadTestButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		ReloadTextFile(FileTest, TestView, false, Samples);
	}

	protected void OnRunButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		if (!ClassifierInitialized)
		{
			TrainingProgress.Fraction = 0.0;

			TrainingProgress.Text = Examples.Value > 0 ? "Setting up classifiers" : "No training data provided";

			if (ModelKernels.Count < 1)
			{
				TrainingProgress.Text = "No models to train!";
			}

			SetupClassifiers();

			TrainingDone = false;
		}

		if (ClassifierInitialized)
		{
			if (!TrainingDone)
				Run();
		}
	}

	protected void OnPauseButtonClicked(object sender, EventArgs e)
	{
		if (Paused)
			return;

		Pause();
	}

	protected void OnResetButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		Pause();

		ResetModels();
	}

	protected void OnKernelBoxChanged(object sender, EventArgs e)
	{
		if (!ControlsEnabled)
			return;

		if (!Paused)
			return;

		var kclass = KernelBox.Active;

		if (kclass >= 0 & kclass < Kernels.Count)
		{
			var kernel = Kernels[kclass];

			DisableControls();

			UpdateKernelParameters(kernel, LabelParam1, Parameter1, LabelParam2, Parameter2);

			EnableControls();
		}
	}

	protected void OnAddModelButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		var category = (int)Category.Value;
		var current = KernelBox.Active;
		var tolerance = Convert.ToDouble(Tolerance.Value, ci);
		var maxpasses = Convert.ToInt32(MaxPasses.Value, ci);
		var regularization = Convert.ToDouble(Regularization.Value, ci);
		var features = Convert.ToInt32(Features.Value);
		var categories = Convert.ToInt32(Categories.Value);

		if (current >= 0 && current < Kernels.Count && category >= 1 && category <= categories && features > 0)
		{
			var model = ModelKernels.Find(obj => obj.Category == category);

			if (model == null)
			{
				var kernel = Kernels[current];

				var kernelParams = SetKernelParameters(current);

				var kclass = new KernelClass(kernel.Name, kernel.Type, kernelParams, kernel.ParameterNames);

				ModelKernels.Add(new ModelKernel(kclass, category, regularization, tolerance, maxpasses));

				UpdateModelsBox(ModelBox, ModelKernels);

				LabelModel.LabelProp = ModelKernels.Count > 0 ? String.Format(ci, "<b>Model(s): {0}</b>", ModelKernels.Count) : "<b>Model</b>";
			}
		}
	}

	protected void OnModelBoxChanged(object sender, EventArgs e)
	{
		if (!ControlsEnabled)
			return;

		if (!Paused)
			return;

		var current = ModelBox.Active;

		if (current >= 0 && current < ModelKernels.Count)
		{
			var model = ModelKernels[current];
			var kernel = model.Kernel;

			Category.Value = model.Category;
			Tolerance.Value = model.Tolerance;
			MaxPasses.Value = model.MaxPasses;
			Regularization.Value = model.C;

			DisableControls();

			var type = (int)kernel.Type;

			if (type >= 0 && type < Kernels.Count)
				KernelBox.Active = type;

			UpdateKernelParameters(model.Kernel, LabelParam1, Parameter1, LabelParam2, Parameter2);

			EnableControls();
		}
	}

	protected void OnRemoveModelButton(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		var current = ModelBox.Active;

		if (current >= 0 && current < ModelKernels.Count)
		{
			ModelKernels.RemoveAt(current);

			LabelModel.LabelProp = ModelKernels.Count > 0 ? String.Format(ci, "<b>Model(s): {0}</b>", ModelKernels.Count) : "<b>Model</b>";

			DisableControls();

			UpdateModelsBox(ModelBox, ModelKernels);

			EnableControls();
		}
	}

	protected void OnSaveModelButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		var current = ModelBox.Active;
		var kclass = KernelBox.Active;

		if (current >= 0 && current < ModelKernels.Count && kclass >= 0 && kclass < Kernels.Count)
		{
			var model = ModelKernels[current];
			var kernel = Kernels[kclass];

			model.Tolerance = Convert.ToDouble(Tolerance.Value, ci);
			model.C = Convert.ToDouble(Regularization.Value, ci);
			model.MaxPasses = Convert.ToInt32(MaxPasses.Value, ci);
			model.Kernel.Type = kernel.Type;

			var kernelParams = SetKernelParameters(kclass);

			model.Kernel.Parameters.Clear();
			model.Kernel.Parameters.AddRange(kernelParams);
			model.Kernel.FreeParameters = kernelParams.Count;
			model.Kernel.ParameterNames.Clear();
			model.Kernel.ParameterNames.AddRange(kernel.ParameterNames);
		}
	}

	protected void OnTrainedParametersBoxChanged(object sender, EventArgs e)
	{
		if (!ControlsEnabled)
			return;

		if (!Paused)
			return;

		var current = TrainedParametersBox.Active;
		var model = TrainedModelBox.Active;

		if (model >= 0 && model < Models.Count)
		{
			if (Models[model].Trained)
				UpdateTrainedModelView(Models[model], current);
		}
	}

	protected void OnTrainedModelBoxChanged(object sender, EventArgs e)
	{
		if (!ControlsEnabled)
			return;

		if (!Paused)
			return;

		var current = TrainedModelBox.Active;

		if (current >= 0 && current < Models.Count)
		{
			SetTrainedModelParameters(Models[current]);

			DisableControls();

			TrainedParametersBox.Active = -1;

			ParametersView.Buffer.Clear();

			EnableControls();
		}
	}

	protected void OnClassifyButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		var model = ClassificationModelsBox.Active;

		if (model >= 0 && model < Models.Count)
		{
			if (Models[model].Trained)
				Classify(Models[model]);
		}
	}

	protected void OnClassifyAllButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		if (Models.Count > 0 && TrainingDone)
		{
			ClassifyAll();
		}
	}

	protected void OnSaveTrainedModelButton(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		if (Models.Count > 0 && ClassifierInitialized && TrainingDone)
		{
			var json = Utility.Serialize(Models, NormalizationData);

			SaveJson(ref FileModels, "Save trained models", ModelFilename, json);
		}
	}

	protected void OnOpenTrainedModelButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		LoadJson(ref FileModels, "Load trained models", ModelFilename);

		LoadClassifier(FileModels);
	}

	protected void OnClearModelsButtonClicked(object sender, EventArgs e)
	{
		if (!Paused)
			return;

		DisableControls();

		ResetModelKernels();

		UpdateModelsBox(ModelBox, ModelKernels);

		HideKernelParameters();

		EnableControls();
	}
}
