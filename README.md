# Support Vector Machine (SVM) Classifier

The GTK-based no-programming-required, Support vector machine (SVM) classifier for Win/Unix/Linux/OSX platforms

**About**

![About](/Screenshots/AboutPage.png)

SVMClassifier software utilizes support vector machines to perform multiclassification. Various kernels are available to map input data into a new space.

**Data**

![Data](/Screenshots/DataPage.png)

Input data from a csv/text file can be loaded, provided you indicate the correct delimiter. Some model parameters are estimated based on the loaded data but you can modify it prior to running the training algorithm. When loading data, the last column in each line is assumed to be the category or class number.

**Training**

![Training](/Screenshots/TrainingPage.png)

Because SVMs are nonlinear, binary classifiers, multiple models are trained in order to perform multiclassification tasks. Input data is mapped into a new space via "kernels" to enhance the separability between the classes. This means that SVMs are usually slower compared to other learning methods. The mapping process also contributes to the performance penalty. It is therefore recommended that multiclass SVMs are only utilized for small data sets.

**Models**

![Models](/Screenshots/ModelsPage.png)

All trained models and learned parameters are now saved in single JSON file. This avoids the hassle of loading and saving multiple files for every trained model. SVMClassifier ustilizes JSON.NET library from https://www.newtonsoft.com/json

# Platform

SVMClassifier software has been tested on Linux, OSX, and Windows platforms.
