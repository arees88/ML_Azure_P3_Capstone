# **Predicting presence of Amphibians using GIS and Satellite images**

This is the Capstone project for the **Machine Learning Engineer with Microsoft Azure** Nanodegree program from Udacity. 

This project uses **Microsoft Azure Machine Learning** to configure, deploy and consume a machine learning model. 

In this project we are using both, the **Hyperdrive** and **Auto ML** APIs to train the models. 
The best of these models is deployed as an endpoint. The model endpoint is then tested to verify that it is working as intended by sending HTTP requests. 

We can choose any model and any data for the project, but the dataset needs to be external and not available in the Azure's ecosystem. 

The dataset used for the project was obtained from **Machine Learning Repository** at **University of California**.

The diagram below outlines the steps performed as part of the project:

![image](https://user-images.githubusercontent.com/60096624/116930600-dde77d80-ac57-11eb-8fb1-bc153b32bab5.png)

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

The dataset used for the project is the **Amphibians Data Set** available from Machine Learning Repository at University of California, Irvine. 

Here is the dataset summary:

![image](https://user-images.githubusercontent.com/60096624/116929286-1be3a200-ac56-11eb-804a-ccc3009b5836.png)

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

The goal of the project is to predict the presence of amphibians species near the water reservoirs based on features obtained from Geographic Information System (GIS) and satellite images

The snapshot of the dataset is below:

![image](https://user-images.githubusercontent.com/60096624/116927469-9d860080-ac53-11eb-8f11-8cafc0dfdb85.png)

The dataset is multilabel and can be used to predict presence of a number of amphibian species in water reservoirs.

Here is the description of the features within the Amphibians Data Set:

![image](https://user-images.githubusercontent.com/60096624/119881071-c8970380-bf24-11eb-843b-3c06039a8f21.png)

In this project we will be using the dataset to predict the presence of **Green frogs** (**Label 1**).


The dataset was downloaded using the following link:  http://archive.ics.uci.edu/ml/datasets/Amphibians

### Access
*TODO*: Explain how you are accessing the data in your workspace.

The dataset has been downloaded from from Machine Learning Repository at University of California, Irvine.

After convering to CSV format the Amphibians Data Set was uploaded to GitHub repository.

Within the notebooks the data is accessed via the URL of the raw file and TabularDataset is used to convert it to Dataset format suitable for 

``
data_loc = "https://raw.githubusercontent.com/arees88/ML_Azure_P3_Capstone/main/Amphibians_dataset_green_frogs.csv"
dataset = Dataset.Tabular.from_delimited_files(data_loc)
``

The dataset called Amphibians_dataset_green_frogs.csv is in GitHub and use xxx to create Dataset passed to AutoML.

For hyperdrive I use access the dataset from the train.py + call clean_data for Hyperdrive run.

In addition I have created test input files for testing ONNX runtime with the Amphibians test data.


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

Setting early termination to True and the timeout to 30 min to limit AutoML duration.
Setting the maximum iterations concurrency to 4, as the maximum nodes configured in the compute cluster must be greater than the number of concurrent operations in the experiment, and the compute cluster has 5 nodes configured. 
I selected accuracy as the primary metric. The AutoML will perform 4 cross validations.


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

This section includes the link to the project screencast. The screencast shows the entire process of the working ML application, including a demonstration of:

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
- Demo of converting model to ONNX format and using ONNX runtime

Video is available at the following link: https://www.youtube.com/watch?v=Ueu9BC5kYeM

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## Future Suggestions
Also tried **Great crested newt** (**Label 7**).

Tried swagger but could not get it to work. The swagger configuration file was not generated automatically.

Two version of Hyperdrive - One SKLearn class and the other using ScriptRunConfig Class.

## References

- Amphibians Data Set, Machine Learning Repository, Center for Machine Learning and Intelligent Systems, University of California, Irvine (http://archive.ics.uci.edu/ml/datasets/Amphibians)
- AutoMLConfig Class, Microsoft Azure (https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py)
- HyperDriveConfig Class, Microsoft Azure (https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- ScriptRunConfig Class, Microsoft Azure (https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- Hyperparameter tuning with Azure Machine Learning (https://docs.microsoft.com/azure/machine-learning/service/how-to-tune-hyperparameters#specify-an-early-termination-policy)
- Scikit-learn Random Forest Classifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- OnnxConverter Class, Microsoft Azure (https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.onnx_convert.onnx_converter.onnxconverter?view=azure-ml-py)

