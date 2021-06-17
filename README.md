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

The dataset has been downloaded from the Machine Learning Repository, Center for Machine Learning and Intelligent Systems at University of California, Irvine.

After convering to CSV format the Amphibians Data Set was uploaded to the GitHub repository.

Within the notebooks the data is accessed via the URL of the raw file and converted to Dataset format using `Tabular` attribute as follows:

```
data_loc = "https://raw.githubusercontent.com/arees88/ML_Azure_P3_Capstone/main/Amphibians_dataset_green_frogs.csv"
dataset = Dataset.Tabular.from_delimited_files(data_loc)
```

In addition I have created sample data input file for testing ONNX runtime which is accessed in a similar way in the AutoML notebook:

```
test_data = "https://raw.githubusercontent.com/arees88/ML_Azure_P3_Capstone/main/Amphibians_testset.csv"
test_dataset = Dataset.Tabular.from_delimited_files(test_data)
```
The below screenshots from ML Studio show the __Amphibians Dataset__ has been created as expected:

![alt text](screenshots/1.1_Dataset_121936116-2d35c900-cd41-11eb-99e4-3a4b905fb34e.png)

![alt text](screenshots/1.3_Dataset_121936267-58201d00-cd41-11eb-9c53-1beaadae5efc.png)

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

Setting early termination to True and the timeout to 30 min to limit AutoML duration.
Setting the maximum iterations concurrency to 4, as the maximum nodes configured in the compute cluster must be greater than the number of concurrent operations in the experiment, and the compute cluster has 5 nodes configured. 
I selected accuracy as the primary metric. The AutoML will perform 4 cross validations.


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![alt text](screenshots/2.1_AutoML_RunDetails_121939279-c4505000-cd44-11eb-85ac-c2eff1c61ee9.png)

![alt text](screenshots/2.2_AutoML_RunDetails_Zoom_121939452-f2ce2b00-cd44-11eb-8719-e8dfcc406cd3.png)

![alt text](screenshots/2.3_AutoML_RunDetails_Accuracy_121939065-7a676a00-cd44-11eb-8aa5-da39bb4ee2b4.png)

Best Model

![alt text](screenshots/3.1_AutoML_Best_model_121941266-f498ee00-cd46-11eb-801e-87e60f898752.png)

![alt text](screenshots/3.2_AutoML_Best_model__Zoom_121941211-e3e87800-cd46-11eb-9226-8f52eaa4ffe3.png)

![alt text](screenshots/3.3_AutoML_Best_model_Studio_121940739-53aa3300-cd46-11eb-9e71-cfd4049d328d.png)



## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![alt text](screenshots/4.1_HyperDrive_RunDetails_121944100-2c556500-cd4a-11eb-8ade-3e22c07f0b1c.png)

![alt text](screenshots/4.6_HyperDrive_RunDetails_121944357-7fc7b300-cd4a-11eb-92d7-66b22f8d56fb.png)

![alt text](screenshots/4.3_HyperDrive_RunDetails_Popup_121945413-a89c7800-cd4b-11eb-94d7-36d573d120c5.png)

![alt text](screenshots/4.4_HyperDrive_RunDetails_Popup_Zoom_121945472-bb16b180-cd4b-11eb-8751-660b11882bc9.png)

![alt text](screenshots/4.8_Hyperdrive_Tune_Graph_121248344-3e8f5900-c89b-11eb-89da-41725cea6f56.png)

![alt text](screenshots/4.5_HyperDrive_RunDetails_Studio_121944820-ffee1880-cd4a-11eb-87a6-71cbe74e99c5.png)

![alt text](screenshots/5.1_HyperDrive_Best_model_121945025-37f55b80-cd4b-11eb-9cfd-ec607d8d07be.png)

![alt text](screenshots/5.2_HyperDrive_Best_model_Zoom121946376-c5857b00-cd4c-11eb-91e8-f6ac2b820ed9.png)

![alt text](screenshots/5.3_HyperDrive_Best_model_Studio_121946514-eb128480-cd4c-11eb-9819-c18b9b42c1c8.png)


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

![alt text](screenshots/6.1_Hyper_Deployment_121951473-e224b180-cd52-11eb-8e0a-2be6134d62eb.png)

![alt text](screenshots/6.2_Hyper_Deployment_Zoom_121951711-2e6ff180-cd53-11eb-9237-78c08ae45576.png)

![alt text](screenshots/6.3_Endpoint_Active_Studio_121946707-1e551380-cd4d-11eb-9cee-d21d4605d788.png)

![alt text](screenshots/6.4_Endpoint_Active_Studio_121951254-9a058f00-cd52-11eb-9be9-bbda7f63a4c5.png)

![alt text](screenshots/6.5_Endpoint_Active_Studio_URI_121951343-b9042100-cd52-11eb-8aac-05972291e9d6.png)

![alt text](screenshots/7.1_Request_Data_121951844-5a8b7280-cd53-11eb-9360-33a63a70675c.png)

![alt text](screenshots/7.2_Request_Data__Zoom_121951885-68d98e80-cd53-11eb-955c-e0a68eb142dc.png)

![alt text](screenshots/7.3_Hyper_Test_Results_121951968-86a6f380-cd53-11eb-842a-f6c2b272f9a0.png)

![alt text](screenshots/7.4_Hyper_Test_Results_Zoom_121952024-9a525a00-cd53-11eb-8435-b3d72ba6d987.png)

![alt text](screenshots/7.5_Endpoint_Logs_121952103-b524ce80-cd53-11eb-821c-9d002e7c4026.png)

![alt text](screenshots/7.6_Endpoint_Logs_Zoom_121952156-c8d03500-cd53-11eb-80ab-7d08a6b86adc.png)


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

![alt text](screenshots/8.1_Get_ONNX_model_121947852-72142c80-cd4e-11eb-8246-b55319cec17d.png)

![alt text](screenshots/8.2_Save_ONNX_model_121948014-a25bcb00-cd4e-11eb-8d2b-5ca9a5044f83.png)

![alt text](screenshots/8.3_Save_ONNX_model_Zoom_121948142-c7503e00-cd4e-11eb-9929-586fbcd62d96.png)

![alt text](screenshots/8.4_ONNX_Test_Data.PNG)

![alt text](screenshots/8.5_ONNX_model_Predict.PNG)


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

