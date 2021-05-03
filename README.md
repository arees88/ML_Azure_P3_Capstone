# Your Project Title Here

*TODO:* Write a short introduction to your project.

This project uses Microsoft Azure Machine Learning to configure, deploy and consume a cloud-based machine learning production model. The project will also include the creation, publishing and consuming of a machine learning pipeline.

You will be using both the hyperdrive and automl API from azureml to build this project. You can choose the model you want to train, and the data you want to use. However, the data you use needs to be external and not available in the Azure's ecosystem. For instance, you can use the Heart Failure Prediction dataset from Kaggle to build a classification model.

The diagram below outlines the steps performed as part of the project:

![image](https://user-images.githubusercontent.com/60096624/116926776-b7731380-ac52-11eb-9b10-da932523248d.png)

## Dataset

The data used for the project is the Amphibians Data Set available from Machine Learning Repository at University of California, Irvine. Here is the dataset summary:

![image](https://user-images.githubusercontent.com/60096624/116929286-1be3a200-ac56-11eb-804a-ccc3009b5836.png)

The goal is to predict the presence of amphibians species near the water reservoirs based on features obtained from GIS systems and satellite images

Predicting presence of amphibian species using features obtained from GIS and satellite images.

The snapshot of the dataset is below:

![image](https://user-images.githubusercontent.com/60096624/116927469-9d860080-ac53-11eb-8f11-8cafc0dfdb85.png)

The dataset is multilabel and can be used to predict presence of a number of amphibian species. 

In this project we will be using the dataset to predict the presence of Green frogs (Label 1).

The dataset was downloaded using the following link:  http://archive.ics.uci.edu/ml/datasets/Amphibians


## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

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
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
