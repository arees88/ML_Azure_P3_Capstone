# **Predicting presence of amphibians using GIS and satellite images**

This is the Capstone project for the **Machine Learning Engineer with Microsoft Azure** Nanodegree program from Udacity. 

This project uses **Microsoft Azure Machine Learning** to configure, deploy and consume a machine learning model. 

In this project we are using both, the **Hyperdrive** and **Auto ML** APIs to train the models. 
The best of these models will be deployed as an endpoint. The model endpoint is then tested to verify if it is working as intented by sending HTTP requests. 

We can choose any model and any data for the project, but the dataset needs to be external and not available in the Azure's ecosystem. 

The dataset used for the project was obtained from **Machine Learning Repository** at **University of California**.

The diagram below outlines the steps performed as part of the project:

![image](https://user-images.githubusercontent.com/60096624/116930600-dde77d80-ac57-11eb-8fb1-bc153b32bab5.png)

## Dataset

The dataset used for the project is the **Amphibians Data Set** available from Machine Learning Repository at University of California, Irvine. Here is the dataset summary:

![image](https://user-images.githubusercontent.com/60096624/116929286-1be3a200-ac56-11eb-804a-ccc3009b5836.png)

The goal of the project is to predict the presence of amphibians species near the water reservoirs based on features obtained from Geographic Information System (GIS) and satellite images

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
