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

The dataset has been downloaded from the Machine Learning Repository at Center for Machine Learning and Intelligent Systems at University of California, Irvine.

After convering to CSV format the Amphibians Data Set was uploaded to the GitHub repository.

Within the notebooks the data is accessed via the URL to the raw file and converted to Dataset format using ``Tabular`` attribute as follows:

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

In the first part of the project AutoML API is used to train the models.

The following configuration was used for the AutoML run:
```
# automl settings 
automl_settings = {
    "enable_early_stopping": True,
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 4,
    "featurization": 'auto',
    "primary_metric": 'accuracy',
    "verbosity": logging.INFO,
    "n_cross_validations": 4
}

# Automl config 
automl_config = AutoMLConfig(  
                    task = "classification",
                    compute_target = compute_cluster, 
                    training_data = dataset,
                    label_column_name = "Label1",   
                    debug_log = "automl_errors.log",
                    enable_onnx_compatible_models = True,
                    **automl_settings  
)
```

The early termination flag was set to true and the timeout was set to 30 minutes to limit the AutoML run duration.

The maximum iterations concurrency was set to 4, as the maximum nodes configured in the compute cluster must be greater than the number of concurrent operations in the experiment, and the compute cluster has 5 nodes configured. 

Accuracy was selected as the primary metric. The AutoML is configured to perform 4 cross validations.

As I am predicting the presence of Green frogs in water reservoirs the label column is set to ``Label1``.

I have also set the ``enable_onnx_compatible_models`` parameter to true as I also would like to convert the best model to ONNX format.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

The AutoML experiment ended at 29 minutes as it reached the stopping criteria (`experiment_timeout_minutes=30`).

During this time AutoML performed 58 iterations evaluating a set of diffrent models. The best performing model was `VotingEnsemble` with accuracy `0.7458`:
```
ITERATION   PIPELINE                                       DURATION      METRIC      BEST
       58   VotingEnsemble                                 0:01:15       0.7458    0.7458
```

Here is the screenshot of the completed AutoML experiment with the best model summary:

![alt text](screenshots/2.1_AutoML_RunDetails_121939279-c4505000-cd44-11eb-85ac-c2eff1c61ee9.png)

![alt text](screenshots/2.2_AutoML_RunDetails_Zoom_121939452-f2ce2b00-cd44-11eb-8719-e8dfcc406cd3.png)

![alt text](screenshots/2.3_AutoML_RunDetails_Accuracy_121939065-7a676a00-cd44-11eb-8aa5-da39bb4ee2b4.png)

Best Model

![alt text](screenshots/3.1_AutoML_Best_model_121941266-f498ee00-cd46-11eb-801e-87e60f898752.png)

![alt text](screenshots/3.2_AutoML_Best_model__Zoom_121941211-e3e87800-cd46-11eb-9226-8f52eaa4ffe3.png)

![alt text](screenshots/3.3_AutoML_Best_model_Studio_121940739-53aa3300-cd46-11eb-9e71-cfd4049d328d.png)

With the AutoML option we restricted the experiment time to 30 minutes which allowed for 58 models to be explored. 
Increasing the experiment time would potentially allow to find another, better performing model.


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

### Hyperparameter Sampler

In the second part of the project Hyperdrive API is used to tune the hyperparameters of the model.

I am using the __RandomForestClassifier__ model for the HyperDrive run.

Azure ML supports three types of parameter sampling - Random, Grid and Bayesian sampling.
I have chosen Random Parameter Sampling because it is faster and supports early termination of low-performance runs.
It supports discrete and continous hyperparameters. 

I am using the RandomParameterSampling method for the HyperDrive run to tune the following three hyperparameters of __RandomForestClassifier__:

```
--n_estimators      - Number of trees in the forest
--max_leaf_nodes    - Grow trees with max_leaf_nodes
--class_weight      - Weights associated with classes
```

The primary metric is 'Accuracy' which is set by executing the train.py script. BanditPolicy is configured as the early termination policy for the run.

In the `train.py` script there are three hyperparameters defined that we can be pass to create the RandomForestClassifier model:
```
    parser.add_argument('--n_estimators',   type=int, default=20,   help="Number of trees in the forest")
    parser.add_argument('--max_leaf_nodes', type=int, default=60,   help="Grow trees with max_leaf_nodes")
    parser.add_argument('--class_weight',   type=str, default=None, help="Weights associated with classes")

```

Two version of Hyperdrive - One SKLearn class and the other using ScriptRunConfig Class

Configure the Hyperdrive run using ScriptRunConfig class to set configuration information for submitting a training run in Azure Machine Learning.

The ***max_iter*** parameter is of type integer and I have used `choice` to specify four discrete values in the sampler as follows: 
```
# Specify parameter sampler
ps = RandomParameterSampling({
        "--n_estimators":    choice(30, 40, 60, 80, 100, 120),
        "--max_leaf_nodes":  choice(50, 60, 100),
        "--class_weight":    choice('balanced', 'balanced_subsample')
})
```

In random sampling, hyperparameter values are chosen randomly, thus saving a lot of computational efforts.
It can also be used as a starting sampling method as we can use it to do an initial search and then continue with other sampling methods.



#### Eearly Termnination Policy

The purpose of early termination policy is to automatically terminate poorly performing runs so we do not waste time and resources for the experiment. 

There are a number of early termination policies such as BanditPolicy, MedianStoppingPolicy and TruncationSelectionPolicy. 
For the project I have chosen the **BanditPolicy** based on the slack factor which is the mandatory parameter.
I have also specified optional parameters, evaluation interval and delay evaluation as follows:
```
policy = BanditPolicy (slack_factor = 0.1, evaluation_interval = 1, delay_evaluation = 5)
```
The above policy basically states to check the job at every iteration after the initial delay of 5 evaluations. 
If the primary metric (accuracy) falls outside of the top 10% range, Azure ML will terminate the job. 

The early termination policy ensures that only the best performing runs will execute to completion and hence makes the process more efficient.

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

### Save Model in ONNX format
As the additional task I have converted the best AutoML model to ONNX format. 

#### Retrieve the Best ONNX Model

To retrieve the best ONNX model from the AutoML experiment run, we use the same __get_output__ method as above. 
In addition, the parameter __return_onnx_model__ has to be set to true to retrieve the best ONNX model, instead of the Python model:
```
best_run, onnx_model = remote_run.get_output(return_onnx_model=True)
```

#### Save the Best ONNX Model

To save the model in ONNX format we need to use the __OnnxConverter__ class with __save_onnx_model__ method to convert the model from pkl format to onnx format:

```
from azureml.automl.runtime.onnx_convert import OnnxConverter

onnx_path = out_dir + "/automl_best_model.onnx"
OnnxConverter.save_onnx_model(onnx_model, onnx_path)
```

#### Predict with ONNX Model

I have saved four data samples in the CSV file __Amphibians_testset.csv__ and used the Tabular to convert it to Azure Dataset for input to ONNX model:

```
test_data = "https://raw.githubusercontent.com/arees88/ML_Azure_P3_Capstone/main/Amphibians_testset.csv"
test_dataset = Dataset.Tabular.from_delimited_files(test_data)
```

The code below shows how to get the necessary ONNX resources and then use the __onnxruntime__ package and __OnnxInferenceHelper__ class to get predictions with ONNX model:
```
import onnxruntime
from azureml.automl.runtime.onnx_convert import OnnxInferenceHelper

def get_onnx_res(run):
    res_path = 'onnx_resource.json'
    run.download_file(name=constants.MODEL_RESOURCE_PATH_ONNX, output_file_path=res_path)
    with open(res_path) as f:
        onnx_res = json.load(f)
    return onnx_res

if python_version_compatible:
    test_df = test_dataset.to_pandas_dataframe()
    mdl_bytes = onnx_model.SerializeToString()
    onnx_res = get_onnx_res(best_run)

    onnxrt_helper = OnnxInferenceHelper(mdl_bytes, onnx_res)
    pred_onnx, pred_prob_onnx = onnxrt_helper.predict(test_df)

    print(pred_onnx)
    print(pred_prob_onnx)
else:
    print('Please use Python version 3.6 or 3.7 to run the inference helper.')
```

Below is the screenshots of Jupyter notebook where the model from the best AutoML run was retrieved: 

![alt text](screenshots/8.1_Get_ONNX_model_121947852-72142c80-cd4e-11eb-8246-b55319cec17d.png)

Below screenshot show the code that converts the model to the ONNX format and saves to a file:

![alt text](screenshots/8.2_Save_ONNX_model_121948014-a25bcb00-cd4e-11eb-8d2b-5ca9a5044f83.png)

![alt text](screenshots/8.3_Save_ONNX_model_Zoom_121948142-c7503e00-cd4e-11eb-9929-586fbcd62d96.png)

![alt text](screenshots/8.4_ONNX_Test_Data.PNG)

![alt text](screenshots/8.5_ONNX_model_Predict.PNG)


## Future Suggestions

As mentioned in the previous section, both experiments in the project had restricted number of iterations when searching for best model performance.

In the case of HyperDrive experiment we used Random sampling and restricted the number of iterations to 20. Using higher number of iterations with more Random sampler choices may help with finding a set of hyperparameters that give better performance. To make sure that we don't miss the best performing hyperparameter settings we could swith to Grid sampling instead. Choosing a diffrent early termination policy may also help by providing savings without terminating promising jobs. For example using the more conservative Median Stopping Policy rather than BanditPolicy.

With the AutoML option we restricted the experiment time to 30 minutes which allowed for 28 models to be explored. Increasing the experiment time would potentially allow to find another, better performing model.

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

