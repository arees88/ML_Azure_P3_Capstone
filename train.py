from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# csv_path = "https://github.com/arees88/ML_Azure_P3_Capstone/blob/main/Amphibians_dataset.csv"
# csv_path = "https://raw.githubusercontent.com/arees88/ML_Azure_P3_Capstone/main/Amphibians_dataset.csv"

csv_path = "Amphibians_dataset.csv"

# ----------------------------------------
# clean_data
# ----------------------------------------

def clean_data(data):

    x_df = data.copy()

    # Drop columns not used as input to the model
    x_df.drop(columns=['ID','Motorway','Label2','Label3','Label4','Label5','Label6','Label7'], inplace = True)

    # x_df.to_csv("Amphibians_dataset_green_frogs.csv", sep=',', header=True, index=False)

    y_df = x_df.pop("Label1")

    return x_df, y_df
    

# ----------------------------------------
# main
# ----------------------------------------

def main():

    # Create TabularDataset using TabularDatasetFactory
    # ds = TabularDatasetFactory.from_delimited_files(path = csv_path)

    ds = pd.read_csv(csv_path)

    # Call clean_data to preprocess the dataset
    x, y = clean_data(ds)

    # Split data into train and test sets.
    x_train, x_test, y_train,y_test = train_test_split(x, y, train_size = 0.8, random_state = 88)

    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators',   type=int, default=20,   help="Number of trees in the forest")
    parser.add_argument('--max_leaf_nodes', type=int, default=60,   help="Grow trees with max_leaf_nodes")
    parser.add_argument('--class_weight',   type=str, default=None, help="Weights associated with classes")

    args = parser.parse_args()

    run.log("Number of trees: ", np.int(args.n_estimators))
    run.log("Max leaf nodes:  ", np.int(args.max_leaf_nodes))
    run.log("Class weight:    ", args.class_weight)

    model = RandomForestClassifier( random_state   = 42,
                                    n_estimators   = args.n_estimators,
                                    max_leaf_nodes = args.max_leaf_nodes,
                                    class_weight   = args.class_weight)

    # run.log(model)

    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    # Files saved in the "outputs" folder are automatically uploaded into run history
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()

