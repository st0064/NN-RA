# Feed-Forward Neural Network (FFNN) for Rate Adaption 

## Introduction

We've utilized the MU-MIMO simulator to train a neural network (NN) model for predicting effective rates based on CSI values. This training scenario involves four user equipment (UE) and a total of 28 models trained for each UE.

## The dataset

The dataset for each UE, organized by MCS, has been provided. For UE1, there are a total of 28 MCS datasets with file names ranging from "1_MCS.csv" to "28_MCS.csv." Similarly, for UE2, another 28 MCS datasets are available with file names from "1_MCS_UE2.csv" to "28_MCS_UE2.csv."

These datasets are prepared and ready for use. You can load the CSV files into your program for both training and testing purposes.

## Setup environment

Create a Python virtual environment for this project. This can be done by the following commands.

```
python3 -m venv venv
source venv/bin/activate
```

Install packages and dependencies.

```
pip install numpy pandas scikit-learn tensorflow
```

## Run the code

Run the code using the following command. By default, the program will read the model stored in the local folder and test the stored model. You can change the code to re-train the model.

```
python ffnn.py
```
The program has 3 run modes:
    mode 1: train & save the model into a folder
    mode 2: train the model but don't save it
    mode 3: (default) load the saved model from the folder

The models have been trained and organized by UE and MCS. For UE1, the trained model names for each MCS range from "1_model" to "28_model." Similarly, for UE2, the model names follow the pattern "1_model_UE2" to "28_model_UE2."

You can easily upload these trained models into your program and conduct testing with your dataset. Additionally, if needed, you have the option to re-train the models. To use a specific model, simply set the following line in the "ffnn.py" file, for example, to utilize the 27th model for MCS27 in UE1:

Line 21: to_train = False                                 # don't re-train, use the saved model
Line 22: #to_train = True                                 # train the model again 
Line 40: model = load_model('27_model')                   # load the model from the given folder

## Test the model for unkown input for each MCS
You have the flexibility to test each individual model with unknown CSI data. The program already contains the necessary CSI information. Specifically, for testing MCS27 in UE1, the test data is readily available:

test_data_27 = [333.804, -6.844, 279.694, -5.928, 278.983, -5.378, 279.736, -5.775, 279.396, -5.938, 281.006, -7.159, 280.36, -6.508, 280.37, -6.1, 279.431, -6.948, 279.886, -6.126, 385.678, -6.494, 355.335, -6.159, 355.107, -5.67, 354.769, -6.049, 354.718, -6.05, 356.267, -6.83, 355.849, -6.267, 354.985, -5.656, 354.566, -6.131, 355.128, -5.95, 659.8, -5.62, 635.616, -5.623, 633.958, -6.875, 635.382, -6.263, 634.566, -6.571, 635.057, -5.867, 635.147, -5.866, 635.68, -6.201, 633.101, -5.99, 634.328, -7.234, 385.21, -5.519, 355.791, -5.847, 354.511, -6.245, 355.218, -6.635, 354.606, -6.485, 355.416, -6.049, 355.296, -6.188, 355.276, -5.946, 353.031, -5.335, 354.573, -6.523]

In a similar manner, for MCS28 in UE2, you can employ the "test_data_28_UE2."

To make predictions, use the following function:
    predicted_output = function_to_predict(test_data_27)

This function utilizes the trained "27_MCS" model to predict the effective rate. To assess accuracy, we've already furnished the actual effective rate for "test_data_27." Now, you can compare the predicted effective rate with the actual effective rate to validate the model.

Here's the provided actual value for MCS 27: 5040.64.
