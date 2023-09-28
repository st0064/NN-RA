# Feed-Forward Neural Network (FFNN) for Rate Adaption 

## Introduction
We have trained the NN model based on the MU-MIMO simulator to predict the effective rate according to CSIs values. The scenario is based on 4 UE and trained 28 models per UE.

## The dataset

The data set of per UE MCS wise is already given. For UE1 total 28 MCS data set are provided and name of files are: 1_MCS.csv,2_MCS.csv, 3_MCS.csv,.......,28_MCS.csv.
For UE2, 28 MCS data set provided and names are: 1_MCS_UE2, 2_MCS_UE2, 3_MCS_UE2, .........., 28_MCS_UE2

The data sets are ready to use. The csv file will be loaded in the program for training and testing.


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

The model name UE and MCS wise are already trained and provided. The trained model names for UE1 MCS wise are:1_model, 2_model, 3_model,...........,28_model
The trained model names for UE2 MCS wise are:1_model_UE2, 2_model_UE2, 3_model_UE2,...........,28_model_UE2

You can upload the model in the program and test the model with your data set. You can also re-train the model. To use the available model just set the below line in the ffnn.py (for example to use UE1 for MCS27 27_model) :
Line 21: to_train = False # don't re-train, use the saved model
Line 22: #to_train = True # train the model again 
Line 40: model = load_model('27_model') # load the model from the given folder

## Test the model for unkown input for each MCS
You can test the individual model for unknown CSI. For this, the CSI information is already given in the program. for UE1 MCS27 the test data is:

test_data_27 = [333.804, -6.844, 279.694, -5.928, 278.983, -5.378, 279.736, -5.775, 279.396, -5.938, 281.006, -7.159, 280.36, -6.508, 280.37, -6.1, 279.431, -6.948, 279.886, -6.126, 385.678, -6.494, 355.335, -6.159, 355.107, -5.67, 354.769, -6.049, 354.718, -6.05, 356.267, -6.83, 355.849, -6.267, 354.985, -5.656, 354.566, -6.131, 355.128, -5.95, 659.8, -5.62, 635.616, -5.623, 633.958, -6.875, 635.382, -6.263, 634.566, -6.571, 635.057, -5.867, 635.147, -5.866, 635.68, -6.201, 633.101, -5.99, 634.328, -7.234, 385.21, -5.519, 355.791, -5.847, 354.511, -6.245, 355.218, -6.635, 354.606, -6.485, 355.416, -6.049, 355.296, -6.188, 355.276, -5.946, 353.031, -5.335, 354.573, -6.523]

Similarly, for UE2 MCS28 you can use test_data_28_UE2.

Use the function to make predictions. For UE1 MCS27 use- 
    predicted_output = function_to_predict(test_data_27)

The function will predict the effective rate based on the trained 27_MCS model.

To check the accuracy we already provided the actual effective rate for the test_data_27. Now, you can compare the predicted effective rate and actual effective rate to validate the model. 

print("Actual value MCS 27: 5040.64")
