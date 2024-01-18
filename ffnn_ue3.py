import pandas as pd
import numpy as np
import math, random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

# Server code to read data from Client C program
import socket
import struct

#HOST = '131.227.60.141'  # localhost
HOST = '127.0.0.1'  # localhost
PORT = 65430        # Port to listen on

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    print(f"Server listening on {HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        print('Connected by', addr)

        with conn:
            while True:
                data = conn.recv(1024).decode()  # Receive data from the C program, Assuming an integer is 4 bytes (32 bits)
                if not data:
                    break
                received_values_25 = list(map(float, data.split()))
            # received_int = struct.unpack('!i', data)[0]  # Convert bytes to integer
                print("Received CSI values from ORAN:", received_values_25) 
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("25_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('25_model_UE3')
                else:
                    model = load_model('25_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    print(f"Predicted 25 = {y_hat[i]:.2f}; Actual 25 = {y_true[i]:.2f}; Diff. 25 = {y_true[i] - y_hat[i]:.2f}")

# Test unkown input data for each MCS. 

                    test_data_25 = received_values_25

# Use the function to make predictions
                    predicted_output_25 = function_to_predict(test_data_25)
                    myvalue_25 = round(predicted_output_25)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("28_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('28_model_UE3')
                else:
                    model = load_model('28_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_28 = received_values_25
                    predicted_output_28 = function_to_predict(test_data_28)
                    myvalue_28 = round(predicted_output_28)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("27_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('27_model_UE3')
                else:
                    model = load_model('27_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_27 = received_values_25
                    predicted_output_27 = function_to_predict(test_data_27)
                    myvalue_27 = round(predicted_output_27)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("26_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('26_model_UE3')
                else:
                    model = load_model('26_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_26 = received_values_25
                    predicted_output_26 = function_to_predict(test_data_26)
                    myvalue_26 = round(predicted_output_26)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("24_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('24_model_UE3')
                else:
                    model = load_model('24_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_24 = received_values_25
                    predicted_output_24 = function_to_predict(test_data_24)
                    myvalue_24 = round(predicted_output_24)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("23_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('23_model_UE3')
                else:
                    model = load_model('23_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_23 = received_values_25
                    predicted_output_23 = function_to_predict(test_data_23)
                    myvalue_23 = round(predicted_output_23)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("22_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('22_model_UE3')
                else:
                    model = load_model('22_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_22 = received_values_25
                    predicted_output_22 = function_to_predict(test_data_22)
                    myvalue_22 = round(predicted_output_22)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("21_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('21_model_UE3')
                else:
                    model = load_model('21_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_21 = received_values_25
                    predicted_output_21 = function_to_predict(test_data_28)
                    myvalue_21 = round(predicted_output_21)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("20_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('20_model_UE3')
                else:
                    model = load_model('20_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_20 = received_values_25
                    predicted_output_20 = function_to_predict(test_data_20)
                    myvalue_20 = round(predicted_output_20)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("19_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('19_model_UE3')
                else:
                    model = load_model('19_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_19 = received_values_25
                    predicted_output_19 = function_to_predict(test_data_19)
                    myvalue_19 = round(predicted_output_19)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("18_MCS_UE3.csv")
                df = df.dropna()          #removes any rows containing missing values
                X = df.drop('y', axis=1)  # We have 80 input columns, considering all except y column
                y = df['y']  # Here "y" is target variable

                print("Shape of X:", X.shape)  # This will print the rows and columns which has a shape of 610 rows and 80 columns.
                print("Shape of y:", y.shape)  # This will print the columns variable (target variable) which has a shape of 29 rows and 1 column

            # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                to_train = False # don't re-train, use the saved model
            #to_train = True # train the model again
                if to_train:
            # Define the neural network model
                    model = Sequential()
                    model.add(Dense(units=64, activation='relu', input_dim=80))  # Adjust input_dim to match the number of input features (80)
                    model.add(Dense(units=128, activation='tanh'))
                    model.add(Dense(units=256, activation='tanh'))
                    model.add(Dense(units=512, activation='relu'))
                    model.add(Dense(units=256, activation='relu'))
                    model.add(Dense(units=64, activation='relu'))
                    model.add(Dense(units=8, activation='relu'))
                    model.add(Dense(units=1)) # Output layer for regression
                    model.compile(loss='mean_squared_error', optimizer='adam')           

            # Train the model (you can adjust batch size and epochs as needed)
                    model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)
                    model.save('18_model_UE3')
                else:
                    model = load_model('18_model_UE3') # load the model from the given folder
                
            # Function to predict 'y' based on input 'x' using the trained model
                def function_to_predict(x):
                # Reshape 'x' to match the input shape of the model (80 input features)
                    x = np.array(x).reshape(1, -1)
                    print(x)
                # Predict 'y' using the model

                    y_pred = model.predict(x)[0][0]            #used to extract a specific value from the prediction
                    print(f"Predicted value: {y_pred:.2f}")
                    return y_pred   

            # Generate random input data and make predictions 
                x_hat = []
                y_true = []

                for _ in range(1):
                    x = [random.uniform(-10.0, 700.0) for _ in range(80)]  # Generate random input within the specified range
                    x_hat.append(x)
                    y_true.append(function_to_predict(x))

            # Predict 'y' using the model
                y_hat = [function_to_predict(x) for x in x_hat]

            # Print the results in
                for i in range(len(y_true)):
                    #print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_18 = received_values_25
                    predicted_output_18 = function_to_predict(test_data_18)
                    myvalue_18 = round(predicted_output_18)

                    myvalue_17 = 5146
                    myvalue_16 = 5111
                    myvalue_15 = 5146
                    myvalue_14 = 466
                    myvalue_13 = 5082
                    myvalue_12 = 313
                    myvalue_11 = 5033
                    myvalue_10 = 5054
                    myvalue_9 = 5167
                    myvalue_8 = 5089
                    myvalue_7 = 5301
                    myvalue_6 = 5096
                    myvalue_5 = 5244
                    myvalue_4 = 5075
                    myvalue_3 = 5040
                    myvalue_2 = 5181
                    myvalue_1 = 4970
                    myvalue_0 = 5102

                    values = [globals()[f"myvalue_{i}"] for i in range(0, 28)]
                    random.shuffle(values)
                    print("Predicted Effective Rates of UE3:", values)
                    max_value = max(values)
                    position = values.index(max_value)
                    response1 = position
                    print("Predicted and forwarded MCS to the xApp:", response1)
                    conn.sendall(struct.pack('!i', response1))   # Send integer as 4 bytes