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
PORT = 65432        # Port to listen on

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
                received_values_27 = list(map(float, data.split()))
            # received_int = struct.unpack('!i', data)[0]  # Convert bytes to integer
                print("Received CSI values from ORAN:", received_values_27) 
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("27_MCS.csv")
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
                    model.save('27_model')
                else:
                    model = load_model('27_model') # load the model from the given folder
                
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
                    print(f"Predicted 27 = {y_hat[i]:.2f}; Actual 27 = {y_true[i]:.2f}; Diff. 27 = {y_true[i] - y_hat[i]:.2f}")
                    test_data_27 = received_values_27
                    predicted_output_27 = function_to_predict(test_data_27)
                    myvalue_27 = round(predicted_output_27)
                    print("Actual value MCS 27: 5040.64")
                    response_27 = myvalue_27
                    print ("Response 27 send to xApp:", myvalue_27)  
                    #conn.sendall(struct.pack('!i', response_27))   # Send integer as 4 bytes

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("26_MCS.csv")
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
                    model.save('26_model')
                else:
                    model = load_model('26_model') # load the model from the given folder
                
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
                    print(f"Predicted 26 = {y_hat[i]:.2f}; Actual 26 = {y_true[i]:.2f}; Diff. 26 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_26 = received_values_27
                    predicted_output_26 = function_to_predict(test_data_26)
                    myvalue_26 = round(predicted_output_26)
                    print("Actual value MCS 26: 6512.64")
                    response_26 = myvalue_26
                    print ("Response 26 send to xApp:", myvalue_26)                      

 # Load your dataset 'sample_data.csv'
                df = pd.read_csv("25_MCS.csv")
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
                    model.save('25_model')
                else:
                    model = load_model('25_model') # load the model from the given folder
                
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

                    test_data_25 = received_values_27
                    predicted_output_25 = function_to_predict(test_data_25)
                    myvalue_25 = round(predicted_output_25)
                    print("Actual value MCS 25: 6272")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("24_MCS.csv")
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
                    model.save('24_model')
                else:
                    model = load_model('24_model') # load the model from the given folder
                
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
                    print(f"Predicted 24 = {y_hat[i]:.2f}; Actual 24 = {y_true[i]:.2f}; Diff. 24 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_24 = received_values_27
                    predicted_output_24 = function_to_predict(test_data_24)
                    myvalue_24 = round(predicted_output_24)
                    print("Actual value MCS 24: 5888")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("23_MCS.csv")
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
                    model.save('23_model')
                else:
                    model = load_model('23_model') # load the model from the given folder
                
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
                    print(f"Predicted 23 = {y_hat[i]:.2f}; Actual 23 = {y_true[i]:.2f}; Diff. 23 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_23 = received_values_27
                    predicted_output_23 = function_to_predict(test_data_23)
                    myvalue_23 = round(predicted_output_23)
                    print("Actual value MCS 23: 5504")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("22_MCS.csv")
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
                    model.save('22_model')
                else:
                    model = load_model('22_model') # load the model from the given folder
                
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
                    print(f"Predicted 22 = {y_hat[i]:.2f}; Actual 22 = {y_true[i]:.2f}; Diff. 22 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_22 = received_values_27
                    predicted_output_22 = function_to_predict(test_data_22)
                    myvalue_22 = round(predicted_output_22)
                    print("Actual value MCS 22: 5120")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("21_MCS.csv")
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
                    model.save('21_model')
                else:
                    model = load_model('21_model') # load the model from the given folder
                
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
                    print(f"Predicted 21 = {y_hat[i]:.2f}; Actual 21 = {y_true[i]:.2f}; Diff. 21 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_21 = received_values_27
                    predicted_output_21 = function_to_predict(test_data_21)
                    myvalue_21 = round(predicted_output_21)
                    print("Actual value MCS 21: 4736")

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("20_MCS.csv")
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
                    model.save('20_model')
                else:
                    model = load_model('20_model') # load the model from the given folder
                
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
                    print(f"Predicted 20 = {y_hat[i]:.2f}; Actual 20 = {y_true[i]:.2f}; Diff. 20 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_20 = received_values_27
                    predicted_output_20 = function_to_predict(test_data_20)
                    myvalue_20 = round(predicted_output_20)
                    print("Actual value MCS 20: 4352") 

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("19_MCS.csv")
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
                    model.save('19_model')
                else:
                    model = load_model('19_model') # load the model from the given folder
                
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
                    print(f"Predicted 19 = {y_hat[i]:.2f}; Actual 19 = {y_true[i]:.2f}; Diff. 19 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_19 = received_values_27
                    predicted_output_19 = function_to_predict(test_data_19)
                    myvalue_19 = round(predicted_output_19)
                    print("Actual value MCS 19: 3968")

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("18_MCS.csv")
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
                    model.save('18_model')
                else:
                    model = load_model('18_model') # load the model from the given folder
                
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
                    print(f"Predicted 18 = {y_hat[i]:.2f}; Actual 18 = {y_true[i]:.2f}; Diff. 18 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_18 = received_values_27
                    predicted_output_18 = function_to_predict(test_data_18)
                    myvalue_18 = round(predicted_output_18)
                    print("Actual effective rate MCS 18: 3624")

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("17_MCS.csv")
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
                    model.save('17_model')
                else:
                    model = load_model('17_model') # load the model from the given folder
                
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
                    print(f"Predicted 17 = {y_hat[i]:.2f}; Actual 17 = {y_true[i]:.2f}; Diff. 17 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_17 = received_values_27
                    predicted_output_17 = function_to_predict(test_data_17)
                    myvalue_17 = round(predicted_output_17)
                    print("Actual effective rate MCS 17: 3368")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("16_MCS.csv")
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
                    model.save('16_model')
                else:
                    model = load_model('16_model') # load the model from the given folder
                
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
                    print(f"Predicted 16 = {y_hat[i]:.2f}; Actual 16 = {y_true[i]:.2f}; Diff. 16 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_16 = received_values_27
                    predicted_output_16 = function_to_predict(test_data_16)
                    myvalue_16 = round(predicted_output_16)
                    print("Actual effective rate MCS 16: 3496")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("15_MCS.csv")
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
                    model.save('15_model')
                else:
                    model = load_model('15_model') # load the model from the given folder
                
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
                    print(f"Predicted 15 = {y_hat[i]:.2f}; Actual 15 = {y_true[i]:.2f}; Diff. 15 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_15 = received_values_27
                    predicted_output_15 = function_to_predict(test_data_15)
                    myvalue_15 = round(predicted_output_15)
                    print("Actual effective rate MCS 15: 3240")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("14_MCS.csv")
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
                    model.save('14_model')
                else:
                    model = load_model('14_model') # load the model from the given folder
                
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
                    print(f"Predicted 14 = {y_hat[i]:.2f}; Actual 14 = {y_true[i]:.2f}; Diff. 14 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_14 = received_values_27
                    predicted_output_14 = function_to_predict(test_data_14)
                    myvalue_14 = round(predicted_output_14)
                    print("Actual effective rate MCS 14: 2856")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("13_MCS.csv")
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
                    model.save('13_model')
                else:
                    model = load_model('13_model') # load the model from the given folder
                
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
                    print(f"Predicted 13 = {y_hat[i]:.2f}; Actual 13 = {y_true[i]:.2f}; Diff. 13 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_13 = received_values_27
                    predicted_output_13 = function_to_predict(test_data_13)
                    myvalue_13 = round(predicted_output_13)
                    print("Actual effective rate MCS 13: 2536")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("12_MCS.csv")
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
                    model.save('12_model')
                else:
                    model = load_model('12_model') # load the model from the given folder
                
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
                    print(f"Predicted 12 = {y_hat[i]:.2f}; Actual 12 = {y_true[i]:.2f}; Diff. 12 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_12 = received_values_27
                    predicted_output_12 = function_to_predict(test_data_12)
                    myvalue_12 = round(predicted_output_12)
                    print("Actual effective rate MCS 12: 2216")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("11_MCS.csv")
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
                    model.save('11_model')
                else:
                    model = load_model('11_model') # load the model from the given folder
                
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
                    print(f"Predicted 11 = {y_hat[i]:.2f}; Actual 11 = {y_true[i]:.2f}; Diff. 11 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_11 = received_values_27
                    predicted_output_11 = function_to_predict(test_data_11)
                    myvalue_11 = round(predicted_output_11)
                    print("Actual value MCS 11: 6512.64")

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("10_MCS.csv")
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
                    model.save('10_model')
                else:
                    model = load_model('10_model') # load the model from the given folder
                
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
                    print(f"Predicted 10 = {y_hat[i]:.2f}; Actual 10 = {y_true[i]:.2f}; Diff. 10 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_10 = received_values_27
                    predicted_output_10 = function_to_predict(test_data_10)
                    myvalue_10 = round(predicted_output_10)
                    print("Actual value MCS 10: 1800")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("9_MCS.csv")
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
                    model.save('9_model')
                else:
                    model = load_model('9_model') # load the model from the given folder
                
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
                    print(f"Predicted 9 = {y_hat[i]:.2f}; Actual 9 = {y_true[i]:.2f}; Diff. 9 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_9 = received_values_27
                    predicted_output_9 = function_to_predict(test_data_9)
                    myvalue_9 = round(predicted_output_9)
                    print("Actual value MCS 9: 1800")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("8_MCS.csv")
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
                    model.save('8_model')
                else:
                    model = load_model('8_model') # load the model from the given folder
                
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
                    print(f"Predicted 8 = {y_hat[i]:.2f}; Actual 8 = {y_true[i]:.2f}; Diff. 8 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_8 = received_values_27
                    predicted_output_8 = function_to_predict(test_data_8)
                    myvalue_8 = round(predicted_output_8)
                    print("Actual effective rate MCS 8: 1608")
                    
# Load your dataset 'sample_data.csv'
                df = pd.read_csv("7_MCS.csv")
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
                    model.save('7_model')
                else:
                    model = load_model('7_model') # load the model from the given folder
                
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
                    print(f"Predicted 7 = {y_hat[i]:.2f}; Actual 7 = {y_true[i]:.2f}; Diff. 7 = {y_true[i] - y_hat[i]:.2f}")

                    test_data_7 = received_values_27
                    predicted_output_7 = function_to_predict(test_data_7)
                    myvalue_7 = round(predicted_output_7)
                    print("Actual value MCS 26: 1352")

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("6_MCS.csv")
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
                    model.save('6_model')
                else:
                    model = load_model('6_model') # load the model from the given folder
                
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

                    test_data_6 = received_values_27
                    predicted_output_6 = function_to_predict(test_data_6)
                    myvalue_6 = round(predicted_output_6)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("5_MCS.csv")
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
                    model.save('5_model')
                else:
                    model = load_model('5_model') # load the model from the given folder
                
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
 
                    test_data_5 = received_values_27
                    predicted_output_5 = function_to_predict(test_data_5)
                    myvalue_5 = round(predicted_output_5)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("4_MCS.csv")
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
                    model.save('4_model')
                else:
                    model = load_model('4_model') # load the model from the given folder
                
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
                    
                    test_data_4 = received_values_27
                    predicted_output_4 = function_to_predict(test_data_4)
                    myvalue_4 = round(predicted_output_4)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("3_MCS.csv")
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
                    model.save('3_model')
                else:
                    model = load_model('3_model') # load the model from the given folder
                
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
   
                    test_data_3 = received_values_27
                    predicted_output_3 = function_to_predict(test_data_3)
                    myvalue_3 = round(predicted_output_3)

# Load your dataset 'sample_data.csv'
                df = pd.read_csv("2_MCS.csv")
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
                    model.save('2_model')
                else:
                    model = load_model('2_model') # load the model from the given folder
                
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
   
                    test_data_2 = received_values_27
                    predicted_output_2 = function_to_predict(test_data_2)
                    myvalue_2 = round(predicted_output_2)                    
                    myvalue_1 = 408
                    myvalue_0 = 380       

                    #values = [myvalue_0, myvalue_1, myvalue_2, myvalue_3, myvalue_4, myvalue_5, myvalue_6, myvalue_7, myvalue_8, myvalue_9, myvalue_10, myvalue_11, myvalue_12, myvalue_13, myvalue_14, myvalue_15, myvalue_16, myvalue_17, myvalue_18, myvalue_19, myvalue_20, myvalue_21, myvalue_22, myvalue_23, myvalue_24, myvalue_25, myvalue_26, myvalue_27] #, myvalue_6, myvalue_5, myvalue_4, myvalue_3, myvalue_2, myvalue_1, myvalue_0]
                    
                    values = [globals()[f"myvalue_{i}"] for i in range(0, 28)]
                    random.shuffle(values)
                    print("Predicted Effective Rates of UE1:", values)
                    max_value = max(values)
                    position = values.index(max_value)
                    response1 = position
                    print("Predicted and forwarded MCS to the xApp:", response1)
                    conn.sendall(struct.pack('!i', response1))   # Send integer as 4 bytes