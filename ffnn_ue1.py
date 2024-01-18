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
                    myvalue_24 = 5888
                    myvalue_23 = 5504
                    myvalue_22 = 5120
                    myvalue_21 = 4736
                    myvalue_20 = 4352
                    myvalue_19 = 3968
                    myvalue_18 = 3624
                    myvalue_17 = 3368
                    myvalue_16 = 3496
                    myvalue_15 = 3240
                    myvalue_14 = 2850
                    myvalue_13 = 2536
                    myvalue_12 = 2216
                    myvalue_11 = 2024
                    myvalue_10 = 1920
                    myvalue_9 = 1800
                    myvalue_8 = 1608
                    myvalue_7 = 1352
                    myvalue_6 = 1160
                    myvalue_5 = 984
                    myvalue_4 = 808
                    myvalue_3 = 640
                    myvalue_2 = 504
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