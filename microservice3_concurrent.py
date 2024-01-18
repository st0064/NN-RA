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

import concurrent.futures

def train_and_predict_model(model_name, received_values):
    print(f"UE3 Training and predicting for model {model_name}")

    # Load your dataset
    df = pd.read_csv(f"{model_name}_MCS_UE3.csv")
    df = df.dropna()
    X = df.drop('y', axis=1)
    y = df['y']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    to_train = False  # don't re-train, use the saved model
    if to_train:
        # Define the neural network model
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=80))
        model.add(Dense(units=128, activation='tanh'))
        model.add(Dense(units=256, activation='tanh'))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)

        # Save the model
        model.save(f"{model_name}_model_UE3")
    else:
        # Load the pre-trained model
        model = load_model(f"{model_name}_model_UE3")

    # Function to predict 'y' based on input 'x' using the trained model
    def function_to_predict(x):
        x = np.array(x).reshape(1, -1)
        print(x)
        y_pred = model.predict(x)[0][0]
        print(f"Predicted value: {y_pred:.2f}")
        return y_pred

    # Generate random input data and make predictions
    #x_hat = [random.uniform(-10.0, 700.0) for _ in range(80)]
    #y_true = function_to_predict(x_hat)

    # Predict 'y' using the model
    y_hat = function_to_predict(received_values)

    # Print the results
    #print(f"Predicted {model_name} = {y_hat:.2f}; Actual {model_name} = {y_true:.2f}; Diff. {model_name} = {y_true - y_hat:.2f}")

    response = round(y_hat)
    print(f"Response send to Matrix: {response}")

    return response

def listen_for_data():
    HOST = '131.227.60.141'
    PORT = 65430

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            print('Connected by', addr)

            with conn:
                while True:
                    data = conn.recv(1024).decode()
                    if not data:
                        break
                    received_values_25 = list(map(float, data.split()))

                    # Create a ThreadPoolExecutor for parallel execution
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Define the list of model names
                        model_names = ["27", "26", "25", "24", "23", "22", "21", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4","3", "2", "1"]
                        
                        # Execute the models in parallel
                        futures = [executor.submit(train_and_predict_model, model_name, received_values_25) for model_name in model_names]
                        concurrent.futures.wait(futures)
                        results = [future.result() for future in futures]

                    # Choose the model with the highest predicted value
                    random.shuffle(results)
                    print("Predicted Effective Rates of UE3:", results)
                    max_result = max(results)
                    position = results.index(max_result)

                    print(f"Predicted and forwarded UE3 MCS to the xApp: {position}")
                    conn.sendall(struct.pack('!i', position))

# Start listening for data and making predictions
listen_for_data()