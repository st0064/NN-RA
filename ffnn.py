############################################################
# install the following dependencies:
# > pip install numpy pandas scikit-learn tensorflow
############################################################

import pandas as pd
import numpy as np
import math, random
from sklearn.model_selection import train_test_split

############################################################
# the csv specifies the function to predict,
# it contains 3 columns:
# - column 1: `x1`, float between -5.0 & 5.0
# - column 2: `x2`, float between -5.0 & 5.0
# - column 3: `y`,  float y = x2*sin(x1) - x1*cos(x2)
############################################################

df = pd.read_csv("sample_data.csv")
X = pd.get_dummies(df.drop("y",axis=1)) # the input columns, drop `y` to keep x1 & x2
y = df["y"]                             # the output column, keep `y` only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

############################################################
# training
############################################################

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

to_train = False # don't re-train, use the saved model
#to_train = True # train the model again
if to_train:
    model = Sequential()

    model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=12, activation='tanh'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=50, epochs=200)  # loss: 0.17

    # model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    # model.add(Dense(units=12, activation='tanh'))
    # model.add(Dense(units=1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, batch_size=20, epochs=10)   # loss: 2.5

    # model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    # model.add(Dense(units=12, activation='tanh'))
    # model.add(Dense(units=1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, batch_size=50, epochs=250)  # loss: 0.2

    #model.save('xy_model')  # save the model to a folder?

else:
    model = load_model('xy_model') # load the model from the given folder

############################################################
# further testing
############################################################

def rand_input():
    '''input range of the function'''
    return random.random()*10-5

def function_to_predict(x1,x2):
    '''the function to predict'''
    return x2*math.sin(x1) - x1*math.cos(x2)

x_hat = []
y_true = []
for _ in range(5):
    x1, x2 = rand_input(), rand_input()
    x_hat.append([x1,x2])
    y_true.append(function_to_predict(x1,x2))

y_hat = model.predict(np.array(x_hat))

for i in range(len(y_true)):
    print(f"x1,x2 = {x_hat[i][0]:.2f}, {x_hat[i][1]:.2f}; ",end='')
    print(f"predicted = {y_hat[i][0]:.2f}; actual={y_true[i]:.2f}; diff={y_true[i]-y_hat[i][0]:.2f}")

