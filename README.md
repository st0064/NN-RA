# Feed-Forward Neural Network (FFNN) for Regression

## Introduction

This is a very simple example showing how FFNN can be used for regression.
In this example, the function to predict is 
$$f(x_1,x_2) = x_2 sin(x_1) - x_1 cos(x_2)$$ 
where $x_1$ and $x_2$ are two inputs between -5.0 and 5.0. 
The following is the plot of the function:

![Plot](https://gitlab.surrey.ac.uk/cf0014/ffnn/-/raw/main/sample_plot.png)


## The dataset

The dataset is already given in `sample_data.csv`. It's ready to use. The csv file will be loaded in the program for training and testing.

You can also regenerate a new set of dataset by using `sample_generator.xlsx`. You can simply open and touch the file to generate new random inputs in the spreadsheet, then save the file as `sample_data.csv`. 

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

## FFNN Design

The following is the default FFNN design in the code:

```python
    model = Sequential()
    model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=12, activation='tanh'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=50, epochs=200)
```

We have 5000 instances in our dataset, 80% (or 4000) are used for training. Since each batch size is 50, the FFNN will be updated 80 times for each epoch. The final few epochs show:

```
...
Epoch 197/200
80/80 [==============================] - 0s 657us/step - loss: 0.2446
Epoch 198/200
80/80 [==============================] - 0s 646us/step - loss: 0.2423
Epoch 199/200
80/80 [==============================] - 0s 624us/step - loss: 0.2428
Epoch 200/200
80/80 [==============================] - 0s 639us/step - loss: 0.2383
```

![Plot](https://gitlab.surrey.ac.uk/cf0014/ffnn/-/raw/main/ffnn.png)

