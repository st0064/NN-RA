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

