#!/usr/bin/python3
"""
This script generates scatter and line plots for a linear model using
data from a CSV file containing model variables.

Usage:
    Execute the script in the terminal with a CSV file as input.
    The script will create scatter plots and fit a linear model to the
    data.

Requirements:
    - Python 3
    - pandas
    - scikit-learn (sklearn)
    - matplotlib

Example:
    $ ./graph_regression.py data.csv

Returns:
    - Scatter plot showing the data points.
    - Line plot representing the linear model fit.

Note:
    Make sure the CSV file contain just two columns for the model
    variables.
    The first and second columns correspond to x and y variables
    respectively.

Author:
    Jose M. Tobón

Date:
    May 18, 2024
"""

import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# File path given by user
file_path = sys.argv[1]

# Loading data with pandas
data = pd.read_csv(file_path)

# Creating x and y variables
X = data.iloc[:, 0].values.reshape(-1, 1)
Y = data.iloc[:, 1].values.reshape(-1, 1)

# Making the model
model = LinearRegression()

# Training the model
model.fit(X, Y)

# Defining regression variables: slope, bias, etc.
m = model.coef_
b = model.intercept_
r_squared = model.score(X, Y)

if b < 0:
    function = f'Y = {m[0][0]:.4f} * X - {-b[0]:.4f}'
else:
    function = f'Y = {m[0][0]:.4f} * X + {b[0]:.4f}'

text = f'{function}\nR² = {r_squared:.4f}'

# Making y variable to plot
y = m * X + b

# Defining title and axis labels
title = input('\nSet title: ')
x_label = input('Set x label: ')
y_label = input('Set y label: ')

# Ploting regression line and function
plt.scatter(X, Y)
plt.plot(X, y, color='orange')
plt.title(title)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.annotate(text, xy=(0, 1), xycoords='axes fraction',
             xytext=(1, -3), textcoords='offset fontsize',)
plt.show()

