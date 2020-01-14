import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fit linear regression model to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predict
y_pred = regressor.predict(x_test)

#plot
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='red')
plt.title('training set')
plt.xlabel('Years of experience')
plt.ylabel('Salary, USD')
plt.show()

#plot
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_train, regressor.predict(x_train), color='red')
plt.title('test set')
plt.xlabel('Years of experience')
plt.ylabel('Salary, USD')
plt.show()