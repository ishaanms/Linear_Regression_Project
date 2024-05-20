# importing libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import math

# reading data
data = pd.read_csv("TSLA.csv")
data.head()
data.info()
data.describe()
data.columns


X = data[['High','Low','Open','Volume']].values
y = data['Close'].values
X
y

# assigning training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
LgR = LinearRegression()
LgR.fit(X_train, y_train)
print(LgR.coef_)
print(LgR.intercept_)
predicted = LgR.predict(X_test)

# combining actual and predicted data
data1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted' : predicted.flatten()})
data1.head(20)

#plotting graph
graph = data1.head(20)
graph.plot(kind='bar')

# mean absolute error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,predicted))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test,predicted)))
