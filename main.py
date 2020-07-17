#Package Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

#Data Import
feature = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_label.csv')

#Test Set Import
X_test = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_test_feature.csv').to_numpy()
y_test = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_test_label.csv').to_numpy()

#Extract Data and Reshape
X = feature.to_numpy()
y = label.to_numpy()

#Normalizing data
Xnorm = preprocessing.scale(X)

#Splitting training and validation data
X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, test_size=0.2, random_state=0)

''' 
Height = feature['Height'].to_numpy()
Len1 = feature['Length1'].to_numpy()
Len2 = feature['Length2'].to_numpy()
Len3 = feature['Length3'].to_numpy()     
Width = feature['Width'].to_numpy()              
weight = label['Weight'].to_numpy()

#Training Score
MSE_Train = metrics.mean_squared_error(Ytrain, yhat)
MAE = metrics.mean_absolute_error(Ytrain, yhat) 

#Testing Score
MSE = metrics.mean_squared_error(Ytest, yhat)
MAE = metrics.mean_absolute_error(Ytest, yhat)
'''
regr = LinearRegression()
regr.fit(X_train, y_train)

y_hat = regr.predict(X_train)
MSE_train = np.mean((y_hat - y_train)**2)
print("Train MSE: {}".format(MSE_train))

y_hat = regr.predict(X_test)
MSE_test = np.mean((y_hat - y_test)**2)
print("Test MSE: {}".format(MSE_test))

#Regularization with Ridge Function
reg = linear_model.Ridge(alpha=.05, fit_intercept=True)      #Alpha is hyper-parameter
reg.fit(X_train, y_train)
w = reg.coef_
print('Training score with Ridge, ', reg.score(X_train, y_train))

#Working with the test set
y_test_hat = reg.predict(X_test)                        #Predicted test label
test_score = reg.score(X_test, y_test)
print('Testing data set score, ', test_score)

#Mean Absolute Error
print('Mean absolute error: ', mean_absolute_error(y_test, y_hat))

#Regression Coefficient / Weights
print('Weights: ', regr.coef_)

#Making a graph
plt.figure()
plt.plot(y_hat)
plt.plot(y_test)
plt.show()
