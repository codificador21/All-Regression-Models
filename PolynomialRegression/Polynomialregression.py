#Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing The Dataset
dataset = pd.read_csv('datasets_88705_204267_Real estate.csv')
X = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,-1].values

#Visualising The Dataset
list = np.arange(len(y))
plt.bar(list, y, color = 'darkgreen', width = 1)
plt.title('House Prices')
plt.show()

#Splitting Of Dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2 , random_state = 0)

#Training The Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_poly = poly_reg.fit_transform(X_train)
regressor.fit(X_poly,y_train)

#Predicting The Dataset
y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1)),1))

#Evaluating the performance of the model
from sklearn.metrics import r2_score
print("Performance of Decision tree regression is : ",r2_score(y_test,y_pred))

#Visualising The Predicted Set
list = np.arange(len(y_test))
plt.plot(list, y_pred, color = 'blue', label = 'Predicted Value')
plt.plot(list, y_test, color = 'green',  label = 'Original Value')
plt.title('Real Estate (Decision Tree)')
plt.legend()
plt.title(' Polynomial regression model \n Model performance(r squared value) : %f '%r2_score(y_test,y_pred))
plt.show()