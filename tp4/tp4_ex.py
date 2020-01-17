

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#~ X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
#~ print (X)
#~ # y = 1 * x_0 + 2 * x_1 + 3
#~ y = np.dot(X, np.array([1, 2])) + 3
#~ print (y)
#~ reg = LinearRegression().fit(X, y)


X = np.array ([[5.5], [6.0], [6.5], [6.0], [5.0], [6.5], [4.5], [5]])
print ("X\n", X)
y = np.array ([[420], [380], [350], [400], [440], [380], [450], [420]])
print ("\ny\n", y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42) #size ==> % appliqué au test

reg = LinearRegression().fit(X, y)

print ("reg\n", reg)

print ("score\n", reg.score(X, y))
print ("coef\n", reg.coef_)
print ("intercept\n", reg.intercept_) 
#~ print (reg.predict(np.array([[3, 5]])))

print ("\nPredict X")
print (reg.predict (X))
print ("\nPredict y")
print (reg.predict (y))
