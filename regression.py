#EXP1 - REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_csv("Advertising.csv")

# Simple Linear Regression
X,y=data[['TV']],data['Sales']
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
m=LinearRegression().fit(Xt,yt)
p=m.predict(Xv)
print("\n--- Simple Linear Regression ---")
print("MSE:",round(mean_squared_error(yv,p),2),", R2:",round(r2_score(yv,p),2))
plt.scatter(X,y);plt.plot(X,m.predict(X), color='red');plt.title("Simple Linear Regression")
plt.xlabel("TV");plt.ylabel("Sales");plt.show()

# Multiple Linear Regression
X,y=data[['TV','Radio','Newspaper']],data['Sales']
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
m=LinearRegression().fit(Xt,yt);p=m.predict(Xv)
print("\n--- Multiple Linear Regression ---")
print("MSE:",round(mean_squared_error(yv,p),2),", R2:",round(r2_score(yv,p),2))
plt.scatter(yv,p);plt.title("Multiple Linear Regression")
plt.xlabel("Actual Sales");plt.ylabel("Predicted Sales");plt.show()

# Polynomial Regression
X,y=data[['TV']],data['Sales']
poly=PolynomialFeatures(2)
Xp=poly.fit_transform(X)
Xt,Xv,yt,yv=train_test_split(Xp,y,test_size=.2,random_state=42)
m=LinearRegression().fit(Xt,yt);p=m.predict(Xv)
print("\n--- Polynomial Regression ---")
print("MSE:",round(mean_squared_error(yv,p),2),", R2:",round(r2_score(yv,p),2))
xs=np.sort(X,axis=0)
plt.scatter(X,y);plt.plot(xs,m.predict(poly.transform(xs)), color='red')
plt.title("Polynomial Regression");plt.xlabel("TV");plt.ylabel("Sales");plt.show()