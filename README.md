# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Use the standard libraries in python for Gradient Design.<br>
2.Set Variables for assigning dataset values.<br>
3.Import linear regression from sklearn.<br>
4.Assign the points for representing the graph.<br>
5.predict the regression for marks by using the representation of the graph.<br>
6.Compare the graphs and hence we obtained the linear regression for the given data.<br>

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Suriya Prakash.B
RegisterNumber:  212220220048
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("df.head():")
df.head()
```
```
print("df.tail(): ")
df.tail()
```
```
print("Array values of x:")
x=df.iloc[:,:-1].values
x
```
```
print("Array value of y:")
y=df.iloc[:,1].values
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print("y_pred:")
y_pred
```
```
print("y_test:")
y_test
```
```
print("Training set graph:")
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Trainig set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
print("Test Set graph:")
plt.scatter(x_test,y_test,color="green")
plt.plot(x_test,regressor.predict(x_test),color="violet")
plt.title("Hours vs Scores (Trainig set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:
1.df.head():<br>
<img src="https://user-images.githubusercontent.com/104640337/230722355-50d6c8df-1e3f-4b47-a516-c63943c4849d.png"><br>
2.df.tail():<br>
<img src="https://user-images.githubusercontent.com/104640337/230722376-d85fa375-500c-4591-af1a-97bed4437c3b.png"><br>
3.Array value of X:<br>
<img src="https://user-images.githubusercontent.com/104640337/230722382-b78a0472-c580-4dad-b380-2acaba052143.png"><br>
4.Array value of Y:<br>
<img src="https://user-images.githubusercontent.com/104640337/230722385-d8e49475-79a7-4d5a-8eec-45b91e640514.png"><br>
5.Values of Y prediction:<br>
<img src="https://user-images.githubusercontent.com/104640337/230722395-eca4f828-d973-4214-9223-5cca60b14cd2.png"><br>
6.Array values of Y:<br>
<img src="https://user-images.githubusercontent.com/104640337/230722402-d975abf6-09d2-46c8-93f6-b2efb0c09903.png"><br>
7.Training set Graph:<br>
<img src="https://user-images.githubusercontent.com/104640337/230722406-322b66e7-b86b-401f-bcfc-466a94afd1d3.png" alt="alt text" width="200" height="200"><br>
8.Test set Graph:<br>
<img src="https://user-images.githubusercontent.com/104640337/230722409-52191baf-6b2b-4c59-af8d-d1041e0e0f98.png" alt="alt text" width="200" height="200"><br>
9.Values of MSE, MAE and RMSE:<br>
<img src="https://user-images.githubusercontent.com/104640337/230722416-2c44d6e5-c34c-486e-a249-5bccfff5c90b.png">


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
