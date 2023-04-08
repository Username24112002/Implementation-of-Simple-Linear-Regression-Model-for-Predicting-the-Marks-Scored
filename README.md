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
![Screenshot (1)](https://user-images.githubusercontent.com/104640337/230721056-d704008e-d926-41e7-a5d4-28b4874c147c.png)<br>
2.df.tail():<br>
![Screenshot (2)](https://user-images.githubusercontent.com/104640337/230721075-db7e183d-c577-41d4-b641-a232d15f7518.png)<br>
3.Array value of X:<br>
![Screenshot (3)](https://user-images.githubusercontent.com/104640337/230721092-5c8f3ee1-806d-4fe4-8c74-d74b65fda87e.png)<br>
4.Array value of Y:<br>
![Screenshot (4)](https://user-images.githubusercontent.com/104640337/230721096-a92f9802-b313-4bf1-98a1-31ed5da81e70.png)<br>
5.Values of Y prediction:<br>
![Screenshot (5)](https://user-images.githubusercontent.com/104640337/230721098-1500c521-6cd4-4ba5-9dd3-8a58b04b8891.png)<br>
6.Array values of Y:<br>
![Screenshot (6)](https://user-images.githubusercontent.com/104640337/230721103-7eb45847-66e8-4b57-b3b5-55752b4b78ef.png)<br>
7.Training set Graph:<br>
![Screenshot (7)](https://user-images.githubusercontent.com/104640337/230721107-6a05f8da-11bb-4d29-91e1-32c76a814b97.png)<br>
8.Test set Graph:<br>
![Screenshot (8)](https://user-images.githubusercontent.com/104640337/230721112-f9b53dda-e874-41c9-9c47-9196aa64f5e0.png)<br>
9.Values of MSE, MAE and RMSE:<br>
![Screenshot (9)](https://user-images.githubusercontent.com/104640337/230721146-7478a6a2-5f51-4cb0-b29d-5c6a3ca4dae1.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
