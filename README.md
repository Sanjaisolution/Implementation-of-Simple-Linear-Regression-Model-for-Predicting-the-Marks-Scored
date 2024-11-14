# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Here's a more concise, two-line explanation for each step:

### Step 1: Import Required Libraries
### Step 2: Load the Data
### Step 3: Define Feature Matrix and Target Variable
### Step 4: Split the Data into Training and Test Sets
### Step 5: Train the Linear Regression Model
### Step 6: Make Predictions on Test Data
### Step 7: Visualize Training Set Results
### Step 8: Visualize Test Set Results
### Step 9: Evaluate Model Performance
  
## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANJAI.R
RegisterNumber: 212223040180
*/
```PY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("Values of Y prediction")
Y_pred
print("Values of Y test")
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()
print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![Screenshot 2024-11-14 135353](https://github.com/user-attachments/assets/969e7316-6a2f-42b9-888e-396198632af4)
![Screenshot 2024-11-14 135416](https://github.com/user-attachments/assets/e8160977-7323-4e8e-a728-1c974002f374)
![Screenshot 2024-11-14 135428](https://github.com/user-attachments/assets/17f6eb0c-f969-4080-aaa0-91f19c3471bc)
![Screenshot 2024-11-14 135442](https://github.com/user-attachments/assets/b010c71e-ca8a-4bfd-97cd-d30bc4a805c9)
![Screenshot 2024-11-14 135455](https://github.com/user-attachments/assets/bae682d1-b896-4b64-b8c7-ee0ee81dd0a1)
![Screenshot 2024-11-14 135502](https://github.com/user-attachments/assets/02920dab-39cf-41e6-9a9f-f51a527cb789)
![Screenshot 2024-11-14 135511](https://github.com/user-attachments/assets/4deccd5d-978f-4e9d-9fa5-78017b6947bc)
![Screenshot 2024-11-14 135524](https://github.com/user-attachments/assets/2d9bc44a-5001-45b5-b9d1-109dd73aba77)
![Screenshot 2024-11-14 135530](https://github.com/user-attachments/assets/ae855b9b-c222-4919-ab99-b8037b9fce57)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
