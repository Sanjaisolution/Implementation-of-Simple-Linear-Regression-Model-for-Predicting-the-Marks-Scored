# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Here's a more concise, two-line explanation for each step:

Step 1: Import Required Libraries

Step 2: Load the Data

Step 3: Define Feature Matrix and Target Variable

Step 4: Split the Data into Training and Test Sets

Step 5: Train the Linear Regression Model

Step 6: Make Predictions on Test Data

Step 7: Visualize Training Set Results

Step 8: Visualize Test Set Results

Step 9: Evaluate Model Performance
  
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
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![Screenshot 2024-11-25 193055](https://github.com/user-attachments/assets/b30a0c63-d01a-4e81-9532-01c9ce10bd6e)
![Screenshot 2024-11-25 193059](https://github.com/user-attachments/assets/547bc782-e779-43c5-861f-c6d02b2f0621)
![Screenshot 2024-11-25 193104](https://github.com/user-attachments/assets/839b7376-d235-41bd-973a-ec755df5536f)
![Screenshot 2024-11-25 193112](https://github.com/user-attachments/assets/24af7c4c-d85c-402c-9e19-3708ed56e02d)
![Screenshot 2024-11-25 193118](https://github.com/user-attachments/assets/69843917-7ce7-489d-95e4-8d58283b7ea2)
![Screenshot 2024-11-25 193124](https://github.com/user-attachments/assets/6652cca8-1df0-43b9-8726-fc3b96dcf598)
![Screenshot 2024-11-25 193129](https://github.com/user-attachments/assets/c56d0e99-ee60-4f68-9274-6702c8794a7b)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
