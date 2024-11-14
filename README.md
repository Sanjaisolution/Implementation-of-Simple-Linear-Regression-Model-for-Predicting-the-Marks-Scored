# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Here's a more concise, two-line explanation for each step:

### Step 1: Import Required Libraries
- Import libraries: `pandas`, `numpy`, `matplotlib.pyplot`, and metrics from `sklearn`.
- These libraries are essential for data handling, visualization, and model evaluation.

### Step 2: Load the Data
- Load the dataset from a CSV file into a DataFrame `df`.
- Display the first and last few rows of the dataset to inspect data structure.

### Step 3: Define Feature Matrix and Target Variable
- Define `X` as the feature matrix containing all columns except the last.
- Define `Y` as the target variable, containing only the last column.

### Step 4: Split the Data into Training and Test Sets
- Use `train_test_split` to split data into training and testing sets with a 2:1 ratio.
- Set a random seed to ensure reproducibility.

### Step 5: Train the Linear Regression Model
- Initialize a `LinearRegression` model and fit it to the training data (`X_train`, `Y_train`).
- The model learns the relationship between input features and target.

### Step 6: Make Predictions on Test Data
- Use the trained model to predict outputs (`Y_pred`) for `X_test`.
- This will give predicted scores based on the test input features.

### Step 7: Visualize Training Set Results
- Plot the actual training data points and the model's fitted line.
- Add appropriate labels for the title, x-axis, and y-axis.

### Step 8: Visualize Test Set Results
- Plot the actual test data points and the model's fitted line for comparison.
- Again, label the plot for clarity.

### Step 9: Evaluate Model Performance
- Calculate `MSE`, `MAE`, and `RMSE` between `Y_test` and `Y_pred`.
- Print each metric to assess model accuracy.
  
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
print('df.head')

df.head()

print("df.tail")
df.tail()

X=df.iloc[:,:-1].values
print("Array of X")
X

Y=df.iloc[:,1].values
print("Array of Y")
Y

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
