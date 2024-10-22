# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VARSHINI S 
RegisterNumber: 212222220056
*/

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

dataset.info()

X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)
mse=mean_squared_error(Y_test, Y_pred)
print("MSE = ",mse)

mae=mean_absolute_error(Y_test, Y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)

plt.scatter(X_train, Y_train, color="green")
plt.plot(X_train, reg.predict(X_train), color="red")
plt.title("Training set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title("Test set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:




### Head:
![Screenshot 2024-08-28 103458](https://github.com/user-attachments/assets/24dfdd22-fcad-43a7-a492-1dd6871c2229)


### Info:

![Screenshot 2024-08-30 174649](https://github.com/user-attachments/assets/9183d136-b41e-426d-af96-7b059355c5e3)

### Value of X:

![Screenshot 2024-08-28 103509](https://github.com/user-attachments/assets/0c5e3fe5-0b77-48fc-ba73-0aa663a563d2)



### Value of Y:

![Screenshot 2024-08-28 103520](https://github.com/user-attachments/assets/bb03584d-408d-4cc9-821a-e1a7db91fdd7)


### Values of MSE, MAE and RMSE :

![Screenshot 2024-08-28 103530](https://github.com/user-attachments/assets/934c291f-a5d1-491b-b794-2c44fe18e93f)




### Training set graph:




![Screenshot 2024-08-30 175001](https://github.com/user-attachments/assets/7a9ccc5b-15c6-4d92-9d16-8e2532fc89f1)

### Testing set graph:

![Screenshot 2024-08-30 175009](https://github.com/user-attachments/assets/b00f35af-d111-45bd-8bc1-5d86846916ae)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
