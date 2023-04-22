# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries. 
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 
3. Import LabelEncoder and encode the dataset. 
4. Import LogisticRegression from sklearn and apply the model on the dataset. 
5. Predict the values of array. 
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 
7. Apply new unknown values.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Y SHAVEDHA
RegisterNumber:  212221230095

```
```
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0 )

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
### Placement Data
<img width="777" alt="image" src="https://user-images.githubusercontent.com/93427376/233775777-e85a3092-33d7-47a5-950d-d27581828eef.png">

### Salary Data
<img width="700" alt="image" src="https://user-images.githubusercontent.com/93427376/233775803-da532fc5-4c11-40b5-b8f3-10769af603df.png">

### Checking the null() Function
<img width="214" alt="image" src="https://user-images.githubusercontent.com/93427376/233775831-988a8360-e9db-435a-ba96-edb9f4986edb.png">

### Data Duplicate
<img width="152" alt="image" src="https://user-images.githubusercontent.com/93427376/233775867-c4d6f271-7d38-4de1-ad87-76f19508b43f.png">

### Print Data
<img width="671" alt="image" src="https://user-images.githubusercontent.com/93427376/233775903-465c1e16-884a-4249-b887-c6a251998f5f.png">

### Data Status
<img width="628" alt="image" src="https://user-images.githubusercontent.com/93427376/233775909-e41282b4-73fa-4de1-826d-fa96b4d42697.png">

### y_prediction array
<img width="412" alt="image" src="https://user-images.githubusercontent.com/93427376/233775923-ef56b6c5-d2fa-4e08-b286-270cda2cc508.png">

### Accuracy Value
<img width="512" alt="image" src="https://user-images.githubusercontent.com/93427376/233775977-828bac61-bb0c-46e4-8ba8-ce17ecfd62fd.png">

### Confusion Array
<img width="308" alt="image" src="https://user-images.githubusercontent.com/93427376/233775996-96c887fd-ae9a-4b2b-bc44-bd6f67a64920.png">

### Classification Report
<img width="420" alt="image" src="https://user-images.githubusercontent.com/93427376/233776009-05eee8ea-e7dd-46a1-8e9d-9522f1d072c5.png">

### Prediction of LR
<img width="192" alt="image" src="https://user-images.githubusercontent.com/93427376/233776027-186ebc8b-4ed9-492d-a13d-9d20c84c7cbb.png">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
