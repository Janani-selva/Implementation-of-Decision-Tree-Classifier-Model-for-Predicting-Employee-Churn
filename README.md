# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.
```
## Program:
```
Developed by: Janani.S
RegisterNumber: 212224230103
```
```
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/4d1d3731-734d-4965-833b-2e2992e98ec9)
![image](https://github.com/user-attachments/assets/d5ce1050-5032-412b-a28d-e79e89e9a440)
![image](https://github.com/user-attachments/assets/2fe367f8-fe00-4775-a70a-b41c94a43aed)


![image](https://github.com/user-attachments/assets/2c2fe39c-9f8f-4675-941d-3d933d86be38)
![image](https://github.com/user-attachments/assets/0fe33983-e220-44bc-9e3e-a4b6b066cf02)
![image](https://github.com/user-attachments/assets/bfca85b4-9b54-42b6-950e-fb47ae42f977)
![image](https://github.com/user-attachments/assets/63a9373c-521e-435f-ac1a-538cf2013c89)
![image](https://github.com/user-attachments/assets/2d08e460-0d86-42f9-aa85-1f7a7ace7ec3)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
