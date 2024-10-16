# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
 ```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```
![output 1](https://github.com/user-attachments/assets/56172747-f1e9-4e9a-8dbe-0d3b492e427f)
```
data.isnull().sum()
```
![output 2](https://github.com/user-attachments/assets/32ea947f-77e8-458e-a80d-6814b10adff3)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![output 3](https://github.com/user-attachments/assets/30cd6709-7267-4f87-bac9-6fb4d593ad9f)
```
data2=data.dropna(axis=0)
data2
```
![output 4](https://github.com/user-attachments/assets/4a94f13a-ba95-460a-bbd6-36e32fc136a5)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![output 5](https://github.com/user-attachments/assets/801c8704-443c-492b-b672-7eeabb7965ab)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![output 6](https://github.com/user-attachments/assets/e9b460dc-c000-43a1-84bb-48d92300305d)
```
data2

```
![output 7](https://github.com/user-attachments/assets/2ab7550c-4d45-4b31-a45c-cda4430b9b58)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![output 8](https://github.com/user-attachments/assets/d75ac4ec-2d43-4e0d-988b-f04eb99d7503)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![output 9](https://github.com/user-attachments/assets/66f6a7a1-4367-4f90-b6a8-9d6128d86e3d)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![output 10](https://github.com/user-attachments/assets/4ac7bb0e-bd4e-4ee5-a1dd-b0b2b22d8d41)
```
y=new_data['SalStat'].values
print(y)
```
![output 11](https://github.com/user-attachments/assets/48aee418-c770-49f7-b72e-a56706e69c81)
```
x=new_data[features].values
print(x)
```
![output 12](https://github.com/user-attachments/assets/a3a68128-6d3f-471e-b0a6-9a6b421603f1)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![output 13](https://github.com/user-attachments/assets/41f4dda7-9212-4f52-9db3-43b502fc524f)
```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![output 14](https://github.com/user-attachments/assets/7704e67c-7664-4cc0-898d-5c6f90349b52)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![output 15](https://github.com/user-attachments/assets/276599fd-fe80-411f-ac1f-8155984cabab)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![output 16](https://github.com/user-attachments/assets/e194e731-8b9a-48d5-ac4e-521192458724)
```
data.shape
```
![output 17](https://github.com/user-attachments/assets/18069e67-f65d-4c30-9e71-32179ac1e7c7)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![output 18](https://github.com/user-attachments/assets/4f062f9a-7db5-486f-a69e-a7bda525dce4)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![output 19](https://github.com/user-attachments/assets/35236133-e2a2-4550-ab79-7c4aa8bc756b)
```
tips.time.unique()
```
![output 20](https://github.com/user-attachments/assets/be48dd7b-0de9-4fe7-ba3c-830f844c76e1)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![output 21](https://github.com/user-attachments/assets/a2213b39-01b0-43f1-a5af-7517b4afc2c8)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![output 22](https://github.com/user-attachments/assets/ae682343-40d7-4c5a-b298-75bb86be7c90)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
