<H3>ENTER YOUR NAME</H3> EZHIL NEVEDHA K
<H3>ENTER YOUR REGISTER NO.</H3> 212223230055
<H3>EX. NO.1</H3>
<H3>DATE</H3> 28.04.2025
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv('churn_modelling.csv')
print(df)
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,:-1].values
print(y)
print(df.isnull().sum())
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)
df.duplicated()
print(df.describe())
df=df.drop(['Surname','Geography','Gender'],axis=1)
df.head()
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(len(X_train))
print(X_test)
print(len(X_test))
```


## OUTPUT:
### Dataset:
<img width="1015" height="351" alt="image" src="https://github.com/user-attachments/assets/d6999571-2ad7-4ea7-912f-c6c4be4bd0a9" />

### X values:
<img width="384" height="141" alt="image" src="https://github.com/user-attachments/assets/76113420-e76f-4215-aafc-9248855f5725" />

### Y values:
<img width="420" height="133" alt="image" src="https://github.com/user-attachments/assets/f7259776-67f6-4b12-be4d-5502d8833d0d" />

### Null values:
<img width="358" height="304" alt="image" src="https://github.com/user-attachments/assets/de38ffd2-1cc8-424d-bf37-f19b5a91b6db" />

### Duplicate values:
<img width="260" height="257" alt="image" src="https://github.com/user-attachments/assets/30ad1b3f-e776-4954-abce-6d354a9ad4b2" />

### Description:
<img width="1005" height="287" alt="image" src="https://github.com/user-attachments/assets/37d1db8e-2b06-4313-a0e8-a549b104d10d" />

### Normalized Data:
<img width="617" height="479" alt="image" src="https://github.com/user-attachments/assets/39584745-ea36-42bb-aab5-1a3efdf99d21" />

### X-train data:
<img width="435" height="142" alt="image" src="https://github.com/user-attachments/assets/48e18107-8015-4b30-baab-15a8d37503cf" />

### X-test:
<img width="413" height="130" alt="image" src="https://github.com/user-attachments/assets/5a754a54-5614-4dea-86dc-2364d34d1fbe" />

### Length of training and testing data:
<img width="225" height="68" alt="image" src="https://github.com/user-attachments/assets/ec36c69f-fcb8-4de8-9865-83b7bf80984f" />
<img width="183" height="82" alt="image" src="https://github.com/user-attachments/assets/78a17399-37e7-455f-b1ed-551cb765cc1b" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


