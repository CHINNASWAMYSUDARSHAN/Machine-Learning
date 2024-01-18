import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
filepath='/kaggle/input/heart-disease/heart.csv'
df=pd.read_csv(filepath)
df.head()
df.tail()
df.info()
df.describe()
df.groupby('target').mean()
df.groupby('gender').mean()
df['target'].value_counts()
df['gender'].value_counts()
sns.countplot(x='target',data=df)
plt.show()
sns.countplot(x='gender',data=df)
plt.show()
plt.scatter(df['bp'],df['heartrate'])
plt.xlabel("Blood Pressure")
plt.ylabel("Heart rate")
plt.title("BP vs Heart Rate")
plt.show()
#Regression plot
sns.regplot(x='bp',y='heartrate',data=df)
plt.title("Regresssion plot for BP vs Heartrate")
plt.show()
pd.crosstab(df.gender,df.target).plot(kind="bar")
plt.legend(['Healthy','Disease'])
plt.xticks(rotation=1)
plt.show()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(15,8))
plt.legend(['Healthy','Disease'])
plt.xticks(rotation=1)
plt.show()
plt.scatter(df.bp[df['target']==0],df.heartrate[df['target']==0],c='g')
plt.scatter(df.bp[df['target']==1],df.heartrate[df['target']==1],c='r')
plt.legend(['healthy','Disease'])
plt.show()
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)
plt.show()
#data cleaning
df.isna().sum()
df['oldpeak']=df['oldpeak'].astype('int64')
df.info()
#find x and y

y=df.iloc[:,-1]
print(x.shape)
print(y.shape)
#splitting data into training data and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=3)
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
#model selection
from sklearn.tree import DecisionTreeClassifier
DTmodel=DecisionTreeClassifier(criterion='gini',random_state=34)
#model training
DTmodel.fit(xtrain,ytrain)
#model testing
accuracy=DTmodel.score(xtest,ytest)
print("accuracy",accuracy*100)
ypred=DTmodel.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ypred,ytest)
print(cm)
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()

