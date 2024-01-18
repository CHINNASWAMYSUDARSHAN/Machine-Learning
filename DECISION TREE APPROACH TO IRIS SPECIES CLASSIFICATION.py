import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/kaggle/input/iris-dataset/iris.csv') 
df.head()
df.tail()
df.info()
df.describe()
df.isna().sum()
df['species'].value_counts()
sns.countplot(x='species',data=df)
plt.show()
sns.boxplot(x='sepal_length',data=df)
plt.show()
sns.boxplot(y='sepal_width',data=df)
plt.show()
sns.boxplot(y='petal_length',data=df)
plt.show()
sns.boxplot(y='petal_width',data=df)
plt.show()
sns.scatterplot(x='sepal_length',y='petal_length',hue='species',data=df)
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.show()
sns.scatterplot(x='sepal_width',y='petal_width',hue='species',data=df)
plt.xlabel('sepal width')
plt.ylabel("petal width")
plt.show()
sns.kdeplot(x='sepal_length',data=df)
sns.kdeplot(x='sepal_width',data=df)
sns.kdeplot(x='petal_length',data=df)
sns.kdeplot(x='petal_width',data=df)
plt.xlabel('iris length and width')
plt.legend(['sepal_length','sepal_width','petal_length','petal_width'])
plt.show()
sns.pairplot(df,hue='species')
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit_transform(df['species'])
df['species']=le.transform(df['species'])
df
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(x.shape,y.shape)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)
from sklearn.tree import DecisionTreeClassifier
DTmodel=DecisionTreeClassifier(criterion='entropy',random_state=1)
#training the data
DTmodel.fit(xtrain,ytrain)
#testing the data
accuracy=DTmodel.score(xtest,ytest)
print("Accuracy",accuracy*100)
ypred=DTmodel.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ypred,ytest)
cm
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()
