
"Import Libaries "
import tensorflow as tf
import neural_structured_learning as nsl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics




print("==================================================")
print(" Spark Based Big Data Analytics ")
print(" Framework for Competitive Intelligence")
print("==================================================")


##1.data slection---------------------------------------------------
#def main():
dataframe=pd.read_csv("data.csv")
print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()


 #2.pre processing--------------------------------------------------
#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")
 

#label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
print("--------------------------------------------------")
print("Before Label Handling ")
print()
print(dataframe_2.head(10))
print("--------------------------------------------------")
print()

#3.Data splitting--------------------------------------------------- 
x=dataframe.drop('VersionInformationSize',axis=1)
y=dataframe.VersionInformationSize
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

##4.feature selection------------------------------------------------
##kmeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)
plt.scatter(x[:, 0], x[:, 1], s=50);

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.title("k-means")
plt.show()


from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
dt.fit(x_train, y_train)
dt_prediction=dt.predict(x_test)
print()
print("---------------------------------------------------------------------")
print("Decision Tree")
print()
Result_2=accuracy_score(y_test, dt_prediction)*100
print(metrics.classification_report(y_test,dt_prediction))
print()
print("DT Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, dt_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, dt_prediction)
plt.plot(fpr, tpr, marker='.', label='DT')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


#---------------------------------------------------------------------------------------------
"SVM Algorithm "
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.svm import SVC
svclassifier = SVC()
svclassifier.fit(x_train,y_train)
y_pred11 = svclassifier.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred11)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred11)
print("Classification Report:",)
print (result1)
print("Accuracy:",accuracy_score(y_test, y_pred11))


import seaborn as sns
fig, ax = plt.subplots(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(result, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Spark', 'Benign']); ax.yaxis.set_ticklabels(['Spark', 'Benign']);


#---------------------------------------------------------------------------------
"Navie Bayies "
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#---------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators = 100)  
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
Result_3=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, rf_prediction)
plt.plot(fpr, tpr, marker='.', label='RF')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()








