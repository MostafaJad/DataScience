#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:20:08 2019

@author: jado
"""

import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics 
import seaborn as sns
import matplotlib.pyplot as plt 


path = "Path/"
filename = 'iris.csv'
fullpath = os.path.join(path,filename)
d_mostafa = pd.read_csv(fullpath,sep=',')
print(d_mostafa.columns.values)
print(d_mostafa.shape)
print(d_mostafa.describe())
print(d_mostafa.dtypes) 
print(d_mostafa.head(5))
print(d_mostafa['Species'].unique())
#Separate the predictors from the target then split the dataset using numpy random function.
colnames=d_mostafa.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]

d_mostafa['is_train'] = np.random.uniform(0, 1, len(d_mostafa)) <= .75
print(d_mostafa.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = d_mostafa[d_mostafa['is_train']==True], d_mostafa[d_mostafa['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

#Build the decision tree using the training dataset
dt_mostafa = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_mostafa.fit(train[predictors], train[target])
preds=dt_mostafa.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])
#Generate a dot file and visualize the tree using the online vizgraph editor and share (download) as picture
with open('/Users/jado/Desktop/DataWarehousingTest2/dtree3.dot', 'w') as dotfile:
    export_graphviz(dt_mostafa, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=d_mostafa[predictors]
Y=d_mostafa[target]
#split the data sklearn module
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)

dt1_mostafa = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20, random_state=99)
dt1_mostafa.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data 
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_mostafa, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_mostafa.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

    
cm = confusion_matrix(testY, testY_predict, labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']);
plt.show()
