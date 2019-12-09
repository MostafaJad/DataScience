#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:51:37 2019

@author: jado
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


path = "Path/"
filename = 'Bank.csv'
fullpath = os.path.join(path,filename)
d_mostafa = pd.read_csv(fullpath,sep=';')
print(d_mostafa.columns.values)
print(d_mostafa.shape)
print(d_mostafa.describe())
print(d_mostafa.dtypes) 
print(d_mostafa.head(5))

print(d_mostafa['education'].unique())
d_mostafa['education']=np.where(d_mostafa['education'] =='basic.9y', 'Basic', d_mostafa['education'])
d_mostafa['education']=np.where(d_mostafa['education'] =='basic.6y', 'Basic', d_mostafa['education'])
d_mostafa['education']=np.where(d_mostafa['education'] =='basic.4y', 'Basic', d_mostafa['education'])
d_mostafa['education']=np.where(d_mostafa['education'] =='university.degree', 'University Degree',d_mostafa['education'])
d_mostafa['education']=np.where(d_mostafa['education'] =='professional.course', 'Professional Course', d_mostafa['education'])
d_mostafa['education']=np.where(d_mostafa['education'] =='high.school', 'High School', d_mostafa['education'])
d_mostafa['education']=np.where(d_mostafa['education'] =='illiterate', 'Illiterate', d_mostafa['education'])
d_mostafa['education']=np.where(d_mostafa['education'] =='unknown', 'Unknown', d_mostafa['education'])
#Check the values of who  purchased the deposit account
print(d_mostafa['y'].value_counts())
#Check the average of all the numeric columns
pd.set_option('display.max_columns',100)
print(d_mostafa.groupby('y').mean())
#Check the mean of all numeric columns grouped by education
print(d_mostafa.groupby('education').mean())
#Plot a histogram showing purchase by education category
pd.crosstab(d_mostafa.education,d_mostafa.y)
pd.crosstab(d_mostafa.education,d_mostafa.y).plot(kind='bar')
plt.title('Purchase Frequency for Education Level')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
#draw a stacked bar chart of the marital status and the purchase of term deposit to see whether this can be a good predictor of the outcome
table=pd.crosstab(d_mostafa.marital,d_mostafa.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
#plot the bar chart for the Frequency of Purchase against each day of the week to see whether this can be a good predictor of the outcome
pd.crosstab(d_mostafa.day_of_week,d_mostafa.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
#Repeat for the month
pd.crosstab(d_mostafa.month,d_mostafa.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
#Plot a histogram of the age distribution
d_mostafa.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

#Deal with the categorical variables,
#Deal with the categorical variables, use a for loop
#1- Create the dummy variables 
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(d_mostafa[var], prefix=var)
    d_mostafa1=d_mostafa.join(cat_list)
    d_mostafa=d_mostafa1
d_mostafa.head(5)    
#  2- Removee the original columns
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
d_mostafa_vars=d_mostafa.columns.values.tolist()
to_keep=[i for i in d_mostafa_vars if i not in cat_vars]
d_mostafa_final=d_mostafa[to_keep]
d_mostafa_final.columns.values
# 3- Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
d_mostafa_final_vars=d_mostafa_final.columns.values.tolist()
Y=['y']
X=[i for i in d_mostafa_final_vars if i not in Y ]
type(Y)
type(X)
#Carryout feature selection and update the data
#1- We have many features so let us carryout feature selection

model = LogisticRegression()
rfe = RFE(model, 12)
rfe = rfe.fit(d_mostafa_final[X],d_mostafa_final[Y] )
print(rfe.support_)
print(rfe.ranking_)
#2- Update X and Y with selected features
cols=['previous', 'euribor3m', 'job_entrepreneur', 'job_self-employed', 'poutcome_success', 'poutcome_failure', 'month_oct', 'month_may',
    'month_mar', 'month_jun', 'month_jul', 'month_dec'] 
X=d_mostafa_final[cols]
Y=d_mostafa_final['y']
type(Y)
type(X)
#1- split the data into 70%training and 30% for testing, note  added the solver to avoid warnings
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 2-Let us build the model and validate the parameters

clf1 = linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(X_train, Y_train)
#3- Run the test data against the new model
probs = clf1.predict_proba(X_test)
print(probs)
predicted = clf1.predict(X_test)
print (predicted)
#4-Check model accuracy
print (metrics.accuracy_score(Y_test, predicted))	
#To avoid sampling bias run cross validation for 10 times, as follows
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X, Y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())
#Generate the confusion matrix as follows:
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.05,1,0)
Y_A =Y_test.values
Y_P = np.array(prob_df['predict'])
confusion_matrix = confusion_matrix(Y_A, Y_P)
print (confusion_matrix)

