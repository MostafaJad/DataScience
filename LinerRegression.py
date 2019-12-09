#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:14:10 2019

@author: jado
"""
#Loading The DataSet
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

path = "Path/"
filename = 'Advertising.csv'
fullpath = os.path.join(path,filename)
d_mostafa = pd.read_csv(fullpath)
d_mostafa.columns.values
d_mostafa.shape
d_mostafa.describe()
d_mostafa.dtypes
d_mostafa.head(5)
print(d_mostafa)

#Finding the correlation 
def corrcoeff(df,var1,var2):
    df['corrn']=(df[var1]-np.mean(df[var1]))*(df[var2]-np.mean(df[var2]))
    df['corrd1']=(df[var1]-np.mean(df[var1]))**2
    df['corrd2']=(df[var2]-np.mean(df[var2]))**2
    corrcoeffn=df.sum()['corrn']
    corrcoeffd1=df.sum()['corrd1']
    corrcoeffd2=df.sum()['corrd2']
    corrcoeffd=np.sqrt(corrcoeffd1*corrcoeffd2)
    corrcoeff=corrcoeffn/corrcoeffd
    return corrcoeff
print(corrcoeff(d_mostafa,'TV','Sales'))
print(corrcoeff(d_mostafa,'Radio','Sales'))
print(corrcoeff(d_mostafa,'Newspaper','Sales'))

#Use  the matplotlib module to visualize the  relationships between each of the inputs and the output (sales), i.e. generate three scattered plots.
plt.plot(d_mostafa['TV'],d_mostafa['Sales'],'ro')
plt.title('TV vs Sales')
plt.plot(d_mostafa['Radio'],d_mostafa['Sales'],'ro')
plt.title('Radio vs Sales')
plt.plot(d_mostafa['Newspaper'],d_mostafa['Sales'],'ro')
plt.title('Newspaper vs Sales')

#Use the ols method and the statsmodel.formula.api library to build a linear regression model 
model1=smf.ols(formula='Sales~TV',data=d_mostafa).fit()
#print(model1.param)
print(model1.pvalues)
print(model1.rsquared)
print(model1.summary())
#Rebuild The Model ....
model3=smf.ols(formula='Sales~TV+Radio',data=d_mostafa).fit()
print(model3.params)
print(model3.rsquared)
print(model3.summary())
## Predicte a new value
X_new2 = pd.DataFrame({'TV': [50],'Radio' : [40]})
# predict for a new observation
sales_pred2=model3.predict(X_new2)
print(sales_pred2)

#In this step we will build the model using scikit-learn package

feature_cols = ['TV', 'Radio']
X = d_mostafa[feature_cols]
Y = d_mostafa['Sales']
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
lm = LinearRegression()
lm.fit(trainX, trainY)
print (lm.intercept_)
print (lm.coef_)
zip(feature_cols, lm.coef_)
[('TV', 0.045706061219705982), ('Radio', 0.18667738715568111)]
lm.score(trainX, trainY)
lm.predict(testX)
#Feature selection: using the scikit , in order to check which predictors are best as input variable 
feature_cols = ['TV', 'Radio','Newspaper']
X = d_mostafa[feature_cols]
Y = d_mostafa['Sales']
estimator = SVR(kernel="linear")
selector = RFE(estimator,2,step=1)
selector = selector.fit(X, Y)
print(selector.support_)
print(selector.ranking_)

