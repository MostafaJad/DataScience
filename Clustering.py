#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:40:22 2019

@author: jado
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

path = "Path/"
filename = 'wine.csv'
fullpath = os.path.join(path,filename)
d_mostafa_wine = pd.read_csv(fullpath,sep=';')
# set the columns display and check the data
pd.set_option('display.max_columns',15)
print(d_mostafa_wine.head())
print(d_mostafa_wine.columns.values)
print(d_mostafa_wine.shape)
print(d_mostafa_wine.describe())
print(d_mostafa_wine.dtypes) 
print(d_mostafa_wine.head(5))
print(d_mostafa_wine['quality'].unique())
pd.set_option('display.max_columns',15)
print(d_mostafa_wine.groupby('quality').mean())
#Plot a histogram to see the number of wine samples in each quality type
plt.hist(d_mostafa_wine['quality'])

#Use seaborn library to generate different plots: histograms, pairplots, heatmaps…etc. and investigate the correlations.
import seaborn as sns
sns.distplot(d_mostafa_wine['quality'])
# plot only the density function
sns.distplot(d_mostafa_wine['quality'], rug=True, hist=False, color = 'r')
# Change the direction of the plot
sns.distplot(d_mostafa_wine['quality'], rug=True, hist=False, vertical = True)
# Check all correlations
sns.pairplot(d_mostafa_wine)
# Subset three column
x=d_mostafa_wine[['fixed acidity','chlorides','pH']]
y=d_mostafa_wine[['chlorides','pH']]
# check the correlations 
sns.pairplot(x)
# Generate heatmaps
sns.heatmap(d_mostafa_wine[['fixed acidity']])
sns.heatmap(x)
sns.heatmap(x.corr())
sns.heatmap(x.corr(),annot=True)
##
import matplotlib.pyplot as plt
plt.figure(figsize=(10,9))
sns.heatmap(x.corr(),annot=True, cmap='coolwarm',linewidth=0.5)
##line two variables
plt.figure(figsize=(20,9))
sns.lineplot(data=y) 
sns.lineplot(data=y,x='chlorides',y='pH')
## line three variables
sns.lineplot(data=x)

#Normalize the data in order to apply clustering, the formula is as follows:
d_mostafa_wine_norm = (d_mostafa_wine - d_mostafa_wine.min()) / (d_mostafa_wine.max() - d_mostafa_wine.min())
d_mostafa_wine_norm.head()
# check some plots after normalizing the data
x1=d_mostafa_wine_norm[['fixed acidity','chlorides','pH']]
y1=d_mostafa_wine_norm[['chlorides','pH']]
sns.lineplot(data=y1) 
sns.lineplot(data=x1)
sns.lineplot(data=y,x='chlorides',y='pH')

from sklearn.cluster import KMeans
from sklearn import datasets
model=KMeans(n_clusters=6)
model.fit(d_mostafa_wine_norm)
model.labels_
# Append the clusters to each record on the dataframe, i.e. add a new column for clusters
md=pd.Series(model.labels_)
d_mostafa_wine_norm['clust']=md
d_mostafa_wine_norm.head(10)
#find the final cluster's centroids for each cluster
model.cluster_centers_
#Calculate the J-scores The J-score can be thought of as the sum of the squared distance between points and cluster centroid for each point and cluster.
#For an efficient cluster, the J-score should be as low as possible.
model.inertia_
#let us plot a histogram for the clusters
import matplotlib.pyplot as plt
plt.hist(d_mostafa_wine_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
# plot a scatter 
plt.scatter(d_mostafa_wine_norm['clust'],d_mostafa_wine_norm['pH'])
plt.scatter(d_mostafa_wine_norm['clust'],d_mostafa_wine_norm['chlorides'])

