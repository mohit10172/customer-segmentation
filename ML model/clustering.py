import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys


#READ DATA
input = pd.read_csv('dataset\Mall_Customers.csv')
X = input.iloc[:, [3,4]].values


#OPTIMIZING MODEL FOR NO. OF CLUSTERS:
def optimizeModel(t_data, wcss_plot=False):
    wcss=[]
    for i in range(1,11):
        cluster = KMeans(n_clusters=i, init='k-means++', random_state=0, verbose=0)
        cluster.fit(t_data)
        wcss.append(cluster.inertia_)   
    
    if wcss_plot==True:
        plt.plot(range(1, 11),wcss)   
        plt.title('Elbow technique ' )
        plt.xlabel('k ')
        plt.ylabel('WCSS')
        plt.savefig('./plots/elbow_technique.png')
        plt.clf()
    
    return wcss

#ELBOW METHOD FUNCTION:

wcss = optimizeModel(X)
for i in range(len(wcss)):
    sum = sum + wcss[i]
avg = sum / len(wcss)

print(avg)
