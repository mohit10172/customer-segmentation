import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import pickle

input = pd.read_csv('dataset\Mall_Customers.csv')
X = input.iloc[:, [3,4]].values

#OPTIMIZING MODEL FOR NO. OF CLUSTERS:
wcss=[]
for i in range(1,11):
  cluster = KMeans(n_clusters=i, init='k-means++', random_state=0, verbose=0)
  cluster.fit(X)
  wcss.append(cluster.inertia_)   
    

plt.plot(range(1, 11),wcss)   
plt.title('Elbow technique ' )
plt.xlabel('k ')
plt.ylabel('WCSS')
plt.savefig('./plots/elbow_technique.png')
plt.clf()