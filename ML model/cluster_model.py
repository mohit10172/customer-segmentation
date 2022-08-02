import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import pickle

#READ DATA
input = pd.read_csv('dataset\Mall_Customers.csv')
X = input.iloc[:, [3,4]].values

#FITTING THE DATA
cluster = KMeans(n_clusters=5)
cluster.fit(X)

pickle.dump(cluster, open('ML model\clustermodel.pkl', 'wb'))
