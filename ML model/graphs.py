import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import pickle

input = pd.read_csv('dataset\Mall_Customers.csv')


model = pickle.load(open('ML model\clustermodel.pkl','rb'))


#GENDER DSITRIBUTION
genders = input.Gender.value_counts()
plt.figure(figsize=(6,3))
sns.barplot(x=genders.index, y=genders.values)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('./plots/gender_dist.png')
plt.clf()

#SCATTER PLOT OF THE DATA
plt.scatter(x=input['Annual Income (k$)'], y=input['Spending Score (1-100)'], s=50, c='blue')
plt.title('Scatter plot of the DATA')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.savefig('./plots/scatter_plot.png')
plt.clf()

#PREDICTING THE DATA
y = model.predict(input.iloc[:, [3,4]].values)
input['Cluster'] = y
input.to_csv('dataset\Clustered_data.csv', index=False)

#VISUALIZING THE CLUSTERS:
plt.figure(figsize=(15,7))
plt.scatter(input["Annual Income (k$)"][input.Cluster == 0], input["Spending Score (1-100)"][input.Cluster == 0], c='blue', s=60,label='Cluster 0')
plt.scatter(input["Annual Income (k$)"][input.Cluster == 1], input["Spending Score (1-100)"][input.Cluster == 1], c='red', s=60,label="Cluster 1")
plt.scatter(input["Annual Income (k$)"][input.Cluster == 2], input["Spending Score (1-100)"][input.Cluster == 2], c='green', s=60,label='Cluster 2')
plt.scatter(input["Annual Income (k$)"][input.Cluster == 3], input["Spending Score (1-100)"][input.Cluster == 3], c='yellow', s=60,label='Cluster 3')
plt.scatter(input["Annual Income (k$)"][input.Cluster == 4], input["Spending Score (1-100)"][input.Cluster == 4], c='black', s=60,label='Cluster 4')
plt.title('Clusters of Customers')
plt.legend()
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig('./plots/cluster_visualization.png')
plt.clf()