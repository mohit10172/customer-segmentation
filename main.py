import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("E:\Customer Segmentation Project\dataset\Mall_Customers.csv")
customerId = train_data.pop('CustomerID')
#print(train_data.head(),"\n",train_data.describe())

sns.countplot(x='Gender', data=train_data)
plt.title('Distribution Of Gender')
plt.show()

plt.scatter(train_data['Annual Income (k$)'], train_data['Spending Score (1-100)'], s=50, c='blue')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title("Annual Income (k$) vs Spending Score (1-100)")
plt.show()


X = train_data[['Annual Income (k$)', 'Spending Score (1-100)']]
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()