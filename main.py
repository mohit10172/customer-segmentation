import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


train_data = pd.read_csv("E:\Customer Segmentation Project\dataset\Mall_Customers.csv")
customerId = train_data.pop('CustomerID')
#print(train_data.head(),"\n",train_data.describe())

train_data.Gender.value_counts().plot(kind='bar')
plt.show()

plt.scatter(train_data['Age'], train_data['Annual Income (k$)'], s=50, c='blue')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title("Age vs Annual Income (k$)")
plt.show()