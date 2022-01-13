import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


train_data = pd.read_csv("E:\Customer Segmentation Project\dataset\Mall_Customers.csv")
customerId = train_data.pop('CustomerID')
print(train_data.head())

plt.scatter(train_data[:,0], train_data[:,1], s=50, c='b')

