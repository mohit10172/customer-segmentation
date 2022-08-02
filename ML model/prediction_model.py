import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import pickle

model = pickle.load(open('E:\Customer Segmentation Project\ML model\clustermodel.pkl','rb'))

print("\n***Enter Customer Details***")
x = input("\nAnnual Income: ")
y = input("\nSpending Score: ")

clusterno = model.predict([[x,y]])
print("\nThis customer belongs to Cluster ",clusterno[0])