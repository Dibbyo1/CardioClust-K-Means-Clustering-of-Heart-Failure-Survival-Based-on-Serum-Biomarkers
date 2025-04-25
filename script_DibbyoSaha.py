#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

for col in df.columns:
    df[col].fillna(df[col].median(), inplace=True)
    
df2 = df[['serum_creatinine', 'ejection_fraction']].values

scaler = StandardScaler()
scaledDF = scaler.fit_transform(df2)

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    preds = kmeans.fit_predict(scaledDF)
    score = silhouette_score(scaledDF, preds)
    print(f'k={k} silhouette score={score}')
    
optimalK = 3 # Found from print statement from line 27 in this file

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaledDF)
centroids = kmeans.cluster_centers_

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(scaledDF)
transformedCentroid = pca.transform(centroids)

plt.figure(figsize=(10,6))
plt.scatter(principalComponents[:,0], principalComponents[:,1], c=clusters, label='Clustered Data')
plt.scatter(transformedCentroid[:,0], transformedCentroid[:,1], c='red', label='Centroids')
plt.show()

"""
References: 
- W3Schools, "Python Machine Learning - K-means," [Online]. Available: https://www.w3schools.com/python/python_ml_k-means.asp. [Accessed: 6th December, 2023].
- GeeksforGeeks, "Determining the Number of Clusters in Data Mining," [Online]. Available: https://www.geeksforgeeks.org/determining-the-number-of-clusters-in-data-mining/. [Accessed: 6th December, 2023].
- A. Geron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd ed. O'Reilly Media, pp. 237-255.
- A. C. Muller and S. Guido, Introduction to Machine Learning with Python. O'Reilly Media.
"""