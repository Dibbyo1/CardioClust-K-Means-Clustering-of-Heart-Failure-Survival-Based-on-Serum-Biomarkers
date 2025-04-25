# CardioClust-K-Means-Clustering-of-Heart-Failure-Survival-Based-on-Serum-Biomarkers
CardioClust is a machine learning project that uses K-Means clustering to analyze the impact of serum creatinine and ejection fraction on heart failure patient survival. This bioinformatics analysis leverages clinical data from the UCI Heart Failure dataset to reveal survival patterns using unsupervised learning techniques. This project focuses on clustering patients with heart failure based on two key features—serum_creatinine and ejection_fraction—to identify survival-impacting biomarker trends. The optimal number of clusters is selected using the Silhouette Score, and the results are visualized using PCA-reduced 2D scatter plots.

# Key Features
K-Means Clustering: Uses sklearn's KMeans to cluster patients into risk categories.</br>
Silhouette Scoring: Determines the best number of clusters (k) based on cohesion/separation.</br>
PCA Visualization: Projects high-dimensional data into 2D space for interpretable plotting.</br>
Data Cleaning: Handles missing values using the median to avoid bias from non-normal distributions.</br>
Outlier Analysis: Visually inspects cluster spread to identify outliers or anomalies in patient data.
