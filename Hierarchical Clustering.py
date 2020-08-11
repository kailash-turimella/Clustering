"""
Heirarchial Clustering

There are two types of heirarchial clustering: Agglomerative and Divisive

Distance Between Clusters:
    1) The distance between the closest points of each cluster
    2) The distance between the furthest points of each cluster
    3) The distance between average of all points of first cluster and the average of all points of second cluster.
    4) The distance between the centroids

Agglomerative Clustering:
    Make each data point a single-point cluster
    Take the two closest data points and make them one cluster
    Keep joining the two closest clusters until you have one big cluster
"""
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

# Importing datast
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using dendrogram to find the optimal number of clusters
"""
The optimal number of clusters is the number of vertical lines that cross the horizontal threshold.
Threshold is a horizontal line which passes through the longest continuous vertical line
 all horizontal lines are extended end-to-end
 it can go through the line at any point on the line
"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))#Which distance to use between clusters   #Minimum varience 
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# Fitting hierachical clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,          # Number of clusters(according to dendogram)
                             affinity = 'euclidean',  # How the distance between points is measured
                             linkage = 'ward')        # Which distance to use between clusters   # Minimum varience 
y_hc = hc.fit_predict(X)

# Visualising Clusters
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
