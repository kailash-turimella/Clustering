"""
Kmeans

Choose the number of clusters, k(n_clusters)
Select the location of the centroids(not necessarily one of the points).
  You will have k centroids
Assign each point to the closest centroid
Move the centroid to the center of all the points assigned to it
Reassign each data point to the new closest centroid. Closest according to euclidean distance(but we can change it)
If any reassignment took place                                                                                    <--| 
 move the centroid to the center of all points assigned to it and reassign each data point to new closest centroid   |
"""
# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Finding the optimal number of clusters
"""
Within-Clusters-Sum-of-Squares
WCSS = ∑distance(Pi,C1)^2 + ∑distance(Pi,C2)^2 + ∑distance(Pi,C3)^2 ....

Sum of the square of the distance between each point of each cluster and their respective centroid
WCSS Decreases with the increase in the number of clusters
Number of clusters is where the change in WCSS isn't as significant anymore
"""
# Calculating and plotting the WCSS with different number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # inertia_ = WCSS
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means
kmeans = KMeans(n_clusters = 5,       # Number of clusters
                init = 'k-means++',   # K-means++ helps us avoid the random initialisation trap(Centroids)
                max_iter = 300,       # Maximum number of iterations of the K-means algorithm for a single run
                n_init = 10)          # Number of time the k-means algorithm will be run with different centroids
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s = 100, c = 'magenta', label = 'Sensible')
# Change the labels according to the plot
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centeroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
