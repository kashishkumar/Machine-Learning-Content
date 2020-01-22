"""If intelligence was a cake,
unsupervised learning would be the cake, supervised learning would be the icing on
the cake, and reinforcement learning would be the cherry on the cake - Yann Le Cun"""

""" Clustering, Anomaly Detection, and Density Estimation"""

"""Customer Segmentation, Dimensionality Reduction (using affinity which is any measure of how well an instance fits into a cluster), SemiSupervised Learning, Search Engines(Image Search), Image Segmentation"""
import pandas as pd
import numpy as np
data=pd.read_csv("total_data_na.csv")
data.columns
X=data.values[:140,1:]
X=X.astype(float)
for i in range(len(X)):
    for j in range(len(X[i])):
        if X[i][j]=='-':
            X[i][j]= 0
X_test=data.values[140:,1:]
from sklearn.preprocessing import scale
X=scale(X,axis=0)
from sklearn.cluster import KMeans
k=4
kmeans=KMeans(n_clusters=k)
ypreds=kmeans.fit_predict(X)
ypreds #Cluster Predictions
kmeans.cluster_centers_ 
kmeans.predict(X_test)
from sklearn.metrics import silhouette_score
inertia=[] 
silhouette_score_=[]
for k in range(2,12):
    kmeans=KMeans(n_clusters=k)
    ypreds=kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)
    silhouette_score_.append(silhouette_score(X,ypreds))

np.where(inertia==min(inertia))

kmeans=KMeans(n_clusters=6)
ypreds=kmeans.fit_predict(X)

import matplotlib.pyplot as plt
plt.plot(inertia)    
plt.plot(silhouette_score_)
    
"""Instead of assigning each instance to a single cluster, which is called hard clustering, it
can be useful to just give each instance a score per cluster: this is called soft clustering."""
"""In the KMeans class, the transform() method measures the distance from each instance to every centroid"""

"""If you have a high-dimensional dataset and you transform it this way, you end up with a k-dimensional dataset: this
can be a very efficient non-linear dimensionality reduction techniques which means an n dimensional vector can be represented with respect to few significant points in the dataset"""
kmeans.transform(X_test)

    
for i in range(6):
    print("i is")
    print(i)
    print(data.values[np.where(ypreds==i),0])    


"""just start by placing the centroids randomly (e.g., by
picking k instances at random and using their locations as centroids). Then label the
instances, update the centroids, label the instances, update the centroids, and so on
until the centroids stop moving."""


"""The computational complexity of the algorithm is generally linear
with regards to the number of instances m, the number of clusters
k and the number of dimensions n."""

good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)

"""uses a performance metric! It is called the model’s inertia: this is the mean squared distance
between each instance and its closest centroid."""

from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)

"""For Better Speed with respect to number of clusters. Use minibatch KMeans clustering when number of clusters are more"""
"""The inertia is not a good performance metric when trying to choose k since it keeps getting lower as we increase k."""

"""Use the elbow rule for coarse selection"""

"""For fine selection, use silhouette score, which is the mean silhouette coefficient over all the instances."""
"""b-a/max(a,b). The silhouette coefficient can vary between -1 and +1: a coefficient close to
+1 means that the instance is well inside its own cluster and far from other clusters,
while a coefficient close to 0 means that it is close to a cluster boundary, and finally a
coefficient close to -1 means that the instance may have been assigned to the wrong
cluster."""


"""silhouette diagram can also be used""" 

silhouette_score(X,ypreds)

from yellowbrick.cluster import SilhouetteVisualizer
visualizer=SilhouetteVisualizer(kmeans,colors='yellowbrick')
visualizer.fit(X)
visualizer.show()


# DB and Dubb scores can also be used as metrics