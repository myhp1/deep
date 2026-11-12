#EXP6 - K MEANS CLUSTERING
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X,_=make_blobs(n_samples=300,centers=4,cluster_std=.6,random_state=42)

k=4
model=KMeans(n_clusters=k,init='k-means++',n_init=10,random_state=42)
labels=model.fit_predict(X)
centroids=model.cluster_centers_

print("K =",k)
print("Inertia:",round(model.inertia_,2))

plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis')
plt.scatter(centroids[:,0],centroids[:,1],c='red',marker='X',s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2"); plt.show()