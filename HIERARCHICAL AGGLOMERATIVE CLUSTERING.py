#EXP5 - HIERARCHICAL AGGLOMERATIVE CLUSTERING
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

X,_=make_blobs(n_samples=150,centers=3,cluster_std=.8,random_state=42)

plt.figure()
plt.subplot(1,2,1)
sch.dendrogram(sch.linkage(X,method='ward'))
plt.title("Dendrogram")

model=AgglomerativeClustering(n_clusters=3,linkage='ward')
labels=model.fit_predict(X)

plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis')
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.tight_layout(); plt.show()