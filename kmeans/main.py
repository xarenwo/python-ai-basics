import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs



x,y = make_blobs(n_samples=1000,centers=6,random_state=0,cluster_std=1)


estimator = KMeans(2)

estimator.fit(x) # Run loyd algorithm and find clusters
y_means = estimator.predict(x)
plt.axis([-20,100,-20,100])
plt.scatter(x[:,0],x[:,1],s=10)
plt.scatter(x[:,0],x[:,1],c=y_means,s=10,cmap='rainbow')


plt.show()
