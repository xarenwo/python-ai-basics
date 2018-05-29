import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


x = np.array([[1,1],[1.1,1.1],[3,3],[4,4],[3,3.5],[3.5,4]])

plt.scatter(x[:,1],x[:,1],s=50)
linkage_matrix = linkage(x,"single")

dendogram = dendrogram(linkage_matrix,truncate_mode='none')
plt.title("Hierarchical clustering")
plt.show()