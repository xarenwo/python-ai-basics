import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# x : sweetness
# y : crunchiness


xFruit = np.array([10,10])
yFruit = np.array([9,1])

xProtein = np.array([1,1])
yProtein = np.array([4,1])

xVegetable = np.array([7])
yVegetable = np.array([10])


X = np.array([[5,9],[5,1],[5,10],[1,1],[5,10],[3,7]])
y = np.array([0,0,1,1,2,2]) # 0 : FRUIT , 1: Protein , 2: Vegetables

plt.plot(xFruit,yFruit,'ro',color = 'blue')
plt.plot(xProtein,yProtein, 'ro', color='green')
plt.plot(xVegetable,yVegetable, 'ro', color='orange')

plt.plot(6,6,'ro',color='grey',markersize=15)

plt.axis([-0.5,15,-0.5,15])

classifier = KNeighborsClassifier(n_neighbors=3) # k value
classifier.fit(X,y)

pred=classifier.predict([[6,4]])

print(pred)
if(pred == 0):
    print('This is a fruit')
elif(pred == 1):
    print('This is a protein')
elif(pred == 2):
    print('This is a vegetable')
plt.show()