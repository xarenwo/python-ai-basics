import numpy as np
from  matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression



# p i = 1 /1 + ex [ - ( b0 + b1 * x )]   sigmoid function


x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,999999])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,7,7.1,7.9,7,7.9,8,999999999])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[9999999999]])
Y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])


plt.plot(x1,y1,'ro',color='blue')
plt.plot(x2,y2,'ro',color='red')


classifier = LogisticRegression()

classifier.fit(X,Y)

pred = classifier.predict_proba(8)
print(pred)

def model(classifier, X):
    return 1 / ( 1 + np.exp(-(classifier.intercept_ + classifier.coef_ * X)))


for i in range(1,100000000,1):
    print(" probability  for %d " %i + "is %f" % float(model(classifier,i)*100) + "%")
    plt.plot(i/10.0-2,model(classifier,i/10.0-2),'ro',color='orange')

print("Percentage of admission probability is %s " %  + "%")

plt.axis([-2,1000000,-0.5,200000])
plt.plot()
plt.show()