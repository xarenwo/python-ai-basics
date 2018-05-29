from scipy import stats
import numpy as np
from matplotlib import pyplot as plot

x = np.array([50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]) # size of house
y = np.array([85000,180000,255000,280500,320000,360000,400000,440000,480000,520000,560000,600000,640000,680000,720000,760000]) # price


slope,intercept, r_value,p_value,std_err = stats.linregress(x,y)

plot.plot(x,y,'ro',color='black')

plot.ylabel('Price')
plot.xlabel('Size m2')

plot.axis([0,800,0,800000])
plot.plot(x,x*slope+intercept, 'b')
plot.plot()
plot.show()


#predict

data = 150
result = data * slope + intercept # y=mx+q
print("New house would cost $%s " % result)