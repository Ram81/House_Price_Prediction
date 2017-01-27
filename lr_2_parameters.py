from numpy import loadtxt, zeros, ones, logspace, linspace, mean, std, arange
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import plot, show, xlabel, ylabel

#Load Dataset
data = loadtxt('Dataset/ex1data2.txt',delimiter=',')

#Plot The Data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100

for c,m,zl,zh in [('r','o',-50,-25)]:
	xs = data[:,0]
	ys = data[:,1]
	zs = data[:,2]
	ax.scatter(xs,ys,zs,c=c,marker=m)

ax.set_xlabel('Size of House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of House')

plt.show()
