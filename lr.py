from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the Dataset
data = loadtxt('Dataset/ex1data.txt',delimiter=',')

#plot dataset
scatter(data[:,0], data[:,1], marker='x', c='b')
title('Profit Distribution')
xlabel('Population in 10,000s')
ylabel('Profit in $10,000s')
show()
