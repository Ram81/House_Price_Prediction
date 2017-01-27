from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the Dataset
data = loadtxt('Dataset/ex1data1.txt',delimiter=',')

#plot dataset
scatter(data[:,0], data[:,1], marker='x', c='b')
title('Profit Distribution')
xlabel('Population in 10,000s')
ylabel('Profit in $10,000s')
show()

#Fitting Linear Regression model
X = data[:,0]
Y = data[:,1]

#number of traing examples
m = X.size
it = ones(shape=(m,2))

it[:,1] = X

#initialize theta parameter
theta = zeros(shape=(2,1))

#gradient Descent Setting
iterations = 1500
alpha = 0.01


