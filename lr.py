from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Evaluate Linear Regression
def compute_Cost(X, Y, theta):
	'''
		Compute Cost function 
	'''
	m=Y.size
	h_theta = X.dot(theta).flatten()
	sqr_Error = (h_theta - y)**2
	J = (1/(2*m)) * sqrError.sum()
	return J
	
def gradient_Descent(X, Y, theta,alpha,iterations):
	'''
		Gradient Descent to learn theta by taking 
		iterations gradient steps with learning rate alpha
	'''
	m = Y.size
	J_history = zeros(shape=(iterations,1))
	for i in range(iterations):
		h_theta = X.dot(theta).flatten()
		
		error_x1 = (h_theta - y) * X[:,0]
		error_x2 = (h_theta - y) * X[:,1]		
		
		theta[0] = theta[0] - alpha/m * error_x1.sum()
		theta[1] = theta[1] - alpha/m * error_x2.sum()
		
		J_history[i,:] = compute_Cost(X,Y,theta)
	
	return theta, J_history


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
m = Y.size
it = ones(shape=(m,2))

it[:,1] = X

#initialize theta parameter
theta = zeros(shape=(2,1))

#gradient Descent Setting
iterations = 1500
alpha = 0.01


