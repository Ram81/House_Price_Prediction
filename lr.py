from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Evaluate Linear Regression
def compute_Cost(X, Y, theta):
	'''
		Compute Cost function 
	'''
	m=Y.size
	h_theta = X.dot(theta).flatten()
	sqr_Error = (h_theta - Y)**2
	J = (1.0/(2*m)) * sqr_Error.sum()
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
		
		error_x1 = (h_theta - Y) * X[:,0]
		error_x2 = (h_theta - Y) * X[:,1]		
		
		theta[0] = theta[0] - alpha * (1.0/m) * error_x1.sum()
		theta[1] = theta[1] - alpha * (1.0/m) * error_x2.sum()
		
		J_history[i,:] = compute_Cost(X,Y,theta)
	
	return theta, J_history

def Predict(X,theta):
	predict = X.dot(theta).flatten()
	return predict

#Load the Dataset
data = loadtxt('Dataset/ex1data1.txt',delimiter=',')

#plot dataset
scatter(data[:,0], data[:,1], marker='x', c='b')
title('Profit Distribution')
xlabel('Population in 10,000s')
ylabel('Profit in $10,000s')

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

#compute & display initial cost
print(compute_Cost(it,Y,theta))
print()
theta, J_History = gradient_Descent(it, Y, theta, alpha, iterations)

print(theta)

#predict values for population sizes 35,000 & 75,000

X1 = array([1,3.5])
X2 = array([1,7.5])

print('For population of 35,000 we predict profit of %f' % (Predict(X1,theta) * 10000))
print('For population of 75,000 we predict profit of %f' % (Predict(X2,theta) * 10000))

#plot the results
result = it.dot(theta).flatten()
plot(data[:,0], result)
show()

#Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)

#Jval initialized to 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

for t1,element in enumerate(theta0_vals):
	for t2,element2 in enumerate(theta1_vals):
		thetaT = zeros(shape=(2,1))
		thetaT[0][0] = element
		thetaT[1][0] =element2
		J_vals[t1,t2] = compute_Cost(it,Y,thetaT)

#plot COntours
J_vals = J_vals.T

contour(theta0_vals,theta1_vals,J_vals,logspace(-2,3,20))
xlabel('Theta 0')
ylabel('THeta 1')
scatter(theta[0][0],theta[1][0])
show()















