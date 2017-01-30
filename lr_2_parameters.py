from numpy import loadtxt, zeros, ones, logspace, linspace, mean, std, arange, array
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import plot, show, xlabel, ylabel

def feature_Normalize(X):
	'''
		Returns normalized version of X
	'''
	mean_r = []
	std_r = []
	
	X_norm = X
	n_c = X.shape[1]
	
	for i in range(n_c):
		m = mean(X[:,i])
		s = std(X[:,i])
		mean_r.append(m)
		std_r.append(s)
		X_norm[:,i] = (X_norm[:,i] - m)/s;
	return X, mean_r, std_r

def compute_Cost(X, y, theta):		
	'''
		Compute Cost function
	'''
	m=y.size
	
	h_theta = X.dot(theta)
	
	sqr_Error = (h_theta - y);
	
	J = (1.0/(2*m)) * sqr_Error.T.dot(sqr_Error)
	
	return J


def gradient_Descent(X, y, theta, alpha, iters):
    '''
    	Performs gradient descent to learn theta by taking iters gradient steps with learning rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(iters, 1))

    for i in range(iters):

        h_theta = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (h_theta - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_Cost(X, y, theta)

    return theta, J_history


def Predict(X,theta):
	prediction = X.dot(theta)
	return prediction	
			

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


#input & output vectors
X = data[:,:2]
y = data[:,2]

#Number of training examples
m=y.size

y.shape = (m, 1)

x, mean_r, std_r = feature_Normalize(X)

#Adding a column of zeros for x0 term

it = ones(shape=(m,3))
it[:,1:3] = x

#Some Gradient Descent setting
iterations = 100
alpha = 0.01

#init & run gradient descent
theta = zeros(shape=(3,1))

theta, J_history = gradient_Descent(it, y, theta, alpha, iterations)

print(theta)
print()
print(J_history)

plot(arange(iterations),J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()


#Predicting Price for a new house of  165 sq-ft & 3 bedroom
X1 = array([1.0, ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])])

print('Predicted Price of a house of  1650 sq-ft with 3 bedrooms is %f '% (Predict(X1,theta)))
