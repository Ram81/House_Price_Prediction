from numpy import loadtxt, zeros, ones, logspace, linspace, mean, std, arange
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


def gradient_Descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_Cost(X, y, theta)

    return theta, J_history

			

#Load Dataset
data = loadtxt('Dataset/ex1data2.txt',delimiter=',')

#Plot The Data
'''fig = plt.figure()
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

#plt.show()
'''

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
















