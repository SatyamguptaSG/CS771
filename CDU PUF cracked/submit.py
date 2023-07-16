import numpy as np

def objective(A,x,b,gamma):
	
	return f(A,x,b) + gamma*np.sum(np.abs(x))


def f(A,x,b):
	
	return 0.5*(np.linalg.norm(A*x-b)**2)


def gradf(AtA,x,Atb):
	
	return AtA*x-Atb


def uppbd(A,AtA,Atb,x,x_1,b,lamdaK):
	
	xDiff = x_1 - x
	return f(A,x,b) + gradf(AtA,x,Atb).T*xDiff + 1.0/(2.0*lamdaK)* np.sum(np.multiply(xDiff,xDiff))

def proxop(v,lamdaK):
	
	zero = np.matrix(np.zeros(np.shape(v)))

	return np.multiply(np.sign(v),np.maximum(np.abs(v)-lamdaK,zero))

def jls_extract_def(x_1, obj):
    return x_1, obj

def AccProxgd(Max_iter,lamdaK,gamma,AtA,Atb,A,beta,b,ABSTOL):

	x = np.matrix(np.zeros((np.shape(A)[1],1)))
	xprev = x

	obj = []
	for k in range(Max_iter):

		y = x + (1/(k+3)) * (x-xprev)

		while True:
			x_1 = proxop(y-lamdaK*gradf(AtA,y,Atb),lamdaK*gamma)
			if f(A,x_1,b) <= uppbd(A,AtA,Atb,y,x_1,b,lamdaK):
				break
			else:
				lamdaK = beta*lamdaK

		obj.append(objective(A,x_1,b,gamma))

		# terminating condition
		if k > 1 and np.linalg.norm(objective(A,x_1,b,gamma) - objective(A,x,b,gamma)) < ABSTOL:
			break

		xprev = x
		x = x_1

	return x_1, obj

# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_trn, y_trn ):
################################
#  Non Editable Region Ending  #
################################

	A = np.matrix(X_trn)
	b = np.matrix(y_trn).T

	m = 1600 	#number of examples

	n = 2048	#number of features
	s = 512
	for i in range(m):
		b[i][0] = b[i][0] * 0.001

	S = np.zeros((m,1))
	S[np.floor(m*np.random.rand(s,1)).astype(int)] = 1  # setting any random 512 values of S to 1
	S = S*np.random.normal(0,5,(m,1))             # normalized S

	x_k = np.zeros((m,1))
	x_k[np.floor(m*np.random.rand(m-s,1)).astype(int)] = 1  # setting any random 512 values of xStar to 1  
	x_k = S*np.random.normal(0,2,(m,1))               # normalized xstar

	x_r = S*np.random.normal(0,2,(m,1))

	AtA  = A.T*A #n*n
	Atb  = A.T*b #n*1
	Max_iter = 2000
	ABSTOL = 1e-3
	RELTOL   = 1e-2

	lamdaK = 1
	beta = 0.5 #decreasing parameter for lambda
	# gamma =  0.1*np.linalg.norm(Atb,np.inf)
	gamma =  0.1

	w,obj= AccProxgd(Max_iter,lamdaK,gamma,AtA,Atb,A,beta,b,ABSTOL)	
	

	return np.array(w).ravel()					# Return the trained model

