import pandas as pd
import numpy as np

################### Data import and cleaning ###################
"""
Goal:
Using either all three or any two of the features, train a logistic regression model usingNewton’s method. Clearly express your gradient function and the Hessian (i.e., matrix ofsecond partial derivatives) which you will need to compute while implementing Newton’smethod.Compare the coefficients obtained with a black-box implementation of logistic regressionwith the coefficients obtained above using Newton’s method.Does the Newton’s method converge with ease? In case of non-convergence, set yourinitial points to be close to the estimates obtained using the black-box implementation andcheck.
"""

df = pd.read_csv('Default.csv')
df['default'] = df['default'].replace({'No': 0, 'Yes': 1})
df['student'] = df['student'].replace({'No': 0, 'Yes': 1})
df.isnull().sum()

# Split dataset to training and testing
X = df.iloc[:, 2:] # I will only use balance and income as the x variables
y = df['default']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

################### Black-box logistic regression ###################
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
model = lr.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
model.intercept_ # -1.07898914e-06
model.coef_ # 0.00048006, -0.00012674

# Accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred) # 0.969

# MSE
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_test_pred) # 0.031

################### Newton's Method ###################
# Define f
# which in this case is the log likelihood function of logistic regression
# z = beta0 + beta1*x1 + beta2*x2, where x1 = df['balance'] and x2 = df['income']
# p(z) = 1/(1+exp(-z)), this is the sigmoid function
# L = product^{n}_{i=1} p(z)^{y_i} + product^{n}_{i=1} (1-p(z))^(1-y_i), likelihood function
# l = sum^{n}_{i=1} (y_i*ln(p(z)) + (1-y_i)*ln(1-p(z))), log-likelihood function
# the goal is to minimize the negative log-likelihood function


################### Newton's Method -- Method 1 ###################
# Define length, independent and dependent variables
n = len(df)
x1 = df['balance']
x2 = df['income']
y = df['default']

# Gradient function
def gradient(beta0, beta1, beta2):
    df_beta0 = (-1)*sum(-y[i]*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                          + np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                   - (1-y[i])*((np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))/
                               (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2) for i in range(n))
    df_beta1 = (-1)*sum(-y[i]*beta1*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                + np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                   - (1-y[i])*beta1*((np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))
                                     /(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2) for i in range(n))
    df_beta2 = (-1)*sum(-y[i]*beta2*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                + np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                   - (1-y[i])*beta2*((np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))/
                                     (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2) for i in range(n))
    gradient = np.array([df_beta0, df_beta1, df_beta2])
    return gradient

# Hessian gradient
def hessian(beta0, beta1, beta2):
    df_beta0_beta0 = (-1)*sum(y[i]*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) + 
                                                       2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) - 
                                                 (1-y[i])*((-np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))*
                                                            (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 +
                                                            2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])) * 
                                                            (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                           (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))
    df_beta0_beta1 = (-1)*sum(y[i]*beta1*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                 + 2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                                                     - (1-y[i])*((-beta1*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  + 2*beta1*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                  * (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                                 (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))
    df_beta0_beta2 = (-1)*sum(y[i]*beta2*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                 + 2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                                                     - (1-y[i])*((-beta2*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  + 2*beta2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                  * (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                                 (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))
    df_beta1_beta0 = (-1)*sum(y[i]*beta1*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                 + 2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                                                     - (1-y[i])*((-beta1*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  + 2*beta1*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                  * (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                                 (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))
    df_beta1_beta1 = (-1)*sum(-y[i]*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) + 
                                                            np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                                                     + y[i]*beta1**2*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                     + 2*beta1**2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                     - (1-y[i])*((  (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  * np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                  * (1-beta1**2) + 2 * beta1**2 * np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                                 ((1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4))  for i in range(n))
    df_beta1_beta2 = (-1)*sum(y[i]*beta1*beta2*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                       + 2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])))
                                                     - (1-y[i])*((-beta1*(-2*beta2)*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  + 2*beta1*beta2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                  * (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                                 (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))
    df_beta2_beta0 = (-1)*sum(y[i]*beta2*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))+ 
                                                                 2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                                                     - (1-y[i])*((-beta2*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  + 2*beta2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  * (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                                 (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))
    df_beta2_beta1 = (-1)*sum(y[i]*beta1*beta2*(np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                       + 2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])))
                                                     - (1-y[i])*((-beta1*beta2*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  + 2*beta1*beta2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                  * (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))/
                                                                 (1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))
    df_beta2_beta2 = (-1)*sum(y[i]*((np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                            + np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))) 
                                                           + beta2*(beta2*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])) 
                                                                    + 2*beta2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i])))) 
                                                     - (1-y[i])*(((np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                   -2*beta2*np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**2 
                                                                  + 2*beta2**2*np.exp(-2*(beta0 + beta1 * x1[i] + beta2 * x2[i]))
                                                                  *(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i]))))   
                                                                 /(1+np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))**4) for i in range(n))

    hessian = np.array([[df_beta0_beta0, df_beta0_beta1, df_beta0_beta2], 
                        [df_beta1_beta0, df_beta1_beta1, df_beta1_beta2],
                        [df_beta2_beta0, df_beta2_beta1, df_beta2_beta2]])
    return hessian


def newton_method1(f, beta0, beta1, beta2):
    max_iteration = 10 # terminating condition 1
    curr_iter = 0
    beta_array = np.array([beta0, beta1, beta2])
    while curr_iter < max_iteration:
        delta_h = hessian(beta_array[0], beta_array[1], beta_array[2])
        delta_g = gradient(beta_array[0], beta_array[1], beta_array[2])
        
        new_beta_array = beta_array - np.linalg.inv(delta_h) @ delta_g.T
        
        delta = abs(f(new_beta_array[0], new_beta_array[1], new_beta_array[2]) - f(beta_array[0], beta_array[1], beta_array[2]))
        print('delta = ', delta)
        print("Iteration", curr_iter, "\nbeta0 = ", new_beta_array[0], "\nbeta1 = ", new_beta_array[1], "\nbeta2 = ", new_beta_array[2], " and l = ", f(new_beta_array[1], new_beta_array[1], new_beta_array[2]))
        beta_array = new_beta_array
        
        curr_iter += 1
        
    print("The optimal coefficient is", 'beta0 = ', new_beta_array[0], 'beta1 = ', new_beta_array[1], 'beta2 = ', new_beta_array[2])
    
f = lambda beta0, beta1, beta2: (-1)*sum(y[i] * np.log(1/(1 + np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))) + (1 - y[i]) * np.log(1 - 1/(1 + np.exp(-(beta0 + beta1 * x1[i] + beta2 * x2[i])))) for i in range(n))
newton_method1(f, 0, 0, 0) # beta0 =  12.221652916700355, beta1 =  0.0, beta2 =  0.0


# This is the results from the black-box implementation
# beta0 = -1.07898914e-06
# beta1 = 0.00048006, beta2 = -0.00012674
# And just by curiosity, here are the value for f, gradient, and hessian using output from black-box implementation
f(-1.07898914e-06,0.00048006,-0.00012674) # -1735.64045
gradient(-1.07898914e-06,0.00048006,-0.00012674) # [-1.90341982e+07, -9.13755718e+03,  2.41239428e+03]
hessian(-1.07898914e-06,0.00048006,-0.00012674) # [[ 3.80318535e+07,  1.82575716e+04, -4.82015711e+03], [ 1.82575716e+04, -1.90333247e+07, -2.31387845e+00], [-4.82015711e+03, -2.31396462e+00,  1.90332544e+07]]

# If I were to run the newton_method function with initial beta of (0,0,0),
# The result does not converge easily.
# By running 10 iterations, the results are getting farther away from the the optimal solution
# If we were to set the initial points close to the optimal points (-1e-06,0.00048,-0.00013), 
# then the result would slowly converge to the optimal solution.
newton_method1(f, -1e-06,0.00048,-0.00013)  # beta0 = -1.90341982e+07, beta1 = -9.13755718e+03, beta2 = 2.41239428e+03]


################### Newton's Method -- Method 2 ###################
def sigmoid(x, beta_array):
    inp = x @ beta_array
    y = 1/(1+np.exp(-inp))
    return y

# Define length, independent and dependent variables
n = len(df)
X = df.iloc[:, 2:] # I will only use balance and income as the x variables
X.insert(0, 'coef', 1)
X = X.to_numpy()
y = df['default']
y = y.to_numpy()

def newton_method2(beta0, beta1, beta2):
    # Define initial beta array
    beta_array = np.array([beta0, beta1, beta2])
    
    # Define terminating conditions
    precision = 0.0001
    max_iteration = 100
    curr_iter = 0
    delta = 1
    
    while curr_iter < max_iteration and delta > precision:
        # define f
        sigmoid_value = sigmoid(X, beta_array)
        f = -(y.T @ np.log(sigmoid_value) + (1 - y.T @ np.log(1 - sigmoid_value)))
        
        # define graident
        gradient = X.T @ (sigmoid_value - y)
        
        # define hessian
        # H = XDX^T, D is the positive definite diagonal matrix 
        hessian = X.T @ np.diag([(sigmoid(X[i,:], beta_array) * (1 - sigmoid(X[i,:], beta_array))) for i in range(n)])  @ X
        
        # define delta
        delta = abs(delta - f)
        
        # Updating step
        new_beta_array = beta_array - np.linalg.inv(hessian) @ gradient
        beta_array = new_beta_array
        curr_iter += 1
        
        # print result
        print('delta = ', delta)
        print("Iteration", curr_iter, "\nbeta0 = ", new_beta_array[0], "\nbeta1 = ", new_beta_array[1], "\nbeta2 = ", new_beta_array[2], " and loss = ", f)

    # print final result        
    print("The optimal coefficient is", 'beta0 = ', new_beta_array[0], 'beta1 = ', new_beta_array[1], 'beta2 = ', new_beta_array[2])
        
# call the function to optimize
newton_method2(0,0,0) # beta0 = -1.90341982e+07, beta1 = -9.13755718e+03, beta2 = 2.41239428e+03
# For method 2, if the initial value for beta are 0, 0, and 0
# It will converge to the optimal value from the black-box implementation.

newton_method2(-1e-06,0.00048,-0.00013) # beta0 = -1.90341982e+07, beta1 = -9.13755718e+03, beta2 = 2.41239428e+03
# And if I try numbers close to the optimal, it will converge as well.

################### Stochastic Gradient Descent ###################
def mini_batch(x, y, batchsize, shuffle=False):
    assert x.shape[0] == y.shape[0]
    if shuffle:
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
    for i in range(0, x.shape[0] - batchsize + 1, batchsize):
        end_idx = min(i + batchsize, x.shape[0])
        if shuffle:
            batch = indices[i : end_idx]
        else:
            batch = slice(i, end_idx)
        yield x[batch], y[batch]

def sgd(beta0, beta1, beta2, learning_rate, batchsize):
    
    # define terminating condition
    iteration = 0
    max_iteration = 1000
    
    # define beta array
    beta_array = np.array([beta0, beta1, beta2])
    
    # iterate
    while iteration < max_iteration :

        for batch in mini_batch(X, y, batchsize, shuffle = True):
            x_batch, y_batch = batch
            
            # define graident
            sigmoid_value = sigmoid(x_batch, beta_array)
            gradient = x_batch.T @ (sigmoid_value - y_batch)
            
            # Updating step
            new_beta_array = beta_array - learning_rate * 1/n * np.sum(gradient)
            print("Iteration", iteration, "\nbeta0 = ", new_beta_array[0], "\nbeta1 = ", new_beta_array[1], "\nbeta2 = ", new_beta_array[2])

            beta_array = new_beta_array
            iteration += 1
    print("The optimal coefficient is", 'beta0 = ', new_beta_array[0], 'beta1 = ', new_beta_array[1], 'beta2 = ', new_beta_array[2])
    
# to optimize at batch size 16
sgd(0, 0, 0, 0.0015, 16) # beta0 =  -0.057103480043064536 beta1 =  -0.057103480043064536 beta2 =  -0.057103480043064536

sgd(-1e-06,0.00048,-0.00013, 0.0015, 16) # beta0 =  -0.013985104741889631 beta1 =  -0.013504104741889622 beta2 =  -0.01411410474188965

# to optimize at batch size 32
sgd(0, 0, 0, 0.0015, 32) # beta0 =  -0.048476434421103595 beta1 =  -0.048476434421103595 beta2 =  -0.048476434421103595

sgd(-1e-06,0.00048,-0.00013, 0.0015, 32) # beta0 =  -0.12029433709118476 beta1 =  -0.11981333709118483 beta2 =  -0.1204233370911848

# to optimize at batch size 64
sgd(0, 0, 0, 0.0015, 64) # beta0 =  -0.10322963898333617 beta1 =  -0.10322963898333617 beta2 =  -0.10322963898333617

sgd(-1e-06,0.00048,-0.00013, 0.0015, 64) # beta0 =  -0.18988083515243245 beta1 =  -0.1893998351524325 beta2 =  -0.1900098351524324

# Overall, the optimal solution when using mini-batch does not converge to the true optimal from black-box implementation
# This could due to the learning rate or the batch size
# Or just the initial starting value for beta
        


# References:
# https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method
# https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
# https://hastie.su.domains/Papers/ESLII.pdf