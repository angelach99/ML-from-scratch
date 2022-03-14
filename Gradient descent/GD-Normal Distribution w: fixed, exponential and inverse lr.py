import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import linspace, meshgrid

############################## Problem 1 ##############################
"""
Question:
The dataset sales.csv contains the daily sales of a product for 1000 days.Visually
demonstrate that the daily sales follows a normal distribution.
In this problem, you will be fitting a normal distribution to the observed dataset by
minimizing the negative log-likelihood function. Using the probability distribution function
for a normal distribution 𝑁 with two parameters: mean and variance, obtain
the negative log-likelihood function. Using gradient descent, minimize the negative loglikelihood
function to estimate the parameters and that best explain the data.
"""

# Import dataset
df = pd.read_csv('sales.csv')
df = df.drop(columns = ['Unnamed: 1', 'Unnamed: 2'])

# To visualize the distribution
plt.hist(df, bins=20)
plt.xlabel("Daily Sales")
plt.ylabel("Count")
plt.title('Histogram of daily sales distribution')
plt.show() 

# Check mu and sigma2 first to make sure the result is similar
df.mean() # 79.379
df.std()**2 # 14.79^2 = 218.67

# Fit a normal distribution
# The negative log-likelihood function is:
# -L(mu, sigma^2) =
# (n/2)*ln(2*pi) + n*ln(sigma) + (1/(2*sigma^2))*sum^(n)_{i=1}((xi-mu)^2)

# First, let's define some values
n = len(df['DAILY SALES'])
x = df['DAILY SALES'].to_numpy()

iteration = 0
precision = 0.00001
delta = 10

sigma2 = 200
mu = 70

f = lambda mu, sigma2: (n/2)*math.log(2*math.pi) + n*math.log(sigma2**(0.5))+ 1/(2*sigma2) * sum((x[i] - mu)**2 for i in range(n))
df_sigma2 = lambda mu, sigma2: n/(2*sigma2) - 1/(2*(sigma2**2)) * sum((x[i] - mu)**2 for i in range(n))
df_mu = lambda mu, sigma2: -1/(sigma2) * sum((x[i] - mu) for i in range(n))

learning_rate = 0.01

# Gradient descent method
while delta > precision:
    sigma_new = sigma2 - learning_rate * df_sigma2(mu, sigma2)
    mu_new = mu - learning_rate * df_mu(mu, sigma2)
    delta = abs(f(mu_new, sigma_new) - f(mu, sigma2))
    iteration += 1
    print("Iteration", iteration, "\nmu = ", mu_new, "\nsigma = ", sigma_new, " and f(mu, sigma^2) = ", f(mu_new, sigma_new))
    mu = mu_new
    sigma2 = sigma_new

# The estimated mu =  79.379 and sigma^2 = 215.526. There is tiny difference between the
# true value and the estimated value, which is due to manual set learning rate and precision.


############################## Problem 2 ##############################
"""
def gradient_descent(x, y, learning_rate, f, df_x, df_y):
    iteration = 0
    precision = 0.00001 # terminating condition
    delta = 10 # difference between two f(x, y)
    max_iteration = 10000
    data = {}
    data[0] = [x, y, f(x, y)]
    x_line, y_line = np.meshgrid(np.linspace(-6,6,100),np.linspace(-6,6,100))
    z_line = f(x_line, y_line)
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(x_line, y_line, z_line)
    while iteration < max_iteration and delta > precision:
        x_new = x - learning_rate * df_x(x, y)
        y_new = y - learning_rate * df_y(x, y)
        delta = abs(f(x_new, y_new) - f(x, y))
        iteration = iteration + 1
        print("Iteration", iteration, "\nx = ", x_new, "\ny = ", y_new, " and f(x, y) = ", f(x_new, y_new))
        x = x_new
        y = y_new 
        data[iteration] = [x_new, y_new, f(x_new, y_new)]
        ax.scatter3D(x_new, y_new, f(x_new, y_new), cmap='Greens')
    print("The local minimum occurs at", 'x = ', x_new, 'y = ', y_new, 'f = ', f(x_new, y_new))

# f(x, y)
gradient_descent(0, 2, 0.05, lambda x, y: (x-5)**2 + 2*(y+3)**2 + x*y, lambda x, y: 2*(x-5)+y, lambda x, y: 4*(y+3)+x)
# The local minimum occurs at x =  7.421371175505048 y =  -4.85416041283541 f =  -23.28566612647574

 # g(x, y)
gradient_descent(0, 2, 0.0015, lambda x, y: (1-(y-3))**2 + 10*((x+4)-(y-3)**2)**2, lambda x, y: 20*(x+4)-20*(y-3), lambda x, y: -2*(4-y)-20*(x+4)+20*(y-3))
# The local minimum occurs at x =  -2.9810775909937477 y =  4.017999926802516 f =  0.0033520991890477383

# Exponential decay learning rate
def gradient_descent_exponential(x, y, learning_rate, f, df_x, df_y):
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    iteration = 0
    precision = 0.00001 # terminating condition
    delta = 10 # difference between two f(x, y)
    max_iteration = 10000
    while iteration < max_iteration and delta > precision:
        x_new = x - learning_rate * df_x(x, y)
        y_new = y - learning_rate * df_y(x, y)
        delta = abs(f(x_new, y_new) - f(x, y))
        iteration = iteration + 1
        print("Iteration", iteration, "\nx = ", x_new, "\ny = ", y_new, " and f(x, y) = ", f(x_new, y_new))
        x = x_new
        y = y_new 
        learning_rate = learning_rate * math.exp(-0.001*iteration) # In this case let k = 1
    print("The local minimum occurs at", 'x = ', x_new, 'y = ', y_new, 'f = ', f(x_new, y_new))

# f(x, y)
gradient_descent_exponential(0, 2, 0.05, lambda x, y: (x-5)**2 + 2*(y+3)**2 + x*y, lambda x, y: 2*(x-5)+y, lambda x, y: 4*(y+3)+x)
# The local minimum occurs at x =  0.7635170555426241 y =  0.29889289060578306 f =  39.94138616581495
# The performance gets worse compare to the constant learning rate.

# g(x, y)
gradient_descent_exponential(0, 2, 0.0015, lambda x, y: (1-(y-3))**2 + 10*((x+4)-(y-3)**2)**2, lambda x, y: 20*(x+4)-20*(y-3), lambda x, y: -2+2*(y-3)-20*(x+4)+20*(y-3))
# The local minimum occurs at x =  -0.273301540521472 y =  2.2841269717801898 f =  106.256596631213
# The performance gets worse compare to the constant learning rate.


# Tune the parameters -- Exponential
def gradient_descent_exponential_tune(x_0, y_0, learning_rate_0, f, df_x, df_y):
    total_iter = 10000
    for k in np.arange(0, 1, 0.005):
        precision = 0.00001 # terminating condition
        x = x_0
        y = y_0
        iteration = 0
        learning_rate = learning_rate_0
        delta = 10 # difference between two f(x, y)
        while delta > precision:
            x_new = x - learning_rate * df_x(x, y)
            y_new = y - learning_rate * df_y(x, y)
            delta = abs(f(x_new, y_new) - f(x, y))
            iteration = iteration + 1
            x = x_new
            y = y_new 
            learning_rate = learning_rate * math.exp(-k*iteration)
        print("Iteration", iteration, "\nx = ", x_new, "\ny = ", y_new, " and f(x, y) = ", f(x_new, y_new), 'k =', k)
        if iteration < total_iter:
            total_iter = iteration
            opt_k = k
            x_new_final = x_new
            y_new_final = y_new
    print("The local minimum occurs at", 'x = ', x_new_final, 'y = ', y_new_final, 'f = ', f(x_new_final, y_new_final), 'minimum iteration', iteration, 'optimal_k =', opt_k)
 
# f(x, y)
gradient_descent_exponential_tune(0, 2, 0.05, lambda x, y: (x-5)**2 + 2*(y+3)**2 + x*y, lambda x, y: 2*(x-5)+y, lambda x, y: 4*(y+3)+x)
# The least number of iteration is 6.

# g(x, y)
gradient_descent_exponential_tune(0, 2, 0.0015, lambda x, y: (1-(y-3))**2 + 10*((x+4)-(y-3)**2)**2, lambda x, y: 20*(x+4)-20*(y-3), lambda x, y: -2+2*(y-3)-20*(x+4)+20*(y-3))
  # The least number of iteration is 6.
    
    
# Inverse decay learning rate
def gradient_descent_inverse(x, y, learning_rate, f, df_x, df_y):
    plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    iteration = 0
    precision = 0.00001 # terminating condition
    delta = 10 # difference between two f(x, y)
    max_iteration = 10000
    while iteration < max_iteration and delta > precision:
        x_new = x - learning_rate * df_x(x, y)
        y_new = y - learning_rate * df_y(x, y)
        delta = abs(f(x_new, y_new) - f(x, y))
        iteration = iteration + 1
        print("Iteration", iteration, "\nx = ", x_new, "\ny = ", y_new, " and f(x, y) = ", f(x_new, y_new))
        x = x_new
        y = y_new 
        learning_rate = learning_rate / (1 + 0.5*iteration)
    print("The local minimum occurs at", 'x = ', x_new, 'y = ', y_new, 'f = ', f(x_new, y_new))
 
# f(x, y)
gradient_descent_inverse(0, 2, 0.05, lambda x, y: (x-5)**2 + 2*(y+3)**2 + x*y, lambda x, y: 2*(x-5)+y, lambda x, y: 4*(y+3)+x)
# The local minimum occurs at x =  0.8893857432162481 y =  0.08008811733116022 f =  35.94226441887843
# The performance gets worse compare to the constant learning rate.

# g(x, y)
gradient_descent_inverse(0, 2, 0.0015, lambda x, y: (1-(y-3))**2 + 10*((x+4)-(y-3)**2)**2, lambda x, y: 20*(x+4)-20*(y-3), lambda x, y: -2+2*(y-3)-20*(x+4)+20*(y-3))
# The local minimum occurs at x =  -0.31457353233761565 y =  2.3269961681195874 f =  107.28900715948336
# The performance gets worse compare to the constant learning rate.


# Tune the parameters -- Inverse
def gradient_descent_inverse_tune(x_0, y_0, learning_rate_0, f, df_x, df_y):
    total_iter = 10000
    for k in np.arange(0, 1, 0.005):
        precision = 0.00001 # terminating condition
        x = x_0
        y = y_0
        iteration = 0
        learning_rate = learning_rate_0
        delta = 10 # difference between two f(x, y)
        while delta > precision:
            x_new = x - learning_rate * df_x(x, y)
            y_new = y - learning_rate * df_y(x, y)
            delta = abs(f(x_new, y_new) - f(x, y))
            iteration = iteration + 1
            x = x_new
            y = y_new 
            learning_rate = learning_rate / (1 + k*iteration)
        print("Iteration", iteration, "\nx = ", x_new, "\ny = ", y_new, " and f(x, y) = ", f(x_new, y_new), 'k =', k)
        if iteration < total_iter:
            total_iter = iteration
            opt_k = k
            x_new_final = x_new
            y_new_final = y_new
    print("The local minimum occurs at", 'x = ', x_new_final, 'y = ', y_new_final, 'f = ', f(x_new_final, y_new_final), 'minimum iteration', iteration, 'optimal_k =', opt_k)
 
    
# f(x, y)
gradient_descent_inverse_tune(0, 2, 0.05, lambda x, y: (x-5)**2 + 2*(y+3)**2 + x*y, lambda x, y: 2*(x-5)+y, lambda x, y: 4*(y+3)+x)
# The least number of iteration is 10.

# g(x, y)
gradient_descent_inverse_tune(0, 2, 0.0015, lambda x, y: (1-(y-3))**2 + 10*((x+4)-(y-3)**2)**2, lambda x, y: 20*(x+4)-20*(y-3), lambda x, y: -2+2*(y-3)-20*(x+4)+20*(y-3))
# The least number of iteration is 10.
 



