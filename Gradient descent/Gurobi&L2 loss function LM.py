import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB

############################## Problem 1 ##############################
"""
Obtain the parameter estimates for a linear regression model with an L1 lossfunction using the advertising dataset. Use Gurobi to solve the associated linear program.Compare these parameter estimates with that obtained in the previous homework
"""

# Import dataset
df = pd.read_csv('advertising.csv')
df = df.drop(columns = ['Unnamed: 0'])

# define X and y
X = df.loc[:, df.columns != 'Sales']
y = df['Sales']

X_arr = X.to_numpy()
y_arr = y.to_numpy()

p = 4 # Number of x variables + coefficient
n = len(X_arr) # Number of observations

# Minimize L1 loss function
# Problem setup
model = gp.Model('Min L1 loss function for LR')

beta = model.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=['beta_' + str(j) for j in range(p)])
f = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name = ['f_' + str(i) for i in range(n)])

# Define 2 constraints
for i in range(n):
    const1 = model.addConstr(f[i] >= (y_arr[i] - (beta[0] + beta[1] * X_arr[i][0] + beta[2] * X_arr[i][1] + beta[3] * X_arr[i][2])))
    const2 = model.addConstr(f[i] >= -(y_arr[i] - (beta[0] + beta[1] * X_arr[i][0] + beta[2] * X_arr[i][1] + beta[3] * X_arr[i][2])))

# Objective function
model.setObjective(sum(f[i] for i in range(n)), GRB.MINIMIZE)

model.optimize()

print("Optimal value for DVs:")
for v in model.getVars():
    print(v.varName, "=", round(v.x,5))

print("\n\n_________________________________________________________________________________")
print(f"The best straight line that minimizes the maximum deviation is:")
print("_________________________________________________________________________________")
print(f"y = {beta[1].x:.4f}*x1 + {beta[2].x:.4f}*x2 + {beta[3].x:.4f}*x3 + ({beta[0].x:.4f})")

# To examine outliers from scatter plot
plt.scatter(X['TV'], y)
plt.show()

plt.scatter(X['Radio'], y)
plt.show()

plt.scatter(X['Newspaper'], y)
plt.show()

# Comparison these parameter with that obtained in the previous homework:
# The parameters are different from what we got in HW1
# Now the linear regression is y = 0.0435*x1 + 0.1973*x2 + -0.0032*x3 + (3.4119)
# While in HW1 it was Sales = 2.94 + 4.76 * TV + 1.89 * Radio - 1.04 * Newspaper
# This is because L1 loss function is more robust to outliers in the data
# Also if there are large numbers in x variables, L2 could cause a problem by squaring them.
# Therefore the fit here is slightly different from what we got in HW1.

############################## Problem 2 ##############################
"""
The make blobs function inscikit-learn can be used to generate data-points simulated from multivariate normal distributions.Your first task is to use the make blobs function to generate 5001 observations(𝑥,𝑦) with two predictor variables and one outcome variable. Set centers = 2. For eachobservation 𝑖, 𝑦𝑖 denotes the class of observations it belongs to.Visualize the data and visually identify whether the observations can be separated witha linear classifier.Implement a linear classifier by formulating and solving the optimization problemdiscussed in the lecture using Gurobi.
"""
 
from sklearn.datasets import make_blobs

n_samples = 500
n_features = 2

X, y = make_blobs(n_samples=500, n_features=2, centers=2, random_state=0)

for i, value in enumerate(y):
    if value == 0:
        y[i] = -1

df2 = pd.DataFrame(X, columns=['predictor1', 'predictor2'])
df2['outcome'] = pd.DataFrame(y)

# To visualize whether the observations can be separated with a linear classifier
plt.scatter(df2['predictor1'], df2['predictor2'], c = df2['outcome'])
plt.xlabel('predictor1')
plt.ylabel('predictor2')
plt.show()

# Another 3D visualization
ax = plt.axes(projection='3d')
ax.scatter(df2['predictor1'], df2['predictor2'], df2['outcome'], c=df2['outcome'], cmap='viridis', linewidth=0.5);

# Just by visualization, the data cannot be linearly separated

# Implement a linear classifier
model2 = gp.Model('Linear Classifier')

alpha = model2.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=["alpha_" + str(i) for i in range(3)])
error = model2.addVar(lb=0, vtype=GRB.CONTINUOUS, name="error")

for i in range(n_samples):
    constr1 = model2.addConstr(y[i]*(alpha[0] * X[i][0] + alpha[1] * X[i][1] + alpha[2]) >= error)
    constr2 = model2.addConstr(y[i]*(alpha[0] * X[i][0] + alpha[1] * X[i][1] + alpha[2] + error) >= 0)

model2.setObjective(error, GRB.MAXIMIZE)
model2.optimize()

print("Optimal value for DVs:")
for v in model2.getVars():
    print(v.varName, "=", round(v.x,5))

# Yes, the implementation correctly predicts whether the dataset can be classified using a linear classifier.





