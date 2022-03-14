# Final Project (Gaussian Mixture Model)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


#################### Implement GMM from scratch ####################
class gaussian_mixture_model:
    """ Implementation of Gaussian Mixture Model from scratch
    
    Parameters
    ----------
        k: number of gaussian distributions (clusters)
        max_iter: maxium number of iteration 
        precision: stopping criteria, when delta is smaller than precision, the algorithm will stop
    
    """
    def __init__(self, k, max_iter, precision):
        self.k = k
        self.max_iter = 100
        self.precision = 1e-05
        
    # define initialization step
    def initialization(self, data):
        # dimensionality
        self.shape = data.shape
        self.row, self.column = self.shape
        
        # Initialize mu and covariance
        rand = np.random.randint(low = 0, high = self.row, size = self.k)
        self.mu = [data[i,:] for i in rand]
        self.cov = [np.cov(data.T) for _ in range(self.k)]

        # Initialize phi and weight
        self.phi = np.full(self.k, 1/self.k) # cluster weight assigned to each cluster
        # This parameter tells us what's the prior probability that the data point in our data set x comes from the kth cluster
        self.weight = np.full(self.shape, 1/self.k)  # the expectation/weight, likelihood that the ith observation belongs to cluster j
        
    
    def log_likelihood(self, data):
        # create an empty array to store likelihood value 
        likelihood = np.zeros((self.row, self.k))
        
        for i in range(self.k):
            dist = multivariate_normal(mean = self.mu[i], cov = self.cov[i])
            likelihood[:,i] = dist.pdf(data)
            
        weight = (likelihood * self.phi)/(likelihood * self.phi).sum(axis=1)[:,np.newaxis]
        return weight, likelihood
        
    
    def e_step(self, data):
        # Update phi and weight, holding mu and covariance fixed
        labels = np.zeros((self.row))
        self.weight, likelihood = self.log_likelihood(data)
        self.phi = self.weight.mean(axis = 0)
        labels = np.argmax(self.weight, axis = 1)
        return likelihood, labels

    def m_step(self, data):
        # Update mu and covariance, holding phi and weight fixed
        for i in range(self.k):
            new_weight = self.weight[:,[i]]
            sum_weight = new_weight.sum()
            self.mu[i] = (data * new_weight).sum(axis = 0)/sum_weight
            self.cov[i] = np.cov(data.T, aweights=(new_weight/sum_weight).flatten(), bias=True)
          
    def visualization(self, data):
        # This visualization only apply to 2 dimensional datasets
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1])
        
        # define initial settings
        delta = 0.05
        x1_min = data[:, 0].min() - 5
        x1_max = data[:, 0].max() + 5
        x2_min = data[:, 1].min() - 5
        x2_max = data[:, 1].max() + 5
        
        # scatter plot dimension
        x = np.arange(x1_min, x1_max, delta)
        y = np.arange(x2_min, x2_max, delta)
        x_grid, y_grid = np.meshgrid(x, y)
        coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T
    
        # prepare a list of colors
        color = ['aliceblue',  'lightgreen', 'yellow', 'slategrey', 'chocolate', 'salmon', 
                 'pink', 'darkorange', 'darkcyan',  'tomato']
        
        # plot each cluster
        for i in range(self.k):
            mean = self.mu[i]
            cov = self.cov[i]
            z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
            plt.contour(x_grid, y_grid, z_grid, colors = color[i])
            
            # plot centroid (mu) of each cluster
            plt.scatter(self.mu[i][0], self.mu[i][1], s = 250, c = 'black', label = 'centroids')
            
        plt.tight_layout()
        
        
    def init_glance(self, data):
        # Generates a scatter plot of the data
        plt.figure(figsize=(10,10))
        plt.scatter(data[:,0], data[:,1], c = 'black')
        plt.title('Initial scatter plot', fontsize = 15)
        plt.show()
        
    def gmm(self, data):
        # define stopping criteria
        cur_iter = 0
        
        delta = 50
        log_likelihood = 0
        
        # Initialization
        self.initialization(data)
        
        # initial plot for 2 dimensional dataset
        if self.column == 2:
            self.init_glance(data)
            
        # loop over iterations
        while cur_iter < self.max_iter and delta > self.precision:
            likelihood, labels = self.e_step(data)
            self.m_step(data)
            
            # update
            cur_iter += 1
            delta = abs(np.sum(np.log(np.sum(likelihood))) - log_likelihood)
            log_likelihood = np.sum(np.log(np.sum(likelihood)))
            print('Iteration = ', cur_iter, ', delta = ', delta)
            
            # visualization if there are only 2 features in the dataset
            if self.column == 2:
                print('There will be visualization for this dataset because it is 2 dimensional.')
                self.visualization(data)
            else:
                print('There will be NO visualization for this dataset because it is NOT 2 dimensional.')
        return labels
        print('Final mu = ', self.mu, 'Final cov = ', self.cov)


    
#################### Unit test ####################

# -------- 1 --------
# Let's use the dataset from HW6 as a simple example
# This has two features, so it will be a two dimensional problem
clustering_gmm = pd.read_csv('clustering_gmm - Copy.csv')
gmm_array = clustering_gmm.to_numpy()

# check shape
gmm_array.shape

data_test1 = gaussian_mixture_model(k = 4, max_iter = 100, precision = 1e-06)
data_test1.gmm(gmm_array)

# Test it with black-box method
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(gmm_array)

# View label of the predictions of the mixture model
labels = gmm.predict(gmm_array)
frame = pd.DataFrame(gmm_array)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

# to visualize
plt.figure(figsize=(10,10))
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],cmap = 'viridis')
plt.title('Clustering using Mixture Models', fontsize = 15)
plt.show()

# -------- 2 --------
from sklearn.datasets import load_iris
iris = load_iris()
iris_data = iris.data

data_test2 = gaussian_mixture_model(k = 3, max_iter = 100, precision = 1e-06)
data_test2.gmm(iris_data)

# Compare it with black-box method
gmm = GaussianMixture(n_components=3)
gmm.fit(iris_data)

# View label of the predictions of the mixture model
gmm.predict(iris_data)

# -------- 3 --------
# Let's use datasets from previous homework
faith = pd.read_csv('faithful.csv')
faith = faith.drop(columns = 'Unnamed: 0')
faith_array = faith.to_numpy()

data_test3 = gaussian_mixture_model(k = 2, max_iter = 100, precision = 1e-06)
data_test3.gmm(faith_array)

# Compare it with black-box method
gmm = GaussianMixture(n_components=2)
gmm.fit(faith_array)

# View label of the predictions of the mixture model
gmm.predict(faith_array)

# -------- 4 --------
# Let's consider image compression question again from HW6
from sklearn.datasets import load_sample_image 
flower = load_sample_image("flower.jpg")        
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(flower);

# check dimension
flower.shape

# change to 2-dimensional
data = flower / 255.0
data = data.reshape(427 * 640, 3)
data.shape

# Define a plot_pixels function 
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # Choose a random subset of pixels 

    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    # Define the axes and markets on the figure
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);
    
def show_image(new_group):
    # Change format
    if type(new_group) == dict:
        image_group = []
        for i in range(len(new_group)):
            new = new_group[i+1].tolist()
            for j in new:
                image_group.append(j)
        image_group = np.array(image_group)
    else:
        image_group = new_group
    
    # Defines the recolored image
    flower_recolored = image_group.reshape(flower.shape) 
    
    # Plots the original image and the recolored image for comparison
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(flower)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(flower_recolored)
    ax[1].set_title('16-color Image', size=16);

plot_pixels(data, title = 'Full input color space')

# If we were to perform GMM directly
data_test4 = gaussian_mixture_model(k = 16, max_iter = 100, precision = 1e-06)
data_test4.gmm(data)
'''
Final mu =  [array([0.02726945, 0.1914294 , 0.11789879]), 
             array([0.14916919, 0.23001353, 0.13038068]), 
             array([0.77044763, 0.4598379 , 0.25971945]), 
             array([0.04534844, 0.16765133, 0.15183144]), 
             array([0.29549306, 0.40257805, 0.13626122]), 
             array([0.6373358 , 0.47063362, 0.24472877]), 
             array([0.00089219, 0.27227342, 0.27335278]), 
             array([0.51568163, 0.39574115, 0.26170874]), 
             array([0.01556837, 0.09274297, 0.07794336]), 
             array([0.12332625, 0.16108205, 0.15677329]), 
             array([0.00632565, 0.25481939, 0.27201095]), 
             array([0.87987632, 0.62020233, 0.38399632]), 
             array([0.00420577, 0.25016442, 0.20315239]), 
             array([0.01769805, 0.16923598, 0.15854881]), 
             array([0.07617526, 0.1699496 , 0.14188155]), 
             array([0.69336919, 0.12912038, 0.01293348])]
'''
