import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib

################### Problem 1 ###################
"""
Visualize the observations in faithful.csv using a scatter plot. It contains two features: oneof the features corresponds to the waiting time between eruptions and the second featurecorresponds to the duration of each eruption at the Old Faithful geyser in YellowstoneNational Park in Wyoming. It contains 272 observations on 2 variables.Implement the 𝑘-means clustering algorithm from scratch and use it to cluster the data.Check for various values of 𝑘 and visually identify the appropriate cluster size.
""" 

faith = pd.read_csv('faithful.csv')
faith = faith.drop(columns = 'Unnamed: 0')
# Scatter plot
plt.scatter(faith['eruptions'], faith['waiting'], label = 'initial data')
plt.xlabel('eruptions')
plt.ylabel('waiting')
plt.legend()
plt.title('Scatter plot of initial data')
plt.show()

# Implementing K-means clustering algorithm from scratch

faith_array = faith.to_numpy()

# A function to initialize centroids
def init_centroid(k, data, n, d):
    # Initialize random k centroids by creating a d * k matrix
    # each column is a centroid
    centroid = np.array([]).reshape(d, 0)
    for i in range(k):
        rand = random.randint(0, n-1)
        centroid = np.c_[centroid, data[rand]]
    return centroid

# A function to calculate euclidean distance
def euclidean_distance(k, centroid, data, n):
    # Create a n * k matrix
    # Each row is an observation, with each column represents 
    # the distance of this observation to each centroid
    eu = np.array([]).reshape(n, 0)
    for cluster in range(k):
        # Euclidean distance d(p,q) = (sum^{n}_{i=1}(q[i] - p[i])**2)**0.5
        eu_distance = np.sum((data - centroid[:, cluster])**2, axis = 1)
        
        # append it to the array
        eu = np.c_[eu, eu_distance]
        
    # show the index of minimum distance of each observation to a centroid
    eu_min = np.argmin(eu, axis = 1) + 1
    return eu, eu_min

# A function to regroup data points to closest centroid,
# and re-calculate the centroid point of new groups
def regroup(k, eu_min, data, centroid, n, d):
    group = {}
    # This new_group_centroid variable stores the nearest centroid point for each datapoint
    # And it will be used for the image compression problem for visualization
    new_group_centroid = data.copy()
    # to create empty array at specific index
    for i in range(k):
        group[i + 1] = np.array([]).reshape(d,0)
    # to insert data point for the lowest distance
    for j in range(n):
        group[eu_min[j]] = np.c_[group[eu_min[j]], data[j]]
        new_group_centroid[j] = centroid[:,eu_min[j] -  1]
    # transpose array for each cluster
    for cluster in range(k):
        group[cluster+1] = group[cluster+1].T
        new_group_centroid[cluster+1] = new_group_centroid[cluster+1].T
    # calculate the new centroid value for each cluster
    for cluster in range(k):
        centroid[:, cluster] = np.mean(group[cluster+1], axis = 0)
    return group, new_group_centroid, centroid
   

def visual(k, data, group, centroid):
    color = []
    for name, hex in matplotlib.colors.cnames.items():
        color.append(name)
    random.shuffle(color)
    label = []
    for i in range(1, 101):
        label.append('cluster ' + str(i))
    for i in range(k):
        # plot data points with different color
        plt.scatter(group[i+1][:,0], group[i+1][:,1], c = color[i], label = label[i])
    # plot centroids
    plt.scatter(centroid[0,:], centroid[1,:], s = 250, c = 'grey', label = 'centroids')
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    plt.legend()
    plt.title('Scatter plot of regrouped data points')
    plt.show()
    
    
def iterate(k, data, n, d):
    # define stopping criteria
    change = 0.00001 # 1. observation probabilities change threshold
    max_iter = 20 # 2. maximum iterations
    delta = 1
    iteration = 1
    
    # initialize centroids
    centroid = init_centroid(k, data, n, d)
    
    while delta > change and iteration < max_iter:
        # calculate distance to centroid
        eu, eu_min = euclidean_distance(k, centroid, data, n)
        # regroup data to nearest centroid, re-calclate centroid
        new_group, new_group_centroid, new_centroid = regroup(k, eu_min, data, centroid, n, d)
        
        # delta is the average percent change in centroid 
        delta = np.mean((new_centroid - centroid)/centroid)
        
        # show visualization of regrouped data point
        visual(k, data, new_group, new_centroid)
        
        print('Iteration = ', iteration, ', delta = ', delta, ', centroid = ', new_centroid)
        
        # update
        iteration += 1
        centroid = new_centroid
    print('The final centroids are', centroid)
    return new_group, new_group_centroid, centroid


# Examine k-mean clustering for different values of k
n = len(faith) # observation size = 272
d = 2 # number of features = 2, ie eruptions and waiting

# k = 2
iterate(2, faith_array, 272, 2) # [[ 4.18212973  2.01129885], [79.17837838 53.28735632]]
    
# k = 3
iterate(3, faith_array, 272, 2) # [[ 4.1765974   2.06631959  4.35353061], [74.81818182 54.39175258 84.15306122]]
  
# k = 4
iterate(4, faith_array, 272, 2)  # [[ 4.793625    4.26358088  2.10659406  4.17715789], [88.875      78.28676471 54.88118812 88.        ]]
    
# k = 5
iterate(5, faith_array, 272, 2) # [[ 2.26145238  4.2914359   4.38736667  1.99635593  4.13528571], [60.83333333 78.05128205 84.58888889 50.6440678  73.42857143]]
    
# since the initial centroids are created at random,
# the clusters is slightly different at each run.
# The appropriate cluster size is 2 because there isn't a huge
# difference between each run. On the other hand,
# when the cluster size is 3, 4, or 5, there could be large differences.

# Reference used for this problem:
# https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42


################### Problem 2 ###################
"""
Go over the Jupyter notebook (clustering.ipynb) provided to you. It containsexplanations and applications of some clustering methods. It also contains an applicationof the 𝑘-means clustering algorithm to an image compression problem (Section 2, AnotherApplication of 𝐾-means clustering). In the Jupyter notebook provided to you, a black-box(mini-batch) k-means algorithm is used for compressing image. Using your 𝑘-means fromscratch implementation from Problem 1, re-perform the same clustering approach toobtain a compressed image. If your implementation fails for this problem, describe why.
"""

# Import the image
import seaborn as sns; sns.set()  # For plotting
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

# Run k-means on image compression problem
# k = 16
new_group16, new_group_centroid16, centroid16 = iterate(16, data, 273280, 3)

'''
The final centroids are [[0.00879707 0.00223799 0.00220779 0.00397391 0.89861092 0.84255758
  0.27239139 0.00202363 0.04066847 0.01509673 0.01102911 0.00391502
  0.29540173 0.03382762 0.74122302 0.02681359]
 [0.06359159 0.29244463 0.33223175 0.2278384  0.76148862 0.58192423
  0.31617455 0.32125897 0.17660379 0.22417841 0.20977707 0.2837622
  0.25324595 0.11199334 0.21700073 0.15535119]
 [0.04167131 0.29943659 0.36732664 0.25542719 0.57064794 0.34637052
  0.24170704 0.31963981 0.17193622 0.14608122 0.20352709 0.24796412
  0.08215145 0.08051795 0.05196255 0.11499755]]
'''

def show_image(new_group, k):
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
    ax[1].set_title(str(k)+'-color Image', size=16);

# Show image for implementation from scratch
show_image(new_group_centroid16, 16)

# Try black-box method
from sklearn.cluster import MiniBatchKMeans

# Perform K-means with K = 16 
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

# Show image for black box method
show_image(new_colors, 16)

# k = 32
new_group32, new_group_centroid32, centroid32 = iterate(32, data, 273280, 3)

show_image(new_group_centroid32, 32)

# k = 64
new_group64, new_group_centroid64, centroid64 = iterate(64, data, 273280, 3)

show_image(new_group_centroid64, 64)

# Overall, the implementation was excellent, colors can be clustered.
# The output of 16/32/64 clusters was shown above.
# And from the image, we can see although the color of the follower isn't
# as great as the original image, we can still see the basic structure of the flower.
# As we increase the k, the image will be more clear.

