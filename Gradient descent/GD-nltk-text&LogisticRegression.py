############################## Problem 1 ##############################
# Pseudo-code for a stochastic gradient descent algorithm
def stochastic_gradient_descent(x, y, n, d, learning_rate, lambd, l, dl_x, dl_y):
    '''
    x[i, j]: d-dimensional input vector (d features)
    y[i]: outcome variable for observation i
    n: total number of observations
    d: the number of features in the dataset
    learning_rate: just the learning rate for the gradient descent
    lambd: a scalar constant
    l: loss function for this question
    dl_x: gradient of l to x
    dl_y: gradient of l to y
    '''
    import random
    for iter in range(10000):
        i = random.randint(0,n)
        x_new = x - learning_rate * dl_x((x[i, j], y[i]) for j in range(d))
        y_new = y - learning_rate * dl_y((x[i, j], y[i]) for j in range(d))
        print("Iteration", iter, "\nx = ", x_new, "\ny = ", y_new, " and loss(x, y) = ", l(x_new, y_new))
        x = x_new
        y = y_new
    return x_new, y_new, l(x_new, y_new)


############################## Problem 2 ##############################
"""
From the nltk2 corpus (or elsewhere), import a dataset containing tweets that are prelabeledand classified as denoting a positive sentiment or a negative sentiment. Afterappropriately segregating the dataset into a training and test set, you will build a logisticregression sentiment classifier. Feel free to explore and create your own features whichcan be used as inputs to the logistic regression model. Feel free to also explore appropriatepre-processing steps to handle the tweet dataset.
"""

# i)
"""
clearly explain what features you areusing in your logistic regression model, and write down the model mathematically
"""

# This will follow the general guideline on 
# https://medium.com/swlh/sentiment-analysis-from-scratch-with-logistic-regression-ca6f119256ab

# Data import
import nltk 
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples

nltk.download('twitter_samples') # download required dataset

posi_tweets = twitter_samples.strings('positive_tweets.json')
nega_tweets = twitter_samples.strings('negative_tweets.json')

num_posi = len(posi_tweets) # 5000 observations 
num_nega = len(nega_tweets) # 5000 observations 

# concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = posi_tweets + nega_tweets

# Split the 5000 tweet dataset into 4000 training and 1000 testing datasets,
# for both positive and negative
test_pos = posi_tweets[4000:]
train_pos = posi_tweets[:4000]
test_neg = nega_tweets[4000:]
train_neg = nega_tweets[:4000]

# Combine them to have 8000 training and 2000 testing sets
train_x = train_pos + train_neg 
test_x = test_pos + test_neg
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# Text preprocessing and cleaning
import re                                  
import string
from nltk.corpus import stopwords          
from nltk.stem.wordnet import WordNetLemmatizer   
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@', '', tweet)
    tokenizer = TweetTokenizer()
    tweet_tokenized = tokenizer.tokenize(tweet)
    stopwords_english = stopwords.words('english') 
    tweet_processsed=[word for word in tweet_tokenized if word not in 
                      stopwords_english and word not in string.punctuation]
    lemma = WordNetLemmatizer()
    tweet_after_stem=[]
    for word in tweet_processsed:
        word=lemma.lemmatize(word)
        tweet_after_stem.append(word)
    return tweet_after_stem

pos_words=[]
for tweet in posi_tweets:
    tweet=process_tweet(tweet)
    for word in tweet:
        pos_words.append(word)
        
neg_words=[]
for tweet in nega_tweets:
    tweet=process_tweet(tweet)
    for word in tweet:
        neg_words.append(word)
        
# Feature selection (create a dictionary with the frequency count of each word)      
freq_pos={}
for word in pos_words:
    if (word,1) not in freq_pos:
        freq_pos[(word,1)]=1
    else:
        freq_pos[(word,1)]=freq_pos[(word,1)]+1
        
freq_neg={}
for word in neg_words:
    if (word,0) not in freq_neg:
        freq_neg[(word,0)]=1
    else:
        freq_neg[(word,0)]=freq_neg[(word,0)]+1
        
# Combine to have one united dictionary
freqs_dict = dict(freq_pos)
freqs_dict.update(freq_neg)

# Create features for logistic regression
def features_extraction(tweet, freqs_dict):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0,0] = 1 
    for word in word_l:
        try:
            x[0,1] += freqs_dict[(word,1)]
        except:
            x[0,1] += 0
        try: 
            x[0,2] += freqs_dict[(word,0)]
        except:
            x[0,2] += 0
    assert(x.shape == (1, 3))
    return x

# columns of X:
    # 0: 1
    # 1: frequency of words in positive class
    # 2: frequency of words in negative class

# Preprocess train_x
train_x_processed = np.zeros((len(train_x), 3)) # size of X: (8000, 3)
for i in range(len(train_x)):
    train_x_processed[i, :]= features_extraction(train_x[i], freqs_dict)

# Preprocess test_x
test_x_processed = np.zeros((len(test_x), 3)) # size of X: (2000, 3)
for i in range(len(test_x)):
    test_x_processed[i, :]= features_extraction(test_x[i], freqs_dict)



# i)
# There will be two features in the logistic regression:
# freq_posi and freq_nega.
# The first feature is the frequency of words in the positive class.
# For instance, the first tweet in training dataset is:
# #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)'
# Then, the first feature is 3764, meaning the frequency count of positive sentiment
# words in this tweet is 3764. 
# Next, the second feature is the frequency of words in the negative class.
# Similarly to the above example, the second feature is 67, meaning the frequency
# count of negative sentiment words in this tweet is 67. 

# The basic formula for logistic regression is:
# y = sum_{i = 1}^{n}beta_i*x_i + beta_0
# and using the sigmoid function we have
# p(y) = 1/(1+exp(-y))

# The formula in this case would be:
# z(beta) = beta_0 + beta_1 * freq_posi(x_1) + beta_2 * freq_nega(x_2)
# p(z) = 1/(1+exp(-(beta_0 + beta_1 * freq_posi(x_1) + beta_2 * freq_nega(x_2))))

# Notice: Since typing in comment might not be clear,
# the above formulation will be hand written in the PDF document as well to show more clarity.


# ii) PLEASE IGNORE THIS PART
# The formula is written in the PDF document

# iii)
"""
Train the logistic regression classifier using a black-box implementation and evaluateits performance on the test dataset.
"""

# Using black-box method to estimate beta
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(train_x_processed, train_y)

# Check coefficients
logistic.coef_  # [-2.69308309e-08,  8.67133384e-03, -5.74259215e-03]

# Accuracy score
predictions = logistic.predict(test_x_processed)
score = logistic.score(test_x_processed, test_y)
print(score) # The accuracy score is 0.9965

# MSE
from sklearn.metrics import mean_squared_error
mean_squared_error(test_y, predictions) # 0.0035

# Using the output from the sklearn black box implementation, the model is
# y = -2.69e-08 + 8.67e-03*freq_posi - 5.74e-03*freq_nega
# The performance of the model on the test dataset is very good
# with a accuracy score of 0.9965 and a very small MSE value of 0.0035.

# iv)
"""
Also train the logistic regression classifier by minimizing the negative log-likelihoodfunction using a numerical optimization procedure: gradient descent or stochastic gradientdescent. Compare with the coefficients obtained in step (iii).
"""

# Since the dataset is not very large, I will use gradient descent method 
# instead of stochastic gradient descent
def sigmoid_fuction(z): 
    sigma = 1/(1+ np.exp(-z))
    return sigma

def gradientDescent_algo(x, y, theta, delta, num_iters):
    m = x.shape[0] # length of x = 3; ie. the number of observations
    for i in range(num_iters):
        x_scalar = np.dot(x,theta) # theta = np.zeros((3, 1)), it is a 3*1 matrix to transform x to a scalar
        sigma = sigmoid_fuction(x_scalar)
        J = -1 / m * (np.dot(y.T, np.log(sigma)) + np.dot((1-y).T, np.log(1-sigma)))
        theta = theta - (delta/m) * np.dot(x.T, sigma-y)
    J = float(J)
    return J, theta

J, theta = gradientDescent_algo(train_x_processed, train_y, np.zeros((3, 1)), 1e-9, 1000)
# J = 0.20097047
# theta = [[ 8.22843373e-08], [ 5.58021321e-04], [-5.16294032e-04]]

# theta is the coefficients for beta in the negative likelihood function.
# By comparing the coefficients, there is a big difference in beta 0, 
# where beta0 is -2.59e-8 in black box method while here it is 8.22e-8. 
# beta 1 and beta 2 are rather similar, where beta 1 was 8.67e-3 above and
# 5.58e-4 here. And beta 2 is -5.74e-3 above and -5.16e-4 here. 
# The reason that there is such a difference is because we manually set delta for
# the gradient descent algorithm, where delta is the learning rate. 
# If I change the learning rate, the results will be different. 
# The number of iteration could make an impact as well. 




