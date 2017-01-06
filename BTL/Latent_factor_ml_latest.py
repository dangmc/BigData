import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt
import csv

path = '/home/dangmc/Documents/Learning/20161/BigData/BTL/Data/ml-latest/'

n_user = 259137
n_item = 40110

average_rating = 0
"""
train_user = []
train_item = []
train_rating = []

test_user = []
test_item = []
test_rating = []

with open(path + 'test.csv') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        test_user.append(row['userId'])
        test_item.append(row['movieId'])
        test_rating.append(row['rating'])
"""
with open(path + 'train.csv') as csv_file:
    reader = csv.DictReader(csv_file)
    line = 0
    for row in reader:
        #train_user.append(row['userId'])
        #train_item.append(row['movieId'])
        #train_rating.append(row['rating'])
        average_rating += float(row['rating'])
        line += 1
    average_rating /= line

print 'average rating = ' + str(average_rating)

lmbda = 0.1 # Regularisation weight
k = 40  # Dimension of the latent feature space
m = int(n_user)
n = int(n_item)  # Number of users and items
n_epochs = 60  # Number of epochs
gamma = 0.0002  # Learning rate

P = np.random.rand(k, m) # Latent user feature matrix
Q = np.random.rand(k, n) # Latent movie feature matrix
Bu = np.zeros(m) # Bias for users
Bi = np.zeros(n) # Bias for movies


# Predict the unknown ratings through the dot product of the latent features for users and items
def prediction(P, Q, user, item):
    return np.dot(P.T, Q) + Bu[user] + Bi[item] + average_rating

# Calculate the RMSE
def rmse(filename, Q, P):
    sum = 0
    line = 0
    with open(path + filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if (line % 1000000 == 0):
                print str(line) + " " + 'Loading...'
            line += 1
            u = int(row['userId'])
            i = int(row['movieId'])
            rating = float(row['rating'])
            sum += (rating - prediction(P[:, u], Q[:, i], u, i)) ** 2
    return np.sqrt(sum / line)

train_errors = []
test_errors = []

for epoch in xrange(n_epochs):
    if (epoch > 40):
        gamma = 0.00002;
    with open(path + 'train.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        line = 0
        for row in reader:
            if (line % 1000000 == 0):
                print str(line) + " " + 'Loading...'
            line += 1
            u = int(row['userId'])
            i = int(row['movieId'])
            rating = float(row['rating'])
            e = rating - prediction(P[:, u], Q[:, i], u, i)  # Calculate error for gradient
            P[:, u] += gamma * (e * Q[:, i] - lmbda * P[:, u]) # Update latent user feature matrix
            Q[:, i] += gamma * (e * P[:, u] - lmbda * Q[:, i])  # Update latent movie feature matrix
            Bu[u] -= gamma * (e + lmbda * Bu[u]) # Update bias for users
            Bi[i] -= gamma * (e + lmbda * Bi[i]) # Update bias for movies
    train_rmse = rmse('train.csv', Q, P) # Calculate root mean squared error from train dataset
    test_rmse = rmse('test.csv', Q, P) # Calculate root mean squared error from test dataset
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    print 'epoch = %d' %(epoch)
    print 'Train RMSE = %f' %(train_rmse)
    print 'Test RMSE = %f' %(test_rmse)
# Check performance by plotting train and test errors

plt.plot(range(n_epochs), train_errors, marker='.', label='Training Data');
plt.plot(range(n_epochs), test_errors, marker='.', label='Test Data');
plt.title('Latent Factor K = 50')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()
"""
# Calculate prediction matrix R_hat (low-rank approximation for R)
R = pd.DataFrame(R)
R_hat=pd.DataFrame(prediction(P,Q))

# Compare true ratings of user 17 with predictions
ratings = pd.DataFrame(data=R.loc[16,R.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16,R.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']
ratings
"""