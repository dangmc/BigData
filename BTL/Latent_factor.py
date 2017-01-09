import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

path = '/home/dangmc/Documents/Learning/20161/BigData/BTL/Data/ml-100k/'

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(path + 'u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

train_data, test_data = cv.train_test_split(df, test_size=0.25)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
average_rating = 0;

# Create training and test matrix
R = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    R[line[1]-1, line[2]-1] = line[3]
    average_rating += line[3]


T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    T[line[1]-1, line[2]-1] = line[3]

I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0
average_rating /= np.sum(I)
print 'average_rating = %f' %(average_rating)
# Predict the unknown ratings through the dot product of the latent features for users and items
def prediction(P, Q, user, item):
    return np.dot(P.T, Q) \
           + Bu[user] + Bi[item] + average_rating

lmbda = 0.1 # Regularisation weight
k = 40  # Dimension of the latent feature space
m, n = R.shape  # Number of users and items
n_epochs = 200  # Number of epochs
gamma = 0.002  # Learning rate

P = np.random.rand(k, m) # Latent user feature matrix
Q = np.random.rand(k, n) # Latent movie feature matrix
Bu = np.zeros(m) # Bias for users
Bi = np.zeros(n) # Bias for movies

# Calculate the RMSE
"""
def compute_rmse(E, Q, P):
    sum = 0
    for u in xrange(m):
        for i in xrange(n):
            if (E[u, i] > 0):
                sum += (E[u, i] - prediction(P[:, u], Q[:, i], u, i)) ** 2
    return np.sqrt(sum / len(R[R > 0]))
"""
def prediction_matrix(R, P, Q):
    E = R.copy();

    for u in xrange(m):
        for i in xrange(n):
            E[u, i] -= prediction(P[:, u], Q[:, i], u, i)
    return E

# Calculate the RMSE
def rmse(I, R, Q, P):
    return np.sqrt(np.sum((I * prediction_matrix(R, P, Q))**2)/len(R[R > 0]))
train_errors = []
test_errors = []

#Only consider non-zero matrix
users, items = R.nonzero()
for epoch in xrange(n_epochs):
    for u, i in zip(users, items):
        e = R[u, i] - prediction(P[:, u], Q[:, i], u, i)  # Calculate error for gradient
        P[:, u] += gamma * (e * Q[:, i] - lmbda * P[:, u]) # Update latent user feature matrix
        Q[:, i] += gamma * (e * P[:, u] - lmbda * Q[:, i])  # Update latent movie feature matrix
        Bu[u] += gamma * (e - lmbda * Bu[u]) # Update bias for users
        Bi[i] += gamma * (e - lmbda * Bi[i]) # Update bias for movies
    train_rmse = rmse(I, R, Q, P) # Calculate root mean squared error from train dataset
    test_rmse = rmse(I2, T, Q, P) # Calculate root mean squared error from test dataset
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    print 'epoch = %d' %(epoch)
    print 'Train RMSE = %f' %(train_rmse)
    print 'Test RMSE = %f' %(test_rmse)
# Check performance by plotting train and test errors

plt.plot(range(n_epochs), train_errors, marker='.', label='Training Data');
plt.plot(range(n_epochs), test_errors, marker='.', label='Test Data');
plt.title('Latent factor - ML100k')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()
