import matplotlib.pyplot as plt;
import numpy as np

n_buckets = 10**4
p = 123457
num_hashfun = 5
word_stream_file = '/home/dangmc/Documents/Learning/20161/BigData/Assignment 4/HW4-q4/HW4-q4/words_stream.txt'
hash_params_file = '/home/dangmc/Documents/Learning/20161/BigData/Assignment 4/HW4-q4/HW4-q4/hash_params.txt'
counts_file = '/home/dangmc/Documents/Learning/20161/BigData/Assignment 4/HW4-q4/HW4-q4/counts.txt'

def     hash_function(a, b, x):
    y = x % p;
    hashval = (a*y + b) % p;
    return hashval % n_buckets

# Read hash_params file
a = []
b = []
hash_params_fo = open(hash_params_file, "rw+")
for line in hash_params_fo:
    params = line.split()
    a.append(params[0])
    b.append(params[1])
# Compute
word_stream_fo = open(word_stream_file, "rw+")
t = 0
c = np.ndarray(shape=(num_hashfun, n_buckets), dtype=np.int32)
for word_id in word_stream_fo:
    t += 1
    for i in xrange(num_hashfun):
        hash_val = hash_function(int(a[i]), int(b[i]), int(word_id))
        c[i][hash_val] += 1

# Compute Error
x = []
y = []
counts_fo = open(counts_file, "rw+")
for line in counts_fo:
    count = line.split()
    approximate = t
    for i in xrange(num_hashfun):
        hash_val = hash_function(int(a[i]), int(b[i]), int(count[0]))
        approximate = min(approximate, c[i][hash_val])
    error = (float(approximate - int(count[1])) / int (count[1]))
    y.append(error)
    x.append(float(count[1]) / t)
plt.plot(x, y, "b.");
plt.xscale("log");
plt.yscale("log");
plt.xlabel("x");
plt.ylabel("y");
plt.show();
