import pandas as pd
from statistics import mean, stdev
import numpy as np
from numpy.random import default_rng
rng = default_rng(10)

# LeNet Simulation

# Service time prediction
predict_service_time = np.poly1d([2497.02, 65.05])

# Train set prediction
mu = 1.010371
w_lr = -0.003141
w_ts = -0.062014
w_lr_ts = -0.030613

def predict_train_size_lenet(acc, lr):
    return np.exp(
        (-w_lr*np.log(lr) - mu + acc)/
        (w_lr_ts*np.log(lr) + w_ts)
    )

def rand_service_time(size=None):
    lr = rng.uniform(0.001, 0.1, size=size)
    accuracy = rng.uniform(0.5, 0.95, size=size)

    ts = predict_train_size_lenet(accuracy, lr)
    return predict_service_time(ts)

LAMBDA = 1.0/1300

A = [rng.exponential(1.0/LAMBDA)]
S = [A[0]]
C = [S[0] + rand_service_time()]
response_time = [C[0] - A[0]]
waiting_time = [S[0] - A[0]]

for i in range(1, 10000):
    A.append(A[i-1] + rng.exponential(1.0/LAMBDA))
    S.append(max(C[i-1], A[i]))

    st = rand_service_time()
    C.append(S[i] + st)
    response_time.append(C[i] - A[i])
    waiting_time.append(S[i] - A[i])
    
print(f"Mean waiting time: {mean(waiting_time)}")
print(f"Mean response time: {mean(response_time)}")

