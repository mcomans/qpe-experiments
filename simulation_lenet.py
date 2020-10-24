import pandas as pd
from statistics import mean, stdev
import numpy as np
from numpy.random import default_rng
rng = default_rng(10)

ITERATIONS = 100000
LAMBDAS = np.arange(0.0001, 1.0/120, 0.0004)

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

with open("results_lenet_simulation.csv", "w") as f:
    f.write("lambda,mean_service_time,mean_waiting_time,mean_response_time\n")

for l in LAMBDAS:
    s = rand_service_time(ITERATIONS)
    print(f"Mean service time: {mean(s)}\n")

    A = [rng.exponential(1.0 / l)]
    S = [A[0]]
    C = [S[0] + s[0]]
    response_time = [C[0] - A[0]]
    waiting_time = [S[0] - A[0]]

    for i in range(1, ITERATIONS):
        A.append(A[i-1] + rng.exponential(1.0/l))
        S.append(max(C[i-1], A[i]))

        st = s[i]
        C.append(S[i] + st)
        response_time.append(C[i] - A[i])
        waiting_time.append(S[i] - A[i])

    print(f"Mean waiting time: {mean(waiting_time)}")
    print(f"Mean response time: {mean(response_time)}")

    with open("results_lenet_simulation.csv", "a") as f:
        f.write(
            f"{l},{mean(s)},{mean(waiting_time)},{mean(response_time)} \n")

