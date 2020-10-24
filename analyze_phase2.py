import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

figure = plt.figure()

df = pd.read_csv("results_birnn_phase_2_duration.csv")
df = df.sort_values(by=['train_size'])

plt.scatter(df['train_size'], df['duration'])

z = np.polyfit(df['train_size'], df['duration'], 1)
p = np.poly1d(z)
plt.plot(df['train_size'],p(df['train_size']),"r--")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Training set size")
plt.ylabel("Duration (s)")
plt.show()
figure.savefig("brnn_phase2_duration.pdf")