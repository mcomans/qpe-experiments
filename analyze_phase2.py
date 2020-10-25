import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

figure = plt.figure(figsize=(4, 4))

lenet = pd.read_csv("results_lenet_queue.csv")
lenet = lenet.sort_values(by=['train_size'])
brnn = pd.read_csv("results_birnn_phase_2_duration.csv")
brnn = brnn.sort_values(by=['train_size'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3))
ax1.set_title('LeNet5')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim([min(lenet['train_size']), max(lenet['train_size'])])
ax1.set_ylim([min(lenet['duration']), max(lenet['duration'])])
ax1.set_ylabel('Duration (s)')
ax1.set_xlabel('Training set size')
ax1.scatter(lenet['train_size'], lenet['duration'])
z = np.polyfit(lenet['train_size'], lenet['duration'], 1)
p = np.poly1d(z)
x = np.linspace(min(lenet['train_size']), max(lenet['train_size']), 500)
ax1.plot(x,p(x),"r--")

ax2.set_title('BRNN')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlim([min(brnn['train_size']), max(brnn['train_size'])])
ax2.set_ylim([min(brnn['duration']), max(brnn['duration'])])
ax2.set_ylabel('Duration (s)')
ax2.set_xlabel('Training set size')
ax2.scatter(brnn['train_size'], brnn['duration'])
z = np.polyfit(brnn['train_size'], brnn['duration'], 1)
p = np.poly1d(z)
x = np.linspace(min(brnn['train_size']), max(brnn['train_size']), 500)
ax2.plot(x,p(x),"r--")

plt.show()
fig.savefig("phase2_duration.pdf")