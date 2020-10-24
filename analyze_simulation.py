import matplotlib.pyplot as plt
import pandas as pd

figure = plt.figure(figsize=(4, 4))

lenet = pd.read_csv("results_lenet_simulation.csv")
brnn = pd.read_csv("results_brnn_simulation.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3))
ax1.set_title('LeNet5')
ax1.set_yscale('log')
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Lambda')
ax1.plot(lenet['lambda'], lenet['mean_response_time'], label='Mean response time')
ax1.plot(lenet['lambda'], lenet['mean_waiting_time'], 'r', label='Mean waiting time')
ax1.legend()

ax2.set_title('BRNN')
ax2.set_yscale('log')
ax2.set_ylabel('Time (s)')
ax2.set_xlabel('Lambda')
ax2.plot(brnn['lambda'], brnn['mean_response_time'], label='Mean response time')
ax2.plot(brnn['lambda'], brnn['mean_waiting_time'], 'r', label='Mean waiting time')
ax2.legend()

plt.show()
fig.savefig("simulation_results.pdf")