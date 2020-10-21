import pandas as pd

df = pd.read_csv('qpe-experiments-lenet-phase1/results_lenet_phase_1.csv')

print(df.groupby(['learning_rate']).mean()[['accuracy']].reset_index())
print(df.groupby(['batch_size']).mean()[['accuracy']].reset_index())
print(df.groupby(['epochs']).mean()[['accuracy']].reset_index())
print(df.groupby(['train_size']).mean()[['accuracy']].reset_index())
