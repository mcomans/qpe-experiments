import re
import datetime
import pandas as pd
import numpy as np

BIGDL_DATE_REGEX = r'(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)'

results = pd.read_csv('results_birnn_phase_2.csv')
if 'duration' not in results:
    results['duration'] = np.nan

for exp in range(9):
    for rep in range(5):
        filename = f'exp{exp+1}-rep{rep+1}.log'
        with open(filename) as f:
            lines = f.read().splitlines()
            start = datetime.datetime.strptime(re.search(BIGDL_DATE_REGEX, lines[0]).group(), '%Y-%m-%d %H:%M:%S')
            end = datetime.datetime.strptime(re.search(BIGDL_DATE_REGEX, lines[-3]).group(), '%Y-%m-%d %H:%M:%S')

            duration = (end-start).total_seconds()
            results.loc[(results['exp'] == exp+1) & (results['replication'] == rep+1), 'duration'] = duration

results.to_csv('results_birnn_phase_2_duration.csv', index=False)
        