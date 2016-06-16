import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import numpy as np

data = OrderedDict({})
month_dict = OrderedDict({'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
                          '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'})

with open('HadSST.3.1.1.0_monthly_globe_ts.txt', 'r') as f:
    for row in f:
        parts = row.strip().split()
        temp = float(parts[1])

        year, month = parts[0].strip().split('/')
        print(year, month, temp)

        if year not in data.keys():
            data[year] = {}

        data[year][month] = temp

temp_df = pd.DataFrame(data)
print(temp_df)
ax = sns.heatmap(temp_df)
ax.set_xticklabels(labels=data, rotation=30, size=8)
plt.draw()
plt.show()

