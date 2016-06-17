"""
Creates a heatmap on the basis of the data set  from http://hadobs.metoffice.com/hadsst3/ containing inf about anomalies
in sea surface temperature.

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {}

with open('input/HadSST.3.1.1.0_monthly_globe_ts.txt', 'r') as f:
    for row in f:
        parts = row.strip().split()
        temp = float(parts[1])

        year, month = parts[0].strip().split('/')
        print(year, month, temp)

        if year not in data.keys():
            data[year] = {}

        data[year][month] = temp


temp_df = pd.DataFrame(data)

ax = sns.heatmap(temp_df)
ax.set_xticklabels(labels=[str(i) if i % 50 == 0 else '' for i in range(1850, 2015)], rotation=30, size=8)
ax.set_yticklabels(labels=['Dec', 'Nov', 'Oct', 'Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr', 'Mar', 'Feb', 'Jan'],
                   rotation=0, size=8)
ax.set_title('Sea surface temperature anomalies (monthly, globe) in years 1850-2015')
plt.draw()
plt.show()

