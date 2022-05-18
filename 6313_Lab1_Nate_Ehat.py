#%% [markdown]
# DATS-6313 - LAB #1
# Nate Ehat

#%% [markdown]
# STATIONARITY

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller

import datetime as dt
import scipy.stats as st

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
# VARIABLE DIRECTORY
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'
filepath = 'tute1'

print("\nDIRECTORY ASSIGNED")

#%%
# 1. Load the time series data called ‘tute1.csv’ [ the dataset can be found on the course GitHub].
# This date relates to the quarterly sales for a small company over period 1981-2005.
    # Sales contains quarterly sales
    # AdBudget is the advertisement budget
    # GDP is the gross domestic product for a small company.

# By saving the file to '.xlsx' format:
# Saves lots of time - prior to import: quickly convert all dates to datetime format

tute_cols = ['Date', 'Sales', 'AdBudget', 'GDP']

tute = pd.read_excel(filepath + '.xlsx', index_col='Date')
#tute = pd.read_csv(filepath + '.csv', index_col=['Unnamed: 0']) #parse_dates=True, infer_datetime_format=True parse_dates=['Unnamed: 0'], infer_datetime_format=True

print("\nIMPORT SUCCESS")

#%%
## INITIAL EDA
print(tute.info())
print(tute.head())

#%%
# IDENTIFY / FILTER 'OUTLIER' DATES BEYOND 2000 (EXCLUDED PER LAB GUIDELINES)
print(tute[0:80])
print('*'*75)
print(tute[80:101])

#%%
# IDENTIFY / FILTER 'OUTLIER' DATES BEYOND 2000 (EXCLUDED PER LAB GUIDELINES)

tute = tute[0:80]
print(tute.info())

#%%
print(tute.columns)
print(tute.index)

#%%
# Plot Sales, AdBudget and GPD versus time step in one graph.
# Add grid and appropriate title, legend to each plot.
# The x-axis is the time, and it should show the time (year).
# The y-axis is the USD($).

#Sales

ax = df["Sales"].plot(legend=True, title= "Daily Sales March 1, 1981- June 8, 1981")
ax.set_ylabel("Sales")
plt.show()
#AdBudget

ax=df['AdBudget'].plot(legend=True, title="Daily AdBudget March 1, 1981- June 8, 1981")
ax.set_ylabel("AdBudget")
plt.show()
#GDP

ax=df['GDP'].plot(legend=True, title="Daily GDP March 1, 1981- June 8, 1981")
ax.set_ylabel('GDP')
plt.show()


#%%
# 2. Find the time series statistics of Sales, AdBudget and GPD and display on the console:
    # a. The Sales mean is : -- and the variance is : -- with standard deviation : ----median:----
    # b. The AdBudget mean is : -- and the variance is : -- with standard deviation : -- median:----
    # c. The GDP mean is :--- and the variance is : -------- with standard deviation : --- median:----


print("The Sales mean is:",df['Sales'].mean(), "and the variance is:",df['Sales'].var(), "with standard deviation:",
df['Sales'].std())
print("The AdBudget mean is:",df['AdBudget'].mean(), "and the variance is:",df['AdBudget'].var(), "with standard deviation:",
df['AdBudget'].std())
print("The GDP mean is:",df['GDP'].mean(), "and the variance is:",df['GDP'].var(), "with standard deviation:",
df['GDP'].std())


#%%
# 3. Prove that the Sales, AdBudget and GDP in this time series dataset is stationary.
# Hint: One way to show a process is stationary, is to plot the rolling mean and rolling variance versus number of samples which is accumulated through time.
# If the rolling mean and rolling variance stabilizes once all samples are included, then this is an indication that a data set is stationary.
# You need to plot the rolling mean and rolling variance in one graph using subplot [2x1] by creating a loop over the number of samples in the dataset and calculate the means & variances versus time.
# Plot all means and variances and show that the means and variances are almost constant.
# To perform this task, you need to create a loop with goes over number of observations in the dataset.
# During the first iteration, the first sample will load and the mean and variance will be calculated.
# During the second iteration, the first two observations will load, and the mean and variance will be calculated and will append to the previous mean and variance.
# Repeat this process till the last observation is added the mean and variance will be calculated.
# You can use the following command to bring new data sample at each iteration.
# The plot the mean and variance over the time at the end.
# Save the above code under a function called ‘Cal-rolling-mean-var ’.
# You will use this function several times throughout the course.






#%%
# 4.


#%%
# 5.


#%%
# 6.


#%%
# 7.


#%%
# 8.


#%%