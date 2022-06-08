#%% [markdown]
# DATS-6313 - LAB #3
# Nate Ehat

#%% [markdown]
# SIMPLE FORECASTING METHODS

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
# import warnings
# warnings.filterwarnings('ignore')

#%%
# DIRECTORY CONFIGURATION
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'

xx_filepath = 'AirPassengers'

print("\nDIRECTORY ASSIGNED")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid')

#%%
# 1. Consider the following joint density function for the two continuous random variable X and Y
# Uniformly distributed between 0 and 1

# a. Find the constant c.
# b. Find the marginal density 𝑓 (𝜆 ) and 𝑓 (𝜆 ) 𝑋1 𝑌2
# c. Graph the marginal density function for random variable X and Y and show that the area under each curve is unity.
# d. What is E[X] ? What is E[Y]?
# e. Find P(𝜆2 < 𝜆1)
# f. Are random variable X and Y independent? Hint: For the dependency, it needs to check

xxx_cols = ['Date', 'Sales', 'AdBudget', 'GDP']

xxx = pd.read_excel(xx_filepath + '.xlsx', index_col='Date')

print("\nIMPORT SUCCESS")

#%%
## INITIAL EDA
print(xxx.info())
print('*'*75)
print(xxx.head())


#%%
print(xxx.columns)
print(xxx.index)

#%%
# Plot Sales, AdBudget and GPD versus time step in one graph.
# Add grid and appropriate title, legend to each plot.
# The x-axis is the time, and it should show the time (year).
# The y-axis is the USD($).

fig, axes = plt.subplots(1,1,figsize=(10,8))
ax = xxx["Sales"].plot(legend=True)
ax = xxx['AdBudget'].plot(legend=True)
ax = xxx['GDP'].plot(legend=True)

plt.title("Sales / AdBudget / GDP (1981-2000)")
plt.xlabel('DATE')
plt.ylabel('USD($)')
plt.tight_layout(pad=1)
#plt.grid()
plt.show()


#%%
# 2.

print("The Sales mean is:", tute['Sales'].mean(),
      "and the variance is:", tute['Sales'].var(),
      "with standard deviation:", tute['Sales'].std())
print('*'*150)
print("The AdBudget mean is:", tute['AdBudget'].mean(),
      "and the variance is:", tute['AdBudget'].var(),
      "with standard deviation:", tute['AdBudget'].std())
print('*'*150)
print("The GDP mean is:", tute['GDP'].mean(),
      "and the variance is:", tute['GDP'].var(),
      "with standard deviation:", tute['GDP'].std())
print('*'*150)

#%%
# 3.

#%%
# 4.

#%% [markdown]
# TBU
# TBU

#%%
# 5.





#%%
print('SALES:')
print(adf_test(['Sales'], tute))
print('*'*100)
print('ADBUDGET:')
print(adf_test(['AdBudget'], tute))
print('*'*100)
print('GDP:')
print(adf_test(['GDP'], tute))
print('*'*100)

#%%
# 6.
print('SALES:')
print(kpss_test(tute['Sales']))
print('*'*100)
print('ADBUDGET:')
print(kpss_test(tute['AdBudget']))
print('*'*100)
print('GDP:')
print(kpss_test(tute['GDP']))
print('*'*100)

#%%



#%%