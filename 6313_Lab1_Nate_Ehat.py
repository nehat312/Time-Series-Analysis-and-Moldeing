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

import scipy.stats as st

print("\nIMPORT SUCCESS")

#%%
# 1. Load the time series data called ‘tute1.csv’ [ the dataset can be found on the course GitHub].
# This date relates to the quarterly sales for a small company over period 1981-2005.
    # Sales contains quarterly sales
    # AdBudget is the advertisement budget
    # GDP is the gross domestic product for a small company.
# Plot Sales, AdBudget and GPD versus time step in one graph.
# Add grid and appropriate title, legend to each plot.
# The x-axis is the time, and it should show the time (year).
# The y-axis is the USD($). The graph should be look like below.




#%%
# 2. Find the time series statistics of Sales, AdBudget and GPD and display on the console:
    # a. The Sales mean is : -- and the variance is : -- with standard deviation : ----median:----
    # b. The AdBudget mean is : -- and the variance is : -- with standard deviation : -- median:----
    # c. The GDP mean is :--- and the variance is : -------- with standard deviation : --- median:----


print(f"K-S test: statistics={kstest_x[0]:.5f}, p-value={kstest_x[1]:.5f}")


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