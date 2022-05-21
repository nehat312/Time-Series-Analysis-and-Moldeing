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
# Save a lot of time with Excel, prior to import
# Quickly convert / conform all dates to datetime format

tute_cols = ['Date', 'Sales', 'AdBudget', 'GDP']

tute = pd.read_excel(filepath + '.xlsx', index_col='Date')

#tute = pd.read_csv(filepath + '.csv', index_col=['Unnamed: 0'])
# #parse_dates=True, infer_datetime_format=True parse_dates=['Unnamed: 0'], infer_datetime_format=True

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
## VISUAL SETTINGS
sns.set_style('whitegrid')

#%%
# Plot Sales, AdBudget and GPD versus time step in one graph.
# Add grid and appropriate title, legend to each plot.
# The x-axis is the time, and it should show the time (year).
# The y-axis is the USD($).

fig, axes = plt.subplots(1,1,figsize=(10,8))
ax = tute["Sales"].plot(legend=True)
ax = tute['AdBudget'].plot(legend=True)
ax = tute['GDP'].plot(legend=True)

plt.title("Sales / AdBudget / GDP (1981-2000)")
plt.xlabel('DATE')
plt.ylabel('USD($)')
plt.tight_layout(pad=1)
#plt.grid()
plt.show()


#%%
# 2. Find the time series statistics of Sales, AdBudget and GPD and display on the console:
    # a. The Sales mean is : -- and the variance is : -- with standard deviation : ----median:----
    # b. The AdBudget mean is : -- and the variance is : -- with standard deviation : -- median:----
    # c. The GDP mean is :--- and the variance is : -------- with standard deviation : --- median:----

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

test = rolling_mean_var(tute)

#%%
# SET COLUMN INDICES FOR CHART TITLES
sales_col_index = tute.columns[0].upper()
AdBudget_col_index = tute.columns[1].upper()
GDP_col_index = tute.columns[2].upper()

#%%
rolling_mean_var_plots(rolling_mean_var(tute['Sales']), sales_col_index)
rolling_mean_var_plots(rolling_mean_var(tute['AdBudget']), AdBudget_col_index)
rolling_mean_var_plots(rolling_mean_var(tute['GDP']), GDP_col_index)

#%%
# 4. Write down observation about the plot of the mean and variance in the previous step.
# Is Sales, GDP and AdBudget stationary or not? Explain why.

#%% [markdown]
# All three variables (Sales, AdBudget, GDP) do appear Stationary from interpretation of the charts exhibited above.
# Observing both rolling mean and variance plots approach stabilization, this likely indicates stationarity.

#%%
# 5. Perform an ADF-test to check if Sales, AdBudget and GDP stationary or not (confidence interval 95% or above). Does your answer for this question reinforce your observations in the previous step?
# Hint: You can use the following code to calculate the ADF-test.

from statsmodels.tsa.stattools import adfuller

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
import warnings
warnings.filterwarnings('ignore')

#%%
# 6. Perform an KPSS-test to check if Sales, AdBudget and GDP stationary or not (confidence interval 95% or above). Does your answer for this question reinforce your observations in the previous steps?
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
# 7. Repeat step 1 - 6 with ‘AirPassengers.csv’ on the GitHub.
# This timeseries dataset is univariate with  passengers as an attribute.

#%%
# 8.
# If the passengers is not stationary, it needs to become stationary by transformation
# a. Perform a 1st order non-seasonal differencing.
    # Is the dataset become stationary? Explain why.
# b. Perform a 2nd order non-seasonal differencing.
    # Is the dataset become stationary? Explain why.
# c. Perform a 3rd order non-seasonal differencing.
    # Is the dataset become stationary? Explain why.
# d. If procedures a, b and c do not make the dataset stationary:
    # Perform a log transformation of the original raw dataset followed by a 1st order differencing
    # Plot the rolling mean and variance.
    # Perform ADF-test and KPSS-test on the transformed dataset and display the results on the console.
        # This step should make the dataset stationary
        # rolling mean and variance is stabilize and the ADF-test confirms stationarity.


#%%