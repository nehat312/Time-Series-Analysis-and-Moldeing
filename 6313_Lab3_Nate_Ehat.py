#%% [markdown]
# DATS-6313 - LAB #2
# Nate Ehat

#%% [markdown]
# TIME SERIES DECOMPOSITION

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

## VISUAL SETTINGS
sns.set_style('whitegrid')

print("\nIMPORT SUCCESS")

#%%
# VARIABLE DIRECTORY
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'
passengers_link = 'https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/AirPassengers'

print("\nDIRECTORY ASSIGNED")

#%%
# 1. Using the Python program to load the ‘Airpassengers.csv’.

passengers = pd.read_csv(passengers_link + '.csv') # #parse_dates=True, infer_datetime_format=True, parse_dates=['Unnamed: 0']
print("\nIMPORT SUCCESS")

#%%
## INITIAL EDA
print(passengers.info())
print('*'*100)
print(passengers.head())

#%%
passengers['Month'] = pd.to_datetime(passengers['Month'])
print(passengers.info())

#%%
## DETERMINE START / END DATES
print(passengers.Month.min())
print('*'*50)
print(passengers.Month.max())

#%%
# Then write a python function that implement moving average of order m.
# The program should be written in a way that when it runs:
      # it should ask the user to input the order of moving average.
      # If m is even, then then the software must ask a user to enter the folding order (second MA) which must be even (Hint: You need to exclude the case m=1,2 and display a message that m=1,2 will not be accepted).
      # If m is odd, no need for the second MA.
      # Then the code should calculate estimated trend-cycle using the following equation where y is the original observation.
      # You are only allowed to use NumPy and pandas for this question.
      # (The use rolling mean inside the pandas is not allowed for this LAB).

def rolling_avg_non_wtd(array, m): # n > 2
    m = int(m)
    odd = True if m % 2 == 1 else False
    if m <= 2:
        return print("ERROR: M MUST BE > 2")
    elif odd == True:
        start = np.array([np.nan] * int((m - 1) / 2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
        end = np.array([np.nan] * int((m - 1) / 2))
        full = np.append(np.append(start, average), end)
        return full
    else:
        start = np.array([np.nan] * int(m / 2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
        end = np.array(([np.nan] * int((m - 1) / 2)))
        full = np.append(np.append(start, average), end)
        return full


#%%
rolling_avg_non_wtd(passengers['#Passengers'], 9)

#%%

# 2. Using the function developed in the previous step plot:
      # Estimated cycle-trend versus the original dataset (plot only the first 50 samples) for:
            # 3-MA, 5-MA, 7-MA, 9-MA
            # All in one graph (use the subplot 2x2).
            # Plot the detrended data on the same graph.
            # Add an appropriate title, x-label, y-label, and legend to the graph.

for i in range(0,9):
rolling_avg_non_wtd(passengers['#Passengers'], 9)


#%%

# 3. Using the function developed in the step 1 plot:
      # Estimated cycle-trend versus the original dataset -- # Plot only the first 50 samples
            # 2x4-MA, 2x6-MA, 2x8-MA, and 2x10-MA
            # All in one graph (use the subplot 2x2).
            # Plot the detrended data on the same graph.
            # Add an appropriate title, x-label, y-label, and legend to the graph.


#%%

# 4. Compare the ADF-test of the original dataset versus the detrended dataset using the 3-MA.
      # Explain your observation.


#%%

# 5.  Apply the STL decomposition method to the dataset.
      # Plot the trend, seasonality, and reminder in one graph.
      # Add an appropriate title, x-label, y-label, and legend to the graph.

#%%
# 6. Calculate the seasonally adjusted data and plot it versus the original data.
      # Add an appropriate title, x- label, y-label, and legend to the graph.

#%%
# 7- Calculate the strength of trend using the following equation and display the following message on the console:
      # The strength of trend for this data set is ________

#%%
# 8-Calculate the strength of seasonality using the following equation and display the following message on the console:
      # The strength of seasonality for this data set is ________

#%%
# 9- Based on the results in the previous steps - is this data set strongly seasonal or strongly trended?
      # Justify your answer.


#%%
# PLOT
#fig, axes = plt.subplots(1,1,figsize=(10,8))
#passengers['#Passengers'].plot(legend=True)

plt.figure(figsize=(10,8))
sns.lineplot(x=passengers['Month'], y=passengers['#Passengers'])
plt.title("AIR PASSENGERS (1949-1960)")
plt.xlabel('DATE')
plt.ylabel('PASSENGERS (#)')
plt.tight_layout(pad=1)
#plt.grid()
plt.show()

#%%
# TIME SERIES STATISTICS
print("#Passengers mean is:", passengers['#Passengers'].mean(),
      "and the variance is:", passengers['#Passengers'].var(),
      "with standard deviation:", passengers['#Passengers'].std())
print('*'*150)

#%%
# SET COLUMN INDICES FOR CHART TITLES
passengers_col_index = passengers.columns[1].upper()
print(passengers_col_index)

#%%
rolling_mean_var_plots(rolling_mean_var(passengers['#Passengers']), passengers_col_index)

#%%
## ADF TEST
print('ADF PASSENGERS:')
print(adf_test(['#Passengers'], passengers))
print('*'*100)

#%%
## KPSS TEST
print('KPSS PASSENGERS:')
print(kpss_test(passengers['#Passengers']))
print('*'*100)

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
## FIRST-ORDER DIFFERENCING
passengers_1diff = differencer(passengers['#Passengers'], 1, passengers['Month'])
print(passengers_1diff)

#%%
## FIRST-ORDER DIFFERENCING - SET COLUMN INDICES FOR CHART TITLES
passengers_1diff_col_index = passengers_1diff.columns[0].upper()
print(passengers_1diff_col_index)

#%%
## FIRST-ORDER DIFFERENCING - GENERATE PLOTS
rolling_mean_var_plots(rolling_mean_var(passengers_1diff), passengers_1diff_col_index)

#%%
print(passengers_1diff.columns)

#%%
## FIRST-ORDER DIFFERENCING - ADF TEST
print('ADF FIRST-ORDER DIFF:')
print(adf_test(['1diff'], passengers_1diff))
print('*'*100)

#%%
## FIRST-ORDER DIFFERENCING - KPSS TEST
print('KPSS FIRST-ORDER DIFF:')
print(kpss_test(passengers_1diff['1diff']))
print('*'*100)

#%%
## SECOND-ORDER DIFFERENCING
passengers_2diff = differencer(passengers_1diff['1diff'], 2, passengers_1diff.index)
print(passengers_2diff)

#%%
## SECOND-ORDER DIFFERENCING - SET COLUMN INDICES FOR CHART TITLES
passengers_2diff_col_index = passengers_2diff.columns[0].upper()
print(passengers_2diff_col_index)

#%%
## SECOND-ORDER DIFFERENCING - GENERATE PLOTS
rolling_mean_var_plots(rolling_mean_var(passengers_2diff), passengers_2diff_col_index)

#%%
## SECOND-ORDER DIFFERENCING - ADF TEST
print('ADF SECOND-ORDER DIFF:')
print(adf_test(['2diff'], passengers_2diff))
print('*'*100)

#%%
## SECOND-ORDER DIFFERENCING - KPSS TEST
print('KPSS SECOND-ORDER DIFF:')
print(kpss_test(passengers_2diff['2diff']))
print('*'*100)

#%%
## THIRD-ORDER DIFFERENCING
passengers_3diff = differencer(passengers_2diff['2diff'], 3, passengers_2diff.index)
print(passengers_3diff)

#%%
## THIRD-ORDER DIFFERENCING - SET COLUMN INDICES FOR CHART TITLES
passengers_3diff_col_index = passengers_3diff.columns[0].upper()
print(passengers_3diff_col_index)

#%%
## THIRD-ORDER DIFFERENCING - GENERATE PLOTS
rolling_mean_var_plots(rolling_mean_var(passengers_3diff), passengers_3diff_col_index)

#%%
## THIRD-ORDER DIFFERENCING - ADF TEST
print('ADF THIRD-ORDER DIFF:')
print(adf_test(['3diff'], passengers_3diff))
print('*'*100)

#%%
## THIRD-ORDER DIFFERENCING - KPSS TEST
print('KPSS THIRD-ORDER DIFF:')
print(kpss_test(passengers_3diff['3diff']))
print('*'*100)

#%%
## LOG TRANSFORMATION
passengers_log = log_transform(passengers['#Passengers'], passengers.index)
print(passengers_log)

#%%
## LOG TRANSFORMATION - SET COLUMN INDICES FOR CHART TITLES
passengers_log_col_index = passengers_log.columns[0].upper()
print(passengers_log_col_index)
print(passengers_log.columns)

#%%
## LOG FIRST-ORDER DIFFERENCING
passengers_log_1diff = differencer(passengers_log['log_transform'], 1, passengers_log.index)
print(passengers_log_1diff)

#%%
## LOG FIRST-ORDER DIFFERENCING - SET COLUMN INDICES FOR CHART TITLES
passengers_log_1diff_col_index = passengers_log_1diff.columns[0].upper()
print(passengers_log_1diff_col_index)

#%%
## LOG FIRST-ORDER DIFFERENCING - GENERATE PLOTS
rolling_mean_var_plots(rolling_mean_var(passengers_log_1diff), passengers_log_1diff_col_index)

#%%
## LOG FIRST-ORDER DIFFERENCING - ADF TEST
print('ADF LOG FIRST-ORDER DIFF:')
print(adf_test(['1diff'], passengers_log_1diff))
print('*'*100)

#%%
## LOG FIRST-ORDER DIFFERENCING - KPSS TEST
print('KPSS LOG FIRST-ORDER DIFF:')
print(kpss_test(passengers_log_1diff['1diff']))
print('*'*100)


#%%