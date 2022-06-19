#%% [markdown]
# DATS-6313 - LAB #4
# Nate Ehat

#%% [markdown]
# GPAC TABLE IMPLEMENTATION

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

from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid') #ticks

#%%
# DIRECTORY CONFIGURATION
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'

print("\nDIRECTORY ASSIGNED")

#%%
# 1. Using the Python program to load the ‘Airpassengers.csv’.
passengers_link = 'https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/AirPassengers'

passengers = pd.read_csv(passengers_link + '.csv', index_col='Month', parse_dates=True) # #parse_dates=True, infer_datetime_format=True, parse_dates=['Unnamed: 0']
print("\nIMPORT SUCCESS")

#%%
## INITIAL EDA
print(passengers.info())
print('*'*100)
print(passengers.head())

#%%
## DETERMINE START / END DATES
print(f'START DATE: {passengers.index.min()}')
print('*'*50)
print(f'END DATE: {passengers.index.max()}')

#%%
# Write a python function that implement moving average of order m.
# The program should be written in a way that when it runs:
      # it should ask the user to input the order of moving average.
      # If m is even, then then the software must ask a user to enter the folding order (second MA)
        # which must be even (Hint: You need to exclude the case m=1,2 and display a message that m=1,2 will not be accepted).
      # If m is odd, no need for the second MA.
      # Then the code should calculate estimated trend-cycle using the following equation where y is the original observation.
      # You are only allowed to use NumPy and pandas for this question.
      # (The use rolling mean inside the pandas is not allowed for this LAB).


#%%
## MOVING AVERAGE FUNCTION #1
def moving_average(data, m1, m2):
    ma = np.empty(len(data))
    ma[:] = np.NaN
    if m1 < 2:
        print('INVALID ORDER')
    elif m1 % 2 != 0:
        k = m1 // 2
        for i in range(k, len(data) - k):
            ma[i] = np.mean(data[i - k:i + k + 1])
    else:
        if m2 % 2 == 1 or m2 < 2:
            print('INVALID FOLDING ORDER')
        else:
            k1 = m1 // 2
            ma_mid = np.empty(len(data))
            ma_mid[:] = np.NaN
            for i in range(k1 - 1, len(data) - k1):
                ma_mid[i] = np.mean(data[i - k1 + 1:i + k1 + 1])
            k2 = m2 // 2
            for i in range(k2 - 1, len(ma_mid) - k2):
                ma[i + 1] = np.mean(ma_mid[i - k2 + 1:i + k2 + 1])
    return ma

#%%
## MOVING AVERAGE FUNCTION #2

def rolling_avg_odd(array, m):
    start = np.array([np.nan] * int((m - 1) / 2))
    average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
    end = np.array([np.nan] * int((m - 1) / 2))
    final = np.append(np.append(start, average), end)
    return final

def rolling_avg_even(array, m):
    start = np.array([np.nan] * int(m / 2))
    average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
    end = np.array(([np.nan] * int((m - 1) / 2)))
    final = np.append(np.append(start, average), end)
    return final

def rolling_avg_odd_or_even(array):
    length = len(array)
    order_1 = int(input("INPUT ORDER OF MOVING AVERAGE:"))
    if order_1 <= 2:
        return print('ERROR: ORDER MUST BE >2')
    elif order_1 % 2 == 0:
        order_2 = int(input('ERROR: FOLDING ORDER MUST BE EVEN / >1'))
        if order_2 < 2 or order_2 % 2 != 0:
            print('INVALID FOLDING ORDER')
            pass
        else:
            output = rolling_avg_even(rolling_avg_even(array, order_1), order_2)
            return output
    elif order_1 % 2 == 1:
        return rolling_avg_odd(array, order_1)

#%%
# 2. Using the function developed in the previous step plot:
      # Estimated cycle-trend versus the original dataset (plot only the first 50 samples) for:
            # 3-MA, 5-MA, 7-MA, 9-MA
            # All in one graph (use the subplot 2x2).
            # Add an appropriate title, x-label, y-label, and legend to the graph.
            # Plot the detrended data on the same graph.

#for i in range(3,10,2):
plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
sns.lineplot(x=passengers.index, y=rolling_avg_non_wtd(passengers['#Passengers'], 3))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('3-MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)
plt.subplot(2, 2, 2)
sns.lineplot(x=passengers.index, y=rolling_avg_non_wtd(passengers['#Passengers'], 5))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('5-MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)
plt.subplot(2, 2, 3)
sns.lineplot(x=passengers.index, y=rolling_avg_non_wtd(passengers['#Passengers'], 7))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('7-MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)
plt.subplot(2, 2, 4)
sns.lineplot(x=passengers.index, y=rolling_avg_non_wtd(passengers['#Passengers'], 9))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('9-MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)

plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# 3. Using the function developed in the step 1 plot:
      # Estimated cycle-trend versus the original dataset -- # Plot only the first 50 samples
            # 2x4-MA, 2x6-MA, 2x8-MA, and 2x10-MA
            # All in one graph (use the subplot 2x2).
            # Plot the detrended data on the same graph.
            # Add an appropriate title, x-label, y-label, and legend to the graph.

plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
sns.lineplot(x=passengers.index, y=moving_average(passengers['#Passengers'], 2, 4))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('2x4 MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)
plt.subplot(2, 2, 2)
sns.lineplot(x=passengers.index, y=moving_average(passengers['#Passengers'], 2, 6))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('2x6 MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)
plt.subplot(2, 2, 3)
sns.lineplot(x=passengers.index, y=moving_average(passengers['#Passengers'], 2, 8))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('2x8 MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)
plt.subplot(2, 2, 4)
sns.lineplot(x=passengers.index, y=moving_average(passengers['#Passengers'], 2, 10))
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('2x10 MA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)

plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# 4. Compare the ADF-test of the original dataset versus the detrended dataset using the 3-MA.

#%%
moving_average(passengers['#Passengers'], 2, 4)
moving_average(passengers['#Passengers'], 2, 6)
moving_average(passengers['#Passengers'], 2, 8)
moving_average(passengers['#Passengers'], 2, 10)
#%%
rolling_avg_non_wtd(passengers['#Passengers'], 3)
orig = pd.DataFrame(index=passengers.index, data=passengers['#Passengers'])
ma3 = pd.DataFrame(index=passengers.index, data=rolling_avg_non_wtd(passengers['#Passengers'], 3))
#orig
ma3_drop = ma3.dropna()
ma3_drop



#%%
## ADF TEST - ORIGINAL
print('ADF PASSENGERS ORIGINAL:')
print(adf_test(['#Passengers'], passengers))
print('*'*100)

## ADF TEST - 3-MA
print('ADF PASSENGERS 3-MA:')
print(adf_test(0, ma3))
print('*'*100)


#%% [markdown]
# Comparing results generated above displaying ADF Statistics across both original and transformed data:
    # * ADF Statistics for both the original data and 3-day moving average are very similar across the board.
    # * For 3-MA data relative to Original data, slight outperformance is observed across both ADF Statistics and P-Values.
        # * ADF STATISTICS: 3-MA -- 0.822058 /// ORIGINAL -- 0.815369


#%%
# 5.  Apply the STL decomposition method to the dataset.
      # Plot the trend, seasonality, and reminder in one graph.
      # Add an appropriate title, x-label, y-label, and legend to the graph.

#%%
## STL DECOMPOSITION
passenger_series =  pd.Series(np.array(passengers['#Passengers']),
                        index = passengers.index,
                        # index=pd.date_range('1981-01-01',
                        # periods=len(temp),
                        # freq='d'),
                        name='PASSENGERS')

print(passenger_series)

#%%

## STL DECOMPOSITION
STL = STL(passenger_series)
res = STL.fit()

#%%
plt.figure(figsize=(12,8))
fig = res.plot()
plt.xlabel('DATE', fontsize=12)
plt.tight_layout(pad=1)
plt.show()

#%%
T = res.trend
S = res.seasonal
R = res.resid

#%%
# 6. Calculate the seasonally adjusted data and plot it versus the original data.
      # Add an appropriate title, x- label, y-label, and legend to the graph.

adjusted_seasonal = passenger_series - S

plt.figure(figsize=(12,8))
sns.lineplot(x=adjusted_seasonal.index, y=adjusted_seasonal, legend='full')
sns.lineplot(x=passengers.index, y=passengers['#Passengers'], legend='full')
plt.title('SEASONALLY ADJUSTED DATA VS. ORIGINAL DATA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# 7- Calculate the strength of trend:

## STRENGTH OF TREND ##
def strength_of_trend(residual, trend):
    var_resid = np.nanvar(residual)
    var_resid_trend = np.nanvar(np.add(residual, trend))
    return 1 - (var_resid / var_resid_trend)


F = np.maximum(0, 1-np.var(R)/np.var(np.array(T)+np.array(R)))

print(f'STRENGTH OF TREND: {100*F:.3f}% or {strength_of_trend(R, T):.5f}')

#%%
# 8-Calculate the strength of seasonality:

## STRENGTH OF SEASONAL ##
def strength_of_seasonal(residual, seasonal):
    var_resid = np.nanvar(residual)
    var_resid_seasonal = np.nanvar(np.add(residual, seasonal))
    return 1 - (var_resid / var_resid_seasonal)

F = np.maximum(0, 1-np.var(R)/np.var(np.array(S)+np.array(R)))

print(f'STRENGTH OF SEASONALITY: {100*F:.3f}% or {strength_of_seasonal(R, S):.5f}')

#%% #%% [markdown]
# 9- Based on the results in the previous steps - is this data set strongly seasonal or strongly trended?
      # Justify your answer.

# Based on results from prior calculations, this data set appears to be both strongly trended and strongly seasonal.
# Strength of Trend (~99.8%) and Strength of Seasonality (~99.7%) are very close to 1, indicating both are apparent.
# Strength of Trend slightly exceeds Strength of Seasonality so the data may be slightly more trended than seasonal.
# Moreover, viewing chart graphics as presented appear to visually confirm both trend and seasonal tendencies.


#%%
