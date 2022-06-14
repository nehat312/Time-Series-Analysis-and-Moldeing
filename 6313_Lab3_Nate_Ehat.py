#%% [markdown]
# DATS-6313 - LAB #3
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
print(passengers.index.min())
print('*'*50)
print(passengers.index.max())

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
## ODD ROLLING AVERAGE ##
def odd_rolling_avg(array, m):
    start = np.array([np.nan] * int((m - 1) / 2))
    average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
    end = np.array([np.nan] * int((m - 1) / 2))
    final = np.append(np.append(start, average), end)
    return final

## EVEN ROLLING AVERAGE ##
def even_rolling_avg(array, m):
    start = np.array([np.nan] * int(m / 2))
    average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
    end = np.array(([np.nan] * int((m - 1) / 2)))
    final = np.append(np.append(start, average), end)
    return final

## ODD OR EVEN ROLLING AVERAGE ##
def odd_or_even_rolling_avg(array):
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
            output = even_rolling_avg(even_rolling_avg(array, order_1), order_2)
            return output
    elif order_1 % 2 == 1:
        return odd_rolling_avg(array, order_1)

#%%
## NON-WEIGHTED ROLLING AVERAGE ##
def rolling_avg_non_wtd(array, m): # n > 2
    m = int(m)
    odd = True if m % 2 == 1 else False
    if m <= 2:
        return print("ERROR: M MUST BE > 2")
    elif odd == True:
        start = np.array([np.nan] * int((m - 1) / 2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
        end = np.array([np.nan] * int((m - 1) / 2))
        final = np.append(np.append(start, average), end)
        return final
    else:
        start = np.array([np.nan] * int(m / 2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
        end = np.array(([np.nan] * int((m - 1) / 2)))
        final = np.append(np.append(start, average), end)
        return final


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

# for ax in axs.tbu:
#     ax.label_outer()

#%%

# 3. Using the function developed in the step 1 plot:
      # Estimated cycle-trend versus the original dataset -- # Plot only the first 50 samples
            # 2x4-MA, 2x6-MA, 2x8-MA, and 2x10-MA
            # All in one graph (use the subplot 2x2).
            # Plot the detrended data on the same graph.
            # Add an appropriate title, x-label, y-label, and legend to the graph.


#%%



#%%

# 4. Compare the ADF-test of the original dataset versus the detrended dataset using the 3-MA.
      # Explain your observation.

## ADF TEST - ORIGINAL
print('ADF PASSENGERS ORIGINAL:')
print(adf_test(['#Passengers'], passengers))
print('*'*100)

## ADF TEST - 3-MA DE-TRENDED
print('ADF PASSENGERS 3-MA:')
print(adf_test(MA_3_df[0], MA_3_df))
print('*'*100)

#%% [markdown]



#%%
# 5.  Apply the STL decomposition method to the dataset.
      # Plot the trend, seasonality, and reminder in one graph.
      # Add an appropriate title, x-label, y-label, and legend to the graph.

temp = data['Temp']
temp = pd.Series(np.array(data['Temp']),
                        index=pd.date_range('1981-01-01',
                        periods=len(temp),
                        freq='d'),
                        name='daily-min-temp')
print(temp.describe())

#%%

passenger_series =  pd.Series(np.array(passengers['#Passengers']),
                        index = passengers.index,
                        # index=pd.date_range('1981-01-01',
                        # periods=len(temp),
                        # freq='d'),
                        name='PASSENGERS')

passenger_series
#%%
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
sns.lineplot(x=adjusted_seasonal.index, y=adjusted_seasonal)
sns.lineplot(x=passengers.index, y=passengers['#Passengers'])
plt.title('SEASONALLY ADJUSTED DATA VS. ORIGINAL DATA', fontsize=18)
plt.xlabel('DATE', fontsize=15)
plt.ylabel('PASSENGERS', fontsize=15)

plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# 7- Calculate the strength of trend using the following equation and display the following message on the console:
      # The strength of trend for this data set is ________

## STRENGTH OF TREND ##
def strength_of_trend(residual, trend):
    var_resid = np.nanvar(residual)
    var_resid_trend = np.nanvar(np.add(residual, trend))
    return 1 - (var_resid / var_resid_trend)

print(f'STRENGTH OF TREND: {strength_of_trend(R, T)}')

#%%
F = np.maximum(0, 1-np.var(R)/np.var(np.array(T)+np.array(R)))
print(f'STRENGTH OF TREND: {100*F:.2f}%')

#%%
# 8-Calculate the strength of seasonality using the following equation and display the following message on the console:
      # The strength of seasonality for this data set is ________

## STRENGTH OF SEASONAL ##
def strength_of_seasonal(residual, seasonal):
    var_resid = np.nanvar(residual)
    var_resid_seasonal = np.nanvar(np.add(residual, seasonal))
    return 1 - (var_resid / var_resid_seasonal)

print(f'STRENGTH OF SEASONALITY: {strength_of_seasonal(R, S)}')

#%%
F = np.maximum(0, 1-np.var(R)/np.var(np.array(S)+np.array(R)))
print(f'STRENGTH OF SEASONALITY: {100*F:.2f}%')

# print(f'The strength of SEASONALITY for this data set is {}')

#%%
# 9- Based on the results in the previous steps - is this data set strongly seasonal or strongly trended?
      # Justify your answer.

#%% [markdown]
# Based on results from prior calculations, this data set appears to be both strongly trended and strongly seasonal.
# Strength of Trend (~99.8%) and Strength of Seasonality (~99.7%) are very close to 1, indicating both are apparent.
# Strength of Trend slightly exceeds Strength of Seasonality so the data may be slightly more trended than seasonal.


#%%



#%%



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


#%%
# T-T/S
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)


#%%
holtt = ets.ExponentialSmoothing(yt,
                                trend=None,
                                seasonal=None,
                                damped_trend=False).fit()

holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

fig, ax = plt.subplots()
ax.plot(yt, label='TRAIN DATA')
ax.plot(yf, label='TEST DATA')
ax.plot(holtf, label='SES METHOD PREDICTION')
plt.legend(loc='best')
plt.show()
