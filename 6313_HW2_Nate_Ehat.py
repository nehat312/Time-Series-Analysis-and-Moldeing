#%% [markdown]
# DATS-6313 - HW #2
# Nate Ehat

#%% [markdown]
# AUTO-CORRELATION FUNCTION

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
import scipy.stats as st
import pandas_datareader as web

import datetime as dt

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
# VARIABLE DIRECTORY
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'
#tute_filepath = 'tute1'

print("\nDIRECTORY ASSIGNED")

#%%



print("\nIMPORT SUCCESS")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid')

#%%
# 1. Let suppose y vectors is given as y(t) = [3, 9, 27, 81, 243].
# Without use of python or any other computer program:
    # manually calculate the ACF for lag 0,1,2,3,4
    # 𝑅𝑦 (0), 𝑅𝑦 (1), 𝑅𝑦 (2), 𝑅𝑦 (3), 𝑅𝑦 (4).
# Hint : The formula for the ACF calculation is given bellow.
# Display the ACF (two sided) on a graph (no python).

y = [3, 9, 27, 81, 243]

#%% [markdown]



#%%
# 2. Using Python program, create a normal white noise with:
        # zero mean
        # standard deviation of 1
        # 1000 samples.
    # Plot the generated WN versus number of samples.
    # Plot the histogram of generated WN.
    # Calculate the sampled mean and sampled std of generated WN.

T = 1000
mean = 0
std = 1
data = np.random.normal(mean, std, size=T)

#%%
# 3. Write a python code to estimate Autocorrelation Function.
# Note: You need to use the equation (1) given in lecture 4. Shown above.
# ACF plot must be double sided, from negative # of lags to positive # of lags with highlighted insignificant region.
    # a. Plot the ACF of the make-up dataset in step 1.
        # Compare the result with the manual calculation.
        # # of lags = 4.
    # b. Plot the ACF of the generated data in step 2.
        # The ACF needs to be plotted using “stem” command.
        # # of lags = 20.
    # c. Record observations about ACF plot, histogram, and time plot of generated WN.


ACF_PACF_Plot(data, 4)


#%%
lags=4
acfunc(data, lags)

#%%
## ACF STEM PLOT
a2 = np.arange(0,lags,1)
a3 = -a2[::-1]
x = np.concatenate((a3[:-1], a2))

plt.figure(figsize=(8,8))
(markers, stemlines, baseline) = plt.stem(data, markerfmt='o')
plt.title(f'ACF PLOT')
plt.setp(markers, color='red', marker='o')
plt.setp(baseline, color='gray', linewidth=2, linestyle='-')
plt.xlabel('LAGS')
plt.ylabel('AUTOCORRELATION VALUE')
#plt.xticks()
m = 1.96 / np.sqrt(len(data))
plt.axhspan(-m, m, alpha=0.2, color='blue')
plt.show()

#%%
# 4. Load the time series dataset from yahoo API.
    # The yahoo API contains the stock value for 6 major companies.
        # stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
    # a. Plot the “Close” value of the stock for all companies versus time in one graph:
        # Subplot [3 figures in row and 2 figures in column].
        # Add grid, x-label, y-label, and title to each subplot.
        # Pick the start date as ‘2000-01-01’ and the end date today.
    # Plot the ACF of the “Close” value of the stock for all companies versus lags in one graph
        # Subplot [3 rows and 2 columns]. Add x-label, y-label, and title to each subplot.
        # Number lags = 50.
#%%
## STOCKS

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
start_date = '2021-01-20'
end_date = '2022-05-22'

print("\nVARIABLES ASSIGNED")

#%%
# Pull ticker data
aapl = web.DataReader('AAPL', data_source='yahoo', start=start_date, end=end_date)
orcl = web.DataReader('ORCL', data_source='yahoo', start=start_date, end=end_date)
tsla = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)
ibm = web.DataReader('IBM', data_source='yahoo', start=start_date, end=end_date)
yelp = web.DataReader('YELP', data_source='yahoo', start=start_date, end=end_date)
msft = web.DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)
df = web.DataReader(stocks, data_source='yahoo', start=start_date, end=end_date)
stock_pulls = [aapl, orcl, tsla, ibm, yelp, msft]

print("\nSTOCKS PULLED")

#%%
print(df.head())

#%%
plt.figure(figsize=(8,8))
orcl['Close'].plot()
aapl['Close'].plot()
tsla['Close'].plot()
yelp['Close'].plot()
msft['Close'].plot()
ibm['Close'].plot()
plt.legend(loc='best')
plt.show()


#%%
acfunc(msft.Close.values, 20)

#%%
msft_col_index = df.columns[5].upper()

rolling_mean_var_plots(msft.Close.values, msft)


#%%
# 5. Write down your observations about:
    # Correlation between stationary and non-stationary time series (if there is any)
    # Autocorrelation function


#%%


#%%


#%%