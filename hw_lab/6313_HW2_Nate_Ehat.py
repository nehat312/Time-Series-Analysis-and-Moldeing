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

print("\nDIRECTORY ASSIGNED")

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

#%% [markdown]

## Please refer to attached screenshot of handwritten ACF calculation.

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
## WHITE NOISE DISTRIBUTION
plt.figure(figsize=(8,8))
plt.plot(data)
plt.title('WHITE NOISE DISTRIBUTION', fontsize=20)
plt.xlabel('SAMPLES', fontsize=16)
plt.ylabel('RANGE', fontsize=16)
plt.tight_layout(pad=1)
plt.show()

#%%
## WHITE NOISE HISTOGRAM
plt.figure(figsize=(8,8))
plt.hist(data, bins=50)
plt.title('WHITE NOISE HISTOGRAM', fontsize=20)
plt.xlabel('RANGE', fontsize=16)
plt.ylabel('FREQUENCY', fontsize=16)
plt.tight_layout(pad=1)
plt.show()

#%%
print(f'SHAPE: {data.shape}')
print(f'MEAN: {data.mean()}')
print(f'STD: {data.std()}')
print(f'VAR: {data.var()}')
print(f'MAX: {data.max()}')
print(f'MIN: {data.min()}')

#%%
# 3. Write a python code to estimate Autocorrelation Function.
    # Note: You need to use the equation (1) given in lecture 4. Shown above.
    # ACF plot must be double sided, from negative # of lags to positive # of lags.
    # Highlight insignificant region.

## AUTO-CORRELATION FUNCTION
def ac_func(series, lag):
    if lag == len(series):
        return 0
    if lag == 0:
        return 1
    series = np.array(series)
    mean = series.mean()
    series_sub_mean = np.subtract(series, mean)
    shifted_right = series_sub_mean[:-lag]
    shifted_left = series_sub_mean[lag:]
    denominator = np.sum(np.square(series_sub_mean))
    numerator = np.sum(np.dot(shifted_right, shifted_left))
    r = numerator / denominator
    return round(r, 3)

#%%
## ACF DATAFRAME FUNCTION
def acf_df(series, lag):
    lag_list = [x for x in range(-lag, lag + 1, 1)]
    acf_value = [1]
    for l in [x for x in range(1, lag + 1, 1)]:
        x = ac_func(series, l)
        acf_value.insert(0, x)
        acf_value.append(x)
    df = pd.DataFrame()
    df['LAGS'] = lag_list
    df['ACF'] = acf_value
    return df

#%%
## STEMPLOT FUNCTION
def acf_stemplot(col, df, n):
    (markers, stemlines, baseline) = plt.stem(df['LAGS'], df['ACF'], markerfmt='o')
    plt.title(f'ACF PLOT') # - {col}
    plt.xlabel('LAGS')
    plt.ylabel('AUTOCORRELATION VALUE')
    plt.setp(markers, color='red', marker='o')
    plt.setp(baseline, color='gray', linewidth=2, linestyle='-')
    plt.fill_between(df['LAGS'], (1.96 / np.sqrt(len(df))), (-1.96 / np.sqrt(len(df))), color='magenta', alpha=0.2)
        #m = 1.96 / np.sqrt(len(df))
        #plt.axhspan(-m, m, alpha=0.2, color='skyblue')
        #plt.savefig(folder + 'images/' + f'{col}.png', dpi=1000)
    #plt.show()


#%%
# 3a. Plot the ACF of the make-up dataset in step 1.
    # Compare the result with the manual calculation.
    # Number of lags = 4.

acf_df_4 = acf_df(data, 4)
acf_stemplot(acf_df_4, acf_df_4, 4)
plt.show()

#%%
# 3b. Plot the ACF of the generated data in step 2.
    # The ACF needs to be plotted using “stem” command.
    # Number of lags = 20.

acf_df_20 = acf_df(data, 20)
acf_stemplot(acf_df_20.ACF, acf_df_20, 20)
plt.show()

#%% [markdown]
# 3c. Record observations about ACF plot, histogram, and time plot of generated WN.

# ACF plot with 20 lags far more optimal than ACF with 4 lags, as implied by highlighted insignificant ranges.


#%%
# 4. Load the time series dataset from yahoo API.
    # The yahoo API contains the stock value for 6 major companies.
        # stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']

#%%
## ASSIGN VARIABLES

tickers = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
columns = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
start_date = '2001-01-01'
end_date = '2022-06-01'

print("\nVARIABLES ASSIGNED")

#%%
## YAHOO API CALLS
stock_df = web.DataReader(tickers, data_source='yahoo', start=start_date, end=end_date)

#aapl = web.DataReader('AAPL', data_source='yahoo', start=start_date, end=end_date)
#orcl = web.DataReader('ORCL', data_source='yahoo', start=start_date, end=end_date)
#tsla = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)
#ibm = web.DataReader('IBM', data_source='yahoo', start=start_date, end=end_date)
#yelp = web.DataReader('YELP', data_source='yahoo', start=start_date, end=end_date)
#msft = web.DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)
#ticker_list = [aapl, orcl, tsla, ibm, yelp, msft]

print("\nTICKERS PULLED")

#%%
print(stock_df.info())

#%%
    # 4a. Plot the “Close” value of the stock for all companies versus time in one graph:
        # Subplot [3 figures in row and 2 figures in column].
        # Add grid, x-label, y-label, and title to each subplot.
        # Pick the start date as ‘2000-01-01’ and the end date today.

df_Close = pd.DataFrame(stock_df.Close)

plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(df_Close.AAPL)
plt.title('AAPL CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
plt.plot(df_Close.ORCL)
plt.title('ORCL CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(df_Close.TSLA)
plt.title('TSLA CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(df_Close.IBM)
plt.title('IBM CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(df_Close.YELP)
plt.title('YELP CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(df_Close.MSFT)
plt.title('MSFT CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()



#%%
    # 4b. Plot the ACF of the “Close” value of the stock for all companies versus lags in one graph
        # Subplot [3 rows and 2 columns]. Add x-label, y-label, and title to each subplot.
        # Number lags = 50.

acf_df_aapl = acf_df(df_Close.AAPL, 50)
acf_df_orcl = acf_df(df_Close.ORCL, 50)
acf_df_tsla = acf_df(df_Close.TSLA, 50)
acf_df_ibm = acf_df(df_Close.IBM, 50)
acf_df_yelp = acf_df(df_Close.YELP, 50)
acf_df_msft = acf_df(df_Close.MSFT, 50)

plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
acf_stemplot(acf_df_aapl.ACF, acf_df_aapl, 50)
plt.title('AAPL CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
acf_stemplot(acf_df_orcl.ACF, acf_df_orcl, 50)
plt.title('ORCL CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
acf_stemplot(acf_df_tsla.ACF, acf_df_tsla, 50)
plt.title('TSLA CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
acf_stemplot(acf_df_ibm.ACF, acf_df_ibm, 50)
plt.title('IBM CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
acf_stemplot(acf_df_yelp.ACF, acf_df_yelp, 50)
plt.title('YELP CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
acf_stemplot(acf_df_msft.ACF, acf_df_msft, 50)
plt.title('MSFT CLOSE', fontsize=21)
plt.xlabel('DATE', fontsize=18)
plt.ylabel('CLOSING SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
aapl_col_index = df_Close.columns[0].upper()
orcl_col_index = df_Close.columns[1].upper()
tsla_col_index = df_Close.columns[2].upper()
ibm_col_index = df_Close.columns[3].upper()
yelp_col_index = df_Close.columns[4].upper()
msft_col_index = df_Close.columns[5].upper()

#%%
def rolling_mean_var_plots_stocks(df, col):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
    fig.suptitle(f'ROLLING MEAN / VARIANCE OVER TIME - {col}')
    ax1.plot(df.index, df['ROLLING MEAN'])
    ax1.set_ylabel('ROLLING MEAN')
    ax2.plot(df.index, df['ROLLING VARIANCE'])
    ax2.set_xlabel('DATE')
    ax2.set_ylabel('ROLLING VARIANCE')
    #plt.show()
    return

#%%

rolling_mean_var_plots_stocks(rolling_mean_var(df_Close.AAPL), aapl_col_index)
plt.show()
#%%
rolling_mean_var_plots_stocks(rolling_mean_var(df_Close.ORCL), orcl_col_index)
plt.show()
#%%
rolling_mean_var_plots_stocks(rolling_mean_var(df_Close.TSLA), tsla_col_index)
plt.show()
#%%
rolling_mean_var_plots_stocks(rolling_mean_var(df_Close.IBM), ibm_col_index)
plt.show()
#%%
rolling_mean_var_plots_stocks(rolling_mean_var(df_Close.YELP), yelp_col_index)
plt.show()
#%%
rolling_mean_var_plots_stocks(rolling_mean_var(df_Close.MSFT), msft_col_index)
plt.show()

#%%
# 5. Write down your observations about:
    # Correlation between stationary and non-stationary time series (if there is any)
    # Autocorrelation function

# None of the stocks appear stationary, with many experiencing explosive and oftentimes volatile growth over time.
# Generally upward trajectory over time for all blue-chip tickers studied herein.


#%%

