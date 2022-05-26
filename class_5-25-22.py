#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import datetime

import pandas_datareader as web

from toolbox import *

print("IMPORT SUCCESS")

#%%
# ASSIGN RANDOM SEED
np.random.seed(123)

#%%
# ASSIGN VARIABLES
N = 500
mean = 0
var = 1
x = np.random.normal(mean, var, N)

#%%
plt.figure(figsize=(8,8))
plt.plot(x)
plt.grid()
plt.show()

#%%
plt.figure(figsize=(8,8))
plt.hist(x, bins=50)
plt.grid()
plt.show()


#%%
y=[1,2,3,4,5]

lags=5
ry = np.array([1,0.1,-0.1,-0.4,-0.4])

#ry = np.linspace(0,10,lags)
#lags=8
ry1 = ry[::-1]
ryy = np.concatenate((ry1[:-1], ry))
print(ryy)

#%%
acfunc(y, lags)

#%%
## ACF STEM PLOT
a2 = np.arange(0,lags,1)
a3 = -a2[::-1]
x = np.concatenate((a3[:-1], a2))

plt.figure(figsize=(8,8))
(markers, stemlines, baseline) = plt.stem(x, ryy, markerfmt='o')
plt.title(f'ACF PLOT')
plt.setp(markers, color='red', marker='o')
plt.setp(baseline, color='gray', linewidth=2, linestyle='-')
plt.xlabel('LAGS')
plt.ylabel('AUTOCORRELATION VALUE')
#plt.xticks()
m = 1.96 / np.sqrt(len(ry))
plt.axhspan(-m, m, alpha=0.2, color='blue')
plt.show()

#%%



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


