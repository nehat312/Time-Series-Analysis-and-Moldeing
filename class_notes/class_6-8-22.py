#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import datetime

import pandas_datareader as web

from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL

import statsmodels.api as sm

from toolbox import *

print("IMPORT SUCCESS")

#%%

url = 'https://github.com/rjafari979/Time-Series-Analysis-and-Moldeing/blob/4278e1dd804726f0ab9ff1543c5d3e7c884ccd5f/daily-min-temperatures.csv'
path = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/daily-min-temperatures.csv'
data = pd.read_csv(path, header=0, index_col=0)#
print(data.info())

#%%
plt.figure()
data.plot()
plt.show()

#%%
temp = data['Temp']
temp = pd.Series(np.array(data['Temp']),
                        index=pd.date_range('1981-01-01',
                        periods=len(temp),
                        freq='d'),
                        name='daily-min-temp')
print(temp.describe())

#%%

STL = STL(temp)
res = STL.fit()
fig = res.plot()
plt.show()

#%%

T = res.trend
S = res.seasonal
R = res.resid

#%%
adjusted_seasonal = temp - S

F = np.maximum(0, 1-np.var(R)/np.var(np.array(T)+np.array(R)))
print(f'STRENGTH OF SEASONALITY: {100*F:.2f}%')

F = np.maximum(0, 1-np.var(R)/np.var(np.array(S)+np.array(R)))
print(f'STRENGTH OF SEASONALITY: {100*F:.2f}%')

#%%
N = 5000
mean = 1
var = 1

np.random.seed(123)
e = np.random.normal(mean, var, N)


#%%
# y(t) + 0.5y(t-1) = e(t)
y = np.zeros(len(e))
for t in range(len(e)):
    if t == 0:
        y[t] = e[t]
    #elif t == 1:
    #    y[t] = e[t]
    elif t == 01:

    else:
        y[t] = -0.5*y[t-1] + e[t]


#%%

plt.figure(figsize=(12,8))
plt.plot(y, label='y(t)', color = 'blue')
plt.plot(e, label='e(t)', color = 'orange')
plt.legend(loc='best')
plt.show()

#%%

acf_df_y = acf_df(y, 20)
#acf_df_e = acf_df(e, 5)


#%%
acf_stemplot(0, acf_df_y, 20)

#%%
acf_stemplot(0, acf_df_e, 20)

#%%

print(f'SAMPLE MEAN OF Y: {np.mean(y):.2f}')


#%%
#%%
N = 5000
mean = 0

var = 1

np.random.seed(123)
e = np.random.normal(mean, var, N)


#%%
# y(t) =  e(t) + 0.5*e(t)
y = np.zeros(len(e))
for t in range(len(e)):
    if t == 0:
        y[t] = e[t]
    #elif t == 1:
    #    y[t] = e[t]
    else:
        y[t] = e[t] + 0.5*e[t-1]


#%%

plt.figure(figsize=(12,8))
plt.plot(y, label='y(t)', color = 'blue')
plt.plot(e, label='e(t)', color = 'orange')
plt.legend(loc='best')
plt.show()

#%%


#%%
acf_stemplot(0, acf_df(y, 20), 20)

#%%
acf_stemplot(0, acf_df(e, 20), 20)

#%%

print(f'SAMPLE MEAN OF Y: {np.mean(y):.2f}')

#%%




#%%





#%%


