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

from toolbox import *

print("IMPORT SUCCESS")

#%%

url = 'https://raw.githubusercontent.com/nehat312/Time-Series-Analysis-and-Moldeing/master/AirPassengers.csv'

passengers = pd.read_csv(url, index_col='Month', parse_dates=True)
print(passengers.info())

#%%
y = passengers['#Passengers']
lags = 40
ACF_PACF_Plot(y, lags)


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


#%%
## MSE ** CHECK FOR ACCURACY ***
MSE = np.square(np.subtract(yf.values, np.ndarray(holtf.values))).mean
print(f'MSE - HOLT-LINEAR METHOD: {MSE}')

#%%
## HOLT LINEAR METHOD
holtt = ets.ExponentialSmoothing(yt,
                                trend='mul',
                                seasonal=None,
                                damped_trend=True).fit()

holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

fig, ax = plt.subplots()
ax.plot(yt, label='TRAIN DATA')
ax.plot(yf, label='TEST DATA')
ax.plot(holtf, label='HOLT-LINEAR METHOD PREDICTION')
plt.legend(loc='best')
plt.show()

#%%
## MSE ** CHECK FOR ACCURACY ***
MSE = np.square(np.subtract(yf.values, np.ndarray(holtf.values))).mean
print(f'MSE - HOLT-LINEAR METHOD: {MSE}')

#%%
## HOLT-WINTER METHOD
holtt = ets.ExponentialSmoothing(yt,
                                trend='mul', # 'add'
                                seasonal='mul',
                                damped_trend=True).fit()

holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

fig, ax = plt.subplots()
ax.plot(yt, label='TRAIN DATA')
ax.plot(yf, label='TEST DATA')
ax.plot(holtf, label='HOLT-WINTER METHOD PREDICTION')
plt.legend(loc='best')
plt.show()

#%%


#%%



#%%



