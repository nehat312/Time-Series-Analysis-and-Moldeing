#%% [markdown]
## CLASS 6-22-22

## SARIMA

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy import signal
from scipy.stats import chi2

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL

from numpy import linalg

# import datetime
# import pandas_datareader as web

from toolbox import *

print("IMPORT SUCCESS")

#%%
## SIMULATE SARIMA MODEL

# y(t) - 0.5y(t-3) + 0.6y(t-6) = e(t)
# e(t) WN(0,1)

np.random.seed(12)
bn = [1]
an = [1, 0, 0, -0.5, 0, 0, 0.6]

# Proj J Toolbox smh
# an, bn = check_num_den_size(an, bn)
system = (bn,an,1)
e = np.random.normal(0,1,size=10000)

_,y_new = signal.dlsim(system, e)

#%%
# flatten import?
y = np.ndarray.flatten(y_new)
#y = pd.DataFrame(y_new)

#%%
y_df = ac_func(y, lag=7)
rolling_mean_var_plots(y_df, col_index=0)


#%%
# ADF / KPSS not necessary since we know it is stationary


#%%
y_train = y[:round(len(y)*.95)]
y_test = y[round(len(y)*.95)]

#%%
## 1-STEP PREDICTION
# y(t) - 0.5y(t-3) + 0.6y(t-6) = e(t)
# e(t) WN(0,1)

model = sm.tsa.SARIMAX(y_train,
                       order=(0,0,0),
                       seasonal_order=(2,0,0,3))
# FIT
model_fit = model.fit()
# PREDICT
y_model_hat = model_fit.predict(start=1, end=len(y_train)-1)
# FORECAST
y_model_h_t = model_fit.forecast(steps=len(y_test))
# RESIDUALS
residual_e = y_train[:-1] - y_model_hat
forecast_e = y_test - y_model_h_t

print(f'VARIANCE - RESIDUAL VS. FORECAST ERRORS: {np.var(residual_e,forecast_e)}')


#%%
## 1-STEP PREDICTION


lags = 50
na = 6
nb = 8
y_hat_t_1 = []

for i in range(0, len(y_train)):
    if i==0:
        y_hat_t_1.append(0)
    elif i==1:
        y_hat_t_1.append(0)
    elif i==2 or i==3 or i==4:
        y_hat_t_1.append(0.5 * y_train[i-2])
    else:
        y_hat_t_1.append(0.5 * y_train[i - 2] - 0.6 * y_train[i - 5])

#%%
plt.figure(figsize=(12,8))
plt.plot(y_train[1:100], label='TRAIN DATA')
plt.plot(y_hat_t_1[2:100], label='1-STEP AHEAD PREDICTION')
plt.title('TRAIN VS. 1-STEP PREDICTION')
plt.legend(loc='best')
plt.show()

#%%
## ACF STEMPLOT
e = y_train[3:] - y_hat_t_1[2:-1]
re = cal_autocorr(e, lags, 'ACF of RES ERROR')

# Q Critical
# Chi Critical

#%%

len(y_test)
#%%
## H-STEP PREDICTION

y_hat_t_h = []

for h in range(1, len(y_test)):
    if h == 1:
        y_hat_t_h.append(0.5 * y_train[-2] - 0.6 * y_train[-5])
    elif h == 2:
        y_hat_t_h.append(0.5 * y_train[-1] - 0.6 * y_train[-4])
    elif h == 3:
        y_hat_t_h.append(0.5 * y_train[0] - 0.6 * y_train[-3])
    elif h == 4:
        y_hat_t_h.append(0.5 * y_hat_t_h[0] - 0.6 * y_train[-2])
    elif h == 5:
        y_hat_t_h.append(0.5 * y_hat_t_h[1] - 0.6 * y_train[-1])
    elif h == 6:
        y_hat_t_h.append(0.5 * y_hat_t_h[2] - 0.6 * y_train[0])
    else:
        y_hat_t_h.append(0.5 * y_hat_t_h[h-4] - 0.6 * y_hat_t_h[h-7])


#%%

## SURVIVAL ANALYSIS
from lifelines import KaplanMeierFitter

#%%
durations = [5,6,6,2,5,4]
event = [1,0,0,1,1,1]


#%%
ax = plt.subplots(111)
kmf = KaplanMeierFitter()
kmf.fit(durations, event, label='# OF MINUTES ON SITE')
kmf.plot_survival_function(ax=ax)
plt.show()


#%%
plt.figure(figsize=(12,8))
plt.plot(y_train[1:100], label='TRAIN DATA')
plt.plot(y_hat_t_1[2:100], label='1-STEP AHEAD PREDICTION')
plt.title('TRAIN VS. 1-STEP PREDICTION')
plt.legend(loc='best')
plt.show()


