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

from numpy import linalg
from scipy import signal

from toolbox import *

print("IMPORT SUCCESS")


#%%


a = [1, -.7, .12]
b = [1, -.8, .15]

a = np.array(a)
b = np.array(b)

#%%

np.roots(a)
np.roots(b)

#%%
####  HW #4

## VIRTUAL CLASS
# ARIMA EXAMPLE / RUN-THROUGH


# sm.tsa.arima.ARIMA(y, order=(na,0,nb), trend=’n’).fit()

# arma_process = sm.tsa.ArmaProcess
arparams = np.array([0.5, 0.2])
maparams = np.array([])

na = len(arparams)
nb = len(maparams)
ar = np.r_[1, arparams]
ma = np.r_[1, maparams]


#%%
### LIVE CLASS CODE - FUTURE WORK
# PROCESS OF USER INPUT CODE FUNCTION:
    # Order of AR Portion
    # Order of MA Portion
    # Coefficient of A1 ... An


#%%


## Summary assessment process:
        # Confidence Interval
        # P-Values
            # High = Insignificant
            #

#%%
# DETERMINANTS


#%%


#%%


#%%


#%%
# Sm.tsa.arima.ARIMA(y, order=(na,0,nb), trend=’n’).fit

#%%



