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
## THEORETICAL MEAN
#x = np.array([[15/16, 1/2], [1/4, 1]])
A = np.array([[15/16, 1/2], [1/4, 1]])
B = np.array([[1, 1/2], [1/2, 1]])


#%%
# DETERMINANTS
print(linalg.det(A))
print(linalg.det(B))

#%%
ry_B = linalg.det(A) / linalg.det(B)

#%%
## EXPERIMENTAL MEAN
    ## Essentially simulate ARMA Process

N = 1000
mean_e = 1
var_e = 1
e = np.random.normal(mean_e, var_e, N)


#%%
## SciPy DLSim - need to implement transfer function

num = [1, 0.25]
den = [1, 0.5]
system = [num, den, 1]
_,y = signal.dlsim(system, e) # Synthesized data

#%%
plt.figure(figsize=(8,8))
plt.plot(y)
plt.show()

#%%
## FIX ACF PLOT ##

plt.figure(figsize=(8,8))
plt.plot(acf_stemplot(0, y, 20))
plt.show()

#%%
np.random.seed(123)

#%%
# Experimental mean is np.mean(y)
# Experimental var is np.var(y)

# y(t) + 0.25y(t-1) + 0.75y(t-2) = e(t) + 0.5e(t-1)

N = 5000
mean_e = 1
var_e = 1

e = np.random.normal(mean_e, var_e, N)

#%%
## NUMERATOR = e
## DENOMINATOR = y

num = [1, 0.5, 0]
den = [1, .25, .75]


#%%
## SciPy DLSim - need to implement transfer function
system = [num, den, 1]
_,y = signal.dlsim(system, e) # Synthesized data

#%%
plt.figure(figsize=(8,8))
plt.plot(y)
plt.show()


#%%

print(f'Experimental mean is {np.mean(y)}')
#.766
print(f'Experimental var is {np.var(y)}')
#2.528

#%%
## STATSMODELS

# y(t) + 0.5y(t-1) = e(t) - .25e(t-1)

ar_params = [0.5] # last coefficient - 1.1 as example of false
ma_params = [0.25]
na, nb = 1, 1
ar = np.r_[1, ar_params]
ma = np.r_[1, ma_params]

print(ar)
print(ma)

#%%
## CONSTRUCT ARMA PROCESS
arma_process = sm.tsa.ArmaProcess(ar, ma)
print(f'IS THIS PROCESS STATIONARY?: {arma_process.isstationary}')

#%%

mean_y = mean_e * (1 + np.sum(ma_params)) / (1 + np.sum(ar_params))
print(mean_y)

#%%
y = arma_process.generate_sample(5000, scale=np.sqrt(var_e)) + mean_y
# scale param is basically computation of std dev

#%%
print(f'Experimental mean is {np.mean(y)}')
#
print(f'Experimental var is {np.var(y)}')
#


#%%
## THEORETICAL ACF
example1, y1 = arma_input_process_and_sample()
ry = arma_process.acf(lags=15)
ry1 = ry[::-1]
ry2 = np.concatenate((np.reshape(ry1, 15), ry[1:]))


#%%
## DOESNT WORK NEED ARRAYS
#B = den
#A = num

print(np.linalg.det(A) / np.linalg.det(B))


#%%



