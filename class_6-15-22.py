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
## L-M ALGORITHM



#%%
# DETERMINANTS


#%%


#%%


#%%


#%%
#%%
## DOESNT WORK NEED ARRAYS
#B = den
#A = num

#%%
print(np.linalg.det(A) / np.linalg.det(B))


#%%



