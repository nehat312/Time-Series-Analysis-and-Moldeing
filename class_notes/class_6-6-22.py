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
import statsmodels.api as sm

from toolbox import *

print("IMPORT SUCCESS")

#%%

url = 'https://github.com/rjafari979/Time-Series-Analysis-and-Moldeing/blob/5de42e256c23f8d6f12be3f5facecf8d877644d4/auto.clean.csv'

cars = pd.read_csv(url)#index_col='Month',
print(cars.info())


#%%
# T-T/S
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

#%%

### OLS MODEL ###
model = sm.OLS(Y_train, X_train).fit()

#%%

## REMOVE HIGHEST P-VALUE FEATURES
    # Iterative process
#%%
## MULTI-COLLINEARITY
np.random.seed(123)

N = 100
x1 = np.random.randn(N, 1)
x2 = np.random.randn(N, 1)
x3 = np.random.randn(N, 1)
x4 = 2*x1 #np.random.randn(N, 1)

#%%
X = np.hstack((x1, x2, x3, x4))

#%%
print(X.shape)

#%%
from numpy import linalg as LA

#%%
print(f'CONDITION NUMBER OF X: {LA.cond(X)}')

#%%
H = np.matmul(X.T, X)
_,d,_ = np.linalg.svd(H)
print(f'SINGULAR VALUES OF X: {d}')

#%%
## CORRELATION MATRIX
H = np.corrcoef(H)
print(f'CORRELATION MATRIX: \n{H}')

#%%
## HEATMAP
plt.figure(figsize=(8,8))
sns.heatmap(H, annot=True)
plt.show()

#%%
## PRINCIPAL COMPONENT ANALYSIS




#%%
## MIN-MAX SCALER

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler

#%%

X = sm.add_constant(X)
scaler = MinMaxScaler(feature_range=(0,1))
dataset = df.reshape(-1,1)
dataset = scaler.fit_transform(dataset)

#%%

