#%% [markdown]
# DATS-6313 - LAB #4
# Nate Ehat

#%% [markdown]
# GPAC TABLE IMPLEMENTATION

# The main purpose of this LAB is to implement the GPAC array covered in lecture using Python program
# Test the accuracy of your code using an ARMA(na,nb) model.
# It is permitted to simulate the ARMA process using the statsmodels.
# Everyone needs to write their own GPAC code that generates GPAC table for various numbers of rows and columns.

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller

from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL

import scipy.stats as st
from scipy import signal

import statsmodels.api as sm

from numpy import linalg

# import datetime as dt

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid') #ticks

print("\nSETTINGS ASSIGNED")

#%%
# DIRECTORY CONFIGURATION
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'

print("\nDIRECTORY CONFIGURED")

#%%
# 1. Develop a python code that generates ARMA (na,nb) process.
    # The program should be written asking user to enter the following info.
    # Hint: Use statsemodels library.

def generate_ARMA():
    n = int(input("INPUT: NUMBER OF SAMPLES"))
    m = int(input("INPUT: MEAN OF WHITE NOISE"))
    v = int(input("INPUT: VARIANCE OF WHITE NOISE"))
    AR_order = int(input("INPUT: ORDER OF AR"))
    MA_order = int(input("INPUT: ORDER OF MA"))
    AR_coeff = []
    MA_coeff = []
    for i in range(AR_order):
        AR_coeff.append(float(input(f'INPUT: AR LAST COEFFICIENT [RANGE (0,1)]')))
    for i in range(MA_order):
        MA_coeff.append(float(input(f'INPUT: MA LAST COEFFICIENT [RANGE (0,1)]')))
    AR_params = np.array(AR_coeff)
    MA_params = np.array(MA_coeff)
    ar = np.r_[1, -AR_params]  # add zero-lag / negate
    ma = np.r_[1, MA_params]  # add zero-lag
    mean_ap_y = m * (1 + np.sum(MA_params)) / (1 + np.sum(AR_params))
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    arma_sample = arma_process.generate_sample(nsample=n, scale=np.sqrt(v)) + mean_ap_y # + m (??) #np.array() ???

    acf = int(input("INPUT: 1/0 FOR ACF"))
    if acf == 1:
        l = int(input("INPUT: NUMBER OF LAGS"))
        ry = arma_process.acf(lags=l)
        ry1 = ry[::-1]
        ry2 = np.concatenate((np.reshape(ry1, l), ry[1:]))
        return ry2
    elif acf == 0:
        return arma_sample

#%%
# 2. Edit the python code in step 1 that implement the GPAC table.
    # ry(j) == estimated autocovariance of y(t) at lag j
    # Output should be the GPAC table.

def GPAC(ry, j0, k0):
    def phi(ry, j, k): # determine Phi
        denominator = np.zeros(shape=(k, k))# placeholder zeroes for denominator
        for a in range(k): # replace denominator matrix with ry(j) values
            for b in range(k):
                denominator[a][b] = ry[abs(j + a - b)]
        numerator = denominator.copy() # copy of denominator for numerator
        numL = np.array(ry[j + 1:j + k + 1]) # generate last column for numerator
        numerator[:, -1] = numL # generate last column for numerator
        phi = np.linalg.det(numerator) / np.linalg.det(denominator)
        return phi

    table0 = [[0 for i in range(1, k0)] for i in range(j0)]

    for c in range(j0):
        for d in range(1, k0):
            table0[c][d - 1] = phi(ry, c, d)

    pac_val = pd.DataFrame(np.array(table0),
                       index=np.arange(j0),
                       columns=np.arange(1, k0))
    return pac_val

#%%
# 3. Using the developed code above, simulate ARMA(1,0) for 1000 samples as follows:
    # Use statsmodels library to simulate above ARMA (1,0) process.
    # 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) = 𝑒(𝑡)
        # e(t) is WN(1,2).

# EXAMPLE #1: ARMA (1,0): y(t) - 0.5y(t-1) = e(t)
    # N = 1000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 1
    # order_MA = 0
    # last_coef_AR = 0.5
    # last_coef_MA = 0

## SIMULATE ARMA
example1 = generate_ARMA()

#%%
# print(f'SHAPE: {len(example1)}')
# print(example1)

#%%
# 4. # Using python program, find the theoretical ACF for y(t) with lags = 15.
    # You can use the .generate_sample (# of samples, scales = std of WN noise) + mean (y)
        # mean(y) = μ𝑒(1+∑ 𝑏𝑖) 1+∑ 𝑎𝑖

## ACF FUNCTION
example1_acf = sm.tsa.stattools.acf(example1, nlags=15)

print(f'THEORETICAL ACF:')
print(example1_acf)

#%%
# 5. Using the theoretical ACF from the previous question:
# Display GPAC table for k=7 and j=7.
    # Do you see a pattern of constant column 0.5 and a row of zeros?
    # What is the estimated na and what is the estimated nb?
    # Hint: You should observe a clear pattern like below figure.
    # Utilize the seaborn package and heatmap to develop the following table.

## GENERATE GPAC TABLE
gpac1 = GPAC(example1_acf, 7, 7) #example1
print(gpac1)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac1, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()


#%%
# 6. # Using Python and statsmodels package, plot ACF and PACF of the process.
    # Plot the ACF and PACF in one figure using subplot 2x1.
    # Number of lags = 20.

#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF OF RAW DATA')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

#%%
ACF_PACF_Plot(example1, 20) #10

#%%
# 7.
# Repeat step 3, 4, 5 and 6 for the following 7 examples
    # with only 10000 samples

#%%
# Example 2: ARMA (0,1): y(t) = e(t) + 0.5e(t-1)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 0
    # order_MA = 1
    # last_coef_AR = 0 or 1 (??)
    # last_coef_MA = 0.5

## SIMULATE ARMA
example2 = generate_ARMA()

#%%
## ACF FUNCTION
example2_acf = sm.tsa.stattools.acf(example2, nlags=15)

print(f'THEORETICAL ACF:')
print(example2_acf)

#%%
## GENERATE GPAC TABLE
gpac2 = GPAC(example2_acf, 7, 7) #example1
print(gpac2)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac2, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()

#%%
## ACF / PACF PLOT
ACF_PACF_Plot(example2, 20)

#%%
# Example 3: ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 1
    # order_MA = 1
    # last_coef_AR = 0.5
    # last_coef_MA = 0.5

## SIMULATE ARMA
example3 = generate_ARMA()

#%%
## ACF FUNCTION
example3_acf = sm.tsa.stattools.acf(example3, nlags=15)

print(f'THEORETICAL ACF:')
print(example3_acf)

#%%
## GENERATE GPAC TABLE
gpac3 = GPAC(example3_acf, 7, 7)
print(gpac3)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac3, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()

#%%
## ACF / PACF PLOT
ACF_PACF_Plot(example3, 20)

# %%
# Example 4: ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 2
    # order_MA = 0
    # last_coef_AR = 0.2
    # last_coef_MA = 1 or 0 (??)

## SIMULATE ARMA
example4 = generate_ARMA()

#%%
## ACF FUNCTION
example4_acf = sm.tsa.stattools.acf(example4, nlags=15)

print(f'THEORETICAL ACF:')
print(example4_acf)

#%%
## GENERATE GPAC TABLE
gpac4 = GPAC(example4_acf, 7, 7)
print(gpac4)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac4, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()

#%%
## ACF / PACF PLOT
ACF_PACF_Plot(example4, 20)

# %%
# Example 5: ARMA (2,1): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 2
    # order_MA = 1
    # last_coef_AR = 0.2
    # last_coef_MA = 0.5

## SIMULATE ARMA
example5 = generate_ARMA()

#%%
## ACF FUNCTION
example5_acf = sm.tsa.stattools.acf(example5, nlags=15)

print(f'THEORETICAL ACF:')
print(example5_acf)

#%%
## GENERATE GPAC TABLE
gpac5 = GPAC(example5_acf, 7, 7) #example1
print(gpac5)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac5, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()

#%%
## ACF / PACF PLOT
ACF_PACF_Plot(example5, 20)

# %%
# Example 6: ARMA (1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 1
    # order_MA = 2
    # last_coef_AR = 0.5
    # last_coef_MA = 0.4

## SIMULATE ARMA
example6 = generate_ARMA()

#%%
## ACF FUNCTION
example6_acf = sm.tsa.stattools.acf(example6, nlags=15)

print(f'THEORETICAL ACF:')
print(example6_acf)

#%%
## GENERATE GPAC TABLE
gpac6 = GPAC(example6_acf, 7, 7) #example1
print(gpac6)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac6, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()

#%%
## ACF / PACF PLOT
ACF_PACF_Plot(example6, 20)

# %%
# Example 7: ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 0
    # order_MA = 2
    # last_coef_AR = 0 or 1 (??)
    # last_coef_MA = 0.4

## SIMULATE ARMA
example7 = generate_ARMA()

#%%
## ACF FUNCTION
example7_acf = sm.tsa.stattools.acf(example7, nlags=15)

print(f'THEORETICAL ACF:')
print(example7_acf)

#%%
## GENERATE GPAC TABLE
gpac7 = GPAC(example7_acf, 7, 7) #example1
print(gpac7)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac7, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()

#%%
## ACF / PACF PLOT
ACF_PACF_Plot(example7, 20)

# %%
# Example 8: ARMA (2,2): y(t)+0.5y(t-1)+0.2y(t-2) = e(t)+0.5e(t-1)-0.4e(t-2)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 2
    # order_MA = 2
    # last_coef_AR = 0.2
    # last_coef_MA = 0.4

## SIMULATE ARMA
example8 = generate_ARMA()

#%%
## ACF FUNCTION
example8_acf = sm.tsa.stattools.acf(example8, nlags=15)

print(f'THEORETICAL ACF:')
print(example8_acf)

#%%
## GENERATE GPAC TABLE
gpac8 = GPAC(example8_acf, 7, 7) #example1
print(gpac8)

#%%
## GPAC HEATMAP
plt.figure()
sns.heatmap(gpac8, annot=True)
plt.title('GPAC MATRIX')
plt.xlabel('K VALUES')
plt.ylabel('J VALUES')
plt.show()

#%%
## ACF / PACF PLOT
ACF_PACF_Plot(example8, 20)


#%%
