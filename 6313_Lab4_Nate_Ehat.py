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
    AR_params = np.array(AR_coeff) # * (-1)
    MA_params = np.array(MA_coeff)
    ar = np.r_[1, -AR_params]  # add zero-lag / negate
    ma = np.r_[1, MA_params]  # add zero-lag
    # ar = np.insert(AR_params, 1)
    # ma = np.insert(MA_params, 1)
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
        return arma_sample #arma_process,

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



#%%

## GENERATE GPAC TABLE
gpac1 = GPAC(example1[14:], 7, 7) #example1
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

ACF_PACF_Plot(example1, 20) #10

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




#%%
# Example 3: ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 1
    # order_MA = 1
    # last_coef_AR = 0.5
    # last_coef_MA = 0.5



# %%
# Example 4: ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 2
    # order_MA = 0
    # last_coef_AR = 0.2
    # last_coef_MA = 1 or 0 (??)



# %%
# Example 5: ARMA (2,1): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 2
    # order_MA = 1
    # last_coef_AR = 0.2
    # last_coef_MA = 0.5



# %%
# Example 6: ARMA (1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 1
    # order_MA = 2
    # last_coef_AR = 0.5
    # last_coef_MA = 0.4

# %%
# Example 7: ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 0
    # order_MA = 2
    # last_coef_AR = 0 or 1 (??)
    # last_coef_MA = 0.4

# %%
# Example 8: ARMA (2,2): y(t)+0.5y(t-1)+0.2y(t-2) = e(t)+0.5e(t-1)-0.4e(t-2)
# N = 10000
    # mean_wn = 1
    # var_wn = 2
    # order_AR = 2
    # order_MA = 2
    # last_coef_AR = 0.2
    # last_coef_MA = 0.4

# %%


#%%
# 8.
# Record observations about ACF and PACF plot of AR, MA and ARMA process
    # Across the above 8 examples.

#%% [markdown]
# EXAMPLE 1:

#%% [markdown]
# EXAMPLE 2:

#%% [markdown]
# EXAMPLE 3:



#%%




#%%

print("====================QUESTION 1==========================")
samples_size = input("Enter # of samples: ")
mean = input("Enter Mean (WN): ")
var = input("Enter Variance (WN): ")
ar_order = input("Enter AR order: ")
ma_order = input("Enter MA order: ")

ar_inputs = []
if int(ar_order) > 0:
    for i in range(int(ar_order)):
        text = "Enter a" + str(i + 1) + ": "
        input_value = input(text)
        ar_inputs.append(float(input_value))

ma_inputs = []
if int(ma_order) > 0:
    for i in range(int(ma_order)):
        text = "Enter b" + str(i + 1) + ": "
        input_value = input(text)
        ma_inputs.append(float(input_value))

ar = np.r_[1, ar_inputs]
ma = np.r_[1, ma_inputs]

samples_size = int(samples_size)
ar_order = int(ar_order)
ma_order = int(ma_order)
mean = float(mean)
var = float(var)

arma_process = sm.tsa.ArmaProcess(ar, ma)
mean_y = mean * (1 + np.sum(ma_inputs)) / (1 + np.sum(ar_inputs))
y = arma_process.generate_sample(samples_size, scale=np.sqrt(var) + mean_y)


acf_lags = 60
ry = arma_process.acf(lags=acf_lags)
toolbox.gpac_calc(ry, 5, 5)

acf_lags = 15
new_ry = toolbox.auto_correlation_cal(y, acf_lags)
toolbox.gpac_calc(new_ry, 7, 7)

lags = 20

acf = arma_process.acf(lags=lags)
pacf = arma_process.pacf(lags=lags)

fig, axs = plt.subplots(2, 1)
fig.subplots_adjust(hspace=1.5, wspace=0.5)
axs = axs.ravel()

ry = arma_process.acf(lags=lags)
a1 = ry
a2 = a1[::-1]
a = np.concatenate((a2[:-1], a1))
x1 = np.arange(0, lags)
x2 = -x1[::-1]
x = np.concatenate((x2[:-1], x1))
(marker, stemlines, baselines) = axs[0].stem(x, a,
                                             use_line_collection=True, markerfmt='o')
plt.setp(marker, color='red', marker='o')
plt.setp(baselines, color='gray', linewidth=2, linestyle='-')
m = 1.96 / np.sqrt(100)
axs[0].axhspan(-m, m, alpha=.2, color='blue')
axs[0].set_title("ACF GPAC")
axs[0].set_ylabel("ACF")
axs[0].set_xlabel("Frequency")

ry = arma_process.pacf(lags=lags)
a1 = ry
a2 = a1[::-1]
a = np.concatenate((a2[:-1], a1))
x1 = np.arange(0, lags)
x2 = -x1[::-1]
x = np.concatenate((x2[:-1], x1))
(marker, stemlines, baselines) = axs[1].stem(x, a,
                                             use_line_collection=True, markerfmt='o')
plt.setp(marker, color='red', marker='o')
plt.setp(baselines, color='gray', linewidth=2, linestyle='-')
m = 1.96 / np.sqrt(100)
axs[1].axhspan(-m, m, alpha=.2, color='blue')
axs[1].set_title("PACF GPAC")
axs[1].set_ylabel("PACF")
axs[1].set_xlabel("Frequency")

plt.show()


samples_size = 5000
arma_process = sm.tsa.ArmaProcess(ar, ma)
mean_y = mean * (1 + np.sum(ma_inputs)) / (1 + np.sum(ar_inputs))
arma_process.generate_sample(samples_size, scale=np.sqrt(var) + mean_y)


acf_lags = 60
ry = arma_process.acf(lags=acf_lags)
toolbox.gpac_calc(ry, 5, 5)


samples_size = 10000
arma_process = sm.tsa.ArmaProcess(ar, ma)
mean_y = mean * (1 + np.sum(ma_inputs)) / (1 + np.sum(ar_inputs))
arma_process.generate_sample(samples_size, scale=np.sqrt(var) + mean_y)

acf_lags = 60
ry = arma_process.acf(lags=acf_lags)
toolbox.gpac_calc(ry, 5, 5)



#%%

#%%
# 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) = 𝑒(𝑡)
    # e(t) is WN(1,2).

N1 = 1000
np.random.seed(42)
mean_e1 = 1
var_e1 = 2
e = np.random.normal(mean_e1, var_e1, N1)

AR_params = [0.5]
MA_params = [0]
na, nb = 1, 0
ar = np.r_[1, AR_params]
ma = np.r_[1, MA_params]

print(ar)
print(ma)

#%%
## CONSTRUCT ARMA PROCESS
arma_process = sm.tsa.ArmaProcess(ar, ma)
print(f'IS ARMA PROCESS STATIONARY?: {arma_process.isstationary}')

#%%
mean_y = mean_e1 * (1 + np.sum(MA_params)) / (1 + np.sum(AR_params))
print(f'MEAN Y: {mean_y:.3f}')

#%%
y = arma_process.generate_sample(N1, scale=np.sqrt(var_e1)) + mean_y

#%%
print(f'EXPERIMENTAL MEAN: {np.mean(y)}')
print('*'*50)
print(f'EXPERIMENTAL VARIANCE: {np.var(y)}')

#%%
## THEORETICAL ACF
ry = arma_process.acf(lags=15)
ry1 = ry[::-1]
ry2 = np.concatenate((np.reshape(ry1, 15), ry[1:]))

print(f'THEORETICAL MEAN: {np.mean(ry2)}')
print('*'*50)
print(f'THEORETICAL VARIANCE: {np.var(ry2)}')