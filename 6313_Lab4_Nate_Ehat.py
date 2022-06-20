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

import datetime as dt
import scipy.stats as st

from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid') #ticks

#%%
# DIRECTORY CONFIGURATION
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'

print("\nDIRECTORY ASSIGNED")

#%%
# 1. Using the Python program to load the ‘Airpassengers.csv’.
passengers_link = 'https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/AirPassengers'

passengers = pd.read_csv(passengers_link + '.csv', index_col='Month', parse_dates=True) # #parse_dates=True, infer_datetime_format=True, parse_dates=['Unnamed: 0']
print("\nIMPORT SUCCESS")

#%%

# 1. Develop a python code that generates ARMA (na,nb) process. The program should be written in a way that askes a user to enter the following information. Hint: Use statsemodels library.
    # a. Enter the number of data samples: ______________
    # b. Enter the mean of white noise: _____________
    # c. Enter the variance of the white noise: _____________
    # d. Enter AR order: ______________
    # e. Enter MA order: _____________
    # f. Enter the coefficients of AR (you need to include a hint how this should be entered):____
    # g. Enter the coefficients of MA (you need to include a hint how this should be entered):____




#%%
# 2.

# Edit the python code in step 1 that implement the GPAC table using the following equation.
# The output should be the GPAC table.

#%%
# 3.

# Using the developed code above, simulate ARMA(1,0) for 1000 samples as follows:
    # 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) = 𝑒(𝑡)
        # e(t) is WN(1,2).
    # Use statsemodels library to simulate above ARMA (1,0) process.
    # You can use the .generate_sample (# of samples, scales = std of WN noise) + mean (y)
        # mean(y) = μ𝑒(1+∑ 𝑏𝑖) 1+∑ 𝑎𝑖

#%%
# 4.
# Using python program, find the theoretical ACF for y(t) with lags = 15.
# Hint: You can use the following function:
    # arma_process.acf(lags=lags)


#%%
# 5.
# Using the theoretical ACF from the previous question, display GPAC table for k=7 and j=7.
# Do you see a pattern of constant column 0.5 and a row of zeros?
# What is the estimated na and what is the estimated nb?
# Hint: You should observe a clear pattern like below figure.
# You need to utilize the seaborn package and heatmap to develop the following table.


gpac = GPAC(ar1000[14:], 7, 7)
plt.figure()
sns.heatmap(gpac, annot=True)
plt.title('GPAC Table')
plt.xlabel('k values')
plt.ylabel('j values')
plt.show()

#%%
# 6.
# Using Python and statsmodels package, plot the ACF and PACF of the process.
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
# 7.
# Repeat step 3, 4, 5 and 6 for the following 7 examples with only 10000 samples
# Example 2: ARMA (0,1): y(t) = e(t) + 0.5e(t-1)
# Example 3: ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)
# Example 4: ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
# Example 5: ARMA (2,1): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
# Example 6: ARMA (1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# Example 7: ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)
# Example 8: ARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)



#%%
# 8.
# Write down your observation about the ACF and PACF plot of the AR, MA and ARMA process in the above 8 examples.


#%% [markdown]
# TBU



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