#%% [markdown]
# DATS-6313 - FINAL EXAM
# Nate Ehat

#%% [markdown]
# FINAL EXAM

# TBU


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
from lifelines import KaplanMeierFitter

# import datetime as dt

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid') #ticks

print("\nSETTINGS ASSIGNED")

#%%
# DIRECTORY CONFIGURATION
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/exams/'

print("\nDIRECTORY CONFIGURED")

#%%
# 1. SURVIVAL ANALYSIS

# 1. Load the ”question1.csv” dataset on the blackboard.
    # The tongue dataset contains 3 columns and 80 rows.
    # The description of each column is a follows:
        # Type: Tumor DNA profile - 1= Aneuploid Tummor, 2 = Diploid Tummor
        # Time: Time to death or on-study time, weeks
        # Delta : Death indicator (0=alive, 1= dead)

df = pd.read_csv("question1.csv")
print(df.head())

#%%
print(df.info())

#%%


# (a) (14 points) Using ”lifelines” package and ”KaplanMeierFitter”:
    # Write a python program that estimate the survival function for:
        # Aneuploid Tumor (type 1 Tumor) and Diploid Tumor (type 2 Tumor).
    # Plot the survival functions for these two Tumor in one graph.
    # Add title, legend x-label and y-label to your plot.

#%%
# (4 points) What is the survival rate at week 50, for type 1 Tumor and Tumor 2?



#%%
# (4 points) Which Tumor is deadlier? Justify your answer using the survival function graph. .





aneuploid_tumor = df[df["type"] == 1]
diploid_tumor = df[df["type"] == 2]
new_df = pd.DataFrame()
new_df["Type1"] = pd.to_numeric(np.array(aneuploid_tumor.delta))
new_df["Type2"] = pd.to_numeric(np.concatenate((diploid_tumor.delta, np.repeat(np.array(diploid_tumor["type"].median()), 24))))


kmf = KaplanMeierFitter()
T = df['time']
E = df['delta']
type = df['type']
ix1 = (type == 1)
ix2 = (type == 2)
kmf.fit(T[ix1], E[ix1], label='Aneuploid')
ax = kmf.plot_survival_function()
kmf.fit(T[ix2], E[ix2], label='Diploid')
ax1 = kmf.plot_survival_function(ax=ax)
plt.xlabel("Time")
plt.ylabel("Survival")
plt.show()

#%%
# 2.
import statsmodels.api as sm
import TS_functions
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL



#%%
df = pd.read_csv("question2.csv")
print(df.head())
start_date = '1981-01-01'
date = pd.date_range(start_date, periods=len(df), freq='D')
print("========ROLLING MEAN & VARIANCE========")
plt.title("Rolling Mean from 1981")
plt.xlabel("Frequency")
plt.ylabel("Mean")
plt.plot(date, df.rolling(1).mean())
plt.grid()
plt.legend(['Mean'], loc='lower right')
plt.show()

plt.title("Rolling Variance from 1981")
plt.xlabel("Frequency")
plt.ylabel("Variance")
plt.plot(date, df.rolling(1).var())
plt.grid()
plt.legend(['Variance'], loc='lower right')
plt.show()

TS_functions.ADF_Cal(df.values)
TS_functions.kpss_test(df.values)

stl_data = df.copy()
stl_data.index = [i for i in range(stl_data.shape[0])]
stl_res = STL(stl_data, period=12).fit()
print("The strength of the seasonality is: ", str(TS_functions.seasonality_strength(df,stl_res.seasonal)))
print("The strength of the trend is: ", str(TS_functions.trend_strength(df,stl_res.trend)))
lags = 20
difference_data = TS_functions.series_differencing(df.values)
ry = sm.tsa.acf(difference_data, nlags=lags)
TS_functions.gpac_calc(ry, 7, 7)
na = 2
nb = 2
TS_functions.ACF_PACF_Plot(difference_data, 50, "ACF/PACF Difference Data")

# model = sm.tsa.ARMA(df.values, (na, nb)).fit(trend='nc', disp=0)
model = sm.tsa.ARIMA(endog=df.values, order=(na, 0, nb)).fit()
predictions = model.predict(start=0, end=len(df) - 1)
errors = df.values - predictions

for i in range(na):
    print("AR COEFFICIENT A{}".format(i), "is:", model.params[i])

for i in range(na):
    print("MA COEFFICIENT B{}".format(i), "is:", model.params[i + na])

intervals = model.conf_int()
for i in range(na):
    print("The Confidence Interval for a{}".format(i), "is:", intervals[i])
    print("The p-value for a{}".format(i), "is:", model.pvalues[i])
    print("The Standard Error for a{}".format(i), "is:", model.bse[i])
    print("\n")

for i in range(na):
    print("The Confidence Interval for b{}".format(i), "is:", intervals[i + na])
    print("The p-value for b{}".format(i), "is:", model.pvalues[i + na])
    print("The Standard Error for b{}".format(i), "is:", model.bse[i + na])
    print("\n")


print("The correlation coefficient between the data and prediction is: " + str(TS_functions.correlation_coefficient_cal(df.values, predictions)))

plt.title("Raw Data vs Forecast")
plt.xlabel("Frequency")
plt.ylabel("Raw Data")
plt.plot(df.values)
plt.plot(predictions)
plt.grid()
plt.legend(["Raw Data", "Predictions"], loc='lower right')
plt.show()


#%%

# 3. ROLLING MEAN / VARIANCE

df = pd.read_csv("question3.csv")
print(df.head())

#%%
print("========ROLLING MEAN & VARIANCE========")
plt.title("Rolling Mean")
plt.xlabel("Frequency")
plt.ylabel("Mean")
plt.plot(df.rolling(1).mean())
plt.grid()
plt.legend(['Mean'], loc='lower right')
plt.show()

plt.title("Rolling Variance")
plt.xlabel("Frequency")
plt.ylabel("Variance")
plt.plot(df.rolling(1).var())
plt.grid()
plt.legend(['Variance'], loc='lower right')
plt.show()

#%%
Helper.ADF_Cal(df.values)
Helper.kpss_test(df.values)

#%%
first_order_differencing = df - df.shift(1)

df = pd.read_csv("question3.csv")
print(df.head())
print("========ROLLING MEAN & VARIANCE First Order Difference========")
plt.title("Rolling Mean")
plt.xlabel("Frequency")
plt.ylabel("Mean")
plt.plot(first_order_differencing.rolling(1).mean())
plt.grid()
plt.legend(['Mean'], loc='lower right')
plt.show()

plt.title("Rolling Variance")
plt.xlabel("Frequency")
plt.ylabel("Variance")
plt.plot(first_order_differencing.rolling(1).var())
plt.grid()
plt.legend(['Variance'], loc='lower right')
plt.show()
Helper.ADF_Cal(first_order_differencing[1:])
Helper.kpss_test(first_order_differencing[1:])

lags = 20
ry = sm.tsa.acf(first_order_differencing, nlags=lags)
Helper.gpac_calc(ry, 7, 7)

Helper.ACF_PACF_Plot(first_order_differencing, 50, "Seasonal Difference ACF & PACF")

plt.title("Raw Data vs Seasonal Difference Data")
plt.xlabel("Frequency")
plt.ylabel("Raw Data")
plt.plot(df.values)
plt.plot(first_order_differencing)
plt.grid()
plt.legend(["Raw Data", "Season Differenced"], loc='lower right')
plt.show()


#%%
