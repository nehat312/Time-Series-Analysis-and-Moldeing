#%% [markdown]
# DATS-6313 - EXAM #2
# NATE EHAT

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL

import scipy.stats as st
from scipy import signal

from sklearn.model_selection import train_test_split

from numpy import linalg
from lifelines import KaplanMeierFitter


from toolbox import *
# import TS_functions

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
# 1. Load the ”question1.csv” dataset on the blackboard.
    # The tongue dataset contains 3 columns and 80 rows.
    # The description of each column is a follows:
        # Type: Tumor DNA profile - 1= Aneuploid Tummor, 2 = Diploid Tummor
        # Time: Time to death or on-study time, weeks
        # Delta : Death indicator (0=alive, 1= dead)

df_1 = pd.read_csv(current_folder + 'question1.csv', index_col=0)
print(df_1.head())
print('*'*50)
print(df_1.info())

#%%
# (a) (14 points) Using ”lifelines” package and ”KaplanMeierFitter”:
    # Write a python program that estimate the survival function for:
        # Aneuploid Tumor (type 1 Tumor) and Diploid Tumor (type 2 Tumor).
    # Plot the survival functions for these two Tumor in one graph.
    # Add title, legend x-label and y-label to your plot.

aneuploid_tumor = df_1[df_1["type"] == 1]
diploid_tumor = df_1[df_1["type"] == 2]
tumor_df = pd.DataFrame()
tumor_df["Type1"] = pd.to_numeric(np.array(aneuploid_tumor.delta))
tumor_df["Type2"] = pd.to_numeric(np.concatenate((diploid_tumor.delta, np.repeat(np.array(diploid_tumor["type"].median()), 24))))

#%%
kmf = KaplanMeierFitter()
T = df_1['time']
E = df_1['delta']
type = df_1['type']
ix1 = (type == 1)
ix2 = (type == 2)
kmf.fit(T[ix1], E[ix1], label='ANEUPLOID')
ax = kmf.plot_survival_function()
kmf.fit(T[ix2], E[ix2], label='DIPLOID')
ax1 = kmf.plot_survival_function(ax=ax)
plt.title('TUMOR SURVIVAL RATE BY TYPE')
plt.xlabel('TIME')
plt.ylabel('SURVIVAL')
plt.show()


#%% [markdown]
# (4 points) What is the survival rate at week 50, for type 1 Tumor and Tumor 2?

# At week 50, survival rate for Type 1 (Aneuploid) is ~68%
# At week 50, survival rate for Type 2 (Diploid) is ~50%

#%% [markdown]
# (4 points) Which Tumor is deadlier? Justify your answer using the survival function graph. .

# Type 2 (Diploid) is deadlier than Type 1 (Aneuploid)
# Blue line of chart representing Type 1 (Aneuploid) exhibits higher survival rates across all time intervals
# Thus, the Orange line representing Type 2 (Diploid) is deadlier, with consistently lower survival rates

#%%
# 2. Load the ”question2.csv” dataset from the blackboard.
    # Using python program and the package of your interest:
        # Fit the dataset into an ARIMA model.
        # Use all the data as the train set.

    # This is a non-seasonal time series dataset with 1000 observations y(t).
    # The data is collected daily starting from Jan 1st 1981 .

df_2 = pd.read_csv(current_folder + 'question2.csv', header=None)
print(df_2.head())
print('*'*50)
print(df_2.info())

#%%
# (a) (2 points) Plot the time series dataset versus time.
    # Make sure that the x-axis not be crowded.
    # Add title, legend, x-label, y-label and grid to your graph.

start_date = '1981-01-01'
date = pd.date_range(start_date, periods=len(df_2), freq='D')
df_2.index = date
print(df_2.head())

#%%
## LINEPLOT - DF_2 OVER TIME
plt.title("DATASET OVER TIME (1981-)")
plt.xlabel("TIME")
plt.ylabel("VALUE")
plt.plot(df_2)
plt.legend(['DF_2'], loc='best')
plt.tight_layout(pad=1)
# plt.grid()
plt.show()

#%%
# (b) (3 points) Plot the rolling mean and variance versus time

## ROLLING MEAN / VARIANCE ##

# SET COLUMN INDICES FOR CHART TITLES
df_2_index = df_2.columns[0] #DF_2'

# print(df_2_index)
# print(df_2[0])

#%%
rolling_mean_var_plots(rolling_mean_var(df_2[0]), df_2_index)
plt.show()

#%%
import warnings
warnings.filterwarnings('ignore')

#%%
# Perform an ADF-Test & KPSS-test.
    # Is the dataset stationary? Explain your answer.

print(f'ADF TEST:')
print(adf_test([0], df_2))
print('*'*100)
print(f'KPSS TEST:')
print(kpss_test(df_2[0]))
print('*'*100)

#%% [markdown]
# Per rolling mean / variance charts, and confirmed by ADF / KPSS tests, this data set is NOT stationary.

#%%
# (c) (3 points) Perform a time series decomposition on the raw data

stl_data = df_2.copy()
stl_data.index = [i for i in range(stl_data.shape[0])]
stl_res = STL(stl_data, period=12).fit()

#%%
plt.figure(figsize=(12,8))
fig = stl_res.plot()
plt.xlabel('DATE', fontsize=12)
plt.tight_layout(pad=1)
plt.show()

#%%
# Find the strength of the trend and strength of the seasonality.
    # Is the dataset seasonal or trended? Explain your answer.
T = stl_res.trend
S = stl_res.seasonal
R = stl_res.resid

#%%
## STRENGTH OF TREND ##
def strength_of_trend(residual, trend):
    var_resid = np.nanvar(residual)
    var_resid_trend = np.nanvar(np.add(residual, trend))
    return 1 - (var_resid / var_resid_trend)

F = np.maximum(0, 1-np.var(R)/np.var(np.array(T)+np.array(R)))

print(f'STRENGTH OF TREND: {100*F:.3f}% or {strength_of_trend(R, T):.5f}')

#%%
## STRENGTH OF SEASONAL ##
def strength_of_seasonal(residual, seasonal):
    var_resid = np.nanvar(residual)
    var_resid_seasonal = np.nanvar(np.add(residual, seasonal))
    return 1 - (var_resid / var_resid_seasonal)

F = np.maximum(0, 1-np.var(R)/np.var(np.array(S)+np.array(R)))

print(f'STRENGTH OF SEASONALITY: {100*F:.3f}% or {strength_of_seasonal(R, S):.5f}')

#%%
# (d) (4 points) If the raw dataset is not stationary, transform it to a stationary dataset

## FIRST-ORDER DIFFERENCING
df_2_1diff = differencer(df_2[0], 1, df_2.index)
print(df_2)

#%%
## FIRST-ORDER DIFFERENCING - SET COLUMN INDICES FOR CHART TITLES
df_2_1diff_col_index = df_2_1diff.columns[0].upper()
print(df_2_1diff_col_index)

#%%
# Plot the rolling mean and variance to re-confirm stationarity.
    # Explain your answer.

## FIRST-ORDER DIFFERENCING - GENERATE PLOTS
rolling_mean_var_plots(rolling_mean_var(df_2_1diff), df_2_1diff_col_index)
plt.show()

#%%
# Display the ADF-test & KPSS-test.

## FIRST-ORDER DIFFERENCING - ADF TEST
print('ADF FIRST-ORDER DIFF:')
print(adf_test(['1diff'], df_2_1diff))
print('*'*100)

## FIRST-ORDER DIFFERENCING - KPSS TEST
print('KPSS FIRST-ORDER DIFF:')
print(kpss_test(df_2_1diff['1diff']))
print('*'*100)

#%% [markdown]
# Rolling mean / variance plots appear to display stationary data over time
# Stationarity is confirmed by deriving and assessing p-values along with ADF + KPSS statistics

#%%
# (e) (3 points) Using the GPAC code display the GPAC table (7,7) and estimate a potential order for AR and MA.
    # Highlight the potential pattern.
    # Take a screen shot of the high- lighted pattern and place it into your solution manual.
    # Hint: You may need to perform non-seasonal differencing before using GPAC.

## GENERATE ARMA PROCESS ##
def generate_ARMA():
    n = int(input("INPUT: NUMBER OF SAMPLES"))
    m = int(input("INPUT: MEAN OF WHITE NOISE"))
    v = int(input("INPUT: VARIANCE OF WHITE NOISE"))
    AR_order = int(input("INPUT: ORDER OF AR"))
    MA_order = int(input("INPUT: ORDER OF MA"))
    AR_coeff = []
    MA_coeff = []
    for i in range(AR_order):
        AR_coeff.append(float(input(f'INPUT: AR LAST COEFFICIENT [RANGE (-1,1)]')))
    for i in range(MA_order):
        MA_coeff.append(float(input(f'INPUT: MA LAST COEFFICIENT [RANGE (-1,1)]')))
    AR_params = np.array(AR_coeff)
    MA_params = np.array(MA_coeff)
    ar = np.r_[1, AR_params]
    ma = np.r_[1, MA_params]
    mean_ap_y = m * (1 + np.sum(MA_params)) / (1 + np.sum(AR_params))
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    arma_sample = arma_process.generate_sample(nsample=n, scale=np.sqrt(v)) + mean_ap_y # + m (??) #np.array() ???

    acf = int(input("INPUT: 1/0 FOR ACF"))
    if acf == 1:
        l = int(input("INPUT: NUMBER OF LAGS"))
        ry = arma_process.acf(lags=l)
        ry1 = ry[::-1]
        ry2 = np.concatenate((np.reshape(ry1, l), ry[1:]))
        return ry
    elif acf == 0:
        return arma_sample

#%%
## GPAC MATRIX ##
def GPAC(ry, j0, k0):
    def phi(ry, j, k):
        den = np.zeros(shape=(k,k))
        for a in range(k):
            for b in range(k):
                den[a][b] = ry[abs(j+a-b)]
        num = den.copy()
        numl = np.array(ry[abs(j+1):abs(j+k+1)])
        num[:, -1] = numl
        phi = np.linalg.det(num)/np.linalg.det(den)
        return phi
    tab = [[0 for i in range(1, k0+1)] for w in range(j0)]
    for c in range(0, j0):
        for d in range(1, k0+1):
            tab[c][d-1] = phi(ry, c, d)
    pac_val = pd.DataFrame(np.array(tab),
                           index = np.arange(j0),
                           columns=np.arange(1, k0+1))
    return pac_val

#%%

## SIMULATE ARMA
# q2_arma = generate_ARMA()

#%%
## ACF FUNCTION
# q2_arma_acf = sm.tsa.stattools.acf(q2_arma, nlags=15)
#
# print(f'THEORETICAL ACF:')
# print(q2_arma_acf)

#%%
# 5. Using the theoretical ACF from the previous question:
# Display GPAC table for k=7 and j=7.
    # Do you see a pattern of constant column 0.5 and a row of zeros?
    # What is the estimated na and what is the estimated nb?
    # Hint: You should observe a clear pattern like below figure.
    # Utilize the seaborn package and heatmap to develop the following table.

## GENERATE GPAC TABLE
# gpac_q2 = GPAC(q2_arma_acf, 7, 7)
# print(gpac_q2)

#%%
## GPAC HEATMAP
# plt.figure()
# sns.heatmap(gpac_q2, annot=True)
# plt.title('GPAC MATRIX')
# plt.xlabel('K VALUES')
# plt.ylabel('J VALUES')
# plt.show()

#%%
# (f) (3 points) Plot the ACF and PACF of the differenced dataset for 50 lags.
    # Using the ’tail- off’ & ’cut-off’ scenario:
        # Estimate the order of AR and MA for this dataset?
        # Does the estimated order confirm the order from the GPAC table? Justify your answer.


#%%
# (g) (3 points) Using the estimated order from the previous sections, and using maximum likelihood estimation:
    # Estimate the parameter of the ARIMA model.
# Print the estimated parameters with the corresponding confident intervals.
    # Hint: You can use your own LM algorithm or the Python package.


#%%
# (h) (4 points) Using the estimated order and estimated parameters derived in the previous sections:
    # Develop the forecast function and predict yˆ(t) (one-step prediction).
    # Calculate the residual errors.
    # Plot the ACF of residual errors for 20 lags.


#%%
# (i) (4 points) Plot y(t) versus yˆ(t) in one graph
    # for the first 200 samples.
    # Perform a χ2 test with α = 0.01 to verify the accuracy of the derived model.
    # Analyze χ2 result on the validation of the derived model.
    # Justify your answer. (χ2 table is attached at the end).
    # What is the final ARIMA model that best represent the data?



#%%
# (j) (4 points) Calculate the correlation coefficient between y(t) and yˆ(t)
    # Display the scatter plot between them.
    # What does the plot and correlation coefficient tell about accuracy of prediction?




#%%
# 3. Load the ”question3.csv” dataset from the blackboard.
    # This is a seasonal time series dataset with seasonality order of 3.
    # Using python program / package of your interest, answer the following questions.

df_3 = pd.read_csv(current_folder + 'question3.csv', header=[0]) #index_col=0,
print(df_3.head())
print('*'*50)
print(df_3.info())

#%%
# (a) (5 points) Stationarity check:
    # Plot the mean and variance versus time (rolling mean and variance)

## ROLLING MEAN / VARIANCE ##

# SET COLUMN INDICES FOR CHART TITLES
df_3_index = df_3.columns[0]
print(df_3_index)

#%%
rolling_mean_var_plots(rolling_mean_var(df_3['y']), df_3_index)
plt.show()

#%%
# Perform ADF-test & KPSS-test.
print(f'ADF TEST:')
print(adf_test(['y'], df_3))
print('*'*100)
print(f'KPSS TEST:')
print(kpss_test(df_3['y']))
print('*'*100)

#%% [markdown]
# Is the dataset stationary? Explain why.
# Per rolling mean / variance charts, and confirmed by ADF / KPSS tests, this data set is NOT stationary.
    # ADF Statistic == 0.197 (far worse than associated confidence intervals)
    # KPSS Statistic == 13.63 (does not compare favorably to associated confidence intervals)
    # P-Value for ADF Test == 0.97 (too high for stationarity)
    # P-Value for KPSS Test == 0.01 (too low for stationarity)

#%%
# (b) (5 points) Perform a first order seasonal differing and check the stationarity
    # by plotting the rolling mean and variance and an ADF-test & KPSS-test.
    # Is the seasonally differenced dataset stationary? Explain why?

df_3_szn_diff1 = df_3 - df_3.shift(3)
print(df_3_szn_diff1)

#%%
## ROLLING MEAN / VARIANCE ##

# SET COLUMN INDICES FOR CHART TITLES
df_3_szn_diff1_index = df_3.columns[0]
print(df_3_szn_diff1_index)

#%%
rolling_mean_var_plots(rolling_mean_var(df_3_szn_diff1['y']), df_3_szn_diff1_index)
plt.show()

#%%
# Perform ADF-test & KPSS-test.
print(f'ADF TEST:')
print(adf_test(['y'], df_3_szn_diff1))
print('*'*100)

df_3_szn_diff1_kpss = df_3_szn_diff1.fillna(0)
print(f'KPSS TEST:')
print(kpss_test(df_3_szn_diff1_kpss['y']))
print('*'*100)

#%%
# (c) (5 points) Display the GPAC table (7,7) using the seasonally differenced dataset.
    # Highlight the pattern.
    # What is the estimated order of the SARIMA model?
    # Take a screen shot and highlight the pattern.
    # Include the screen shot into your solution manual.

def difference_alt(dataset, interval):
    diff = []
    for i in range(interval, len(dataset)):
        values = dataset[i] - dataset[i-interval]
        diff.append(values)
    return diff

# lags = 20
# df3_diff_data = difference_alt(df_3.values, 1)
# ry = sm.tsa.acf(df3_diff_data, nlags=lags)

#%%
## GENERATE GPAC TABLE
# gpac1 = GPAC(ry, 7, 7)
# print(gpac1)

#%%
## GPAC HEATMAP
# plt.figure()
# sns.heatmap(gpac1, annot=True)
# plt.title('GPAC MATRIX')
# plt.xlabel('K VALUES')
# plt.ylabel('J VALUES')
# plt.show()


#%%
# (d) (5 points) Plot the ACF and PACF of the seasonally differenced dataset for 50 lags.
    # What is the estimated order of the SARIMA model?

# ACF_PACF_Plot(df3_diff_data, 50, 'ACF/PACF DIFFERENCE DATA')

#%%
# (e) (5 points) Using the estimated order from the previous section and using maximum likelihood estimation:
    # Estimate the parameter of the SARIMA model.
    # Print the estimated parameters with the corresponding confident intervals.
    # Display the SARIMA model.
    # Hint: You need to fed the seasonally differenced dataset into the LM algorithm
        # or the python package for the parameter estimation.

na=2
nb=2
model = sm.tsa.ARIMA(endog=df_3.values, order=(na, 0, nb)).fit()
# model = sm.tsa.ARMA(df.values, (na, nb)).fit(trend='nc', disp=0)
predictions = model.predict(start=0, end=len(df_3) - 1)
errors = df_3.values - predictions


for i in range(na):
    print("AR COEFFICIENT A{}".format(i), "is:", model.params[i])

for i in range(na):
    print("MA COEFFICIENT B{}".format(i), "is:", model.params[i + na])

#%%
## CONFIDENCE INTERVALS ##
intervals = model.conf_int()
for i in range(na):
    print("CONFIDENCE INTERVAL FOR a{}".format(i), "is:", intervals[i])
    print("P-VALUE FOR a{}".format(i), "is:", model.pvalues[i])
    print("STANDARD ERROR FOR a{}".format(i), "is:", model.bse[i])
    print("\n")

for i in range(na):
    print("CONFIDENCE INTERVAL FOR b{}".format(i), "is:", intervals[i + na])
    print("P-VALUE FOR b{}".format(i), "is:", model.pvalues[i + na])
    print("STANDARD ERROR FOR b{}".format(i), "is:", model.bse[i + na])
    print("\n")

#%%
# (f) (5 points) Plot the original raw data versus the seasonally difference dataset.
    # Take a screen shot of the plot and add it to your solution manual.

plt.title("Raw Data vs Seasonal Difference Data")
plt.xlabel("Frequency")
plt.ylabel("Raw Data")
plt.plot(df_3.values)
plt.plot(df_3_szn_diff1)
plt.grid()
plt.legend(["Raw Data", "Season Differenced"], loc='lower right')
plt.show()


#%%
## CORRELATION COEFFICIENT ##
def correlation_coefficent(x, y):
    x_mean = np.nanmean(np.array(x))
    y_mean = np.nanmean(np.array(y))
    x_r = np.subtract(x, x_mean)
    y_r = np.subtract(y, y_mean)
    numerator_xy = np.dot(x_r, y_r)
    denominator_x = np.nansum((x_r) ** 2)
    denominator_y = np.nansum((y_r) ** 2)
    denominator_xy = (denominator_x * denominator_y) ** (1 / 2)
    if denominator_xy != 0:
        return round((numerator_xy / denominator_xy), 2)
    else:
        return print('DIVIDE BY ZERO')


# print(f'CORRELATION COEFFICIENT BETWEEN THE DATA AND PREDICTION: {correlation_coefficent(df_3.values, predictions)}')

#%%
plt.title("Raw Data vs Forecast")
plt.xlabel("Frequency")
plt.ylabel("Raw Data")
plt.plot(df_3.values)
plt.plot(predictions)
plt.grid()
plt.legend(["Raw Data", "Predictions"], loc='lower right')
plt.show()

#%%
