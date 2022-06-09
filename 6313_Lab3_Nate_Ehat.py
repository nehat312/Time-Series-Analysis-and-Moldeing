#%% [markdown]
# DATS-6313 - LAB #3
# Nate Ehat

#%% [markdown]
# SIMPLE FORECASTING METHODS

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

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
# import warnings
# warnings.filterwarnings('ignore')

#%%
# DIRECTORY CONFIGURATION
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid')

#%%
# 1. Let suppose a time series dataset is given as below (make-up dataset).
    # Without a help of Python and using the average forecast method:
    # Perform one-step ahead prediction and fill out the table.
    # To perform the correct cross-validation:
        # start with first observation {y1}
            # predict the second observation {y2} (can now calculate first error).
            # Add the next observation {y1, y2}
            # Predict {y3} (you can now calculate the second error).
        # Continue this pattern through the dataset.
    # Then calculate the MSE of the 1-step prediction and MSE of h-step forecast.

#%% [markdown]
# Refer to attached .xlsx file for manually computed forecasting methods.

#%%
# 2. Write a python code that perform the task in step 1.
    # Plot the test set, training set and the h-step forecast in one graph with different marker/color.
    # Add an appropriate title, legend, x-label, y-label to each graph.
    # No need to include the 1-step prediction in this graph.

dummy_x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
dummy_y = [112, 118, 132, 129, 121, 135, 148, 136, 119, 104, 118, 115, 126, 141]

data = pd.Series(dummy_y, index=dummy_x)
print(data)

#%%
## ROLLING AVERAGE ##
def rolling_average(series, h=0):
    prediction = []
    length = list(range(1, (len(series) + 1), 1))
    series = np.array(series)
    for index in length:
        prediction.append(series[:index].mean())
    # if h > 0:
    #     h = list(range(1, (h + 1), 1))
    #     for step in h:
    #         prediction.append(series.mean())
    # else:
    #     print('NO STEPS')
    return prediction

#%%
print(rolling_average(data))

#%%
# 3. Using python, calculate MSE of prediction errors and forecast errors.

## MSE ##
def mse_calc(obs_series, pred_series):
    return round((np.sum(np.square(np.subtract(np.array(pred_series), np.array(obs_series))) / len(obs_series))), 2)

def mse(errors):
    return np.sum(np.power(errors, 2)) / len(errors)

#%%
# 4. Using python, calculate variance of prediction error and variance of forecast error.

## ERROR VARIANCE ##
def error_variance(array_1, array_2):
    return round(np.nanvar(array_2 - array_1), 2)

## RESIDUALS / SQUARED ERROR ##
def error(observation, prediction):
    try:
        residual = np.subtract(np.array(observation), np.array(prediction))
        squared_error = np.square(residual)
        return residual, squared_error
    except:
        residual = np.subtract(np.array(observation[1:]), np.array(prediction))
        squared_error = np.square(residual)
        return residual, squared_error

#%%
# 5. Calculate the Q value for estimate on training set
    # Display the Q-value on the console.
    # Number of lags = 5)

## RESIDUALS ##
def residuals(array_1, array_2):
    return array_2 - array_1

## Q-VALUE ##
def q_value(residuals, lag):
    ACF_df = acf_df(residuals, lag)
    T = len(residuals)
    squared_acf = np.sum(np.square(ACF_df['ACF']))
    return T * squared_acf

#%%
print(f'AVERAGE METHOD MSE ERRORS: {mse_calc(data, rolling_average(data)):.2f}')
print(f'AVERAGE METHOD VARIANCE: {error_variance(data, rolling_average(data)):.2f}')
print(f'AVERAGE METHOD Q-VALUE: {q_value(residuals(data, rolling_average(data)), 5):.2f}')

#%%
# 6. Repeat step 1 through 5 with the Naïve method.

## NAIVE ROLLING ##
def naive_rolling(series, h=0):
    prediction = [np.nan]
    length = list(range(1, (len(series) + 1), 1))
    series = np.array(series)
    for index in length:
        prediction.append(series[(index - 1)])
    # if h > 0:
    #     h = list(range(1, (h + 1), 1))
    #     for step in h:
    #         prediction.append(series[-1])
    # else:
    #     print('NO STEPS')

    return prediction[:-1]

#%%
print(naive_rolling(data))

#%%
print(f'NAIVE METHOD MSE: {mse_calc(data, naive_rolling(data)):.2f}')
print(f'NAIVE METHOD VARIANCE: {error_variance(data, naive_rolling(data)):.2f}')
print(f'NAIVE METHOD Q-VALUE: {q_value(residuals(data, rolling_average(data)), 5):.2f}')

#%%
# 7. Repeat step 1 through 5 with the drift method.

## DRIFT ROLLING ##
def drift_rolling(series, h=0):
    series = np.array(series)
    length = list(range(1, (len(series) + 1), 1))
    # series = np.append(series, [np.nan] * 1)
    prediction = [np.nan, np.nan]
    for index in length:
        drift = series[index - 1] + (h * ((series[index - 1] - series[0]) / (index - 1)))
        prediction.append(drift)
    # if h > 0:
    #     h = list(range(1, (h + 1), 1))
    #     for step in h:
    #         prediction.append(prediction[-1])
    # else:
    #     print('NO STEPS')
    return prediction[:-2]

#%%
print(drift_rolling(data))

#%%
print(f'DRIFT METHOD MSE: {mse_calc(data, drift_rolling(data)):.2f}')
print(f'DRIFT METHOD VARIANCE: {error_variance(data, drift_rolling(data)):.2f}')
print(f'DRIFT METHOD Q-VALUE: {q_value(residuals(data, drift_rolling(data)), 5):.2f}')

#%%
# 8. Repeat step 1 through 5 with the simple exponential method.
      # Consider alpha = 0.5
      # initial condition = first sample in training set

## SES ROLLING ##
def ses_rolling(series, extra_periods=1, alpha=0.5):
    series = np.array(series)  # Transform input into array
    cols = len(series)  # Historical period length
    series = np.append(series, [np.nan] * extra_periods)  # Append np.nan into demand array to accept future periods
    f = np.full(cols + extra_periods, np.nan)  # Forecast array
    f[1] = series[0]  # Initialize first forecast
    for t in range(2, cols + 1): # Create all t+1 forecasts until end of time series / historical period
        f[t] = alpha * series[t - 1] + (1 - alpha) * f[t - 1]
    f[cols + 1:] = f[t]  # Forecast for all extra periods
    return f[:-extra_periods]

#%%
print(ses_rolling(data))

#%%
print(f'SES METHOD MSE: {mse_calc(data, ses_rolling(data)):.2f}')
print(f'SES METHOD VARIANCE: {error_variance(data, ses_rolling(data)):.2f}')
print(f'SES METHOD Q-VALUE: {q_value(residuals(data, ses_rolling(data)), 5):.2f}')

#%%
# 9. Using SES method:
      # Plot the test set, training set, h-step forecast in one graph for:
      # Alpha = 0, 0.25, 0.75 and 0.99.
      # You can use a subplot 2x2.
      # Add an appropriate title, legend, x-label, y-label to each graph.
      # No need to include the 1-step prediction in this graph.



#%%
# 10. Create a table and compare the four forecast method above by displaying:
      # Q values
      # MSE
      # Mean
      # Number of prediction errors
      # Variance of prediction errors

forecast_table = pd.DataFrame()


#%%
# 11. Using the python program developed in the previous LAB,
      # Plot the ACF of prediction errors.



#%%
# 12. Compare the above 4 methods by looking at:
      # Variance of prediction error versus the variance
      # Number of forecast error and pick the best estimator.
      # Justify your answer.



#%% [markdown]


#%%

