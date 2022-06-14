#%%
## LIBRARY IMPORTS ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot

import scipy.stats as st

print("\nIMPORT SUCCESS")

#%% [markdown]

# All FUNCTIONS: return arrays (imputed array + h_steps)
# GRAPH FUNCTIONS: return extended index arrays (original array + h_steps)

#%%
## DIFFERENCE ##
def differencer(series, interval, index):
    s_len = len(series)
    diff = [0] * interval
    [diff.append(series[j + interval] - series[j]) for j in range(0, s_len - interval)]
    df = pd.DataFrame(index=index)
    df[f'{interval}diff'] = diff
    return df

#%%
## LOG TRANSFORM ##
def log_transform(series, index):
    log_series = np.log(series)
    df = pd.DataFrame(index=index)
    df['log_transform'] = log_series
    return df

#%%
## ROLLING MEAN / VARIANCE ##
def rolling_mean_var(series):
    length = len(series)
    # date = []
    mean = []
    var = []
    for y in range(0, length):
        row_slice = slice(0, y)
        if y != 1:
            mean.append(series[row_slice].mean())
            var.append(series[row_slice].var())
        else:
            mean.append(series[row_slice].mean())
            var.append(0)
        # date.append(df.iloc[y, date_index])
    dict_rolling = {'ROLLING MEAN': mean, 'ROLLING VARIANCE': var}
    roll_mean_var_df = pd.DataFrame(dict_rolling, columns=['ROLLING MEAN', 'ROLLING VARIANCE'])
    return roll_mean_var_df

#%%
## ROLLING MEAN + VARIANCE PLOTS ##
def rolling_mean_var_plots(df, col_index):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
    fig.suptitle(f'ROLLING MEAN / VARIANCE OVER TIME - {col_index}')
    ax1.plot(df.index, df['ROLLING MEAN'])
    ax1.set_ylabel('ROLLING MEAN')
    ax2.plot(df.index, df['ROLLING VARIANCE'])
    ax2.set_xlabel('DATE')
    ax2.set_ylabel('ROLLING VARIANCE')
    #plt.show()
    return

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

#%%
## KPSS TEST ##
def kpss_test(timeseries):
    kpss_test = kpss(timeseries, regression='c', nlags='auto')
    kpss_output = [x for x in kpss_test[0:3]]
    crit_dict = kpss_test[3]
    crit_values = list(crit_dict.keys())
    for x in crit_values:
        kpss_output.append(crit_dict.get(x))
    kpss_cols = ['Test Statistic', 'p-value', 'Lags', '10%', '5%', '2.5%', '1%']
    kpss_dict = {x: y for x, y in zip(kpss_cols, kpss_output)}
    df = pd.DataFrame.from_dict([kpss_dict])
    print(kpss_dict)
    return df

#%%
## ADF TEST ##
def adf_test(x, df):
    df = df.dropna()
    result = adfuller(df[x])
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    output = [x, result[0], result[1]]
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        output.append(value)

    cols = ['Column', 'ADF Statistic', 'p-value', '1% CV', '5% CV', '10% CV']
    dicta = {x: y for x, y in zip(cols, output)}
    df = pd.DataFrame(dicta, columns=['Column', 'ADF Statistic', 'p-value', '1% CV', '5% CV', '10% CV'],
                      index=['Column'])
    return df

#%%
## ADF / KPSS STATISTIC ##
def adf_kpss_statistic(timeseries):
    adf = adfuller(timeseries)[0]
    kpss_ = kpss(timeseries, regression='c', nlags="auto")[0]
    stats = [adf, kpss_]
    return stats


#%%
##  DIFFERENCE FUNCTION - ALTERNATE METHOD ##
def difference_alt(dataset, interval):
    diff = []
    for i in range(interval, len(dataset)):
        values = dataset[i] - dataset[i-interval]
        diff.append(values)
    return diff

#%%
## AUTO-CORRELATION FUNCTION ##
def ac_func(series, lag):
    if lag == len(series):
        return 0
    if lag == 0:
        return 1
    series = np.array(series)
    mean = series.mean()
    series_sub_mean = np.subtract(series, mean)
    shifted_right = series_sub_mean[:-lag]
    shifted_left = series_sub_mean[lag:]
    denominator = np.sum(np.square(series_sub_mean))
    numerator = np.sum(np.dot(shifted_right, shifted_left))
    r = numerator / denominator
    return round(r, 3)

#%%
## ACF DATAFRAME ##
def acf_df(series, lag):
    lag_list = [x for x in range(-lag, lag + 1, 1)]
    acf_value = [1]
    for l in [x for x in range(1, lag + 1, 1)]:
        x = ac_func(series, l)
        acf_value.insert(0, x)
        acf_value.append(x)
    df = pd.DataFrame()
    df['LAGS'] = lag_list
    df['ACF'] = acf_value
    return df

#%%
## ACF STEMPLOT ##
def acf_stemplot(col, df, n):
    (markers, stemlines, baseline) = plt.stem(df['LAGS'], df['ACF'], markerfmt='o')
    plt.title(f'ACF PLOT') # - {col}
    plt.xlabel('LAGS')
    plt.ylabel('AUTOCORRELATION VALUE')
    plt.setp(markers, color='red', marker='o')
    plt.setp(baseline, color='gray', linewidth=2, linestyle='-')
    plt.fill_between(df['LAGS'], (1.96 / np.sqrt(len(df))), (-1.96 / np.sqrt(len(df))), color='magenta', alpha=0.2)
        #m = 1.96 / np.sqrt(len(df))
        #plt.axhspan(-m, m, alpha=0.2, color='skyblue')
        #plt.savefig(folder + 'images/' + f'{col}.png', dpi=1000)
    plt.show()

#%%
## ACF / PACF PLOTS ##
def acf_pacf_plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.title('ACF/PACF - RAW DATA')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
    return


#%%
## ROLLING AVERAGE ##
def rolling_average(series, h=0):
    prediction = []
    length = list(range(1, (len(series) + 1), 1))
    series = np.array(series)
    for index in length:
        prediction.append(series[:index].mean())
    if h > 0:
        h = list(range(1, (h + 1), 1))
        for step in h:
            prediction.append(series.mean())
    else:
        print('NO STEPS')
    return prediction

#%%
## ERROR VARIANCE ##
def error_variance(array_1, array_2):
    return round(np.nanvar(array_2 - array_1), 2)

#%%
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
## AVERAGE PREDICTION ##
def average_prediction(train_series, h_steps):
    prediction = [np.nan]
    average = np.array(train_series).mean()
    for index, x in np.ndenumerate(train_series):
        prediction.append(train_series[:index[0]].mean())
    forecast = np.full(fill_value=average, shape=(h_steps,))
    return np.array(prediction[:]), np.array(forecast[:-1])

#%%
## TRAIN-TEST-PREDICT ##
def plot_train_test_predict(train_index, train, test_index, test, predict_index, predict, title):
    fig, ax = plt.subplots()
    ax.plot(train_index, train, color='navy')
    ax.plot(test_index, test, color='orange')
    ax.plot(predict_index, predict, color='green')
    plt.title(f'{title}')
    plt.xlabel('TIME STEPS')
    plt.ylabel('VALUE')
    plt.legend(['TRAIN', 'TEST', 'PREDICTION'])
    return plt.show()

#%%
## Q-VALUE ##
def q_value(residuals, lag):
    ACF_df = acf_df(residuals, lag)
    T = len(residuals)
    squared_acf = np.sum(np.square(ACF_df['ACF']))
    return T * squared_acf

#%%
## NAIVE PREDICTION ##
def naive_prediction(train_series, h_steps):
    train_series = np.array(train_series)
    prediction = []
    for index, x in np.ndenumerate(train_series):
        prediction.append(train_series[index[0] - 1])
    forecast = np.full(fill_value=train_series[-1], shape=(h_steps,))
    return np.array(prediction), np.array(forecast)

#%%
## NAIVE ROLLING ##
def naive_rolling(series, h=0):
    prediction = [np.nan]
    length = list(range(1, (len(series) + 1), 1))
    series = np.array(series)
    for index in length:
        prediction.append(series[(index - 1)])
    if h > 0:
        h = list(range(1, (h + 1), 1))
        for step in h:
            prediction.append(series[-1])
    else:
        print('NO STEPS')

    return prediction[:-1]

#%%
## DRIFT ROLLING ##
def drift_rolling(series, h=0):
    series = np.array(series)
    length = list(range(2, (len(series) + 1), 1))
    # series = np.append(series, [np.nan] * 1)
    prediction = [np.nan, np.nan]
    for index in length:
        drift = series[index - 1] + (h * ((series[index - 1] - series[0]) / (index - 1)))
        prediction.append(drift)
    if h > 0:
        h = list(range(1, (h + 1), 1))
        for step in h:
            prediction.append(prediction[-1])
    else:
        print('NO STEPS')

    return prediction[:-2]

#%%
## DRIFT PREDICTION ##
def drift_prediction(series, h_steps):
    prediction = []
    series = np.array(series)
    # series = np.insert(series, [np.nan] * 1)
    steps = list(range(1, (h_steps + 1), 1))
    for h in steps:
        drift = series[-1] + (h * ((series[-1] - series[0]) / (len(series) - 2)))
        prediction.append(drift)

    return prediction

#%%
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
## SES PREDICTION ##
def ses_prediction(series, h_steps=1, alpha=0.5):
    series = np.array(series)  # Transform the input into a numpy array
    cols = len(series)  # Historical period length
    series = np.append(series, [np.nan] * h_steps)  # Append np.nan into the demand array to cover future periods
    f = np.full(cols + h_steps, np.nan)  # Forecast array
    f[1] = series[0]  # initialization of first forecast
    # Create all the t+1 forecasts until end of historical period
    for t in range(2, cols + 1):
        f[t] = alpha * series[t - 1] + (1 - alpha) * f[t - 1]
    f[cols + 1:] = f[t]  # Forecast for all extra periods
    return f[-h_steps:]

#%%
## MSE - MEAN SQUARED ERROR ##
def mse_calc(obs_series, pred_series):
    return round((np.sum(np.square(np.subtract(np.array(pred_series), np.array(obs_series))) / len(obs_series))), 2)

#%%
## RESIDUALS ##
def residuals(array_1, array_2):
    return array_2 - array_1

#%%
## LEAST SQUARED ERROR ##
def lse(x_matrix, y_array):
    invert = np.linalg.inv(np.dot(np.transpose(x_matrix), x_matrix))
    transpose_y_array = np.dot(np.transpose(x_matrix), y_array)
    return np.dot(invert, transpose_y_array)

#%%
## SCORING METRICS ##
def aic_bic_rsquared_df(fitted_model):
    return pd.DataFrame.from_dict(
        {'INDEX': [0], 'AIC': fitted_model.aic, 'BIC': fitted_model.bic, 'ADJ_R_SQUARED': fitted_model.rsquared_adj})

#%%
## COEFFICIENTS ## [Multiple Regression Model]
def coefficients(X, Y):
    coef = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)
    return coef

#%%
## WORST FEATURE ##
def worst_feature(p_value_series):
    return p_value_series.idxmax()

#%%
## MSE - ERRORS ##
def mse(errors):
    return np.sum(np.power(errors, 2)) / len(errors)

#%%

def rolling_avg_non_wtd(array, m): # n > 2
    m = int(m)
    odd = True if m % 2 == 1 else False
    if m <= 2:
        return print("ERROR: M MUST BE > 2")
    elif odd == True:
        start = np.array([np.nan] * int((m - 1) / 2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
        end = np.array([np.nan] * int((m - 1) / 2))
        full = np.append(np.append(start, average), end)
        return full
    else:
        start = np.array([np.nan] * int(m / 2))
        average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
        end = np.array(([np.nan] * int((m - 1) / 2)))
        full = np.append(np.append(start, average), end)
        return full

#%%
## ODD ROLLING AVERAGE ##
def odd_rolling_avg(array, m):
    start = np.array([np.nan] * int((m - 1) / 2))
    average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
    end = np.array([np.nan] * int((m - 1) / 2))
    final = np.append(np.append(start, average), end)
    return final

#%%
## EVEN ROLLING AVERAGE ##
def even_rolling_avg(array, m):
    start = np.array([np.nan] * int(m / 2))
    average = np.array(np.lib.stride_tricks.sliding_window_view(array, m).mean(axis=1))
    end = np.array(([np.nan] * int((m - 1) / 2)))
    final = np.append(np.append(start, average), end)
    return final

#%%
## ODD OR EVEN ROLLING AVERAGE ##
def odd_or_even_rolling_avg(array):
    length = len(array)
    order_1 = int(input("INPUT ORDER OF MOVING AVERAGE:"))
    if order_1 <= 2:
        return print('ERROR: ORDER MUST BE >2')
    elif order_1 % 2 == 0:
        order_2 = int(input('ERROR: FOLDING ORDER MUST BE EVEN / >1'))
        if order_2 < 2 or order_2 % 2 != 0:
            print('INVALID FOLDING ORDER')
            pass
        else:
            output = even_rolling_avg(even_rolling_avg(array, order_1), order_2)
            return output
    elif order_1 % 2 == 1:
        return odd_rolling_avg(array, order_1)



#%%
## STRENGTH OF TREND ##
def strength_of_trend(residual, trend):
    var_resid = np.nanvar(residual)
    var_resid_trend = np.nanvar(np.add(residual, trend))
    return 1 - (var_resid / var_resid_trend)

#%%
## STRENGTH OF SEASONAL ##
def strength_of_seasonal(residual, seasonal):
    var_resid = np.nanvar(residual)
    var_resid_seasonal = np.nanvar(np.add(residual, seasonal))
    return 1 - (var_resid / var_resid_seasonal)


#%%



#%%
### PROCESS TIME SERIES - WORK IN PROGRESS
def process_time_series(y_col, time_col, df):
    plt.figure(figsize=(8, 4), layout='constrained')
    plt.plot(time_col, y_col, data=df)
    plt.title(f'{y_col} by {time_col}')
    plt.xlabel(f'{time_col}')
    plt.ylabel(f'{y_col}')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.grid()
    #print(f"{y_col} SAVED")
    return plt.show()


#%%


