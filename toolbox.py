#%%
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
def differencer(series, interval, index):
    s_len = len(series)
    diff = [0] * interval
    [diff.append(series[x + interval] - series[x]) for x in range(0, s_len - interval)]
    df = pd.DataFrame(index=index)
    df[f"{interval}_diff"] = diff
    return df

#%%
def log_transform(series, index):
    log_series = np.log(series)
    df = pd.DataFrame(index=index)
    df['log_transform'] = log_series
    return df



#%%
def cal_rolling_mean_var(series):
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
    dict_rolling = {'Rolling Mean': mean, 'Rolling Variance': var}
    new_df = pd.DataFrame(dict_rolling, columns=['Rolling Mean', 'Rolling Variance'])

    return new_df

#%%
def correlation_coefficent_cal(x, y):
    x_mean = np.nanmean(np.array(x))
    y_mean = np.nanmean(np.array(y))
    x_r = np.subtract(x, x_mean)
    y_r = np.subtract(y, y_mean)
    numerator = np.dot(x_r, y_r)
    denom_x = np.nansum((x_r) ** 2)
    denom_y = np.nansum((y_r) ** 2)
    denominator = (denom_x * denom_y) ** (1 / 2)
    if denominator != 0:
        return round((numerator / denominator), 2)
    else:
        return print('Divide by zero??')

#%%
def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = [x for x in kpsstest[0:3]]
    crit_dict = kpsstest[3]
    crit_values = list(crit_dict.keys())
    for x in crit_values:
        kpss_output.append(crit_dict.get(x))
    kpss_cols = ['Test Statistic', 'p-value', 'Lags', '10%', '5%', '2.5%', '1%']
    kpss_dict = {x: y for x, y in zip(kpss_cols, kpss_output)}
    df = pd.DataFrame.from_dict([kpss_dict])
    print(kpss_dict)
    return df

#%%
def adf_calc(x, df):
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


def adf_kpss_statistic(timeseries):
    adf = adfuller(timeseries)[0]
    kpss_ = kpss(timeseries, regression='c', nlags="auto")[0]
    stats = [adf, kpss_]
    return stats

#%%
def simple_time_series(y_column, time_column, df):
    """
    :param y_column:
    :param time_column:
    :param df:
    :return:
    """
    plt.figure(figsize=(8, 4), layout='constrained')
    plt.plot(time_column, y_column, data=df)
    plt.title(f'{y_column} by {time_column}')
    plt.xlabel(f'{time_column}')
    plt.ylabel(f'{y_column}')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.grid()
    print(f"{y_column} saved!")
    return plt.show()

#%%
def plotting_rolling(name, df):
    """
    :param name:
    :param df:
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Rolling mean and var vs time')

    ax1.plot(df.index, df['Rolling Mean'])
    ax1.set_ylabel('Rolling Mean')

    ax2.plot(df.index, df['Rolling Variance'])
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rolling Variance')
    plt.show()
    return

#%%


#%%