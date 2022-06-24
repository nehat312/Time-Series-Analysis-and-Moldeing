#%% [markdown]

## REIT TRADING COMPS ##

#%%
## LIBRARY IMPORTS ##

## BASE PACKAGES ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## SKLEARN ##
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

## SCIPY ##
from scipy import stats as stats
import scipy.stats as st
from scipy import signal
from scipy.stats import chi2

## STATSMODELS ##
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL

## SUPPLEMENTAL ##
from numpy import linalg

## TOOLBOX ##
from toolbox import *

## NOT-IN-USE ##
# import statistics
# import pandas_datareader as web
# import requests
# import json
# import time
# import datetime as dt
# from google.colab import drive

print("\nIMPORT SUCCESS")

#%%
## FOLDER CONFIGURATION ##

## CURRENT FOLDER / PATH
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/FTP/'

print("\nDIRECTORY CONFIGURED")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid') #ticks

print("\nSETTINGS ASSIGNED")

#%%
## ANALYSIS PARAMETERS ##
start_date = '2000-01-01'
end_date = '2022-03-31'

mo_qtr_map = {'01': '1', '02': '1', '03': '1',
              '04': '2', '05': '2', '06': '2',
              '07': '3', '08': '3', '09': '3',
              '10': '4', '11': '4', '12': '4'}

print("\nPARAMETERS ASSIGNED")

#%%
## REAL ESTATE SECTORS / TICKERS ##

apartment = ["EQR",	"AVB", "ESS", "MAA", "UDR",	"CPT", "AIV", "BRG", "APTS"]
office = ["BXP", "VNO",	"KRC", "DEI", "JBGS", "CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE", "PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"] #"CLI"
hotel = ["HST",	"RHP",	"PK", "APLE", "SHO", "PEB", "RLJ", "DRH", "INN", "HT", "AHT", "BHR"]    #"XHR",
mall = ["SPG", "MAC", "PEI"]    #CBL	TCO	"WPG",
strip_center = ["REG", "FRT",	"KIM",	"BRX", "AKR", "UE", "ROIC", "CDR", "SITC", "BFS"]   #"WRI", "RPAI",
net_lease = ["O", "WPC", "NNN",	"STOR",	"SRC", "PINE", "FCPT", "ADC", "EPRT"]  # "VER",
industrial = ["PLD", "DRE",	"FR", "EGP"]
self_storage = ["EXR",	"CUBE",	"REXR",	"LSI"]
data_center = ["EQIX", "DLR" "AMT"]     #"CONE", "COR"
healthcare = ["WELL", "PEAK", "VTR", "OHI", "HR"]   #"HTA",

sectors = [apartment, office, hotel, mall, strip_center, net_lease, industrial, self_storage, data_center, healthcare]

reit_tickers = ["EQR",	"AVB",	"ESS",	"MAA",	"UDR",	"CPT",	"AIV",	"BRG", "APTS",
               "BXP",	"VNO",	"KRC",	"DEI",	"JBGS",	"CUZ",	"HPP",	"SLG",	"HIW",	"OFC",	"PGRE",	"PDM",	"WRE",	"ESRT",	"BDN", "EQC",
               "HST",	"RHP",	"PK",	"APLE",	"SHO",	"PEB",	"RLJ",	"DRH",	"INN",	"HT",	"AHT",	"BHR",
               "SPG",	"MAC", "PEI", "SKT", "SRG",
               "REG", "FRT",	"KIM",	"BRX",	"AKR",	"UE",	"ROIC",	"CDR",	"SITC",	"BFS",
               "O",	"WPC",	"NNN",	"STOR",	"SRC", "PINE", "FCPT", "ADC", "EPRT",
               "PLD",	"DRE",	"FR",	"EGP",  "GTY",
               "EXR",	"CUBE",	"REXR",	"LSI",
               "EQIX", "DLR", "AMT",
               "WELL",	"PEAK",	"VTR",	"OHI",	"HR"]

print("\nVARIABLES ASSIGNED")

#%%
sector_dict = {'apartment': ["EQR",	"AVB",	"ESS",	"MAA",	"UDR",	"CPT",	"AIV",	"BRG", "APTS"],
               'office': ["BXP",	"VNO",	"KRC", "DEI", "JBGS",	"CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE",	"PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"],
               'hotel': ["HST",	"RHP",	"PK",	"APLE",	"SHO",	"PEB",	"RLJ", "DRH",	"INN", "HT", "AHT",	"BHR"],
               'mall': ["SPG", "MAC", "PEI"],
               'strip_center': ["REG", "FRT",	"KIM",	"BRX",	"AKR",	"UE",	"ROIC",	"CDR",	"SITC",	"BFS"],
               'net_lease': ["O",	"WPC",	"NNN",	"STOR",	"SRC",  "PINE", "FCPT", "ADC", "EPRT"],
               'industrial': ["PLD", "DRE",	"FR",	"EGP"],
               'self_storage': ["EXR",	"CUBE",	"REXR",	"LSI"],
               'data_center': ["EQIX", "DLR" "AMT"],
               'healthcare': ["WELL",	"PEAK",	"VTR",	"OHI", "HR"]}

#%%
# IMPORT DATA (DATAFRAMES BY RE SECTOR)
all_sectors_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='ALL SECTORS', parse_dates = True, index_col = [0], header=[3])
office_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='OFFICE', parse_dates = True, index_col = [0], header=[2])
residential_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='RESIDENTIAL', parse_dates = True, index_col = [0], header=[2])
lodging_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='LODGING', parse_dates = True, index_col = [0], header=[2])
net_lease_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='NET LEASE', parse_dates = True, index_col = [0], header=[2])
strip_center_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='STRIP CENTER', parse_dates = True, index_col = [0], header=[2])
mall_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='MALL', parse_dates = True, index_col = [0], header=[2])
healthcare_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='HEALTH CARE', parse_dates = True, index_col = [0], header=[2])
industrial_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='INDUSTRIAL', parse_dates = True, index_col = [0], header=[2])
self_storage_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='SELF STORAGE', parse_dates = True, index_col = [0], header=[2])
data_center_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='DATA CENTER', parse_dates = True, index_col = [0], header=[2])

print("\nIMPORT SUCCESS")

#%%
## SAVE COPIES OF IMPORTS

sector_comps = all_sectors_import
office_comps = office_import
residential_comps = residential_import
lodging_comps = lodging_import
net_lease_comps = net_lease_import
strip_center_comps = strip_center_import
mall_comps = mall_import
healthcare_comps = healthcare_import
industrial_comps = industrial_import
self_storage_comps = self_storage_import
data_center_comps = data_center_import

sector_df_list = [office_comps, residential_comps,  lodging_comps, net_lease_comps, strip_center_comps,
                  mall_comps, healthcare_comps, industrial_comps, self_storage_comps, data_center_comps]

print("\nCOPIES SAVED")

#%%
# all_sector_return_df = pd.concat(sector_df_list)
# all_sector_return_df = pd.concat(office_comps['AVERAGE_RETURN_1D'], residential_comps['AVERAGE_RETURN_1D'])
# print(all_sector_return_df['AVERAGE_RETURN_1D'])

#%%
# print(sector_df_list[:])
# print(sectors)

#%%
print(sector_comps)

#%%




#%%
## AUTO-CORRELATION FUNCTION ##


#%%
## AUTO-CORRELATION PLOT ##




#%%
all_sectors_returns_df = pd.concat(sector_df_dict)
all_sectors_returns_df


# for col in sector_df_dict:
#     if col == ['AVERAGE_RETURN_1D']:

#%%
print(all_sectors_returns_df.columns)

#%%
# PLOT
fig, axes = plt.subplots(2,5,figsize=(16,8))
# plt.figure(figsize=(10,8))




sns.lineplot(x=passengers['Month'], y=passengers['#Passengers'])
plt.title("AIR PASSENGERS (1949-1960)")
plt.xlabel('DATE')
plt.ylabel('PASSENGERS (#)')
plt.tight_layout(pad=1)
#plt.grid()
plt.show()


#%%
## TIME SERIES STATISTICS ##
print("MEAN:", passengers['#Passengers'].mean(),
      "VARIANCE:", passengers['#Passengers'].var(),
      "STD DEV:", passengers['#Passengers'].std())
print('*'*150)


#%%
# FILTER START DATE
new_start_date = '7/1/2009' #'4/1/2009' #'1/2/2009'
office_comps_after_2009 = office_comps[office_comps.index >= new_start_date]
office_comps_after_2009


#%%
# SET COLUMN INDICES FOR CHART TITLES
office_col_index = office_comps_after_2009.columns[33].upper() #16
print(office_col_index)

#%%
rolling_mean_var_plots(rolling_mean_var(office_comps_after_2009['AVERAGE_RETURN_1D']), office_col_index)

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
## ADF TEST ##
print('ADF - ___:')
print(adf_test(['AVERAGE_RETURN_1D'], office_comps))
print('*'*100)

#%%
## KPSS TEST ##

print('KPSS - ___:')
print(kpss_test(office_comps['AVERAGE_RETURN_1D']))
print('*'*100)

#%%
## FEATURE REDUCTION ANALYSIS ##


#%%
## PRINCIPAL COMPONENT ANALYSIS ##

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
reit_num_cols = reit_comps.select_dtypes(include=numerics)
reit_num_cols.info()


#%%
X = reit_num_cols[reit_num_cols._get_numeric_data().columns.to_list()[:-1]]
Y = reit_num_cols['earningBeforeInterestTaxes'] # earningBeforeInterestTaxes
print(X.describe())

#%%
# reit_num_cols = reit_num_cols.dropna(inplace=True)


#%%
X = StandardScaler().fit_transform(X)


#%%
## PCA STATISTICS ##
pca = PCA(n_components='mle', svd_solver='full') # 'mle'
pca.fit(X)
X_PCA = pca.transform(X)

print('ORIGINAL DIMENSIONS:', X.shape)
print('*'*100)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print('*'*100)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')

#%%
## PCA PLOT ##
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)

plt.figure(figsize=(12,8))
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)
#plt.grid()
plt.show()


#%%
## SINGULAR VALUE DECOMPOSITION ANALYSIS [SVD] ##
    # CONDITION NUMBER
    # ORIGINAL DATA

from numpy import linalg as LA

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'ORIGINAL DATA: SINGULAR VALUES\n {d}')
print('*'*75)
print(f'ORIGINAL DATA: CONDITIONAL NUMBER\n {LA.cond(X)}')

#%%
# TRANSFORMED DATA
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'TRANSFORMED DATA: SINGULAR VALUES\n {d_PCA}')
print('*'*75)
print(f'TRANSFORMED DATA: CONDITIONAL NUMBER\n {LA.cond(X_PCA)}')

#%%
# CONSTRUCTION OF REDUCED DIMENSION DATASET

#pca_df = pca.explained_variance_ratio_

a, b = X_PCA.shape
column = []

for i in range(b):
    column.append(f'PRINCIPAL COLUMN {i+1}')

df_PCA = pd.DataFrame(data=X_PCA, columns=column)
df_PCA = pd.concat([df_PCA, Y], axis=1)

df_PCA.info()


#%%



#%%





#%%
### DATA VIZ ###

#%%
#fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12,8))
plt.figure(figsize=(12,8))
sns.lineplot(data=reit_comps,
             x='calendarDate',
             y='debt',   #netIncome. ## EBIT/Share #'operatingIncome' 'operatingExpense'
             hue='sector',
             style='sector',
             palette='mako')

plt.title('REIT _____ (2010-2022)')
plt.xlabel('DATE')
plt.ylabel(f'_____')
plt.tight_layout(pad=1)
#plt.grid()
plt.show()



#%%