#%% [markdown]

## REIT TRADING COMPS ##

#%%
## LIBRARY IMPORTS ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as stats
import statistics


import statsmodels
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

import pandas_datareader as web

import scipy.stats as st
from scipy import signal
from scipy.stats import chi2

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL
from numpy import linalg

## TOOLBOX
from toolbox import *

# import requests
# import json
# import time
# import datetime as dt
# from google.colab import drive

print("\nIMPORT SUCCESS")

#%%
## FOLDER CONFIGURATION ##

## CURRENT FOLDER / PATH
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid') #ticks

## ANALYSIS PARAMETERS ##
start_date = '2000-01-01'
end_date = '2022-03-31'

mo_qtr_map = {'01': '1', '02': '1', '03': '1',
              '04': '2', '05': '2', '06': '2',
              '07': '3', '08': '3', '09': '3',
              '10': '4', '11': '4', '12': '4'}

#%%

#%%


#%%


## REAL ESTATE SECTORS / TICKERS ##

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

sector_dict = {'apartment': ["EQR",	"AVB",	"ESS",	"MAA",	"UDR",	"CPT",	"AIV",	"BRG", "APTS"],
               'office': ["BXP",	"VNO",	"KRC", "DEI", "JBGS",	"CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE",	"PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"],
               'hotel': ["HST",	"RHP",	"PK",	"APLE",	"SHO",	"PEB",	"RLJ", "DRH",	"INN", "HT", "AHT",	"BHR"],
               'mall': ["SPG", "MAC", "PEI"],
               'strip_center': ["REG", "FRT",	"KIM",	"BRX",	"AKR",	"UE",	"ROIC",	"CDR",	"SITC",	"BFS"],
               'net_lease': ["O",	"WPC",	"NNN",	"STOR",	"SRC",  "PINE", "FCPT", "ADC", "EPRT"],
               'industrial': ["PLD",	"DRE",	"FR",	"EGP"],
               'self_storage': ["EXR",	"CUBE",	"REXR",	"LSI"],
               'data_center': ["EQIX", "DLR" "AMT"],
               'healthcare': ["WELL",	"PEAK",	"VTR",	"OHI", "HR"]}

#%%
# IMPORT DATA (BY SECTOR)

office_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='OFFICE', parse_dates = True, index_col = [0], header=[2])
residential_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='RESIDENTIAL', parse_dates = True, index_col = [0], header=[2])
lodging_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='LODGING', parse_dates = True, index_col = [0], header=[2])
net_lease_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='NET LEASE', parse_dates = True, index_col = [0], header=[2])
strip_center_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='STRIP CENTER', parse_dates = True, index_col = [0], header=[2])
residential_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='MALL', parse_dates = True, index_col = [0], header=[2])
healthcare_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='HEALTH CARE', parse_dates = True, index_col = [0], header=[2])
industrial_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='INDUSTRIAL', parse_dates = True, index_col = [0], header=[2])
self_storage_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='SELF STORAGE', parse_dates = True, index_col = [0], header=[2])
data_center_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='DATA CENTER', parse_dates = True, index_col = [0], header=[2])

# reit_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', parse_dates = True, index_col = 'reportPeriod')



#%%




#%%




#%%




#%%




#%%