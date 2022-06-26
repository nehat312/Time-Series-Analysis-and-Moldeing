#%% [markdown]
# DATS-6313 - LAB #4
# Nate Ehat

#%% [markdown]
#

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter

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
df = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.describe())

#%%
df.head(5).plot()
plt.legend()
plt.show()

#%%
## PLOT
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['Churn']=df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0 )
df.TotalCharges.fillna(value=df['TotalCharges'].median(),inplace=True)
durations = df['tenure']
event_observed = df['Churn']

#%%
km = KaplanMeierFitter()
km.fit(durations, event_observed,label='Customer Retention')
km.plot()
plt.show()

#%%
kmf = KaplanMeierFitter()
T = df['tenure'] ## time to event
E = df['Churn'] ## event occurred or censored
groups = df['Contract'] ## Create the cohorts from the 'Contract' column
ix1 = (groups == 'Month-to-month') ## Cohort 1
ix2 = (groups == 'Two year') ## Cohort 2
ix3 = (groups == 'One year') ## Cohort 3

#%%
kmf.fit(T[ix1], E[ix1], label='Month-to-month')
ax = kmf.plot()
kmf.fit(T[ix2], E[ix2], label='Two year')
ax1 = kmf.plot(ax=ax)
kmf.fit(T[ix3], E[ix3], label='One year')
kmf.plot(ax=ax1)
plt.title('Survival Analysis: Customer Retention by Length of Contract')
plt.legend()
plt.show()

#%%
# Rate of retention is higher for longer contracts.


#%%
kmf1 = KaplanMeierFitter()
groups = df['StreamingTV']
i1 = (groups == 'No')
i2 = (groups == 'Yes')

#%%
kmf.fit(T[i1], E[i1], label='No Streaming')
ax = kmf.plot()
kmf.fit(T[i2], E[i2], label='Streaming')
ax1 = kmf.plot(ax=ax)
plt.title('Survival Analysis: Customer Retention by Streaming TV Option')
plt.legend()
plt.show()

#%%
# Rate of retention is higher for streaming customers.

#%%
url = 'https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/dd.csv'
dd = pd.read_csv(url)

dd.head(5)

#%%
D = dd['duration']
O = dd['observed']
regions = dd['un_continent_name']
asia = (regions == 'Asia')
europe = (regions == 'Europe')
africa = (regions == 'Africa')
americas = (regions == 'Americas')
oceania = (regions == 'Oceania')

#%%
kmf2 = KaplanMeierFitter()

kmf2.fit(D[asia],O[asia],label='asia')
ax3 = kmf2.plot()
kmf2.fit(D[europe],O[europe],label='europe')
ax3 = kmf2.plot(ax=ax3)
kmf2.fit(D[africa],O[africa],label='africa')
ax3 = kmf2.plot(ax=ax3)
kmf2.fit(D[americas],O[americas],label='americas')
ax3 = kmf2.plot(ax=ax3)
kmf2.fit(D[oceania],O[oceania],label='oceania')
ax3.set_ylabel('Survival')
ax3.set_title('Region Survival versus Time')
ax3 = kmf2.plot(ax=ax3)
plt.show()

#%%
# Africa has the highest rate of regime survival. Monarchies stay in power longer.