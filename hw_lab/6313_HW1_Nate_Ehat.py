#%% [markdown]
# DATS-6313 - HW #1
# Nate Ehat

#%% [markdown]
# CORRELATION COEFFICIENT

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

import datetime

from toolbox import *

print("\nIMPORT SUCCESS")

#%%
# VARIABLE DIRECTORY
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/'
tute_filepath = 'tute1'

print("\nDIRECTORY ASSIGNED")

#%%
## VISUAL SETTINGS
sns.set_style('whitegrid')

#%%
# 1. Write a python function “correlation_coefficent_cal(x,y)” to implement correlation coefficient
# Between random variable x and y.
# Function should be written in a general form than can work for any dataset x and dataset y.
# The return value for this function is r.

### CORRELATION COEFFICIENT FUNCTION
def correlation_coefficent(x, y):
    x_mean = np.nanmean(np.array(x))
    y_mean = np.nanmean(np.array(y))
    x_r = np.subtract(x, x_mean)
    y_r = np.subtract(y,       y_mean)
    numerator_xy = np.dot(x_r, y_r)
    denominator_x = np.nansum((x_r) ** 2)
    denominator_y = np.nansum((y_r) ** 2)
    denominator_xy = (denominator_x * denominator_y) ** (1 / 2)
    if denominator_xy != 0:
        return round((numerator_xy / denominator_xy), 2)
    else:
        return print('DIVIDE BY ZERO')

#%%
# 2. Test the developed code in question 1 with the following make-up dataset for x, y, z, h & g.
# Verify a, b and c using a python program:
      # a. The correlation coefficient between x and y must be equal to 1.
      # b. The correlation coefficient between x and z must be equal to -1.
      # c. The correlation coefficient between g and h must be equal to 0.

x = [1,2,3,4,5]
y = [1,2,3,4,5]
z = [-1,-2,-3,-4,-5]
g = [1,1,0,-1,-1,0,1]
h = [0,1,1,1,-1,-1,-1]

#%%
print(f'X/Y CORRELATION COEFFICIENT: {correlation_coefficent(x, y):.0f}')
print('*'*75)
print(f'X/Z CORRELATION COEFFICIENT: {correlation_coefficent(x, z):.0f}')
print('*'*75)
print(f'G/H CORRELATION COEFFICIENT: {correlation_coefficent(g, h):.0f}')

#%%
## DATA IMPORT / PRE-PROCESSING
      # Load the time series data called ‘tute1.csv’
      # By saving the file to '.xlsx' format:
      # Save a lot of time with Excel, prior to import
      # Quickly convert / conform all dates to datetime format

tute_cols = ['Date', 'Sales', 'AdBudget', 'GDP']
tute = pd.read_excel(tute_filepath + '.xlsx', index_col='Date')
print("\nIMPORT SUCCESS")

#%%
## INITIAL EDA
print(tute.info())
print('*'*75)
print(tute.head())

#%%
# IDENTIFY / FILTER 'OUTLIER' DATES BEYOND 2000 (EXCLUDED PER LAB GUIDELINES)
tute = tute[0:80]
print(tute.info())
print('*'*75)
print(tute.columns)
print('*'*75)
print(tute.index)

#%%
# 3. Graph the scatter plot between Sales & GDP.
# Calculate the correlation coefficient between Sales and GDP
# Update the graph title between calculated correlation coefficients.  (Hint: f string).
# Update the x and y axis with an appropriate label.
# Does the calculated correlation coefficient make sense with respect to the scatter plot? Justify your answer.

print(f'SALES/GDP CORRELATION COEFFICIENT: {correlation_coefficent(tute.Sales, tute.GDP)}')
print(f'SALES/GDP PEARSON R: {st.pearsonr(tute.Sales, tute.GDP)[0]:.2f}')

#%%
## SCATTERPLOT - SALES VS. GDP
plt.figure(figsize=(12,8))
sns.scatterplot(data=tute, x='Sales', y='GDP', palette='mako')
plt.title(f'SALES/GDP CORRELATION COEFFICIENT [R]: {correlation_coefficent(tute.Sales, tute.GDP)}')
plt.xlabel('SALES')
plt.ylabel('GDP')
#plt.grid()
plt.tight_layout(pad=1)
plt.show()

#%%
# 4. Graph the scatter plot between Sales and AdBudget.
# Calculate the correlation coefficient between Sales and AdBudget
# Update the graph title between calculated correlation coefficients. (Hint: f string) .
# Update the x and y axis with an appropriate label.
# Does the calculated correlation coefficient make sense with respect to the scatter plot? Justify your answer.
#%%
print(f'SALES/ADBUDGET CORRELATION COEFFICIENT: {correlation_coefficent(tute.Sales, tute.AdBudget)}')
print(f'SALES/ADBUDGET PEARSON R: {st.pearsonr(tute.Sales, tute.AdBudget)[0]:.2f}')

#%%
## SCATTERPLOT - SALES VS. ADBUDGET
plt.figure(figsize=(12,8))
sns.scatterplot(data=tute, x='Sales', y='AdBudget', palette='mako')
plt.title(f'SALES/ADBUDGET CORRELATION COEFFICIENT [R]: {correlation_coefficent(tute.Sales, tute.AdBudget)}')
plt.xlabel('SALES')
plt.ylabel('ADBUDGET')
#plt.grid()
plt.tight_layout(pad=1)
plt.show()

#%%
# 5. Graph the scatter plot between GDP and AdBudget.
# Calculate the correlation coefficient between GDP and AdBudget
# Update the graph title between calculated correlation coefficients. (Hint: f string) .
# Update the x and y axis with an appropriate label.
# Does the calculated correlation coefficient make sense with respect to the scatter plot? Justify your answer.

#%%
print(f'GDP/ADBUDGET CORRELATION COEFFICIENT: {correlation_coefficent(tute.GDP, tute.AdBudget)}')
print(f'GDP/ADBUDGET PEARSON R: {st.pearsonr(tute.GDP, tute.AdBudget)[0]:.2f}')

#%%
## SCATTERPLOT - GDP VS. ADBUDGET
plt.figure(figsize=(12,8))
sns.scatterplot(data=tute, x='GDP', y='AdBudget', palette='mako')
plt.title(f'GDP/ADBUDGET CORRELATION COEFFICIENT [R]: {correlation_coefficent(tute.GDP, tute.AdBudget)}')
plt.xlabel('GDP')
plt.ylabel('ADBUDGET')
#plt.grid()
plt.tight_layout(pad=1)
plt.show()

#%%
# 6. Using the Seaborn package and pairplot() function:
# Graph the correlation matrix for the tute1.csv dataset.
# Plot the Dataframe using the following options:
      # a. kind="kde"
      # b. kind="hist"
      # c. diag_kind="hist"

## PAIRPLOT - KDE
plt.figure(figsize=(12,8))
sns.pairplot(data=tute, kind='kde', palette='mako')
plt.suptitle(f'CORRELATION MATRIX')
#plt.grid()
plt.tight_layout(pad=1)
plt.show()


## PAIRPLOT - HIST
plt.figure(figsize=(12,8))
sns.pairplot(data=tute, kind='hist', palette='mako')
plt.suptitle(f'CORRELATION MATRIX')
#plt.grid()
plt.tight_layout(pad=1)
plt.show()

## PAIRPLOT - DIAG_HIST
plt.figure(figsize=(12,8))
sns.pairplot(data=tute, diag_kind='hist', palette='mako')
plt.suptitle(f'CORRELATION MATRIX')
#plt.grid()
plt.tight_layout(pad=1)
plt.show()

#%%
# 7. Using the Seaborn package and heatmap() function:
# Graph the correlation matrix for the tute1.csv dataset.

## HEATMAP - CORRELATION COEFFICIENT MATRIX
tute_corr = tute.corr()
#fig, ax = plt.subplots()
plt.figure(figsize=(12,8))
sns.heatmap(tute_corr, annot = True, cmap = 'mako', vmin=-1, vmax=1, linecolor = 'white', linewidth = 2);
plt.suptitle(f'HEATMAP CORRELATION MATRIX')
#plt.grid()
plt.tight_layout(pad=1)
plt.show()


#%%