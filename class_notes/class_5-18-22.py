#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import datetime

print("IMPORT SUCCESS")

#%%
# ASSIGN RANDOM SEED
np.random.seed(123)

#%%
# ASSIGN VARIABLES
mean = 0
std = 1
T = 1000
x = np.random.normal(mean, std, T)

#%%
plt.figure(figsize=(12,8))
plt.plot(x)
plt.grid()
plt.show()


#%%
plt.figure(figsize=(12,8))
plt.hist(x, bins=50)
plt.grid()
plt.show()


#%%
# Create 4 normally distributed features as x1, x2, ... xD (D = # of features)
# Generate dataframe
# Daily time
# variance increases by 1 each x
# mean shifts so each hist is visible

gaussian_df = pd.DataFrame()

def random_function(df):
    for x in x_four:
        mean[x]
        gaussian_df[x] = x


#%%
T = 1000

mean_x1 = 0
mean_x2 = 5
mean_x3 = 10
mean_x4 = 15

std_x1 = 1
std_x2 = 2
std_x3 = 3
std_x4 = 4

x1 = np.random.normal(mean_x1, std_x1, T)
x2 = np.random.normal(mean_x2, std_x2, T)
x3 = np.random.normal(mean_x3, std_x3, T)
x4 = np.random.normal(mean_x4, std_x4, T)

#%%
gaussian_df['x1'] = x1
gaussian_df['x2'] = x2
gaussian_df['x3'] = x3
gaussian_df['x4'] = x4

print(gaussian_df.head())
print('*'*100)
print(gaussian_df.info())
print('*'*100)
print(gaussian_df.describe())

#%%

x_four = [x1, x2, x3, x4]
plt.figure(figsize=(12,8))
plt.hist(gaussian_df,
         bins=100,

         )
plt.legend(loc='best')
plt.title('X1 / X2 / X3 / X4 DISTRIBUTION')
plt.xlabel('DISTRIBUTION RANGE')
plt.ylabel('FREQUENCY')
plt.grid()
plt.show()


#%%
# PROFESSOR J SOLUTION:
n = 4  # 10
T = 1000 # 50
M = np.zeros((T, n))
col = []

for i in range(n):
    col.append('x' + str(i + 1))
    x = np.random.normal(5 * 1, i + 1, T)
    if i==1:
        M[:, i] = M[:, i-1]/2 # forcing second feature to be half of first feature
    else:
        M[:, i] = x


time = pd.date_range(end=datetime.today(), periods=T)

df = pd.DataFrame(data=M, columns=col, index=time)

print(df.head())
print('*'*100)
print(df.info())
print('*'*100)
print(df.describe())

#%%
## PLOT
plt.figure(figsize=(12,8))
df.plot()
plt.grid()
plt.show()

#%%
## HISTOGRAM
plt.figure(figsize=(12,8))
df.plot.hist(bins=50, alpha=.5)
plt.grid()
plt.show()

#%%
## SCATTER MATRIX
plt.figure(figsize=(12,8))
pd.plotting.scatter_matrix(df, alpha=0.5)
plt.grid()
plt.show()

#%%
## ADD COEFFICIENT TITLES

def corr_func(x,y, **kws):
    r, _ = stats.pearsonr(x,y)
    ax = plt.gca()
    ax.annotate(f'r = {r:.2f}',
                xy = (0.1,0.9),
                xycoords = ax.transAxes)


#%%
## HEATMAP
g = sns.PairGrid(df, palette='mako')
g.map_upper(plt.scatter, s=10)
g.map_lower(sns.kdeplot, cmap='flare')
g.map_diag(sns.distplot, kde=False)
g.map_lower(corr_func())


#%%
plt.figure(figsize=(12,8))
sns.pairplot(gaussian_df, kind='kde')
plt.grid()
plt.show()

#%%
plt.figure(figsize=(12,8))
sns.pairplot(gaussian_df, kind='hist')
plt.grid()
plt.show()


#%%
## CORRELATION COEFFICIENT MATRIX

df_corr = gaussian_df.corr()

fig, ax = plt.subplots()

#plt.figure(figsize=(12,8))
sns.heatmap(df_corr, annot = True, cmap = 'mako', vmin=-1, vmax=1, linecolor = 'white', linewidth = 2);
plt.grid()
plt.show()


#%%

#%%



#%%



#%%