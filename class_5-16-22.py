#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("IMPORT SUCCESS")

#%%
df = pd.read_excel('/Users/nehat312/GitHub/Complex-Data-Visualization-/class_data/CO2_1970-2015_dataset_of_CO2_report_2016.xls',
                   header=[0],
                   index_col=0,
                   squeeze=True,
                   )

print(df.info())
#print(df[12:20])

#%%
CO2_malaysia = df.loc['Malaysia']
CO2_germany = df.loc['Germany']
Year = np.arange(1970, 2016)


print(CO2_malaysia)
#%%

plt.figure(figsize=(8,8))
df.plot(Year, CO2_malaysia, label='Malaysia')
df.plot(Year, CO2_germany, label='Germany')
plt.xlabel("DATE")
plt.ylabel("C02 EMISSION LEVEL")
plt.legend(loc='best')
plt.grid()
plt.show()


#%%
## TOOLBOX - CREATE FUNCTIONS:

#import syssys.path.appen(r'filepath')
#from toolbox import Plot_Rolling_Mean_Var, ADF_calc, KPSS_test, difference

#%%

Plot_Rolling_Mean_Var()


#%%
## DIFFERENCE FUNCTION

def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        values = dataset[i] - dataset[i-interval]
        diff.append(values)
    return diff

#%%
# FIRST-ORDER DIFFERENCING
CO2_malaysia_diff = difference(CO2_malaysia)
Plot_Rolling_Mean_Var(np.array(CO2_malaysia_diff), 'Malaysia')
ADF_Calc(CO2_malaysia_diff)
KSSP_Test(CO2_malaysia_diff)

#%%
# SECOND-ORDER DIFFERENCING
CO2_malaysia_diff_diff = difference(CO2_malaysia_diff)
Plot_Rolling_Mean_Var(np.array(CO2_malaysia_diff_diff), 'Malaysia')
ADF_Calc(CO2_malaysia_diff_diff)
KSSP_Test(CO2_malaysia_diff_diff)

#%%
# THIRD-ORDER DIFFERENCING
CO2_malaysia_diff_diff_diff = difference(CO2_malaysia_diff_diff)
Plot_Rolling_Mean_Var(np.array(CO2_malaysia_diff_diff_diff), 'Malaysia')
ADF_Calc(CO2_malaysia_diff_diff_diff)
KSSP_Test(CO2_malaysia_diff_diff_diff)

#%%