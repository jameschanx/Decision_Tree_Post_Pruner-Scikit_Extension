"""
@author: James
"""

import pandas as pd
import numpy as np
#from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

#***********************Dataset 1 - Wine***************************
wine_data = pd.read_csv('winequality-white.csv', sep=';')

X_wine = wine_data.iloc[:,:-1]
y_wine = wine_data.iloc[:,-1]

plt.figure()
plt.bar(y_wine.value_counts().index, y_wine.value_counts())

#convert target values to -1, 0, 1 (bad, moderate, good)
bins = np.linspace(y_wine.min(), y_wine.max(), num=4)
y_wine = pd.cut(y_wine, bins, labels=[-1,0,1], right = True, include_lowest = True)

#***********************Dataset 2 - Abalone************************
abalone_data = pd.read_csv('abalone.csv', sep=',')

X_abalone = abalone_data.iloc[:,:-1]
y_abalone = abalone_data.iloc[:,-1]

plt.figure()
plt.bar(y_abalone.value_counts().index, y_abalone.value_counts())

sex_OHE = pd.get_dummies(X_abalone['Sex'])

X_abalone.drop(['Sex'], inplace=True, axis=1)
X_abalone = X_abalone.join(sex_OHE, how='inner')

#convert target values to -1, 0, 1 (young, mid, old)
#bins = np.linspace(y_abalone.min(), y_abalone.max(), num=4)
#y_abalone = pd.cut(y_abalone, bins, labels=[-1,0,1], right = True, include_lowest = True)

#binary classification
y_abalone = pd.cut(y_abalone, [y_abalone.min(), 9.9, y_abalone.max()], labels=[-1,1], right = True, include_lowest = True)

#-----------------------------save to HDF-------------------------------------
X_abalone.to_hdf('X_abalone.hdf','X_abalone',complib='blosc',complevel=9)
y_abalone.to_hdf('y_abalone.hdf','y_abalone',complib='blosc',complevel=9, format='table')
X_wine.to_hdf('X_wine.hdf','X_wine',complib='blosc',complevel=9)
y_wine.to_hdf('y_wine.hdf','y_wine',complib='blosc',complevel=9, format='table')
