import numpy as np
from time import time
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

df = pd.read_csv('C:/RPI/CompSoc/Ethical AI/fairVoting/NeurIPS_tradeoffnew.csv')

#%%

remove_n = int(len(df)*50/51)
drop_indices = np.random.choice(df.index, remove_n, replace=False)
df = df.drop(drop_indices)

#%%
x = df['uf']
xx = 1-df['uf']/df['uf'].max()

y = df['sw']

z = df['eps']

xmin = min(xx)
xmax = max(xx)

ymin = min(y)
ymax = max(y)

nx = 1000
ny = 1000

xi = np.linspace(xmin, xmax, nx)
yi = np.linspace(ymin, ymax, ny)

X,Y= np.meshgrid(xi,yi)
Z = griddata((x, y), z, (X, Y),method='nearest')

#%%

# plt.scatter(xx,y)
cs = plt.contour(X, Y, Z, levels=[0, 2, 3, 5], extend='both')

plt.scatter(xx, y)
