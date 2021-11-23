import numpy as np
from time import time
from matplotlib import pyplot as plt
import pandas as pd

#%% (fairness-Condorcet efficiency plot)
fig, ax = plt.subplots()

x0 = [0.0831,0.0305,0.0829]
y0 = [1,0.27,0.92]

l0 = ['Copeland','MaxFair','Borda']

x1 = [0.072,0.070,0.068,0.063,0.058,0.054,0.047,0.041,0.037,0.034]
y1 = [0.991,0.973,0.952,0.905,0.846,0.778,0.670,0.555,0.441,0.344]
beta = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

x2 = [0.076024,0.069747,0.063683,0.058117,0.053208,0.04873,0.044766,0.041624,0.038967,0.036869,0.035229,0.033974,0.032955,0.032106,0.031649,0.031243,0.030987,0.03084,0.030748,0.030693]
y2 = [0.888,0.845,0.785,0.714,0.644,0.573,0.508,0.457,0.414,0.378,0.349,0.329,0.311,0.297,0.288,0.282,0.277,0.274,0.273,0.272]
alpha = [0.98,0.96,0.94,0.92,0.9,0.88,0.86]

ax.scatter(-1*np.array(x1),y1)
for i, b in enumerate(beta):
    ax.annotate(f"{b}-ML", (-x1[i], y1[i]), horizontalalignment='left')

ax.scatter(-1*np.array(x2),y2)
for i in range(len(alpha)):
    ax.annotate(f"{alpha[i]}-FB", (-x2[2*i+1], y2[2*i+1]), horizontalalignment='right')
    
ax.scatter(-1*np.array(x0),y0)

for i, txt in enumerate(l0):
    ax.annotate(txt, (-x0[i], y0[i]))

ax.set_xlabel('Mean Fairness')
ax.set_ylabel('Condorcet Efficiency')

plt.show()

#%% Privacy plot - 1 (not used)
df = pd.read_csv('C:/RPI/CompSoc/Ethical AI/fairVoting/NeurIPS_tradeoffnew.csv')
# df.rename(columns={"0": "threshold", "1": "eps", "2":"uf", "3":"sw"}, inplace = True)
# df['uf'] = df['uf']
# df['sw'] = df['sw']
summary = df.groupby(['threshold','eps']).mean()
summary = summary.reset_index()

summary['eff'] = summary['Condorcet']/summary['exist']

nf = [summary[summary['eps']==i] for i in summary['eps'].unique()]

eps_all = df['eps'].unique()
eps_all[0] = 'inf'
colors = ['b', 'g', 'r', 'c']
markers = ['o','v','s','P']
fig, ax = plt.subplots()

for i in range(1,len(colors)):
    ax.scatter(-nf[i]['uf'],nf[i]['sw'], c = colors[i], marker= markers[i], label = r"$\varepsilon = %d$"%eps_all[i])
ax.scatter(-nf[0]['uf'],nf[0]['sw'], c = colors[0], marker= markers[0], label = r"$\varepsilon = \infty$")

alpha = np.arange(0,1.01,0.1)
for i,a in enumerate(alpha[5:]):
    ax.annotate(f"{a:.1f}-FP", (np.array(-nf[0]['uf'])[i+5],np.array(nf[0]['sw'])[i+5]))

ax.set_xlabel('Mean Fairness')
ax.set_ylabel('Mean Social Welfare')
ax.legend()

# ax.set_ylim([0,1.0])

plt.show()

#%% Privacy plot - 2 (not used)
df = pd.read_csv('C:/RPI/CompSoc/Ethical AI/autoVoting/fairness-privacy-sw-summary.csv')

summary = df.groupby(['threshold','eps']).mean()
summary = summary.reset_index()

nf = [summary[summary['eps']==i] for i in summary['eps'].unique()]

eps_all = list(df['eps'].unique())
eps_all[0] = 'inf'
colors = ['b', 'g', 'r', 'c']
markers = ['o','v','s','P']
fig, ax = plt.subplots()

for i in range(1,len(colors)):
    ax.scatter(nf[i]['imb-ave'],nf[i]['sw-ave'], c = colors[i], marker= markers[i], label = r"$\varepsilon = %d$"%eps_all[i])

ax.errorbar(nf[0]['imb-ave'],nf[0]['sw-ave'], xerr=nf[0]['imb-std'], yerr=nf[0]['sw-stdev'], c = colors[0], marker= markers[0], label = r"$\varepsilon = \infty$")

alpha = np.arange(0,1.01,0.1)
for i,a in enumerate(alpha[5:]):
    ax.annotate(f"{a:.1f}-FP", (np.array(nf[0]['imb-ave'])[i+5],np.array(nf[0]['sw-ave'])[i+5]))

ax.set_xlabel('Mean Fairness')
ax.set_ylabel('Mean Social Welfare')
ax.legend()

# ax.set_ylim([0,1.0])

plt.show()