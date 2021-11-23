import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from fairness_SW_tradeoff import *
from plackettluce import *

df1 = pd.read_csv('g1_pref.csv')
df2 = pd.read_csv('g2_pref.csv')

gamma1 = []
for row in df1.values:
    gamma1.append(np.exp(row))

gamma2 = []
for row in df2.values:
    gamma2.append(np.exp(row))
    
m = len(df1.columns.values)
#%%
    
trials = 5

for t in range(trials):
    
    ballots1 = []
    ballots2 = []
    
    for gamma in gamma1:
        vote = draw_pl_vote(m, gamma)
        ballots1.append(vote)
    
    for gamma in gamma2:
        vote = draw_pl_vote(m, gamma)
        ballots2.append(vote)
    
    ballots1 = np.array(ballots1)
    ballots2 = np.array(ballots2)
    
    primary = borda_utility
    
    utilg1 = primary(ballots1)
    utilg2 = primary(ballots2)
    util = primary(np.concatenate((ballots1,ballots2)))
    
    uf = calculate_unfairness(utilg1, utilg2, util)
    
    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.plot(range(m),util)
    ax.set_ylabel('SW')
    
    ax2 = plt.subplot(2,1,2)
    ax2.plot(range(m), uf)
    ax.set_ylabel('UF')
    ax.set_xlabel('candidate')
    