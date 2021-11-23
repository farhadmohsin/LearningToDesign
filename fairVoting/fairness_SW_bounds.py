import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from fairness_SW_tradeoff import *

#%%

def local_noise_plurality(ballot, p):
    """
    Description:
        Adds noise to a ballot for plurality vote, only the top candidate is needed, so 
            only changing ballot[0]
        With probability p, ballot is unchanged,
        with probability (1-p), the other m candidates are drawn from uniformly
    """
    x = np.random.uniform()
    if(x<p):
        return ballot
    
    newballot = ballot.copy()
    m = len(newballot)
    i = np.floor((x-p)/(1-p)*m)
    newballot[0] = int(i)
    return newballot

def diff_private_rankings(ballots1, ballots2, eps):
    """
    Description:
        Calculate alternate ballots with coin-flipping algorithm
        This depends on how much data is available, so dependent on utility function
            Examples:
            For plurality, only m possibilities
            For Borda, m! possibilities (too-large)
            For k-Borda, mPk
            For k-approval, mCk
    Parameters:
        ballots1:   ballots for group1
        ballots2:   ballots for group2
        eps:        privacy parameter
    """
    m = len(ballots1[0])
    
    p = (np.exp(eps)-1)/(m+np.exp(eps)-1) # probability that original ballot remains
    noisy_b1 = np.zeros(ballots1.shape)
    for i,ballot in enumerate(ballots1):
        noisy_b1[i] = local_noise_plurality(ballot, p)
        
    noisy_b2 = np.zeros(ballots2.shape)
    for i,ballot in enumerate(ballots2):
        noisy_b2[i] = local_noise_plurality(ballot, p)
    
    return noisy_b1, noisy_b2 

#%% Fairness-privacy-SW trade-off
# We can try plurality, k-approval, k-Borda
  
n1 = 500
n2 = 200
m = 8

eps_all = [10,7,3,1]

rng = np.arange(0,1.01,0.1)
primary = plurality_utility

trials = 100

sw_util = 0
sw_uf = 0

fair_util = 0
fair_uf = 0

nsw_util = np.zeros(len(eps_all))
nsw_uf = np.zeros(len(eps_all))

nfair_util = np.zeros(len(eps_all))
nfair_uf = np.zeros(len(eps_all))



for t in range(trials):
    
    g1, g2, candidates = uniform_voters_candidates(n1,n2,m)
#    plot_all(g1, g2,candidates) #plotting for 2d space only
    ballots1, ballots2 = random_pref_profile(g1, g2, candidates)
    
    utilg1 = primary(ballots1)
    utilg2 = primary(ballots2)
    util = primary(np.concatenate((ballots1,ballots2)))
    
    uf = calculate_unfairness(utilg1, utilg2, util)
    
    winner = util==np.max(util)
    sw_util += np.max(util)
    sw_uf += np.nanmin(uf[winner])

    opt_uf = uf == np.nanmin(uf)
    fair_uf += np.nanmin(uf)
    fair_util += np.max(util[opt_uf]) 
    
#    print(np.argmax(util), np.argmin(uf))
#    print(util)
#    print(uf)    
    
        
#    uf_sums.append(uf_mins)
#    
    for e,eps in enumerate(eps_all):
#        
#        sw_sum = np.zeros(len(rng))
#        uf_sum = np.zeros(len(rng))
#       sw_util = 0
        ntsw_util = 0
        ntsw_uf = 0
        
        ntfair_util = 0
        ntfair_uf = 0
 
        noise_trials = 100
        for nt in range(noise_trials):
            noisy_b1, noisy_b2 = diff_private_rankings(ballots1, ballots2, eps)   
#        
            n_utilg1 = primary(noisy_b1)
            n_utilg2 = primary(noisy_b2)
            n_util = primary(np.concatenate((noisy_b1,noisy_b2)))
            
            n_uf = calculate_unfairness(n_utilg1, n_utilg2, n_util)
#            
##            print(n_util)
##            print(n_uf)
            #calculating expected SW, UF
            winner = n_util==np.max(n_util)
            ntsw_util += np.max(util[winner])
            ntsw_uf += np.nanmin(uf[winner])
        
            opt_uf = n_uf == np.nanmin(n_uf)
            ntfair_uf += np.min(uf[opt_uf])
            ntfair_util += np.max(util[opt_uf]) 
        
        nsw_util[e] += ntsw_util/noise_trials
        nsw_uf[e] += ntsw_uf/noise_trials
        
        nfair_util[e] += ntfair_util/noise_trials
        nfair_uf[e] += ntfair_uf/noise_trials
        
print(np.array([sw_util, sw_uf]))
print(np.array([fair_util, fair_uf]))

print(nsw_util)
print(nsw_uf)
print(nfair_util)
print(nfair_uf)