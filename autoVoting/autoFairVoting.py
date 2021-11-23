import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from autoVoting import *
from fairness_SW_tradeoff import *

# create_voters_candidates and random_pref_profile imported from fairness_SW_tradeoff

n1 = 10
n2 = 3
m = 4

g1, g2, candidates = create_voters_candidates(n1, n2, m)
P1, P2 = random_pref_profile(g1, g2, candidates)

#%%
def max_fair_rule(P1,P2, thresh):
    P = np.concatenate((P1,P2))
    primary = borda_utility #   primary is the utility function that 
                            #   we'll use for unfairness and SW
    
    util1 = primary(P1)
    util2 = primary(P2)
    util = primary(P)

    uf = calculate_unfairness(util1, util2, util)
    util_thresh = np.max(util) * thresh 
    uf_thresh = uf[util > util_thresh] # consider only canidates greater than SW threshold
    opt_uf = np.nanmin(uf_thresh) # find min uf among eligible candidates
    
    return np.argwhere(uf==opt_uf)[0]
#%% Autovoting

def create_Dataset(n1, n2, m, T):
    """
    Description:
        Create "true data" with N voters, m alternatives
        T election data created
    """
    # "Training Data"    
    # array for preference profile and winner pairs
    PWP = []
    
    for t in range(T):
        ballots = random_pref_profile(N, m)
        winner = max_fair_rule(ballots)
        posvec = position_vector(ballots)
        wmg = weighted_majority_graph(ballots)
        
    #    x = np.concatenate((posvec.flatten(),wmg.flatten()), axis = 0)
        for w in winner:
    #        X.append(x)
    #        Y.append(w)
            new_pwp = profile_winner_pair(posvec, wmg, w)
            new_pwp.set_pref_profile(ballots)
            PWP.append(new_pwp)
    return PWP