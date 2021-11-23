import numpy as np
from time import time
from satisfaction_calc import plurality_winner, Borda_winner, Copeland_winner, veto_winner, two_approv_winner
from fairness_SW_tradeoff import gen_PL_ballot, PL_ballots_gen

# this functions return [w], [scores]
np.random.seed(0) 

#%%-----OUTLINE-----------

# Introduction - imbalance actually leads to good things
# Properties - Pareto optimality(?) (need to actually write it down)

# Unfairness values - check specific scenarios
#   particularly worst bounds?

#   For different vec{u}, compare different "fair" voting rules
#       Take average imbalance?
#       Or rather, take worst case imbalance and plot as function of f
#       How to know which one is better?

#   For different vec{u}, compare pluarlity, Borda, Copeland winner
#       Take average imbalance?
#       Draw imbalance utility scatter plot (for all of them, separately)
#   

#%%

m = 4
n1 = 100
n2 = 30

rand_util = []

plu = np.zeros(m)
plu[0] = 1

bor = np.zeros(m)
for i in range(m):
    bor[i] = m-i-1
    
veto = np.ones(m)
veto[-1] = 0

kapp = np.zeros(m)
for i in range(int(m/2)):
    kapp[i] = 1
    
rand_util.append(plu)
rand_util.append(bor)
rand_util.append(veto)
rand_util.append(kapp)

for i in range(16):
    rand_util.append(np.sort(np.random.choice(20, m))[::-1])
    
#%%
def gen_pref_profile(n, m):
    '''
        n = no. of voters
        m = no. of alternatives
        
        random pref profile
    '''
    ballots = []
    for i in range(n):
        ballots.append(np.random.permutation(m))
    
    return np.array(ballots)

def util(ballots, u):
    n, m = ballots.shape
    score = np.zeros(m)
    for v in ballots:
        for idx, alt in enumerate(v):
            score[alt] += u[idx]
    return score, n

def imbalance(ballots1, ballots2, u):
    '''
        ballots_i = n_i*m pref profile
        util = util function, m-vector
    '''
    u1, n1 = util(ballots1, u)
    u2, n2 = util(ballots2, u)
    
    imb = np.zeros(m)
    for i in range(m):
        if(u1[i]==0 and u2[i]==0):
            imb[i] = 0
        else:
            imb[i] = np.abs(u1[i]/n1 - u2[i]/n2)/((u1[i]+u2[i]) / (n1+n2))
    return imb

def u_fair_winner(ballots1, ballots2, u):
    imb = imbalance(ballots1, ballots2, u)
    return np.argmin(imb)

#%% generate random utility functions for testing

util_set = 1000
test_util = []

for i in range(util_set):
    test_util.append(np.sort(np.random.choice(20, m))[::-1])
    
#%% now test

trials = 10

voting_rules = [plurality_winner, Borda_winner, Copeland_winner, veto_winner, two_approv_winner]

# no_u = len(rand_util)
no_u = len(voting_rules)
no_v = len(test_util)

worst_case = np.zeros([no_u, no_v])

for t in range(trials): # no of trials
    tic = time()
    # ballots1 = gen_pref_profile(n1, m)
    # ballots2 = gen_pref_profile(n2, m)
    
    # PL ballots
    ballots1, ballots2 = PL_ballots_gen(n1, n2, m)

    # This for loop was for fair voting rules
    # for iu, u in enumerate(rand_util): # the original utils
    #     ufw = u_fair_winner(ballots1, ballots2, u)

    # Now for regular voting rules
    for iu, u in enumerate(voting_rules): # the traditional voting rules
        ballots = np.concatenate((ballots1, ballots2))
        rw, _ = u(ballots) # returns list of winners
        for iv, v in enumerate(test_util): # the test utils
            v_imb = imbalance(ballots1, ballots2, v)
            
            # this is for fair winners
            # if(v_imb[ufw] > worst_case[iu, iv]):
            #     worst_case[iu, iv] = v_imb[ufw]
            
            # this is for voting rules
            for w in rw:
                if(v_imb[w] > worst_case[iu, iv]):
                    worst_case[iu, iv] = v_imb[w]
            
    toc = time()
    print(toc - tic)
        
#%%
with open('voting_rules_different_util.npy', 'wb') as f:
    np.save(f, worst_case)
    np.save(f, rand_util)
    np.save(f, test_util)

#%%
with open('PL_different_util.npy', 'rb') as f:
    dat1 = np.load(f)
    dat2 = np.load(f)
    dat3 = np.load(f)
    
#%%
with open('voting_rules_different_util.npy', 'rb') as f:
    newdat1 = np.load(f)
    newdat2 = np.load(f)
    newdat3 = np.load(f)

#%%
with open('voting_rules_different_util_2.npy', 'rb') as f:
    newdat1 = np.load(f)
    newdat2 = np.load(f)
    newdat3 = np.load(f)