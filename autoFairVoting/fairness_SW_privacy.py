import pandas as pd
import numpy as np
from time import time
from fairness_SW_tradeoff import plurality_utility
from fairness_new_tradeoffs import fair_plurality
from satisfaction_calc import condorcet_exist, gen_pref_profile, gen_random_vote

#%%
def util_uf(votes1, votes2):
    """
    Redefining util_uf function for top-1 utility

    Parameters
    ----------
    votes1 : preference profile for group 1.
    votes2 : preference profile for group 2.

    Returns
    -------
    TYPE
        average social welfare and unfairness for each alternative

    """
    util1 = plurality_utility(votes1)
    util2 = plurality_utility(votes2)
    util = plurality_utility(np.concatenate((votes1,votes2)))
    
    m = len(util)
    uf = np.zeros(m)
    for j in range(m):
        if(util[j]==0):
            uf[j] = np.nan
        else:
            uf[j] = np.abs(util1[j] - util2[j]) / util[j]
    return util, np.array(uf)

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
    
    m = len(ballot)
    # print(ballot.shape)
    newballot = gen_random_vote(m)
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
    # print(f"p={p}")
    noisy_b1 = np.zeros(ballots1.shape)
    for i,ballot in enumerate(ballots1):
        noisy_b1[i] = local_noise_plurality(ballot, p)
        
    noisy_b2 = np.zeros(ballots2.shape)
    for i,ballot in enumerate(ballots2):
        noisy_b2[i] = local_noise_plurality(ballot, p)
    
    return noisy_b1, noisy_b2

#%%

def privacy_main(n1, n2, m):
    
    eps_all = np.arange(4)
    
    df = pd.DataFrame()
    trials = 10
    
    for _ in range(trials):    
        ballots1 = gen_pref_profile(n1, m)
        ballots2 = gen_pref_profile(n2, m)
        
        print(f"trial = {_}")
        
        exist, C = condorcet_exist(np.concatenate((ballots1, ballots2)))
        
        for thresh in np.arange(0,1.01,0.1):    
                          
            util, _ = util_uf(ballots1, ballots2)
            w, uf = fair_plurality(np.concatenate((ballots1, ballots2)), n1, n2, thresh)

            df = df.append(pd.Series([thresh, 0, uf[w], util[w], exist, 1 if w in C else 0]), ignore_index=True)
            for eps in eps_all[1:]:
                nb1, nb2 = diff_private_rankings(ballots1, ballots2, eps)
                nw, _ = fair_plurality(np.concatenate((nb1, nb2)), n1, n2, thresh)
                df = df.append(pd.Series([thresh, eps, uf[nw], util[nw], exist, 1 if nw in C else 0]), ignore_index=True)


    df.rename(columns={0: "threshold", 1: "eps", 2:"uf", 3:"sw", 4: "exist", 5:"Condorcet"}, inplace = True)
    return df

if __name__ == "__main__":
    n1 = 100
    m = 4
    
    for n2 in range(20,101,20):
        df = privacy_main(n1, n2, m)
        df.to_csv(f'Privacy-fairness-efficiency-{n1}-{n2}-{m}.csv',index=False)
    
    
