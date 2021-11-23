import xgboost as xgb

import multiprocessing
import concurrent.futures

from time import time

import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
#from sklearn.datasets import load_iris, load_digits, load_boston
#def create_voters_candidates(n1, n2, m, d=2):
#def random_pref_profile(g1,g2,candidates):

from fairness_SW_tradeoff import create_voters_candidates, random_pref_profile
from satisfaction_calc import gen_pref_profile, maximin_winner, Copeland_winner, \
    singleton_lex_tiebreaking, Condorcet_efficiency, permutation
from autoVoting import weighted_majority_graph, position_vector
from fairness_new_tradeoffs import borda_utility, util_uf, fair_borda, fairness_efficiency

#%%
def max_fair_winner(votes):
    n1 = 100
    n2 = 30
    util, uf = util_uf(votes[:n1], votes[n1:n1+n2])
    return [np.nanargmin(uf)], uf

def convert_votes_to_features(votes1, votes2):
    WMG1 = weighted_majority_graph(votes1)
    posvec1 = position_vector(votes1)
    # WMG1 /= n1
    # posvec1 /= n1
    
    WMG2 = weighted_majority_graph(votes2)
    posvec2 = position_vector(votes2)
    # WMG2 /= n2
    # posvec2 /= n2
    
    return WMG1, posvec1, WMG2, posvec2

def permute_rows(feats, perm, m):
    y = feats.copy()
    y[perm] = y[list(range(m))]
    return y

def permute_rows_cols(feats, perm, m):
    y = feats.copy()
    # y[list(range(m))] = y[perm]
    # y[:,list(range(m))] = y[:,perm]
    y[perm] = y[list(range(m))]
    y[:,perm] = y[:,list(range(m))]
    return y

def permute_features(WMG1, posvec1, WMG2, posvec2, perm):
    w1 = WMG1.copy()
    pv1 = posvec1.copy()
    w2 = WMG2.copy()
    pv2 = posvec2.copy()
    
    w1 = permute_rows_cols(w1, perm, m)
    w2 = permute_rows_cols(w2, perm, m)
    pv1 = permute_rows(pv1, perm, m)
    pv2 = permute_rows(pv2, perm, m)
    
    return w1, pv1, w2, pv2

#%% generate and save base data

def generate_data(beta, n1, n2, m):
        # tic = time()
    data_random = np.random.choice(2,p=[1-beta,beta])
    # mix_cnt += data_random
    gen = np.random.choice(2)
    # gen = 1
    # sample n2 from [10,20,...,200]
    # n2_all = np.array(range(1,21))*10
    # n2 = np.random.choice(n2_all)    
    
    # tic = time()
    if(gen):
        votes1 = gen_pref_profile(n1, m)
        votes2 = gen_pref_profile(n2, m)    
    else:
        g1, g2, candidates = create_voters_candidates(n1, n2, m)
        votes1, votes2 = random_pref_profile(g1, g2, candidates) 
    # toc = time()
    # print(f"Time to generate: {toc-tic} s")
    
    WMG1, posvec1, WMG2, posvec2 = convert_votes_to_features(votes1, votes2)
    x = np.concatenate((WMG1.flatten(), posvec1.flatten(), WMG2.flatten(), \
                        posvec2.flatten()))
    
    x = np.append(x, [n1, n2])
    
    votes = np.concatenate((votes1,votes2))
    winners, scores = Copeland_winner(votes)
    
    exist = 0
    if(len(winners) == 1):
        if(scores[winners[0]] == m-1):
            exist = 1
    
    w = singleton_lex_tiebreaking(votes, winners)
    
    wfair, _ = fair_borda(votes, n1, n2, 0)
    util, uf = util_uf(votes1, votes2)
    
    # fair_ufmean[gen] += uf[wfair]
    # w_ufmean[gen] += uf[w]
    
    return np.concatenate((x,[w, wfair, gen, exist],uf))

# for t in range(20):

if __name__ == "__main__":
    
    n1 = 100
    n2 = 30
    m = 4
    profiles = 40
    # we will generate 4! permutation of each profile
    # so essentially will have 25000*24 = 0.6M data points
    
    perms = permutation(list(range(m)))
    # voting rule: maximin_winner
    # tiebreaking rule: singleton_lex_tiebreaking
    
    X = []
    W1 = []
    W2 = []
    Wmix = []
    
    cond_exists = []
    
    mix_cnt = 0
    
    df = pd.DataFrame()
    
    fair_ufmean = np.zeros(2)
    w_ufmean = np.zeros(2)
    
    # tic = time()
    # beta = 0.7
    # for t in range(10*profiles):
    
    #     vals = generate_data(beta,n1,n2,m)
        
    #     df = df.append(pd.Series(vals),ignore_index=True)
    # toc = time()
    # print(toc - tic)
    
    
    df = pd.DataFrame()
    tic = time()
    beta = 0.7    
    
    cnt = 6000
    # actual count would be cnt*process, which now is cnt*40
    # let's get to 240000, so cnt = 6000
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in range(cnt):
            results = [executor.submit(generate_data,beta,n1,n2,m) for _ in range(40)]
        
            for f in concurrent.futures.as_completed(results):
                df = df.append(pd.Series(f.result()),ignore_index=True)
        
    toc = time()
    print(f"Time to gen data: {toc-tic}")
    df.to_csv(f'xgboostFairInput{int(toc)}.csv',index=False)
