import xgboost as xgb

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
def alpha_fair_winner(votes):
    n1 = 100
    n2 = 30
    alpha = 0.8
    w, uf = fair_borda(votes, n1, n2, alpha)
    return w

def alpha_fair_winner2(votes):
    n1 = 100
    n2 = 30
    alpha = 0.8
    w, uf = fair_borda(votes, n1, n2, alpha)
    return [w], uf

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

# for t in range(20):
n1 = 200
n2 = 60
m = 4
profiles = 1000
# we will generate 4! permutation of each profile
# so essentially will have 25000*24 = 0.6M data points

perms = permutation(list(range(m)))
# voting rule: maximin_winner
# tiebreaking rule: singleton_lex_tiebreaking


for beta in np.arange(0,1.01,0.1):
    print(f"beta = {beta}")
    X = []
    W1 = []
    W2 = []
    Wmix = []
    
    cond_exists = []
    
    mix_cnt = 0
    
    df = pd.DataFrame()
    df_uf = pd.DataFrame()
    
    tic = time()
    
    fair_ufmean = np.zeros(2)
    w_ufmean = np.zeros(2)
    
    for t in range(profiles):
        
        # beta = 0.1
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
        
        fair_ufmean[gen] += uf[wfair]
        w_ufmean[gen] += uf[w]
        
        X.append(x)
        W1.append(w)
        W2.append(wfair)
        
        if(exist):
            if(w==wfair):
                Wmix.append(w)
                mix_cnt += 1
            else:
                if(data_random):
                    Wmix.append(w)
                    mix_cnt += 1
                else:
                    Wmix.append(wfair)
        else:
            Wmix.append(wfair)
        
        df = df.append(pd.Series(np.concatenate((x,[w,wfair]))),ignore_index=True)
        df_uf = df_uf.append(pd.Series(np.concatenate((uf,[w, gen, exist]))), ignore_index=True)
        
        # tic = time()
        for pi in perms[1:]:
            wmg1,pv1,wmg2,pv2 = permute_features(WMG1, posvec1, WMG2, posvec2, pi)       
            x = np.concatenate((wmg1.flatten(), pv1.flatten(), wmg2.flatten(), \
                                pv2.flatten()))
            x = np.append(x, [n1, n2])
            
            X.append(x)
            W1.append(pi[w])
            W2.append(pi[wfair])
            if(exist):
                if(w==wfair):
                    Wmix.append(pi[w])
                else:
                    if(data_random):
                        Wmix.append(pi[w])
                    else:
                        Wmix.append(pi[wfair])
            else:
                Wmix.append(pi[wfair])
            # if(data_random):
            #     Wmix.append(pi[w])
            # else:
            #     Wmix.append(pi[wfair])
        # # toc = time()
        # # print(f"Time to permute*24: {toc-tic} s")
    
    # df.to_csv('xgboostFairInput.csv',index=False)
    toc = time()
    print(f"Time to gen data: {toc-tic}")
    
    X = np.array(X)
    W1 = np.array(W1)
    W2 = np.array (W2)
    Wmix = np.array(Wmix)
    
    print(mix_cnt, profiles)
    print(fair_ufmean/(profiles/2))
    print(w_ufmean/(profiles/2))
    # print(np.sum(W1==W2))
    #%% read data from written file and generate permutations
    # df = pd.read_csv('xgboostFairInput.csv')
    
    # X = []
    # W1 = []
    # W2 = []
    # Wmix = []
    
    # tic = time()
    
    # for row in df.values:
    #     # 68 = m*2 + m*2 + m*2 + m*2 + 2 + 2
    #     data_random = np.random.choice(2,p=[0.1,0.9])
        
    #     WMG1 = np.reshape(row[:m*m],(m,m))
    #     posvec1 = np.reshape(row[m*m: 2*m*m],(m,m))
    #     WMG2 = np.reshape(row[2*m*m: 3*m*m],(m,m))
    #     posvec2 = np.reshape(row[3*m*m: 4*m*m],(m,m))
    #     n1 = row[4*m*m]
    #     n2 = row[4*m*m+1]
    #     w = int(row[4*m*m+2])
    #     wfair = int(row[4*m*m+3])
        
    #     x = np.concatenate((WMG1.flatten(), posvec1.flatten(), WMG2.flatten(), \
    #                         posvec2.flatten()))
    #     x = np.append(x, [n1, n2])
        
    #     X.append(x)
    #     W1.append(w)
    #     W2.append(wfair)
    #     if(data_random):
    #         Wmix.append(w)
    #     else:
    #         Wmix.append(wfair)
        
    #     for pi in perms[1:]:
    #         wmg1,pv1,wmg2,pv2 = permute_features(WMG1, posvec1, WMG2, posvec2, pi)       
    #         x = np.concatenate((wmg1.flatten(), pv1.flatten(), wmg2.flatten(), \
    #                             pv2.flatten()))
    #         x = np.append(x, [n1, n2])
            
    #         X.append(x)
    #         W1.append(pi[w])
    #         W2.append(pi[wfair])
    #         if(data_random):
    #             Wmix.append(pi[w])
    #         else:
    #             Wmix.append(pi[wfair])
    # toc = time()
    # print(f'{len(df)} {toc-tic}')
    
    # X = np.array(X)
    # W1 = np.array(W1)
    # W2 = np.array (W2)
    # Wmix = np.array(Wmix)
    #%% checking fairBorda correctness
    
    # g1, g2, candidates = create_voters_candidates(n1, n2, m)
    # votes1, votes2 = random_pref_profile(g1, g2, candidates)
    # util, uf = util_uf(votes1, votes2)
    
    # print(util, np.argmax(util))
    # print(uf, np.nanargmin(uf))
    
    
    #%%
    # m_test = 5
    # n_test = 15
    # perms4 = permutation(list(range(m_test)))
    
    # # pi = perms4[3]
    
    # # votes = gen_pref_profile(n_test,m_test)
    # # p_votes = []
    # # for v in votes:
    # #     vv = []
    # #     for j in v:
    # #         vv.append(pi[j])
    # #     p_votes.append(vv)
    
    # # p_votes = np.array(p_votes)
    
    # # # print(position_vector(votes))
    # # pvv1 = position_vector(p_votes)
    # # pvv2 = permute_rows(position_vector(votes),pi,m_test)
    
    # # # print(weighted_majority_graph(votes))
    # # wmgg1 = weighted_majority_graph(p_votes)
    # # wmgg2 = permute_rows_cols(weighted_majority_graph(votes),pi,m_test)
    # for c,pi in enumerate(perms4):
    #     cnt_c1 = 0
    #     cnt_c2 = 0
    #     for t in range(100):    
    #         votes = gen_pref_profile(n_test,m_test)
            
    #         # P = make_perm_matrix(perms3[-1])
    #         # print(perms3[-1])
    #         # print(P)
            
    #         p_votes = []
    #         for v in votes:
    #             vv = []
    #             for j in v:
    #                 vv.append(pi[j])
    #             p_votes.append(vv)
            
    #         p_votes = np.array(p_votes)
            
    #         # print(position_vector(votes))
    #         pvv1 = position_vector(p_votes)
    #         pvv2 = permute_rows(position_vector(votes),pi,m_test)
            
    #         # print(weighted_majority_graph(votes))
    #         wmgg1 = weighted_majority_graph(p_votes)
    #         wmgg2 = permute_rows_cols(weighted_majority_graph(votes),pi,m_test)
            
    #         # print(f'{pi} {t} {np.sum(wmgg1==wmgg2)} {np.sum(pvv1==pvv2)}')
    #         if(np.sum(wmgg1==wmgg2) < m_test*m_test):
    #             cnt_c1 += 1
                
    #         if(np.sum(pvv1==pvv2) < m_test*m_test):
    #             cnt_c2 += 1
    #     print(f'{pi}, {cnt_c1}, {cnt_c2}')
    
    #%% learning maximin
    # rng = np.random.RandomState(31337)
    # kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    # for train_index, test_index in kf.split(X):
    #     # param_dist = {'gamma': 0.2, 'max_depth': 6, 'min_child_weight': 7}
    #     # clf = xgb.XGBClassifier(**param_dist)
    #     clf = xgb.XGBClassifier()
    #     xgb_model = clf.fit(X[train_index], W1[train_index])
    #     predictions = xgb_model.predict(X[test_index])
    #     actuals = W1[test_index]
    #     print(confusion_matrix(actuals, predictions))
    #     print(accuracy_score(actuals, predictions))
        
    # #%% learning fairness
    # rng = np.random.RandomState(31337)
    # kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    # for train_index, test_index in kf.split(X):
    #     # param_dist = {'gamma': 0.2, 'max_depth': 6, 'min_child_weight': 7}
    #     # clf = xgb.XGBClassifier(**param_dist)
    #     clf2 = xgb.XGBClassifier()
    #     xgb_model2 = clf2.fit(X[train_index], W2[train_index])
    #     predictions2 = xgb_model2.predict(X[test_index])
    #     actuals = W2[test_index]
    #     print(confusion_matrix(actuals, predictions2))
    #     print(accuracy_score(actuals, predictions2))
        
    #%% learning mixedRule
    rng = np.random.RandomState(31337)
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X):
        # param_dist = {'gamma': 0.2, 'max_depth': 6, 'min_child_weight': 7}
        # clf = xgb.XGBClassifier(**param_dist)
        clf3 = xgb.XGBClassifier()
        xgb_model3 = clf3.fit(X[train_index], Wmix[train_index])
        predictions3 = xgb_model3.predict(X[test_index])
        actuals = Wmix[test_index]
        print(confusion_matrix(actuals, predictions3))
        print(accuracy_score(actuals, predictions3))
    
    #%% Compute condorcet_efficiency and unfairness for 
    factm = len(perms)
    uf_new = 0
    cnt_uf = 0
    cnt_nan = 0
    cnt_cond_up = 0
    cnt_cond_dn = 0
    
    #df_uf columns - [0-3] = uf, 4 = Copeland_winner, 5 = generating_method, 6 = Conodrcet exist? 
    
    for i,ix in enumerate(test_index):
        df_ix = int(ix/factm)
        pi = perms[ix%factm]
        df_ufx = df_uf.iloc[df_ix].values
        if(df_ufx[-2]<1):
            continue
        uf_val = df_ufx[pi[predictions3[i]]]
        if(np.isnan(uf_val)):
            cnt_nan += 1
        else:
            uf_new += df_ufx[pi[predictions3[i]]]
            cnt_uf += 1
            
        if(df_ufx[-1]==1):
            cnt_cond_dn += 1
            if(df_ufx[-3] == predictions3[i]):
                cnt_cond_up += 1
        
        # print(df_ufx[pi[predictions3[i]]])
        # print(i, actuals[i], predictions3[i])
    print(uf_new, cnt_uf, cnt_nan)
    print(uf_new/cnt_uf)
    print(cnt_cond_dn, cnt_cond_up)
    print(cnt_cond_up / cnt_cond_dn)
    
#%%

def ML_voting(xgbModel, votes, n1, n2):
    # WMG1 = weighted_majority_graph(votes[:n1])
    # posvec1 = position_vector(votes[:n1])
    # WMG1 /= n1
    # posvec1 /= n1
    
    # WMG2 = weighted_majority_graph(votes[n1:n2])
    # posvec2 = position_vector(votes[n1:n2])

    # WMG2 /= n2
    # posvec2 /= n2
    
    # x = np.concatenate((WMG1.flatten(), posvec1.flatten(), WMG2.flatten(), posvec2.flatten()))
    # x = np.append(x, [n1, n2])
    WMG1, posvec1, WMG2, posvec2 = convert_votes_to_features(votes1, votes2)
    x = np.concatenate((WMG1.flatten(), posvec1.flatten(), WMG2.flatten(), \
                            posvec2.flatten()))
    x = np.append(x, [n1, n2])
    
    prob = xgbModel.predict_proba(np.array([x]))
    return prob[0]

def testMLVoting(votes):
    w_all = ML_voting(xgb_model2, votes, 100, 30)
    w = np.argmax(w_all)
    return [w], w_all
    
#%% testing maximin
# n1 = 100
# n2 = 30
# new_trials = 1000
# cnt_ml = 0

# for t in range(new_trials):
#     gen = np.random.choice(2)
#     # gen = 1
#     # sample n2 from [10,20,...,200]
#     # n2_all = np.array(range(1,21))*10
#     # n2 = np.random.choice(n2_all)    
    
#     # tic = time()
#     votes1 = gen_pref_profile(n1, m)
#     votes2 = gen_pref_profile(n2, m)
#     # WMG1, posvec1, WMG2, posvec2 = convert_votes_to_features(votes1, votes2)
#     # x = np.concatenate((WMG1.flatten(), posvec1.flatten(), WMG2.flatten(), posvec2.flatten()))
#     # x = np.append(x, [n1, n2])
    
#     votes = np.concatenate((votes1,votes2))
#     winners, scores = maximin_winner(votes)
#     winners_ml, _ = testMLVoting(votes)
    
#     # print(scores)
#     # print(_)
    
#     if(len(set(winners_ml).intersection(set(winners)))>0):
#         cnt_ml += 1

# print(cnt_ml, new_trials)

#%% testing alpha-fairBorda
# n1 = 100
# n2 = 30
# new_trials = 1000
# cnt_ml = 0

# for t in range(new_trials):
#     if(gen):
#         votes1 = gen_pref_profile(n1, m)
#         votes2 = gen_pref_profile(n2, m)    
#     else:
#         g1, g2, candidates = create_voters_candidates(n1, n2, m)
#         votes1, votes2 = random_pref_profile(g1, g2, candidates)
#     # WMG1, posvec1, WMG2, posvec2 = convert_votes_to_features(votes1, votes2)
#     # x = np.concatenate((WMG1.flatten(), posvec1.flatten(), WMG2.flatten(), posvec2.flatten()))
#     # x = np.append(x, [n1, n2])
    
#     wfair, _ = fair_borda(votes, n1, n2, 0.8)
#     winners_ml, _ = testMLVoting(votes)
    
#     # print(scores)
#     # print(_)
    
#     if(len(set(winners_ml).intersection(set([wfair])))>0):
#         cnt_ml += 1

# print(cnt_ml, new_trials)

#%%

def fairness_eff_prob(xgbModel, n1 = 200, n2 = 60, m=4, profiles=1000, threshold = 0.8):
    eff = []
    unfairness_fair = []
    unfairness_w = []
    for test in range(20):
        # print("test = %d"%test)
        cnt = 0
        fair_ufmean = 0
        w_ufmean = 0
        for t in range(profiles):
            votes = gen_pref_profile(n1+n2, m)
            winner_distributions = ML_voting(xgbModel, votes, n1, n2)
            # print(winner_distributions)
            w_fair, uf = fair_borda(votes, n1,n2, threshold)
            fair_uf = uf[w_fair]
            
            w_uf = 0
            # for j in range(m):
            #     w_uf += uf[j]*winner_distributions[j]
            w = np.argmax(winner_distributions)
            w_uf += uf[w]
            if(np.argmax(winner_distributions) == w_fair):
                cnt += 1
            
            fair_ufmean += fair_uf
            w_ufmean += w_uf
            # print("profile %d"%t, fair_ufmean, w_ufmean)
        
        unfairness_fair.append(fair_ufmean/profiles)
        unfairness_w.append(w_ufmean/profiles)
        eff.append(cnt/profiles)
        # print(voting_rule.__name__,"N = %d, m = %d"%(N,m))
        # print("profiles = %d, Condorcet_winner = %d, V_winner = %d"%(profiles, cnt, cnt_v))
        
        # eff.append(cnt_v/cnt)
        print(fair_ufmean, w_ufmean, cnt, profiles)
    return eff, unfairness_fair, unfairness_w

#%%
# E, FU, WU =  fairness_eff_prob(xgb_model2)
# print("MLVote: n1 = 200, n2 = %d, fair_eff = %lf, fair_uf = %lf, w_uf = %lf"%(n2, np.mean(E), np.mean(FU), np.mean(WU)))
# E, FU, WU = fairness_efficiency(testMLVoting, n1=200, n2=60, profiles = 50)

# #%%
# Condorcet_efficiency(alpha_fair_winner2, N=260, m=4, profiles=1000)
# Condorcet_efficiency(testMLVoting, N=260, m=4, profiles=1000)
# Condorcet_efficiency(max_fair_winner, N=260, m=4, profiles=1000)
# xgb_model = xgb.XGBRegressor()
# clf = GridSearchCV(xgb_model,
#                    {"max_depth"        : [ 2, 4, 6, 8],
#                     "min_child_weight" : [ 1, 3, 5, 7 ],
#                     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]}, 
#                    verbose=0)
# clf.fit(X,W)
# print(clf.best_score_)
# print(clf.best_params_)
