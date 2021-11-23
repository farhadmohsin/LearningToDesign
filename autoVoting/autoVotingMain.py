from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from autoVoting import *

np.set_printoptions(precision=3)
#%%
# all function and class definitions in autoVoting.py
#def random_pref_profile(n, m):
#def Borda_winner(votes):
#def plurality_winner(votes):
#def Copeland_winner(votes):
#def maximin_winner(votes):
#def weighted_majority_graph(votes):
#def position_vector(votes):
#def ranking_count(votes):
#class profile_winner_pair:
#    def __init__(self, position_vector, WMG, W):
#    def set_pref_profile(self, pref_profile):
#def neutral_sample_gen(prof_win_pair):
#def consistent_sample_gen(pwp1, pwp2):
#def monotonic_sample_gen(pwp):
#def upscale_sample_gen(pwp):

#%% Autovoting

def create_Dataset(N, m, T):
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
        winner = Borda_winner(ballots)[0]
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

#%%
# Break down initial X,Y as train and test
def train_test_split(PWP, K, N):
    """
    Description:
        Split "true data" into train and test sets
        Generate synthetic data from train data using axioms to create complete train data
    Input:
        Preference profile, winner pairs
    Parameter:
        K: no of new (pref profile, winner) pairs to be created
    """
    
    dataset_size = len(PWP)
    train_size = round(dataset_size * 0.7)
    train_idx = slice(0,train_size)
    test_idx = slice(train_size,dataset_size)
    
    # generate new data based on axioms
    
    PWP_train = PWP[train_idx]
    
    #def neutral_sample_gen(prof_win_pair):
    #def consistent_sample_gen(pwp1, pwp2):
    #def monotonic_sample_gen(pwp):
    
    tic = time()
    for i in range(round(K/3)):
        #neutral_sample_generation
        rand_idx = np.random.randint(len(PWP_train))
        new_pwp = neutral_profile_gen(PWP_train[rand_idx])
        PWP_train.append(new_pwp)
        
        #consistent_sample_generation
        max_iter = 100
        it = 0
        while(it < max_iter):
            rand_idx = np.random.randint(len(PWP_train), size = 2)
            if(PWP_train[rand_idx[0]].winner == PWP_train[rand_idx[1]].winner):
                break
            it += 1
        new_pwp = consistent_profile_gen(PWP_train[rand_idx[0]],PWP_train[rand_idx[1]])
        PWP_train.append(new_pwp)
        
        #monotonic_sample_gen
        #only possible when pref profile available, so from true data
        rand_idx = np.random.randint(train_size)
        new_pwp = monotonic_sample_gen(PWP_train[rand_idx])
        PWP_train.append(new_pwp)
    toc = time()
    print("Synthetic data generation time. K = %d, time = %d"%(K,(toc - tic)*1000))
    X_train = [np.concatenate((pwp.position_vector.flatten(), \
                               pwp.WMG.flatten()), axis = 0) for pwp in PWP_train]
    X_train = np.array(X_train)/N
    X_test = [np.concatenate((pwp.position_vector.flatten(), \
                               pwp.WMG.flatten()), axis = 0) for pwp in PWP[test_idx]]
    X_test = np.array(X_test)/N
    
    Y_train = [pwp.winner for pwp in PWP_train]
    Y_test = [Borda_winner(pwp.pref_profile)[0] for pwp in PWP[test_idx]]
    
    return X_train, Y_train, X_test, Y_test


#%% "Testing" on random new profiles

def accuracy_Borda(logreg, S):
    """
    Description:
        Generate S new election data and compare winner with prediction by learnt model
    """
    cnt = 0
    for s in range(S):
    #    print("Starting case ", s)
        ballots = random_pref_profile(N, m)
        winner, score = Borda_winner(ballots)
    #    print("winner = ", winner, "scores: ", score)
    #    print(np.argsort(-score))
        posvec = position_vector(ballots)
        wmg = weighted_majority_graph(ballots)
        xx = [np.concatenate((posvec.flatten(),wmg.flatten()), axis = 0)]
        predict_probs = logreg.predict_proba(xx)
    #    print(np.argsort(-predict_probs[0]))
    #    print("highest probability: ",np.argmax(predict_probs[0]))
        if(np.argmax(predict_probs[0]) in winner):
            cnt += 1
         
    print(cnt,S)
    return cnt/S

#%% Testing on test dataset

def accuracy_test(X_test, Y_test, logreg):
    """
    Description:
        Calculate accuracy on holdout test set
    """
    predict_probs_test = logreg.predict_proba(X_test)
    
    cnt = 0
    for c in range(len(Y_test)):
        y_pred = np.argmax(predict_probs_test[c])
        if y_pred in Y_test[c]:
            cnt += 1
        
    print(cnt, len(Y_test))
    return cnt/len(Y_test)

#%%
vals = []

#%%
# for the "real" data, assume N = 1000, m = 6
# T is the the number of "real" elections
# let K be no of new (pref profile, winner) pairs to be created
N = 100
#K = 300
#T = 50
#m = 4

T_range = []
T_start = 10
for i in range(8):
    T_range.append(T_start)
    T_start *= 2

K_range = [0]
K_start = 10
for i in range(3):
    K_range.append(K_start)
    K_start *= 10

#%%
df0 = pd.DataFrame()

for m in [3]:
#    for T in range(10,101,10):
    for T in T_range:
        print("m =",m,"T =",T)
        
#        for K in range(0, 501, 50):
        for K in K_range:
            
            print(m,T,K)
            
            trials = 20
            for t in range(trials):
                PWP = create_Dataset(N, m, T)
                
                tic = time()
                X_train, Y_train, X_test, Y_test = train_test_split(PWP, K, N)
                
                # Fit model to logistic regression
                # using scikitlearn's LogisticRegression function
                logreg = LogisticRegression(C = 10)
                logreg.fit(X_train, Y_train)
                
                toc = time()
                
                acc1_t = accuracy_test(X_test, Y_test, logreg)
                
                # S is the number of new preference profile to test
                S = 100
                acc2_t = accuracy_Borda(logreg, S)
                
                vals.append([m, (toc-tic)*1000, T, K, t, acc1_t, acc2_t])
                
                # print(toc - tic, acc1_t, acc2_t)
                
    df = pd.DataFrame(vals, columns = ['m', 'time', 'T','K','trial','acc_1','acc_2'])    
    df0 = df0.append(df)

df0.to_csv('borda_20200324_2.csv',index=False)