from autoVoting import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

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
def train_test_split(PWP):
    """
    Description:
        Split "true data" into train and test sets
        Generate synthetic data from train data using axioms to create complete train data
    """
    dataset_size = len(PWP)
    train_size = round(dataset_size * 0.7)
    train_idx = slice(0,train_size)
    test_idx = slice(train_size,dataset_size)
    
    # generate new data based on axioms
    # let K be no of new (pref profile, winner) pairs to be created
    
    PWP_train = PWP[train_idx]
    K = 500
    
    #def neutral_sample_gen(prof_win_pair):
    #def consistent_sample_gen(pwp1, pwp2):
    #def monotonic_sample_gen(pwp):
    
    for i in range(round(K/3)):
        #neutral_sample_generation
        rand_idx = np.random.randint(len(PWP_train))
        new_pwp = neutral_sample_gen(PWP_train[rand_idx])
        PWP_train.append(new_pwp)
        
        #consistent_sample_generation
        while(1):
            rand_idx = np.random.randint(len(PWP_train), size = 2)
            if(PWP_train[rand_idx[0]].winner == PWP_train[rand_idx[1]].winner):
                break
        new_pwp = consistent_sample_gen(PWP_train[rand_idx[0]],PWP_train[rand_idx[1]])
        PWP_train.append(new_pwp)
        
        #monotonic_sample_gen
        #only possible when pref profile available, so from true data
        rand_idx = np.random.randint(train_size)
        new_pwp = monotonic_sample_gen(PWP_train[rand_idx])
        PWP_train.append(new_pwp)
    
    X_train = [np.concatenate((pwp.position_vector.flatten(), \
                               pwp.WMG.flatten()), axis = 0) for pwp in PWP_train]
    Y_train = [pwp.winner for pwp in PWP_train]
    
    X_test = [np.concatenate((pwp.position_vector.flatten(), \
                               pwp.WMG.flatten()), axis = 0) for pwp in PWP[test_idx]]
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
# for the "real" data, assume N = 1000, m = 6
N = 1000
m = 8

# T is the the number of "real" elections
T = 50
PWP = create_Dataset(N, m, T)

X_train, Y_train, X_test, Y_test = train_test_split(PWP)

# Fit model to logistic regression

# using scikitlearn's LogisticRegression function
logreg = LogisticRegression(C = 10)
logreg.fit(X_train, Y_train)

print(accuracy_test(X_test, Y_test, logreg))


# S is the number of new preference profile to test
S = 100
print(accuracy_Borda(logreg, S))