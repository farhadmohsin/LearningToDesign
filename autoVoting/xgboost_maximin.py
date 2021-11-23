import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.datasets import load_iris, load_digits, load_boston

from satisfaction_calc import gen_pref_profile, maximin_winner, singleton_lex_tiebreaking
from autoVoting import weighted_majority_graph, position_vector
#%%

N = 100
m = 4
profiles = 50000

# voting rule: maximin_winner
# tiebreaking rule: singleton_lex_tiebreaking

X = []
W = []

for t in range(profiles):
    votes = gen_pref_profile(N, m)
    WMG = weighted_majority_graph(votes)
    posvec = position_vector(votes)
    x = np.concatenate((WMG.flatten(), posvec.flatten()))
    
    X.append(x)
    
    winners, scores = maximin_winner(votes)
    w = singleton_lex_tiebreaking(votes, winners)
    W.append(w)
    
X = np.array(X)
W = np.array(W)

rng = np.random.RandomState(31337)
kf = KFold(n_splits=5, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    # param_dist = {'gamma': 0.2, 'max_depth': 6, 'min_child_weight': 7}
    # clf = xgb.XGBClassifier(**param_dist)
    clf = xgb.XGBClassifier()
    xgb_model = clf.fit(X[train_index], W[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = W[test_index]
    print(confusion_matrix(actuals, predictions))
    print(accuracy_score(actuals, predictions))
    
#%%

# xgb_model = xgb.XGBRegressor()
# clf = GridSearchCV(xgb_model,
#                    {"max_depth"        : [ 2, 4, 6, 8],
#                     "min_child_weight" : [ 1, 3, 5, 7 ],
#                     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]}, 
#                    verbose=0)
# clf.fit(X,W)
# print(clf.best_score_)
# print(clf.best_params_)
