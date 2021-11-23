import numpy as np
from satisfaction_calc import *
from fairness_new_tradeoffs import *
from time import time

np.set_printoptions(precision=4)

def Consistency_fair(m=6, profiles=200, threshold = 0.8):
    eff = []
    for test in range(10):
        #Now we have to store the profiles as well
        cnt = 0
        cnt_v = 0
        votes_all = []
        n_all = []
             
        tic = time()
        for t in range(profiles):
            n1 = 200
            n2 = np.random.randint(4,20)*10
            n_all.append([n1,n2])
            votes = gen_pref_profile(n1+n2, m)
            votes_all.append(votes)
        toc = time()
        print("Profiles generated in %lf s"%(toc-tic))
        
        # calculate all winners given some voting rule and tiebreaking scheming
        winner_sets = [[] for i in range(m)]
        tic = time()
        cnt_borda = 0
        for t in range(profiles):
            winner, _ = fair_borda(votes_all[t],n_all[t][0],n_all[t][1],threshold)
            winner_sets[winner].append(t)
            utils = borda_utility(votes_all[t])
            if(winner != np.argmax(utils)):
                cnt_borda += 1
        toc = time()
        print("Winners computed in %lf s"%(toc-tic))
        print(f"Borda mismatch found: {cnt_borda}")
        print([len(winner_sets[ww]) for ww in range(m)])
        
        # cnt_borda = 0
        for w,wset in enumerate(winner_sets):
            len_set = len(wset)
            for i1 in range(len_set):
                for i2 in range(i1+1,len_set):
                    tic = time()
                    # joined = np.append(votes_all[wset[i1]],votes_all[wset[i2]],axis = 0)
                    n11 = n_all[wset[i1]][0]
                    n12 = n_all[wset[i1]][1]
                    n21 = n_all[wset[i2]][0]
                    n22 = n_all[wset[i2]][1]
                    joined1 = np.append(votes_all[wset[i1]][:n11],votes_all[wset[i2]][:n21],axis = 0)
                    joined2 = np.append(votes_all[wset[i2]][n11:n11+n12],votes_all[wset[i2]][n21:n21+n22],axis = 0)
                    N1 = n11+n12
                    N2 = n21+n22
                    
                    joined = np.append(joined1, joined2, axis = 0)
                    # print(N1, N2, len(joined))
                    w_new, _ = fair_borda(joined,N1,N2,threshold)
                    if(w_new == w):
                        cnt_v += 1
                    # else:
                    #     print(n_all[wset[i1])
                    #     print(n_all[wset[i2])
                        
                    #     util, uf = util_uf(votes_all[wset[i1]])
                    #     break
                    
                    cnt += 1
                    toc = time()
                    # print("Alternative %d. Comparison %d takes %lf s, cnt_v = %d"%(w, cnt, toc-tic, cnt_v))
        print("profiles = %d, Comparisons = %d, Consistent = %d"%(profiles, cnt, cnt_v))
        eff.append([cnt_v,cnt])
    return np.array(eff)

if __name__ == '__main__':
    for threshold in np.arange(0.96,1,0.01):
        print(f"\tstart of threshold = {threshold}")
        E = Consistency_fair(m=4, profiles = 500, threshold = threshold)
        # print(np.sum(E,axis = 1))
    # for t in range(100):
    #     n1 = 200
    #     n2 = np.random.randint(4,20)*10
    #     votes = gen_pref_profile(n1+n2, 4)
    #     w, uf = fair_borda(votes, n1, n2, 1)        
    #     utils = borda_utility(votes)
        
    #     if(w != np.argmax(utils)):
    #         print(w)
    #         print(utils)
    #         break