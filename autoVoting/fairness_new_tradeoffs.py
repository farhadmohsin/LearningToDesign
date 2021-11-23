import numpy as np
from satisfaction_calc import *
from fairness_SW_tradeoff import gen_group_profiles
from time import time
np.set_printoptions(precision=6)

#Functions needed
#Condorcet_efficiency - if Condorcet winner exists, do we choose them
#Fairness_efficiency - count of positive cases? or expectation of fairness?
#Consistency - how many pairs are consistent

#Voting rules
#Borda
#maximin
#alpha-fairBorda (min UF while SW >= alpha*maxSW)

#vary over n1/n2 and alpha

def borda_utility(votes):
    """
    Description:
        Return Borda utility for each candidate
    """
    n, m = votes.shape
    utilities = np.zeros(m)
    for vote in votes:
        for j in range(m):
            utilities[vote[j]] += m-j-1 
    return utilities/n

def util_uf(votes1, votes2):
    util1 = borda_utility(votes1)
    util2 = borda_utility(votes2)
    util = borda_utility(np.concatenate((votes1,votes2)))
    
    m = len(util)
    uf = np.zeros(m)
    for j in range(m):
        if(util[j]==0):
            uf[j] = np.nan
        else:
            uf[j] = np.abs(util1[j] - util2[j]) / util[j]
    return util, np.array(uf)

def fair_borda(votes, n1, n2, threshold):
    #minimize unfairness for SW > threshold*maxSW
    util, uf = util_uf(votes[:n1], votes[n1:n1+n2])
    m = len(util)
    min_uf = np.inf
    w = m+1
    for j,u in enumerate(util):
        if(u < np.max(util)*threshold):
            continue
        if(np.isnan(uf[j])):
            continue
        if(uf[j] < min_uf):
            w = j
            min_uf = uf[j]
    
    # print("utilities: ", util)
    # print("unfairness: ", uf)
    # print(w)
    if(w>m-1):
        print(f"n1: {n1}, n2: {n2}, threshold: {threshold}, winner: {w}")
        print(util)
        print(uf)
    return w, uf

def fairness_efficiency(voting_rule, n1 = 200, n2 = 100, m=4, profiles=1000, threshold = 0.8):
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
            winner, _ = voting_rule(votes)
            w = singleton_lex_tiebreaking(votes, winner)
            w_fair, uf = fair_borda(votes, n1,n2, threshold)
        
            # print(f"{test}, {t}: n1={n1}, n2={n2}, w_fair={w_fair}, w={w}")
            fair_uf = uf[w_fair]
            w_uf = uf[w]
            if(w == w_fair):
                cnt += 1
            fair_ufmean += fair_uf
            w_ufmean += w_uf
            # print("profile %d"%t, w, w_fair, min_uf)
        
        unfairness_fair.append(fair_ufmean/profiles)
        unfairness_w.append(w_ufmean/profiles)
        eff.append(cnt/profiles)
        print(f"{test}: n1={n1}, n2={n2}, count={cnt}, total={profiles}")
        # print(voting_rule.__name__,"N = %d, m = %d"%(N,m))
        # print("profiles = %d, Condorcet_winner = %d, V_winner = %d"%(profiles, cnt, cnt_v))
        
        # eff.append(cnt_v/cnt)
    return eff, unfairness_fair, unfairness_w

def fairness_efficiency_all(n1 = 100, n2 = 30, m=4, profiles = 1000):
    print(f"n1={n1},n2={n2},m={m}")
    sw_w = []
    unfairness_w = []
    sw_fair = []
    unfairness_fair = []
    
    eff = []
    # voting_rules = [plurality_winner, Borda_winner, Copeland_winner, \
    #                 maximin_winner, STV_winner]
    
    alpha = np.arange(0,1.01,0.1)
    
    utils = []
    
    for test in range(10):
        tic = time()
        # print("test = %d"%test)
        cond_cnt = 0
        cnt = np.zeros(len(alpha))
        # cnt = np.zeros(len(voting_rules))
        fair_ufmean = 0
        fair_swmean = 0
        w_ufmean = np.zeros(len(alpha))
        w_swmean = np.zeros(len(alpha))
        
        # w_ufmean = np.zeros(len(voting_rules))
        # w_swmean = np.zeros(len(voting_rules))
        
        maxu = 0
        minu = 0
        
        for t in range(profiles):
            # votes = gen_pref_profile(n1+n2, m)
            votes = gen_group_profiles(n1,n2,m)
            
            exist, cond = condorcet_exist(votes)
            if(exist):
                cond_cnt += 1
            
            _, util = Borda_winner(votes)
            maxu += np.max(util)
            minu += np.min(util)
            
            w_fair, uf = fair_borda(votes, n1,n2, 0)
            # print(f'util: {util}')
            # print(f'uf: {uf}')
            
            # for r,rule in enumerate(voting_rules):
            #     winner, _ = rule(votes) # plurality
            #     w = singleton_lex_tiebreaking(votes, winner)
            #     w_ufmean[r] += uf[w]
            #     w_swmean[r] += util[w]
                
            for r,a in enumerate(alpha):
            # for r,rule in enumerate(voting_rules):
                w, _ = fair_borda(votes, n1,n2, a)
                # winners, scores = rule(votes)
                # w = singleton_lex_tiebreaking(votes, winners)
                # print(r, w)
                w_ufmean[r] += uf[w]
                w_swmean[r] += util[w]
                
                if(exist):
                    # print(w,cond)
                    if(w in cond):
                        cnt[r] += 1
            
            # print(f"{test}, {t}: n1={n1}, n2={n2}, w_fair={w_fair}, w={w}")
            
            fair_ufmean += uf[w_fair]
            fair_swmean += util[w_fair]
            # print("profile %d"%t, w, w_fair, min_uf)
        
        unfairness_fair.append(fair_ufmean/profiles)
        unfairness_w.append(w_ufmean/profiles)
        
        sw_fair.append(fair_swmean/profiles)
        sw_w.append(w_swmean/profiles)
        
        utils.append([maxu/profiles, minu/profiles])
        
        eff.append(cnt/cond_cnt)
        
        toc = time()
        print(f"time taken for each iteration: {toc-tic}")
        # print(f"{test}: n1={n1}, n2={n2}, count={cnt}, total={profiles}")
        # print(voting_rule.__name__,"N = %d, m = %d"%(N,m))
        # print("profiles = %d, Condorcet_winner = %d, V_winner = %d"%(profiles, cnt, cnt_v))
        
        # eff.append(cnt_v/cnt)
    return np.array(unfairness_fair), np.array(unfairness_w), np.array(sw_fair), \
        np.array(sw_w), np.array(utils), np.array(eff)
#%%
def alpha_fair_winner0(votes):
    n1 = 200
    n2 = 60
    alpha = 0.8
    w, uf = fair_borda(votes, n1, n2, alpha)
    return [w], uf

# n1 = 200
# n2 = 60
# m = 4
# Condorcet_efficiency(alpha_fair_winner, N=n1+n2, m=m, profiles = 250)

def main():
    # for thresh in [0.8]:
    #     print(thresh)
    #     for n2 in [60]:
    #         E, FU, WU = fairness_efficiency(maximin_winner, n2 = n2, threshold = thresh)
    #         print("maximin: n1 = 200, n2 = %d, fair_eff = %lf, fair_uf = %lf, w_uf = %lf"%(n2, np.mean(E), np.mean(FU), np.mean(WU)))    
            
    #         E, FU, WU = fairness_efficiency(Borda_winner, n2 = n2, threshold = thresh)
    #         print("Borda: n1 = 200, n2 = %d, fair_eff = %lf, fair_uf = %lf, w_uf = %lf"%(n2, np.mean(E), np.mean(FU), np.mean(WU)))    
    n1 = 100
    n2 = 30
    m = 4
    # for i in range(10):
    #     P = gen_pref_profile(n1+n2, m)
    #     fair_borda(P, n1, n2, 0.8)
    # E, UF, UW = fairness_efficiency(alpha_fair_winner0, n1=200, n2=60, profiles = 50)
    # print(E, UF, UW)
    
    UF, UW, SWF, SWW, U = fairness_efficiency_all()
    print(UF)
    print(UW)
    print(SWF)
    print(SWW)
    print(U)
    return UF, UW, SWF, SWW, U
    
if __name__ == "__main__":
    # UF, UW, SWF, SWW, U = main()
    
    # n2_all = [20,60,100,140]
    # n2_all = [60]
    
    # for n2 in n2_all:
    # print(f'n2 = {n2}')
    UF, UW, SWF, SWW, U, EFF = fairness_efficiency_all(n1 = 100, n2 = 30, m=4)
    print(np.mean(UF,axis=0))
    print(np.mean(UW,axis=0))
    print(np.mean(SWF,axis=0))
    print(np.mean(SWW,axis=0))
    print(np.mean(U,axis=0))
    print(np.mean(EFF, axis=0))