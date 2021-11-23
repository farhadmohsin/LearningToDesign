import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from satisfaction_calc import maximin_winner, Copeland_winner

np.set_printoptions(precision=4)

"""
Generate random preference profiles
    All results can focus on two groups for now
    Data generation schemes
        1. Candidates and Voters on metric space
            Voters divided into two clusters, candidates random across the space
        2. Single peaked preferences
            Again, voters divided into clusters
        3. Completely random preferences
        4. Some sort of adversarial preferences? That might give us extreme values?
            Think some more
    
Calculate unfairness and SW for each candidate (under different utility functions?)
    For OPT SW, find maximum unfairness compared to least unfairness
    For 0.9*OPT SW, repeat
    For 0.8*OPT SW, repeat...
"""

def create_voters_candidates(n1, n2, m, d=2):
    """
    Description:
        Generate m candidates and random voters with two groups
    Parameters:
        n1:  voters in group 1
        n2:  voters in group 2
        m:   number of alternatives
        d:   dimension of the metric space
    """
    candidates = np.random.uniform(size = [m, d]) #get m candidates
    
    cluster_diff = 0
    cluster_diff_thresh = 0.4
    while (cluster_diff < cluster_diff_thresh): 
        #ensuring that the groups are a certain distance apart
        clusters = np.random.uniform(size = [2,d]) # get center for groups
        cluster_diff = np.linalg.norm(clusters[0] - clusters[1])
    #print(clusters)
    g1 = np.random.multivariate_normal(clusters[0], np.eye(d)*0.02, size = n1)
    g2 = np.random.multivariate_normal(clusters[1], np.eye(d)*0.02, size = n2)
    
    return g1, g2, candidates

def uniform_voters_candidates(n1, n2, m, d=2):
    """
    Description:
        Generate m candidates and random voters with two groups
        Voters in a group aren't clustered together
    Parameters:
        n1:  voters in group 1
        n2:  voters in group 2
        m:   number of alternatives
        d:   dimension of the metric space
    """
    candidates = np.random.uniform(size = [m, d]) #get m candidates

    g1 = np.random.uniform(size = [n1, d])
    g2 = np.random.uniform(size = [n2, d])
    
    return g1, g2, candidates

def close_voters_candidates(n1, n2, m, d = 2):
    """
    Description:
        Generate m candidates and random voters with two groups
        Groups are close
    """
    candidates = np.random.uniform(size = [m, d]) #get m candidates
    
    cluster_diff = np.inf
    cluster_diff_thresh = 0.2
    while (cluster_diff > cluster_diff_thresh): 
        #ensuring that the groups are a certain distance apart
        clusters = np.random.uniform(size = [2,d]) # get center for groups
        cluster_diff = np.linalg.norm(clusters[0] - clusters[1])
    #print(clusters)
    g1 = np.random.multivariate_normal(clusters[0], np.eye(d)*0.01, size = n1)
    g2 = np.random.multivariate_normal(clusters[1], np.eye(d)*0.01, size = n2)
    
    return g1, g2, candidates

def simple_voters_candidates(n1,n2,m,d=2):
    """
    Description:
        Generate m candidates and random voters with two groups
        All voters in a group has the same value
    """
    candidates = np.random.uniform(size = [m, d]) #get m candidates
    
    cluster_diff = 0
    cluster_diff_thresh = 0.4
    while (cluster_diff < cluster_diff_thresh): 
        #ensuring that the groups are a certain distance apart
        clusters = np.random.uniform(size = [2,d]) # get center for groups
        cluster_diff = np.linalg.norm(clusters[0] - clusters[1])
    #print(clusters)
#    g1 = np.random.multivariate_normal(clusters[0], np.eye(d)*0.01, size = n1)
    g1 = np.repeat([np.random.multivariate_normal(clusters[0], np.eye(d)*0.01)], n1, axis = 0)
#    g2 = np.repeat([np.random.multivariate_normal(clusters[1], np.eye(d)*0.01)], n2, axis = 0)
    g2 = np.random.multivariate_normal(clusters[1], np.eye(d)*0.01, size = n2)
    return g1, g2, candidates

def random_pref_profile(g1,g2,candidates):
    
    ballot_g1 = []
    for agent in g1:
        dist = np.linalg.norm(agent - candidates, axis = 1)
        ballot_g1.append(np.argsort(dist))
    
    ballot_g2 = []
    for agent in g2:
        dist = np.linalg.norm(agent - candidates, axis = 1)
        ballot_g2.append(np.argsort(dist))
    
    return np.array(ballot_g1), np.array(ballot_g2)

def plot_all(g1, g2, candidates):
    '''
    plots all candidates and voters when d=2 in create_voters_candidates
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(candidates[:,0], candidates[:,1], c = 'r')  
    plt.scatter(g1[:,0], g1[:,1], c = 'g') 
    plt.scatter(g2[:,0], g2[:,1], c = 'b')   
    for i, cnd in enumerate(candidates):
        ax.annotate(i, (cnd[0], cnd[1]))
    plt.show()

def calculate_unfairness(util1, util2, util):
    """
    Description:
        Calculate SW and unfairness for all candidates
    Parameters:
        util1:  utilities for group1
        util2:  utilities for group2
        util:   utilities for whole population
    """
    m = len(util)
    uf = np.zeros(m)
    for j in range(m):
        if(util[j]==0):
            uf[j] = np.nan
        else:
            uf[j] = np.abs(util1[j] - util2[j]) / util[j]
    return np.array(uf)

def kapprov_utility(ballots):
    """
    Description:
        Return k-approval utility for each candidate
    
    k should be a paramter, but for ease, we do it here
    """
    n, m = ballots.shape
    k = np.floor(m/2)+1
    utilities = np.zeros(m)
    for vote in ballots:
        for j in range(int(k)):
            utilities[vote[j]] += 1 
    return utilities/n
    
def borda_utility(ballots):
    """
    Description:
        Return Borda utility for each candidate
    """
    n, m = ballots.shape
    utilities = np.zeros(m)
    for vote in ballots:
        for j in range(m):
            utilities[vote[j]] += m-j-1 
    return utilities/n

def kborda_utility(ballots):
    """
    Description:
        Return k-Borda utility for each candidate
    """
    n, m = ballots.shape
    utilities = np.zeros(m)
    for vote in ballots:
        for j in range(m):
            utilities[vote[j]] += m-j-1 
    return utilities/n

def plurality_utility(ballots):
    """
    Description:
        Return plurality utility for each candidate
        
        k should be a paramter, but for ease, we do it here
    """
    n, m = ballots.shape
    utilities = np.zeros(m)
    for vote in ballots:
        utilities[int(vote[0])] += 1
    return utilities/n

#%%
def gen_group_profiles(n1,n2,m):
    g1, g2, candidates = create_voters_candidates(n1,n2,m)
    #plot_all(g1, g2,candidates) #plotting for 2d space only
    ballots1, ballots2 = random_pref_profile(g1, g2, candidates)
    return np.concatenate((ballots1, ballots2))
#%% SW-Fairness trade-offs

def main():
    trials = 1000
    
    n1 = 100
    n2 = 30
    m = 4
    
    rng2 = np.arange(0, (n1+n2)/n2, 0.05)
    sw_mins = np.zeros(len(rng2))
    
    rng = np.arange(0,1.01,0.05)
    
    primary = plurality_utility
    secondary = borda_utility
    
    uf_mins = np.zeros(len(rng)) 
    
    opt_uf_sim = 0
    
    opt_uf_max = 0
    
    util_fair_min = np.inf
    
    primary_uf_max = 0
    sec_uf_max = 0
    
    for t in range(trials):
        
        g1, g2, candidates = simple_voters_candidates(n1,n2,m)
        #plot_all(g1, g2,candidates) #plotting for 2d space only
        ballots1, ballots2 = random_pref_profile(g1, g2, candidates)   
        
        utilg1 = primary(ballots1)
        utilg2 = primary(ballots2)
        util = primary(np.concatenate((ballots1,ballots2)))
        
    #    print(utilg2[np.argsort(utilg2)[-4:]])
    #    print(np.sum(utilg2[np.argsort(utilg2)[-4:]]))
        
        plu_util = secondary(np.concatenate((ballots1,ballots2)))
        
        uf = calculate_unfairness(utilg1, utilg2, util)
        
        opt_uf = np.nanmin(uf)
        
        util_fair = util[np.nanargmin(uf)]
        if(util_fair < util_fair_min):
            util_fair_min = util_fair
            util_min1 = utilg1[np.nanargmin(uf)]
            util_min2 = utilg2[np.nanargmin(uf)]
        
        if(opt_uf > opt_uf_max):
            opt_uf_max = opt_uf
            opt_util1 = utilg1
            opt_util2 = utilg2
            opt_uf_all = uf
        
        winner_uf_ub = 2*(m-1)/(n1/(n1+n2)*m)
        winner_uf = uf[np.argmax(util)]
        sec_winner_uf = uf[np.argmax(plu_util)]
        
        if(winner_uf > primary_uf_max):
            primary_uf_max = winner_uf
            
        if(sec_winner_uf > sec_uf_max):
            sec_uf_max = sec_winner_uf    
        
        for cnt,r in enumerate(rng):
            sw_lim = np.max(util) * r
            uf_comp = np.inf
            for i,sw in enumerate(util):
                if(sw >= sw_lim and uf[i] < uf_comp):
                    uf_comp = uf[i]
            
            if(uf_comp > uf_mins[cnt]):
                uf_mins[cnt] = uf_comp
        
    #    for cnt,uf_lim in enumerate(rng2):
    #        sw_comp = 0
    #        for i,sw in enumerate(util):
    #            if(sw >= sw_comp and uf[i] < uf_lim):
    #                sw_comp = sw
    #                print(np.array([uf_lim, sw_comp, np.max(util)]))
    ##        print("r = %f, sw_lim = %.2f, uf = %.2f"%(r,sw_lim,uf_min))
    #        
    #        if(sw_comp/np.max(util) > sw_mins[cnt]):
    #            sw_mins[cnt] = sw_comp/np.max(util)
    #%%            
    #    print(np.array([opt_uf_max, opt_uf_ub]))
    print(opt_uf_all)
    print(np.nanargmin(opt_uf_all))
    print(opt_util1)
    print(np.argsort(opt_util1))
    print(opt_util2)
    print(np.argsort(opt_util2))
    #    
    #    print(util_fair_min * (n1+n2), util_min1*n1, util_min2*n2)
    
    #k = np.floor(m/2)+1
    #z = n1/(n1+n2)
    #y = (2*k-m)/k
    #print("k-approval bound: %.3f"%((1-y)/(z+(1-z)*y)))
    
    print(np.array([n1,n2,opt_uf_max,primary_uf_max, sec_uf_max]))
    print(np.array([(n1+n2)/n1, (n1+n2)/n2]))
    
    z = n1/(n1+n2)
    print(np.array([1/(1+z),2/(3-2*z)]))
    #%%
#    fig = plt.figure(figsize = [10, 8])
#    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
#    
#    ax = fig.add_subplot(111)
#    fig.subplots_adjust(top=0.85)
#    ax.set_title('welfare - fairness tradeoff')
#    
#    ax.set_xlabel('SW ratio')
#    ax.set_ylabel('fairness')
#    
#    ax.plot(rng, -uf_mins)
#    #ax.axis([0, 10, 0, 10])
#    
#    plt.show()

if __name__ == "__main__":
    # main()
    trials = 1000
    
    n1 = 100
    n2 = 30
    m = 4
    
    primary = borda_utility
    secondary_voting = Copeland_winner
    primary_uf_max = 0
    sec_uf_max = 0 
    
    for tt in range(100):
        for t in range(trials):
            
            g1, g2, candidates = simple_voters_candidates(n1,n2,m)
            #plot_all(g1, g2,candidates) #plotting for 2d space only
            ballots1, ballots2 = random_pref_profile(g1, g2, candidates)   
            
            utilg1 = primary(ballots1)
            utilg2 = primary(ballots2)
            util = primary(np.concatenate((ballots1,ballots2)))
            
            _, sec_util = secondary_voting(np.concatenate((ballots1,ballots2)))
        #    print(utilg2[np.argsort(utilg2)[-4:]])
        #    print(np.sum(utilg2[np.argsort(utilg2)[-4:]]))
            
            uf = calculate_unfairness(utilg1, utilg2, util)
            
            
            winner_uf = uf[np.argmax(util)]
            sec_winner_uf = uf[np.argmax(sec_util)]
            
            if(winner_uf > primary_uf_max):
                primary_uf_max = winner_uf
                
            if(sec_winner_uf > sec_uf_max):
                sec_uf_max = sec_winner_uf    
            
        print(f"borda: {primary_uf_max}, Copeland: {sec_uf_max}")
