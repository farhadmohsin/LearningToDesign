import numpy as np
from matplotlib import pyplot as plt

uf_opt_max = 0

m = 8
z = 1/1.3
trials = 2000000

for t in range(trials):
#    u1_arr = np.random.choice(m,m,replace=False)
    u1_arr = np.random.random(size = m)
    u1_arr /= np.sum(u1_arr)
    u1_arr *= m*(m-1)/2
    
    u2_arr = np.random.random(size = m)
    u2_arr /= np.sum(u2_arr)
    u2_arr *= m*(m-1)/2
    
    uf = [((u1_arr[i]) - u2)/((u1_arr[i])*z + u2*(1-z)) for i,u2 in enumerate(u2_arr)]
    uf = np.abs(uf)
    uf_opt = np.min(uf)
    if(uf_opt > uf_opt_max):
        uf_opt_max = uf_opt
        print(uf_opt)
        print(uf)
        print(u1_arr)
        print(u2_arr)
        
#%%
u1 = m-1
u2_arr = np.arange(0,m-1+0.01,0.02)

uf = [((m-1) - u2)/((m-1)*z + u2*(1-z)) for u2 in u2_arr]

plt.plot(u2_arr,uf)

#%%
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

def plurality_utility(ballots):
    """
    Description:
        Return plurality utility for each candidate
        
        k should be a paramter, but for ease, we do it here
    """
    n, m = ballots.shape
    utilities = np.zeros(m)
    for vote in ballots:
        utilities[vote[0]] += 1
    return utilities/n

#%%
z_all = np.arange(0.5,1.01,0.05)

for m in range(3,9):
    for z in z_all:
        print("%d %.2lf"%(m,z))
        print("bound 1: %.2lf"%(m/(2*(1-z)*(m-1) + z*(m-2))))
        print("bound 2: %.2lf"%(2/(3-2*z)))
        print("borda: %.2f, pl: %.2f"%(1/z, 1/(1-z)))


