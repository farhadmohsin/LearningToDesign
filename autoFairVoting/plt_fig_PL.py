from matplotlib import pyplot as plt
import numpy as np

methods = ["PL"]
names = ["Plackett-Luce"]

#%%

name_dict = dict(zip(methods,names))
n1 = 100
n2 = 40
m = 4

alpha = np.arange(0,1.01,0.1)
beta = np.arange(0,1.01,0.1)


fig, ax = plt.subplots(figsize = (16,8))

MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#%%
for idx,method in enumerate(methods):
    print(method)
        
    with open(f"data/{n1}-{n2}-{m}-{method}-fairness.npy", 'rb') as f:
        UW = np.load(f)
        EFF = np.load(f) #Condorcet
        SWW = np.load(f) #SW
    
        
    x2 = UW
    # y2 = EFF
    y2 = SWW/(n1+n2)
    
    # va = "top"
    # ha = "right"
    va = "bottom"
    ha = "left"
    
    ax.scatter(1-(n2/(n1+n2))*x2,y2, label = r'$\alpha-FB$', s = 60*np.ones(len(x2)))
       
    #annotate Borda
    ax.annotate("Borda", (1-(n2/(n1+n2))*x2[-1], y2[-1]), xycoords='data',
            xytext=(50, 10), textcoords='offset pixels',
            arrowprops=dict(arrowstyle="->"))
    # for CE
    # ax.annotate('Borda',
    #         xy=(1-(n2/(n1+n2))*x2[len(alpha)-1], y2[len(alpha)-1]), xycoords='data',
    #         xytext=(-100, -50), textcoords='offset pixels',
    #         arrowprops=dict(arrowstyle="->"))
    #annotate maxfair
    ax.annotate("MaxFair", (1-(n2/(n1+n2))*x2[0], y2[0]), xycoords='data',
            xytext=(5, 25), textcoords='offset pixels',
            arrowprops=dict(arrowstyle="->"))
    # for CE
    # ax.annotate("MaxFair", (1-(n2/(n1+n2))*x2[0], y2[0]), xycoords='data',
    #         xytext=(-200, -30), textcoords='offset pixels',
    #         arrowprops=dict(arrowstyle="->"))
    #annotate Copeland
    # ax.annotate("Copeland", (1-(n2/(n1+n2))*x2[-1], y2[-1]), 
    #                  verticalalignment="top",horizontalalignment=ha)

    
    # print(f'\t{(1-(n2/(n1+n2))*x2[-1])}, {y2[-1]}')
    #annotate FB (0-1)

    #TODO: Uncomment this later
    for i in range(4,len(alpha)-1):
        # print(alpha[i])
        ax.annotate("%.1f-FB"%(alpha[i]), (1-(n2/(n1+n2))*x2[i], y2[i]), 
                          verticalalignment=va,horizontalalignment=ha)
    
    
    # ax.set_xlabel('Mean Fairness')
    # ax.set_ylabel('Condorcet Efficiency')
    
    # ax.set_title(f"n1={n1}, n2={n2}, m={m}, data={method}")
    
    with open(f"data/{n1}-{n2}-{m}-{method}-fairness-ML.npy", 'rb') as f:
        uf = np.load(f)
        eff = np.load(f) #Condorcet
        sw = np.load(f) #Social welfare
    
    x1 = uf
    y1 = sw
    # y1 = eff
    
    # fig, ax = plt.subplots()
    # va = "bottom"
    # ha = "left"
    va = "top"
    ha = "right"  
    
    ax.scatter(1-(n2/(n1+n2))*x1,y1, label = r'$\beta-ML$', s = 60*np.ones(len(x1)), marker='^')
    
    #TODO: Uncomment this later
    for i in range(1,len(beta)):
        if(i>=8):
            continue
        ax.annotate("%.1f-ML"%(beta[i]), (1-(n2/(n1+n2))*x1[i], y1[i]), 
                          verticalalignment=va,horizontalalignment=ha)
    ax.annotate("%.1f-ML"%(beta[10]), (1-(n2/(n1+n2))*x1[10], y1[10]), xycoords='data',
            xytext=(-5, -100), textcoords='offset pixels',
            arrowprops=dict(arrowstyle="->"))
    
    # plot Borda and maxfair
    ax.scatter(1-(n2/(n1+n2))*x2[[0,-1]], y2[[0,-1]], color = 'green', s = 90*np.ones(2))
    
    # ax.set_ylim(0.3,1.08)
    
    ax.set_xlabel('Mean Fairness')
    ax.set_ylabel('Average Rank Utility')
    # ax.set_ylabel('Condorcet Efficiency')
    
    # ax.set_ylabel('Average Rank Utility')
    
    ax.set_title(f"n1={n1}, n2={n2}, m={m}, data={name_dict[method]}")
    
    # plt.savefig(f'imgs/{n1}-{n2}-{m}-{method}-fairness-ML.pdf')
    plt.legend()
    ax.legend(fontsize = 20)
    plt.grid(linestyle = ':')
    plt.show()
    
# plt.savefig(f"n1={n1}, n2={n2}, m={m}-Cond.pdf", format="pdf")
