from matplotlib import pyplot as plt
import numpy as np

methods = ["Random","PL","Gauss"]
names = ["uniform","Plackett-Luce","Gaussian"]

#%%

name_dict = dict(zip(methods,names))
n1 = 100
n2 = 40
m = 4

alpha = np.arange(0,1.01,0.1)
beta = np.arange(0,1.01,0.1)

# for n2 in range(20,101,20):
for n2 in [40]:
    fig, ax = plt.subplots(1, 3, figsize=(19,5))
    
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    for idx,method in enumerate(methods):
        print(method)
        
        with open(f"data/{n1}-{n2}-{m}-{method}-fairness.npy", 'rb') as f:
            UW = np.load(f)
            EFF = np.load(f) #Condorcet
            SWW = np.load(f) #SW
        
            
        x2 = UW
        y2 = EFF
        # y2 = SWW/(n1+n2)
        
        va = "top"
        ha = "right"
        # va = "bottom"
        # ha = "left"
        
        ax[idx].scatter(1-(n2/(n1+n2))*x2,y2, label = r'$\alpha-FB$')
           
        #annotate Borda
        # ax.annotate("Borda", (1-(n2/(n1+n2))*x2[len(alpha)-1], y2[len(alpha)-1]), 
        #                      verticalalignment="top",horizontalalignment=ha)
        ax[idx].annotate('Borda',
                xy=(1-(n2/(n1+n2))*x2[len(alpha)-1], y2[len(alpha)-1]), xycoords='data',
                xytext=(-70, 30), textcoords='offset pixels',
                arrowprops=dict(arrowstyle="->"))
        #annotate maxfair
        ax[idx].annotate("MaxFair", (1-(n2/(n1+n2))*x2[0], y2[0]), 
                         verticalalignment=va,horizontalalignment=ha)
        #annotate Copeland
        # ax.annotate("Copeland", (1-(n2/(n1+n2))*x2[-1], y2[-1]), 
        #                  verticalalignment="top",horizontalalignment=ha)
    
        
        # print(f'\t{(1-(n2/(n1+n2))*x2[-1])}, {y2[-1]}')
        #annotate FB (0-1)
        
        #TODO: Uncomment this later
        # for i in range(0,len(alpha)-1):
        #     # print(alpha[i])
        #     ax[idx].annotate("%.1f-FB"%(alpha[i]), (1-(n2/(n1+n2))*x2[i], y2[i]), 
        #                      verticalalignment=va,horizontalalignment=ha)
        
        # ax.set_xlabel('Mean Fairness')
        # ax.set_ylabel('Condorcet Efficiency')
        
        # ax.set_title(f"n1={n1}, n2={n2}, m={m}, data={method}")
        
        with open(f"data/{n1}-{n2}-{m}-{method}-fairness-ML.npy", 'rb') as f:
            uf = np.load(f)
            eff = np.load(f) #Condorcet
            sw = np.load(f) #Social welfare
        
        x1 = uf
        # y1 = sw
        y1 = eff
        
        # fig, ax = plt.subplots()
        va = "bottom"
        ha = "left"
        # va = "top"
        # ha = "right"  
        
        ax[idx].scatter(1-(n2/(n1+n2))*x1,y1, label = r'$\beta-ML$')
        
        #TODO: Uncomment this later
        # for i in range(len(beta)):
        #     if(i==9):
        #         continue
        #     ax[idx].annotate("%.1f-ML"%(beta[i]), (1-(n2/(n1+n2))*x1[i], y1[i]), 
        #                      verticalalignment=va,horizontalalignment=ha)
        
        ax[idx].set_xlabel('Mean Fairness')
        # ax[idx].set_ylabel('Average Rank Utility')
        ax[idx].set_ylabel('Condorcet Efficiency')
        
        # ax.set_ylabel('Average Rank Utility')
        
        ax[idx].set_title(f"n1={n1}, n2={n2}, m={m}, data={name_dict[method]}")
        
        # plt.savefig(f'imgs/{n1}-{n2}-{m}-{method}-fairness-ML.pdf')
        plt.legend()
           
        plt.show()
        
    # plt.savefig(f"n1={n1}, n2={n2}, m={m}-Cond.pdf", format="pdf")
