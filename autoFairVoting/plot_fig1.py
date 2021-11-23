from matplotlib import pyplot as plt
import numpy as np

methods = ["Random","PL","Gauss"]
names = ["uniform","Plackett-Luce","Gaussian"]

#%% Copeland Data
C = []

C.append("""UF 0.038776701231440225
UW [0.104  0.1046]
SWF 211.11135000000004
SWW [225.7273 224.8317]
U [225.7273 194.3018]
EFF [0.9107 1.    ]""")

C.append("""UF 0.17034063254455023
UW [0.3655 0.3811]
SWF 239.87593
SWW [294.6964 294.2868]
U [294.6964 125.1153]
EFF [0.9629 1.    ]""")

C.append("""UF 0.2745352321470079
UW [0.5493 0.6119]
SWF 281.16382
SWW [328.794  327.0677]
U [328.794   78.4729]
EFF [0.9181 1.    ]""")
  
Cxyy = [[0.1046,224.8317,1],[0.3811,294.2868,1],[0.6119,327.0677,1]]


#%%

name_dict = dict(zip(methods,names))
n1 = 100
n2 = 40
m = 4

alpha = np.arange(0,1.01,0.1)
beta = np.arange(0,1.01,0.1)
alpha2 = np.arange(0.82,0.99,0.02)

alpha_start = [8,3,3]

fig, ax = plt.subplots(1, 1, figsize=(8,5))

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
    
    if(not(idx==1)):
        continue
    
    with open(f"data/{n1}-{n2}-{m}-{method}-fairness.npy", 'rb') as f:
        UW = np.load(f)
        EFF = np.load(f) #Condorcet
        SWW = np.load(f) #SW
        
    with open(f"data/{n1}-{n2}-{m}-{method}-fairness-2.npy", 'rb') as f:
        UW2 = np.load(f)
        EFF2 = np.load(f) #Condorcet
        SWW2 = np.load(f) #SW

    yC = Cxyy[idx][1] # for SW
    # yC = Cxyy[idx][2] # for Cond
    xC = Cxyy[idx][0]     
        
    x2 = np.concatenate((UW,np.mean(UW2,axis=0),[xC]))
    # y2 = np.concatenate((EFF,np.mean(EFF2,axis=0),[yC]))
    y2 = np.concatenate((SWW,np.mean(SWW2,axis=0),[yC]))/(n1+n2)
    
    # va = "top"
    # ha = "right"
    va = "bottom"
    ha = "left"
    
    ax.scatter(1-(n2/(n1+n2))*x2,y2, label = r'$\alpha-FB$')
       
    #annotate Borda
    # ax.annotate("Borda", (1-(n2/(n1+n2))*x2[len(alpha)-1], y2[len(alpha)-1]), 
    #                      verticalalignment="top",horizontalalignment=ha)
    ax.annotate('Borda',
            xy=(1-(n2/(n1+n2))*x2[len(alpha)-1], y2[len(alpha)-1]), xycoords='data',
            xytext=(-70, 30), textcoords='offset pixels',
            arrowprops=dict(arrowstyle="->"))
    #annotate maxfair
    ax.annotate("MaxFair", (1-(n2/(n1+n2))*x2[0], y2[0]), 
                     verticalalignment=va,horizontalalignment=ha)
    #annotate Copeland
    # ax.annotate("Copeland", (1-(n2/(n1+n2))*x2[-1], y2[-1]), 
    #                  verticalalignment="top",horizontalalignment=ha)
    
    ax.annotate('Copeland',
            xy=(1-(n2/(n1+n2))*x2[-1], y2[-1]), xycoords='data',
            xytext=(-50, 0), textcoords='offset pixels',
            arrowprops=dict(arrowstyle="->"))
    
    print(f'\t{(1-(n2/(n1+n2))*x2[-1])}, {y2[-1]}')
    #annotate FB (0-1)
    for i in range(alpha_start[idx],len(alpha)-1):
        print(alpha[i])
        ax.annotate("%.1f-FB"%(alpha[i]), (1-(n2/(n1+n2))*x2[i], y2[i]), 
                         verticalalignment=va,horizontalalignment=ha)
    #annotate FB (0.82-0.98)
    for i in range(1,len(alpha2),4):
        # print(alpha2[i])
        ax.annotate("%.2f-FB"%(alpha2[i]), (1-(n2/(n1+n2))*x2[i+len(UW)], y2[i+len(UW)]), 
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
    
    ax.scatter(1-(n2/(n1+n2))*x1,y1, label = r'$\beta-ML$')
    for i in range(len(beta)):
        if(i==9):
            continue
        ax.annotate("%.1f-ML"%(beta[i]), (1-(n2/(n1+n2))*x1[i], y1[i]), 
                         verticalalignment=va,horizontalalignment=ha)
    
    ax.set_xlabel('Mean Fairness')
    ax.set_ylabel('Average Rank Utility')
    
    # ax.set_ylabel('Average Rank Utility')
    
    ax.set_title(f"n1={n1}, n2={n2}, m={m}, data={name_dict[method]}")
    
    # plt.savefig(f'imgs/{n1}-{n2}-{m}-{method}-fairness-ML.pdf')
    plt.legend()
       
    plt.show()
    