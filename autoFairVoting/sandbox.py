import numpy as np
from time import time
from matplotlib import pyplot as plt
import pandas as pd
import mplcursors
# #%%
# from fairness_new_tradeoffs import fairness_efficiency_all
# from fairness_SW_privacy import privacy_main
# from xgboost_fair import learning_main
# #%% Plot Figure 2 
# #(it will miss a few points, the fairness, efficiency for traditional voting rules)

# # Get output of learning mechanism
# beta, x1, y1 = learning_main()

# # Get output of alpha-fairBorda mechanism
# alpha, UF, UW, SWF, SWW, U, EFF = fairness_efficiency_all(n1 = 100, n2 = 30, m=4)

# x2 = UW
# y2 = EFF

# fig, ax = plt.subplots()

# ax.scatter(-1*np.array(x1),y1)
# for i, b in enumerate(beta):
#     ax.annotate(f"{b}-ML", (-x1[i], y1[i]), horizontalalignment='left')

# ax.scatter(-1*np.array(x2),y2)
# for i in range(len(alpha)):
#     ax.annotate(f"{alpha[i]}-FB", (-x2[2*i+1], y2[2*i+1]), horizontalalignment='right')

# ax.set_xlabel('Mean Fairness')
# ax.set_ylabel('Condorcet Efficiency')

# plt.show()

#%% Plot Figure 3 (and also fig 7 and 8)
# df = pd.read_csv('C:/RPI/CompSoc/Ethical AI/fairVoting/NeurIPS_tradeoffnew.csv')

# Get output of privacy tradeoff
m = 4
n1 = 100
# for n2 in range(20,101,40):
for n2 in [40]:
    
    filname = f'C:/RPI/CompSoc/Ethical AI/fairVoting/privacy_data/fairness-privacy-sw-{n1}-{n2}-{m}-PL.csv'
    
    df = pd.read_csv(filname)
        
    df.rename(columns={"0": "threshold", "1": "eps", "2":"uf", "3":"sw", "4":"exist", "5":"Condorcet"}, inplace = True)
    # df['Condorcet'] = df['Condorcet'].apply(lambda x: x*10000 if x<1 else x) 

    # df['uf'] = df['uf']
    # df['sw'] = df['sw']
    summary = df.groupby(['threshold','eps']).mean()
    summary = summary.reset_index()
    
    summary['eff'] = summary['Condorcet']/summary['exist']
    
    nf = [summary[summary['eps']==i] for i in summary['eps'].unique()]
    
    
    eps_all = df['eps'].unique()
    eps_all[0] = 'inf'
    colors = ['b', 'g', 'r', 'c']
    markers = ['o','v','s','P']
    fig, ax = plt.subplots(figsize = (16,8))
    
    # change this for other plots
    ax.set_xlim([0.78,0.94])
    ax.set_xticks(np.arange(0.76,0.93,0.04))
    ax.set_ylim([0.2,0.5])
    
    for i in range(1,len(colors)):
        ax.scatter(1-(n2/(n1+n2))*nf[i+1]['uf'],nf[i+1]['sw'], s = 60*np.ones(len(nf[i+1]['sw'])),
                   c = colors[i], marker= markers[i], label = r"$\varepsilon = %d$"%eps_all[i+1])
    ax.scatter(1-(n2/(n1+n2))*nf[0]['uf'],nf[0]['sw'], c = colors[0], s = 60*np.ones(len(nf[0]['sw'])),
               marker= markers[0], label = r"$\varepsilon = \infty$")
    
    # alpha = np.arange(0,1.01,0.1)
    alpha = summary.threshold.unique()
    #TODO: Uncomment this later
    # for i,a in enumerate(alpha):
    #     ax.annotate(f"{a:.2f}-FP", (np.array(1-(n2/(n1+n2))*nf[0]['uf'])[i],np.array(nf[0]['sw'])[i]))
    
    labels = ['{:.1f}-FP'.format(f) for f in np.arange(0,1.01,0.1)]
    # mplcursors.cursor(ax, multiple=True).connect("add", 
    #                                              lambda sel: sel.annotation.set_text(labels[sel.target.index]))
    c2 = mplcursors.cursor(ax, multiple=True)
    @c2.connect("add")
    def _(sel):
        sel.annotation.get_bbox_patch().set(fc="white")
        sel.annotation.set_text(labels[sel.target.index])
        
    ax.set_xlabel('Mean fairness')
    ax.set_ylabel('Average top-1 utility')
    ax.set_title(f'n1={n1},n2={n2},data=Plackett-Luce')
    ax.legend(fontsize = 20)
    
    MEDIUM_SIZE = 20
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=24)     # fontsize of the axes title
    plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    
    plt.grid(linestyle = ':')
    plt.show()
    
    # plt.savefig(f"n1={n1}, n2={n2}, m={4}-privacy-kapprov.pdf", format="pdf")