import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import csv
from ..config import config

prefix = config.get_OUTPUT_PATH()

svm_pft_filename = 'SVM/PFT_common_gs_res.csv'
rf_pft_filename = 'RF/PFT_common_wo_tt_res.csv'
svm_tct_filename = 'SVM/TCT_common_nous_res.csv'
rf_tct_filename = 'RF/TCT_common_wo_tt_res.csv'

def read_SVM_csv(filename) :
    xpos_map = {'0.5':1, '1':4, '5':7, '10':10, '15':13, '20':16}
    ypos_map = {'0.1':1, '0.01':4, '0.001':7, '0.0001':10, '1E-05':13}
    x = []
    y = []
    z = []
    reader = csv.DictReader(open(prefix + filename, 'r'))
    #next(reader)
    for row in reader :
        x.append(xpos_map[row['C']])
        y.append(ypos_map[row['Gamma']])
        z.append(float(row['Accuracy']))
        #print(row['C'], row['Gamma'], row['Accuracy'])
    return x, y, z, [1.5, 4.5, 7.5, 10.5, 13.5, 16.5], xpos_map.keys(), [1.5, 4.5, 7.5, 10.5, 13.5], ypos_map.keys()

def read_RF_csv(filename, n_estimators) :
    xpos_map = {'4':1, '8':4, '16':7, 'None':10}
    ypos_map = {'auto':1, '0.3':4, '0.4':7, '0.5':10, '0.6':13, '0.7':16, 'None':19}
    x = []
    y = []
    z = []
    reader = csv.DictReader(open(prefix + filename, 'r'))
    #next(reader)
    for row in reader :
        #print(row['#_estimators'])
        if int(row['#_estimators']) == n_estimators :
            x.append(xpos_map[row['Max_depth']])
            y.append(ypos_map[row['Max_features']])
            z.append(float(row['Accuracy']))
            #print(row['#_estimators'], row['Max_depth'], row['Max_features'], row['Accuracy'])
    return x, y, z, [1.5, 4.5, 7.5, 10.5], xpos_map.keys(), [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5], ypos_map.keys()

def plot_svm_gs_graph(dataset, dpi=600) :
    if dataset == 'PFT' :
        filename = svm_pft_filename
    elif dataset == 'TCT' :
        filename = svm_tct_filename
    x, y, z, xtick, xlabel, ytick, ylabel = read_SVM_csv(filename)
    #print(xtick)
    #print(xlabel)
    #print(ytick)
    #print(ylabel)
    #print(x)
    #print(y)
    #print(z)
    fig = plt.figure()
    fig.set_dpi(dpi)
    ax1 = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('bone')
    norm = Normalize(vmin=min(z), vmax=max(z))
    cs = cmap(norm(z))
    ax1.bar3d(x, y, np.zeros_like(z), 1, 1, z, color=cs, shade=True)
    ax1.set_xlabel('C', labelpad=20)
    ax1.set_ylabel('Gamma', labelpad=20)
    ax1.set_zlabel('Accuracy', labelpad=15)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xlabel)
    ax1.set_yticks(ytick)
    ax1.set_yticklabels(ylabel)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm)).ax.tick_params(labelsize=20)
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label, ax1.zaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels() + ax1.get_zticklabels()) :
        item.set_fontsize(20)
    #for a,b,c in zip(x, y, z) : ## values on each bar
        #ax1.text(a+0.1,b+0.1,c+0.02,'%2.3f'%c, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.1))
    plt.show()

def plot_rf_gs_graph(dataset, lower_z=0, upper_z=None, dpi=600) :
    n_estimators = [5, 10, 20, 30]
    
    if dataset == 'PFT' :
        filename = rf_pft_filename
    elif dataset == 'TCT' :
        filename = rf_tct_filename
    
    fig = plt.figure()
    fig.set_size_inches(7.4, 7.4)
    fig.set_dpi(dpi)
    for i in range(len(n_estimators)) :
        x, y, z, xtick, xlabel, ytick, ylabel = read_RF_csv(filename, n_estimators[i])
        #print(xtick)
        #print(xlabel)
        #print(ytick)
        #print(ylabel)
        #print(x)
        #print(y)
        #print(z)
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        cmap = plt.get_cmap('bone')
        norm = Normalize(vmin=min(z), vmax=max(z))
        cs = cmap(norm(z))
        bar = ax.bar3d(x, y, np.full_like(z, lower_z), 1, 1, np.array(z)-lower_z, color=cs, shade=True)
        ax.set_title('No. of trees = ' + str(n_estimators[i]))
        ax.set_xlabel('Max depth')
        ax.set_ylabel('Max Features')
        ax.set_zlabel('Accuracy')
        ax.set_xticks(xtick)
        ax.set_xticklabels(xlabel)
        ax.set_yticks(ytick)
        ax.set_yticklabels(ylabel)
        ax.set_zlim(lower_z, upper_z)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm))
    plt.show()

def execute(dataset, model) :
    if model == 'SVM' :
        plot_svm_gs_graph(dataset)
    elif model == 'RF' :
        if dataset == 'PFT' :
            plot_rf_gs_graph(dataset, lower_z=0.75, upper_z=0.83)
        elif dataset =='TCT' :
            plot_rf_gs_graph(dataset, lower_z=0.55, upper_z=0.65)
            
execute('PFT', 'SVM')
execute('PFT', 'RF')
execute('TCT', 'SVM')
execute('TCT', 'RF')
