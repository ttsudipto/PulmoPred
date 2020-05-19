import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from .pickler import load_saved_models
from ..config import ml_config as mlc

def plot_roc_model(model, do_fit=False) :
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if do_fit == True :
        model.learn()
    for i in range(len(model.estimators)) :
        X_test = model.data[model.test_indices[i]]
        y_test = model.target[model.test_indices[i]]
        probas = model.get_decision_score(model.estimators[i], X_test)
        if mlc.is_SVM_id(model.estimator_id) :
            fpr, tpr, thresholds = roc_curve(y_test, probas)
        elif mlc.is_RandomForest_id(model.estimator_id) :
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        elif mlc.is_NaiveBayes_id(model.estimator_id) :
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        else :
            raise ValueError('Invalid estimator ID')
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('AUC(CV ' + str(i) + ') = ' + str(roc_auc))
        #plt.plot(mean_fpr, tprs[0])
        #plt.show()

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    #------ Print as CSV ------#
    #heading = ['FPR']
    #for i in range(len(model.estimators)) :
        #heading.append('TPR' + str(i) + ' AUC = ' + str(round(aucs[i], 4)))
    #heading.append('Mean TPR' + ' AUC = ' + str(round(mean_auc, 4)) + ' (+/- ' + str(round(std_auc, 4)) + ')')
    #print(', '.join(heading))
    #for i in range(100) :
        #row = [mean_fpr[i]]
        #for j in range(len(model.estimators)) :
            #row.append(tprs[j][i])
        #row.append(mean_tpr[i])
        #print(', '.join(map(str,row)))
    #------ Print as CSV ------#
    
    return mean_fpr, mean_tpr, mean_auc, std_auc

def plot_roc_US(models) :
    tprs = []
    aucs = []
    m_std_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    m_index = 0
    for m in models :
        print('Model' + str(m_index))
        fpr, tpr, m_auc, m_std_auc = plot_roc_model(m)
        tprs.append(tpr)
        aucs.append(m_auc)
        m_std_aucs.append(m_std_auc)
        m_index = m_index + 1
        print('\tModel AUC = ' + str(m_auc) + ' (+/- ' + str(m_std_auc) + ')')
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print('\tMean AUC = ' + str(mean_auc) + ' (+/- ' + str(std_auc) + ')')
    
    #------ Print as CSV ------#
    #heading = ['FPR']
    #for i in range(len(models)) :
        #heading.append('TPR' + str(i) + ' AUC = ' + str(round(aucs[i], 4)) + ' (+/- ' + str(round(m_std_aucs[i], 4)) + ')')
    #heading.append('Mean TPR' + ' AUC = ' + str(round(mean_auc, 4)) + ' (+/- ' + str(round(std_auc, 4)) + ')')
    #print(', '.join(heading))
    #for i in range(100) :
        #row = [mean_fpr[i]]
        #for j in range(len(models)) :
            #row.append(tprs[j][i])
        #row.append(mean_tpr[i])
        #print(', '.join(map(str,row)))
    #------ Print as CSV ------#
    
    return mean_fpr, mean_tpr, mean_auc, std_auc

def execute(estimator_id) :
    models = load_saved_models(estimator_id)
    plot_roc_US(models)
