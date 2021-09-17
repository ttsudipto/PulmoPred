import numpy as np
import copy
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from . import reader
from .model import Model, get_under_sampling_folds
from .pickler import load_saved_models
from ..config import ml_config as mlc

us_svm_hyperparameters = mlc.get_optimal_hyperparameters('us', mlc.get_SVM_id())
us_rf_hyperparameters = mlc.get_optimal_hyperparameters('us', mlc.get_RandomForest_id())
us_nb_hyperparameters = mlc.get_optimal_hyperparameters('us', mlc.get_NaiveBayes_id())
us_mlp_hyperparameters = mlc.get_optimal_hyperparameters('us', mlc.get_MLP_id())
no_us_svm_hyperparameters = mlc.get_optimal_hyperparameters('no-us', mlc.get_SVM_id())
no_us_rf_hyperparameters = mlc.get_optimal_hyperparameters('no-us', mlc.get_RandomForest_id())
no_us_nb_hyperparameters = mlc.get_optimal_hyperparameters('no-us', mlc.get_NaiveBayes_id())
no_us_mlp_hyperparameters = mlc.get_optimal_hyperparameters('no-us', mlc.get_MLP_id())

us_SVM_params = {
    'C' : us_svm_hyperparameters['C'],
    'gamma' : us_svm_hyperparameters['gamma']
    }
us_SVM_thresholds = us_svm_hyperparameters['thresholds']
no_us_SVM_params = {
    'C' : no_us_svm_hyperparameters['C'],
    'gamma' : no_us_svm_hyperparameters['gamma']
    }
no_us_SVM_threshold = no_us_svm_hyperparameters['thresholds'][0]
us_RF_params = {
    'n_estimators' : us_rf_hyperparameters['n_estimators'],
    'max_depth' : us_rf_hyperparameters['max_depth'],
    'max_features' : us_rf_hyperparameters['max_features']
    }
no_us_RF_params = {
    'n_estimators' : no_us_rf_hyperparameters['n_estimators'],
    'max_depth' : no_us_rf_hyperparameters['max_depth'],
    'max_features' : no_us_rf_hyperparameters['max_features']
    }
us_GNB_params = {
    'var_smoothing' : us_nb_hyperparameters['smoothing']
    }
no_us_GNB_params = {
    'var_smoothing' : no_us_nb_hyperparameters['smoothing']
    }
us_MLP_params = {
    'activation' : us_mlp_hyperparameters['activation'],
    'hidden_layer_sizes' : us_mlp_hyperparameters['hidden_layer_sizes'],
    'learning_rate_init' : us_mlp_hyperparameters['learning_rate_init']
    }
no_us_MLP_params = {
    'activation' : no_us_mlp_hyperparameters['activation'],
    'hidden_layer_sizes' : no_us_mlp_hyperparameters['hidden_layer_sizes'],
    'learning_rate_init' : no_us_mlp_hyperparameters['learning_rate_init']
    }

res = None
under_sample_folds = None

def take_input() :
    global res
    global under_sample_folds
    res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=False)
    under_sample_folds = get_under_sampling_folds(res.target, 1, mlc.get_n_US_folds())

def plot_roc_model(model, verbose=False) :
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

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
        elif mlc.is_MLP_id(model.estimator_id) :
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
    
    if verbose == True :
        print('\tMean AUC = ' + str(mean_auc) + ' (+/- ' + str(std_auc) + ')')
    
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

def create_model_us(estimator_id) :
    models = []
    
    for i in range(len(under_sample_folds)) :
        if mlc.is_SVM_id(estimator_id) : ##SVM
            m = Model(estimator_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_SVM_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_RandomForest_id(estimator_id) : ## RF
            m = Model(estimator_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_RF_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_NaiveBayes_id(estimator_id) : ## GNB
            m = Model(estimator_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_GNB_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_MLP_id(estimator_id) : ## MLP
            m = Model(estimator_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_MLP_params.items() :
                m.set_estimator_param(p, v)
        
        m.learn()
        models.append(copy.deepcopy(m))
    
    return models

def create_model_no_us(estimator_id) :
    if mlc.is_SVM_id(estimator_id) :
        m = Model(estimator_id, res.data, res.target)
        for p, v in no_us_SVM_params.items() :
            m.set_estimator_param(p, v)
    elif mlc.is_RandomForest_id(estimator_id) :
        m = Model(estimator_id, res.data, res.target)
        for p, v in no_us_RF_params.items() :
            m.set_estimator_param(p, v)
    elif mlc.is_NaiveBayes_id(estimator_id) :
        m = Model(estimator_id, res.data, res.target)
        for p, v in no_us_GNB_params.items() :
            m.set_estimator_param(p, v)
    elif mlc.is_MLP_id(estimator_id) :
        m = Model(estimator_id, res.data, res.target)
        for p, v in no_us_MLP_params.items() :
            m.set_estimator_param(p, v)
    
    m.learn()
    return m

def execute(estimator_id, dataset) :
    #models = load_saved_models(estimator_id)
    #plot_roc_US(models)
    
    take_input()
    
    if dataset == 'us' :
        models = create_model_us(estimator_id)
        plot_roc_US(models)
    elif dataset == "no-us" :
        model = create_model_no_us(estimator_id)
        plot_roc_model(model, verbose=True)

#execute('SVM', 'us')
#execute('RF', 'us')
#execute('GNB', 'us')
#execute('MLP', 'us')

#execute('SVM', 'no-us')
#execute('RF', 'no-us')
#execute('GNB', 'no-us')
#execute('MLP', 'no-us')
