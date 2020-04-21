from .model import Model
from . import reader
from .model import get_under_sampling_folds
from statistics import mean, stdev
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.feature_selection import f_classif, f_regression

pft_SVM_params = {
    'C' : 5,
    'gamma' : 0.0001
    }
pft_SVM_thresholds = [0.1, 0.2, 0.3, -0.1, 0.2, 0.1]
tct_SVM_threshold = 0.3
tct_SVM_params = {
    'C' : 10,
    'gamma' : 0.0001
    }
pft_RF_params = {
    'n_estimators' : 20,
    'max_depth' : 4,
    'max_features' : 0.4
    }
tct_RF_params = {
    'n_estimators' : 30,
    'max_depth' : 8,
    'max_features' : 0.5
    }
pft_GNB_params = {
    'var_smoothing' : 0
    }
tct_GNB_params = {
    'var_smoothing' : 0.1
    }

res = None
b_res = None
under_sample_folds = None

def take_input(dataset) :
    global res
    global b_res
    global under_sample_folds
    if dataset == 'PFT' :
        res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=False)
        b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
        under_sample_folds = get_under_sampling_folds(res.target, 1, 6)
    elif dataset == 'TCT' :
        res = reader.read_data('TCT_O_NO_uncommon.csv', verbose=False)
        b_res = reader.read_data('TCT_O_NO_common.csv', verbose=False)

def perform_ANOVA(blind=False) :
    if blind == False :
        fval, pval = f_classif(res.data, res.target)
    else :
        fval, pval = f_classif(b_res.data, b_res.target)
    print('Attribute,f-value,p-value')
    for i in range(len(pval)) :
        print(res.attributes[i+3] + ',' + str(fval[i]) + ',' + str(pval[i]))

def perform_f_regresion(blind = False) :
    if blind == False :
        fval, pval = f_regression(res.data, res.target)
    else :
        fval, pval = f_classif(b_res.data, b_res.target)
    print('Attribute,f-value,p-value')
    for i in range(len(pval)) :
        print(res.attributes[i+3] + ',' + str(fval[i]) + ',' + str(pval[i]))

def perform_pft_model(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    for i in range(len(under_sample_folds)) :
        if model_id == 'SVM' :
            m = Model('SVM', res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_SVM_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT SVM'
        elif model_id == 'RF' :
            m = Model('RF', res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_RF_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT RF'
        elif model_id == 'GNB' :
            m = Model('GNB', res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_GNB_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT NB'
        m.learn_k_fold()
        if model_id == 'SVM' :
            acc, sens, spec = m.predict_k_fold(pft_SVM_thresholds[i])
        else :
            acc, sens, spec = m.predict_k_fold()
        accuracies.append(acc)
        sensitivities.append(sens)
        specificities.append(spec)
    print(name, mean(accuracies), stdev(accuracies), mean(sensitivities), stdev(sensitivities), mean(specificities), stdev(specificities))

def perform_tct_model(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    if model_id == 'SVM' :
        m = Model('SVM', res.data, res.target)
        for p, v in tct_SVM_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT SVM'
    elif model_id == 'RF' :
        m = Model('RF', res.data, res.target)
        for p, v in tct_RF_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT RF'
    elif model_id == 'GNB' :
        m = Model('GNB', res.data, res.target)
        for p, v in tct_GNB_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT NB'
    m.learn_k_fold()
    for i in range(len(m.estimators)) :
        y_test = m.target[m.test_indices[i]]
        if model_id == 'SVM' :
            y_pred = m.predict(m.estimators[i], m.data[m.test_indices[i]], tct_SVM_threshold)
        else :
            y_pred = m.predict(m.estimators[i], m.data[m.test_indices[i]])
        sensitivities.append(recall_score(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificities.append((tn/1.0) / (tn+fp))
    print(name, mean(accuracies), stdev(accuracies), mean(sensitivities), stdev(sensitivities), mean(specificities), stdev(specificities))

def perform_pft_model_blind(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    for i in range(len(under_sample_folds)) :
        if model_id == 'SVM' :
            m = Model('SVM', res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_SVM_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT SVM blind'
        elif model_id == 'RF' :
            m = Model('RF', res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_RF_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT RF blind'
        elif model_id == 'GNB' :
            m = Model('GNB', res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_GNB_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT NB blind'
        m.learn_k_fold()
        if model_id == 'SVM' :
            acc, sens, spec = m.predict_blind_data(b_res.data, b_res.target, pft_SVM_thresholds[i])
        else :
            acc, sens, spec = m.predict_blind_data(b_res.data, b_res.target)
        accuracies.append(acc)
        sensitivities.append(sens)
        specificities.append(spec)
    print(name, mean(accuracies), stdev(accuracies), mean(sensitivities), stdev(sensitivities), mean(specificities), stdev(specificities))

def perform_tct_model_blind(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    if model_id == 'SVM' :
        m = Model('SVM', res.data, res.target)
        for p, v in tct_SVM_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT SVM blind'
    elif model_id == 'RF' :
        m = Model('RF', res.data, res.target)
        for p, v in tct_RF_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT RF blind'
    elif model_id == 'GNB' :
        m = Model('GNB', res.data, res.target)
        for p, v in tct_GNB_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT NB blind'
    m.learn_k_fold()
    for i in range(len(m.estimators)) :
        #y_test = m.target[m.test_indices[i]]
        if model_id == 'SVM' :
            y_pred = m.predict(m.estimators[i], b_res.data, tct_SVM_threshold)
        else :
            y_pred = m.predict(m.estimators[i], b_res.data)
        sensitivities.append(recall_score(b_res.target, y_pred))
        accuracies.append(accuracy_score(b_res.target, y_pred))
        tn, fp, fn, tp = confusion_matrix(b_res.target, y_pred).ravel()
        specificities.append((tn/1.0) / (tn+fp))
    print(name, mean(accuracies), stdev(accuracies), mean(sensitivities), stdev(sensitivities), mean(specificities), stdev(specificities))

def execute() :
    take_input('PFT')
    
    #perform_pft_model('SVM')
    #perform_pft_model('RF')
    #perform_pft_model('GNB')
    
    perform_pft_model_blind('SVM')
    perform_pft_model_blind('RF')
    perform_pft_model_blind('GNB')
    
    perform_ANOVA()
    perform_ANOVA(blind=True)
    #perform_f_regresion()
    
    take_input('TCT')
    
    #perform_tct_model('SVM')
    #perform_tct_model('RF')
    #perform_tct_model('GNB')
    
    perform_tct_model_blind('SVM')
    perform_tct_model_blind('RF')
    perform_tct_model_blind('GNB')
    
    perform_ANOVA()
    perform_ANOVA(blind=True)
    #perform_f_regresion()
