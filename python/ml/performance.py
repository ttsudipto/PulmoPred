from .model import Model
from . import reader
from .model import get_under_sampling_folds
from statistics import mean, stdev
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.feature_selection import f_classif, f_regression
from ..config import ml_config as mlc

pft_svm_hyperparameters = mlc.get_optimal_hyperparameters('PFT', mlc.get_SVM_id())
pft_rf_hyperparameters = mlc.get_optimal_hyperparameters('PFT', mlc.get_RandomForest_id())
pft_nb_hyperparameters = mlc.get_optimal_hyperparameters('PFT', mlc.get_NaiveBayes_id())
tct_svm_hyperparameters = mlc.get_optimal_hyperparameters('TCT', mlc.get_SVM_id())
tct_rf_hyperparameters = mlc.get_optimal_hyperparameters('TCT', mlc.get_RandomForest_id())
tct_nb_hyperparameters = mlc.get_optimal_hyperparameters('TCT', mlc.get_NaiveBayes_id())

pft_SVM_params = {
    'C' : pft_svm_hyperparameters['C'],
    'gamma' : pft_svm_hyperparameters['gamma']
    }
pft_SVM_thresholds = pft_svm_hyperparameters['thresholds']
tct_SVM_threshold = tct_svm_hyperparameters['thresholds'][0]
tct_SVM_params = {
    'C' : tct_svm_hyperparameters['C'],
    'gamma' : tct_svm_hyperparameters['gamma']
    }
pft_RF_params = {
    'n_estimators' : pft_rf_hyperparameters['n_estimators'],
    'max_depth' : pft_rf_hyperparameters['max_depth'],
    'max_features' : pft_rf_hyperparameters['max_features']
    }
tct_RF_params = {
    'n_estimators' : tct_rf_hyperparameters['n_estimators'],
    'max_depth' : tct_rf_hyperparameters['max_depth'],
    'max_features' : tct_rf_hyperparameters['max_features']
    }
pft_GNB_params = {
    'var_smoothing' : pft_nb_hyperparameters['smoothing']
    }
tct_GNB_params = {
    'var_smoothing' : tct_nb_hyperparameters['smoothing']
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
        under_sample_folds = get_under_sampling_folds(res.target, 1, mlc.get_n_US_folds('PFT'))
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
        if mlc.is_SVM_id(model_id) : ##SVM
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_SVM_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT SVM'
        elif mlc.is_RandomForest_id(model_id) : ## RF
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_RF_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT RF'
        elif mlc.is_NaiveBayes_id(model_id) : ## GNB
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_GNB_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT NB'
        m.learn()
        if mlc.is_SVM_id(model_id) :
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
    if mlc.is_SVM_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in tct_SVM_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT SVM'
    elif mlc.is_RandomForest_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in tct_RF_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT RF'
    elif mlc.is_NaiveBayes_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in tct_GNB_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT NB'
    m.learn()
    for i in range(len(m.estimators)) :
        y_test = m.target[m.test_indices[i]]
        if mlc.is_SVM_id(model_id) :
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
        if mlc.is_SVM_id(model_id) :
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_SVM_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT SVM blind'
        elif mlc.is_RandomForest_id(model_id) :
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_RF_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT RF blind'
        elif mlc.is_NaiveBayes_id(model_id) :
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_GNB_params.items() :
                m.set_estimator_param(p, v)
            name = 'PFT NB blind'
        m.learn()
        if mlc.is_SVM_id(model_id) :
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
    if mlc.is_SVM_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in tct_SVM_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT SVM blind'
    elif mlc.is_RandomForest_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in tct_RF_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT RF blind'
    elif mlc.is_NaiveBayes_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in tct_GNB_params.items() :
            m.set_estimator_param(p, v)
        name = '2CT NB blind'
    m.learn()
    for i in range(len(m.estimators)) :
        #y_test = m.target[m.test_indices[i]]
        if mlc.is_SVM_id(model_id) :
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
    
    perform_pft_model(mlc.get_SVM_id())
    perform_pft_model(mlc.get_RandomForest_id())
    perform_pft_model(mlc.get_NaiveBayes_id())
    
    perform_pft_model_blind(mlc.get_SVM_id())
    perform_pft_model_blind(mlc.get_RandomForest_id())
    perform_pft_model_blind(mlc.get_NaiveBayes_id())
    
    #perform_ANOVA()
    #perform_ANOVA(blind=True)
    
    #perform_f_regresion()
    
    #--------------------------------------#
    
    take_input('TCT')
    
    perform_tct_model(mlc.get_SVM_id())
    perform_tct_model(mlc.get_RandomForest_id())
    perform_tct_model(mlc.get_NaiveBayes_id())
    
    perform_tct_model_blind(mlc.get_SVM_id())
    perform_tct_model_blind(mlc.get_RandomForest_id())
    perform_tct_model_blind(mlc.get_NaiveBayes_id())
    
    #perform_ANOVA()
    #perform_ANOVA(blind=True)
    
    #perform_f_regresion()
