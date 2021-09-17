from .model import Model
from . import reader
from .model import get_under_sampling_folds
from statistics import mean, stdev
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.feature_selection import f_classif, f_regression
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
b_res = None
under_sample_folds = None

def take_input(dataset) :
    global res
    global b_res
    global under_sample_folds
    if dataset == 'PFT' :
        res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=False)
        b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
        under_sample_folds = get_under_sampling_folds(res.target, 1, mlc.get_n_US_folds())
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
        fval, pval = f_regression(b_res.data, b_res.target)
    print('Attribute,f-value,p-value')
    for i in range(len(pval)) :
        print(res.attributes[i+3] + ',' + str(fval[i]) + ',' + str(pval[i]))

def perform_us_model(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    mccs = []
    for i in range(len(under_sample_folds)) :
        if mlc.is_SVM_id(model_id) : ##SVM
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_SVM_params.items() :
                m.set_estimator_param(p, v)
            name = 'SVM-US'
        elif mlc.is_RandomForest_id(model_id) : ## RF
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_RF_params.items() :
                m.set_estimator_param(p, v)
            name = 'RF-US'
        elif mlc.is_NaiveBayes_id(model_id) : ## GNB
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_GNB_params.items() :
                m.set_estimator_param(p, v)
            name = 'NB-US'
        elif mlc.is_MLP_id(model_id) : ## MLP
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_MLP_params.items() :
                m.set_estimator_param(p, v)
            name = 'MLP-US'
        m.learn()
        if mlc.is_SVM_id(model_id) :
            acc, sens, spec, f1s, mcc = m.predict_k_fold(us_SVM_thresholds[i])
        else :
            acc, sens, spec, f1s, mcc = m.predict_k_fold()
        accuracies.append(acc)
        sensitivities.append(sens)
        specificities.append(spec)
        f1_scores.append(f1s)
        mccs.append(mcc)
    #print(sensitivities)
    #print(specificities)
    print(name, mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs), stdev(accuracies), stdev(sensitivities), stdev(specificities), stdev(f1_scores), stdev(mccs))

def perform_no_us_model(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    mccs = []
    if mlc.is_SVM_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_SVM_params.items() :
            m.set_estimator_param(p, v)
        name = 'SVM-No-US'
    elif mlc.is_RandomForest_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_RF_params.items() :
            m.set_estimator_param(p, v)
        name = 'RF-No-US'
    elif mlc.is_NaiveBayes_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_GNB_params.items() :
            m.set_estimator_param(p, v)
        name = 'NB-No-US'
    elif mlc.is_MLP_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_MLP_params.items() :
            m.set_estimator_param(p, v)
        name = 'MLP-No-US'
    m.learn()
    for i in range(len(m.estimators)) :
        y_test = m.target[m.test_indices[i]]
        if mlc.is_SVM_id(model_id) :
            y_pred = m.predict(m.estimators[i], m.data[m.test_indices[i]], no_us_SVM_threshold)
        else :
            y_pred = m.predict(m.estimators[i], m.data[m.test_indices[i]])
        sensitivities.append(recall_score(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificities.append((tn/1.0) / (tn+fp))
        f1_scores.append(f1_score(y_test, y_pred))
        mccs.append(matthews_corrcoef(y_test, y_pred))
    print(name, mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs), stdev(accuracies), stdev(sensitivities), stdev(specificities), stdev(f1_scores), stdev(mccs))

def perform_us_model_blind(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    mccs = []
    for i in range(len(under_sample_folds)) :
        if mlc.is_SVM_id(model_id) :
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_SVM_params.items() :
                m.set_estimator_param(p, v)
            name = 'SVM-US blind'
        elif mlc.is_RandomForest_id(model_id) :
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_RF_params.items() :
                m.set_estimator_param(p, v)
            name = 'RF-US blind'
        elif mlc.is_NaiveBayes_id(model_id) :
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_GNB_params.items() :
                m.set_estimator_param(p, v)
            name = 'NB-US blind'
        elif mlc.is_MLP_id(model_id) :
            m = Model(model_id, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_MLP_params.items() :
                m.set_estimator_param(p, v)
            name = 'MLP-US blind'
        m.learn()
        if mlc.is_SVM_id(model_id) :
            acc, sens, spec, f1s, mcc = m.predict_blind_data(b_res.data, b_res.target, us_SVM_thresholds[i])
        else :
            acc, sens, spec, f1s, mcc = m.predict_blind_data(b_res.data, b_res.target)
        accuracies.append(acc)
        sensitivities.append(sens)
        specificities.append(spec)
        f1_scores.append(f1s)
        mccs.append(mcc)
    #print(sensitivities)
    #print(specificities)
    print(name, mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs), stdev(accuracies), stdev(sensitivities), stdev(specificities), stdev(f1_scores), stdev(mccs))

def perform_no_us_model_blind(model_id) :
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    mccs = []
    if mlc.is_SVM_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_SVM_params.items() :
            m.set_estimator_param(p, v)
        name = 'SVM-No-US blind'
    elif mlc.is_RandomForest_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_RF_params.items() :
            m.set_estimator_param(p, v)
        name = 'RF-No-US blind'
    elif mlc.is_NaiveBayes_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_GNB_params.items() :
            m.set_estimator_param(p, v)
        name = 'NB-No-US blind'
    elif mlc.is_MLP_id(model_id) :
        m = Model(model_id, res.data, res.target)
        for p, v in no_us_MLP_params.items() :
            m.set_estimator_param(p, v)
        name = 'MLP-No-US blind'
    m.learn()
    for i in range(len(m.estimators)) :
        #y_test = m.target[m.test_indices[i]]
        if mlc.is_SVM_id(model_id) :
            y_pred = m.predict(m.estimators[i], b_res.data, no_us_SVM_threshold)
        else :
            y_pred = m.predict(m.estimators[i], b_res.data)
        sensitivities.append(recall_score(b_res.target, y_pred))
        accuracies.append(accuracy_score(b_res.target, y_pred))
        tn, fp, fn, tp = confusion_matrix(b_res.target, y_pred).ravel()
        specificities.append((tn/1.0) / (tn+fp))
        f1_scores.append(f1_score(b_res.target, y_pred))
        mccs.append(matthews_corrcoef(b_res.target, y_pred))
    print(name, mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs), stdev(accuracies), stdev(sensitivities), stdev(specificities), stdev(f1_scores), stdev(mccs))

def execute() :
    take_input('PFT')
    
    perform_us_model(mlc.get_SVM_id())
    perform_us_model(mlc.get_RandomForest_id())
    perform_us_model(mlc.get_NaiveBayes_id())
    perform_us_model(mlc.get_MLP_id())
    
    perform_no_us_model(mlc.get_SVM_id())
    perform_no_us_model(mlc.get_RandomForest_id())
    perform_no_us_model(mlc.get_NaiveBayes_id())
    perform_no_us_model(mlc.get_MLP_id())
    
    perform_us_model_blind(mlc.get_SVM_id())
    perform_us_model_blind(mlc.get_RandomForest_id())
    perform_us_model_blind(mlc.get_NaiveBayes_id())
    perform_us_model_blind(mlc.get_MLP_id())
    
    perform_no_us_model_blind(mlc.get_SVM_id())
    perform_no_us_model_blind(mlc.get_RandomForest_id())
    perform_no_us_model_blind(mlc.get_NaiveBayes_id())
    perform_no_us_model_blind(mlc.get_MLP_id())
    
    #perform_ANOVA()
    #perform_ANOVA(blind=True)
    
    #perform_f_regresion()

#execute()
