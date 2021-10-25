from ..config import ml_config as mlc
from .pickler import load_saved_models
from .reader import read_data
from .model import get_under_sampling_folds, Model
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
#from statistics import mean

import warnings
warnings.filterwarnings('ignore')

res = read_data('PFT_O_NO_uncommon.csv', verbose=True)
b_res = read_data('PFT_O_NO_common.csv', verbose=False)

under_sample_folds = get_under_sampling_folds(res.target, 1, 6)

RF_params = dict()
RF_params['class_weight'] = 'balanced_subsample'
RF_params['bootstrap'] = True
RF_params['n_jobs'] = -1
RF_params['random_state'] = 42
RF_params['criterion'] = 'gini' # also, 'entropy' -> info. gain
RF_params['min_impurity_decrease'] = 0. # also, float

SVM_params = dict()
SVM_params['class_weight'] = 'balanced'
SVM_params['max_iter'] = -1
SVM_params['random_state'] = 42
SVM_params['kernel'] = 'linear'

def split_cv_folds(data, target) :
    train_indices = []
    test_indices = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=mlc.get_random_state())
    for train_index, test_index in skf.split(data, target) :
        train_indices.append(train_index)
        test_indices.append(test_index)
    return skf, train_indices, test_indices

#def predict_with_cv(model, data, target, test_indices) :
    #n_folds = len(test_indices)
    #print(n_folds)
    #print([x.shape for x in test_indices])
    #sensitivities = []
    #specificities = []
    #accuracies = []
    #f1_scores = []
    #mccs = []
    #for i in range(n_folds) :
        #y_test = target[test_indices[i]]
        #y_pred = model.predict(data[test_indices[i]])
        #sensitivities.append(recall_score(y_test, y_pred))
        #accuracies.append(accuracy_score(y_test, y_pred))
        #f1_scores.append(f1_score(y_test, y_pred))
        #mccs.append(matthews_corrcoef(y_test, y_pred))
        #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        #specificities.append((tn/1.0) / (tn+fp))
    ##print((mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs)))
    #print(accuracies), print(sensitivities), print(specificities), print(f1_scores), print(mccs)
    #return (mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs))

def scorer(clf, X, y) :
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    sens = recall_score(y, y_pred)
    f1s = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spec = (tn/1.0) / (tn+fp)
    #print(acc, sens, spec, f1s, mcc)
    #return {'acc' : acc, 'sens' : sens, 'spec' : spec, 'f1s' : f1s, 'mcc' : mcc}
    return mcc

def grid_search_RF_us(max_depths, max_features, estimator_sizes) :
    for md in max_depths :
        for mf in max_features :
            for ne in estimator_sizes :
                mccs = []
                rankings = []
                n_features = []
                for i in range(len(under_sample_folds)) :
                    RF_params['max_depth'] = md
                    RF_params['max_features'] = mf
                    RF_params['n_estimators'] = ne
                    skf, train_indices, test_indices = split_cv_folds(res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
                    model = RF(**RF_params)
                    rfe = RFECV(model, min_features_to_select=1, cv=skf, n_jobs=-1, scoring=scorer)
                    rfe.fit(res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
                    mccs.append(rfe.cv_results_['mean_test_score'][rfe.n_features_-1])
                    rankings.append(rfe.ranking_)
                    n_features.append(rfe.n_features_)
                print(md, mf, ne, mean(mccs), n_features, rankings)
                #print(md, mf, ne, mean(accuracies), mean(sensitivities), mean(specificities), mean(f1_scores), mean(mccs))

def grid_search_RF_no_us(max_depths, max_features, estimator_sizes) :
    for md in max_depths :
        for mf in max_features :
            for ne in estimator_sizes :
                RF_params['max_depth'] = md
                RF_params['max_features'] = mf
                RF_params['n_estimators'] = ne
                skf, train_indices, test_indices = split_cv_folds(res.data, res.target)
                model = RF(**RF_params)
                rfe = RFECV(model, min_features_to_select=1, cv=skf, n_jobs=-1, scoring=scorer)
                rfe.fit(res.data, res.target)
                #print(rfe.cv_results_)
                print(md, mf, ne, rfe.cv_results_['mean_test_score'][rfe.n_features_-1], rfe.n_features_, rfe.ranking_)
                #acc, sens, spec, f1s, mcc = predict_with_cv(rfe, res.data, res.target, test_indices)
                #acc, sens, spec, f1s, mcc = rfe.cv_results_[]
                #print(md, mf, ne, acc, sens, spec, f1s, mcc, rfe.ranking_)

def grid_search_SVM_no_us(C) :
    for c in C :
        SVM_params['C'] = c
        skf, train_indices, test_indices = split_cv_folds(res.data, res.target)
        model = SVC(**SVM_params)
        rfe = RFECV(model, min_features_to_select=1, cv=skf, n_jobs=-1, scoring=scorer)
        rfe.fit(res.data, res.target)
        print(c, rfe.cv_results_['mean_test_score'][rfe.n_features_-1], rfe.n_features_, rfe.ranking_)
        #acc, sens, spec, f1s, mcc = predict_with_cv(rfe, res.data, res.target, test_indices)
        #acc, sens, spec, f1s, mcc = rfe.cv_results_[]
        #print(md, mf, ne, acc, sens, spec, f1s, mcc, rfe.ranking_)

##-------------------------##
##        Grid search      ##
##-------------------------##
estimator_sizes=[5, 10, 20, 30]
max_depths=[4, 8, 16, None]
max_features=['auto', 0.3, 0.4, 0.5, 0.6, 0.7, None]
estimator_sizes=[10]
max_depths=[16]
max_features=[0.3]
#grid_search_RF_us(max_depths, max_features, estimator_sizes)
#grid_search_RF_no_us(max_depths, max_features, estimator_sizes)

C = [1, 5, 10, 15, 20]
#grid_search_SVM_no_us(C)

##-------------------------##
##      Optimal params     ##
##-------------------------##
max_depth=16
max_features=0.3
estimator_size=10
cols = [4, 6, 7, 9, 10, 11]
#execute_RFE_RF_no_us(max_depth, max_features, estimator_size, cols)



#estimator_id = 'RF'
#models = load_saved_models(estimator_id)
#print(['fev1_pre_value','fev1_pre_percent','fev1_post_value','fev1_post_percent','fvc_pre_value','fvc_pre_percent','fvc_post_value','fvc_post_percent','fef_pre_value','fef_pre_percent','fef_post_value','fef_post_percent'])
#for i in range(len(models)) :
    #model = RF(**(models[i].RF_params))
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=mlc.get_random_state())
    #rfe = RFECV(model, min_features_to_select=7, cv=skf, n_jobs=-1)
    #rfe.fit(models[i].data, models[i].target)
    #print(rfe.ranking_)
    
