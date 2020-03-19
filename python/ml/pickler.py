from joblib import dump, load
from .model import Model
from .model import get_under_sampling_folds
from .model_metadata import ModelMetadata
from ..config import config
from . import reader
import pickle
import copy
#import numpy as np

path_prefix = config.get_OUTPUT_PATH() + 'models/'
#filenames = ['f1.joblib', 'f2.joblib', 'f3.joblib', 'f4.joblib', 'f5.joblib']

res = b_res = under_sample_folds = None

def take_input() :
    global res
    global b_res
    global under_sample_folds
    res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=True)
    b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
    under_sample_folds = get_under_sampling_folds(res.target, 1, 6)

def dump_model_to_file(prefix, model, metadata_filename = None, threshold = None) :
    estimators = model.estimators;
    for i in range(len(estimators)) :
        filename = 'f' + str(i+1) + '.joblib'
        dump(estimators[i], prefix + filename)
    md = ModelMetadata(model, threshold)
    metadata_file = open(prefix + metadata_filename, 'wb')
    pickle.dump(md, metadata_file)
    metadata_file.close()

def load_model_from_file(prefix, metadata_filename, threshold = None) :
    metadata_file = open(prefix + metadata_filename, 'rb')
    md = pickle.load(metadata_file, encoding='latin1')
    model = md.get_model()
    for i in range(model.n_folds) :
        filename = 'f' + str(i+1) + '.joblib'
        model.estimators.append(load(prefix + filename))
    return model

def create_SVM_model(data, target, kernel, C, gamma) :
    m = Model('SVM', data, target)
    m.set_estimator_param('kernel', kernel)
    m.set_estimator_param('C', C)
    m.set_estimator_param('gamma', gamma)
    return m

def create_RF_model(data, target, n_estimators, max_depth, max_features) :
    m = Model('RF', data, target)
    m.set_estimator_param('n_estimators', n_estimators)
    m.set_estimator_param('max_depth', max_depth)
    m.set_estimator_param('max_features', max_features)
    return m

def save_SVM_models(data, target, with_CV=False) :
    kernel = 'rbf'
    C = 5
    gamma = 0.0001
    SVM_thresholds = [0.1, 0.2, 0.3, -0.1, 0.2, 0.1]
    model_path = 'SVM/'
    for i in range(len(under_sample_folds)) :
        m = create_SVM_model(data[under_sample_folds[i]], target[under_sample_folds[i]], kernel, C, gamma)
        if with_CV == True :
            m.learn_k_fold()
            #print(i)
            dump_model_to_file((path_prefix + model_path + 'with_CV/' + str(i) + '/'), m, 'metadata.pkl', SVM_thresholds[i])
        else :
            m.learn_without_CV()
            dump_model_to_file((path_prefix + model_path + 'without_CV/' + str(i) + '/'), m, 'metadata.pkl', SVM_thresholds[i])

def save_RF_models(data, target, with_CV=False) :
    n_estimators = 20
    max_depth = 4
    max_features = 0.4
    model_path = 'RF/'
    for i in range(len(under_sample_folds)) :
        m = create_RF_model(data[under_sample_folds[i]], target[under_sample_folds[i]], n_estimators, max_depth, max_features)
        if with_CV == True :
            m.learn_k_fold()
            #print(i)
            dump_model_to_file((path_prefix + model_path + 'with_CV/' + str(i) + '/'), m, 'metadata.pkl')
        else :
            m.learn_without_CV()
            dump_model_to_file((path_prefix + model_path + 'without_CV/' + str(i) + '/'), m, 'metadata.pkl')

def perform_SVM(data, target, with_CV=False) :
    kernel = 'rbf'
    C = 5
    gamma = 0.0001
    SVM_thresholds = [0.1, 0.2, 0.3, -0.1, 0.2, 0.1]
    model_path = 'SVM/'
    for i in range(len(under_sample_folds)) :
        m = create_SVM_model(data[under_sample_folds[i]], target[under_sample_folds[i]], kernel, C, gamma)
        if with_CV == True :
            m.learn_k_fold()
        else :
            m.learn_without_CV()
        print(str(i)+' -> ', m.predict_k_fold(SVM_thresholds[i]))

def perform_RF(data, target, with_CV=False) :
    n_estimators = 20
    max_depth = 4
    max_features = 0.4
    model_path = 'RF/'
    for i in range(len(under_sample_folds)) :
        m = create_RF_model(data[under_sample_folds[i]], target[under_sample_folds[i]], n_estimators, max_depth, max_features)
        if with_CV == True :
            m.learn_k_fold()
        else :
            m.learn_without_CV()
        print(str(i)+' -> ', m.predict_k_fold())

def check_saved_models(estimator_id, with_CV=False) :
    if estimator_id == 'SVM' :
        model_path = 'SVM/'
    elif estimator_id == 'RF' :
        model_path = 'RF/'
    else :
        raise ValueError('Invalid estimator')
    for i in range(len(under_sample_folds)) :
        if with_CV==True :
            m = load_model_from_file((path_prefix + model_path + 'with_CV/' + str(i) + '/'), 'metadata.pkl')
        else :
            m = load_model_from_file((path_prefix + model_path + 'without_CV/' + str(i) + '/'), 'metadata.pkl')
        print(str(i)+' -> ', m.predict_k_fold(m.optimal_threshold))

def load_saved_models(estimator_id, with_CV=False) :
    if estimator_id == 'SVM' :
        model_path = 'SVM/'
    elif estimator_id == 'RF' :
        model_path = 'RF/'
    else :
        raise ValueError('Invalid estimator')
    models = []
    for i in range(6) :
        if with_CV ==True :
            models.append(load_model_from_file((path_prefix + model_path + 'with_CV/' + str(i) + '/'), 'metadata.pkl'))
        else :
            models.append(load_model_from_file((path_prefix + model_path + 'without_CV/' + str(i) + '/'), 'metadata.pkl'))
    return models

def execute(estimator_id, with_CV=False) :
    take_input()
    
    if estimator_id == 'SVM' :
        #save_SVM_models(res.data, res.target, with_CV)

        perform_SVM(res.data, res.target, with_CV)
        check_saved_models('SVM', with_CV)

        #models = load_saved_models('SVM', with_CV)
        #for m in models :
            #print(m.predict_k_fold(m.optimal_threshold))
    
    elif estimator_id == 'RF' :
        #save_RF_models(res.data, res.target, with_CV)

        perform_RF(res.data, res.target, with_CV)
        check_saved_models('RF', with_CV)

        #models = load_saved_models('RF', with_CV)
        #for m in models :
            #print(m.predict_k_fold(m.optimal_threshold))

    #print(np.array_equal(n_model.data, m.data))
    #print(np.array_equal(n_model.target, m.target))
