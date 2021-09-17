from joblib import dump, load
from .model import Model
from .model import get_under_sampling_folds
from .model_metadata import ModelMetadata
from ..config import config
from ..config import ml_config as mlc
from . import reader
from pathlib import Path
import pickle
import copy
#import numpy as np

path_prefix = config.get_OUTPUT_PATH() + 'models/'
#filenames = ['f1.joblib', 'f2.joblib', 'f3.joblib', 'f4.joblib', 'f5.joblib']

dataset_name = mlc.get_optimal_dataset() # 'us'

res = b_res = under_sample_folds = None
svm_hyperparams = mlc.get_optimal_hyperparameters(dataset_name, mlc.get_SVM_id())
rf_hyperparams = mlc.get_optimal_hyperparameters(dataset_name, mlc.get_RandomForest_id())
gnb_hyperparams = mlc.get_optimal_hyperparameters(dataset_name, mlc.get_NaiveBayes_id())
mlp_hyperparams = mlc.get_optimal_hyperparameters(dataset_name, mlc.get_MLP_id())

def get_version() :
    import sklearn
    #print(sklearn.__version__)
    return sklearn.__version__+'/'

def take_input() :
    global res
    global b_res
    global under_sample_folds
    res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=True)
    b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
    under_sample_folds = get_under_sampling_folds(res.target, 1, mlc.get_n_US_folds())

def dump_model_to_file(prefix, model, metadata_filename = None, threshold = None) :
    estimators = model.estimators;
    for i in range(len(estimators)) : ## store CV estimators
        filename = 'f' + str(i+1) + '.joblib'
        Path(prefix).mkdir(parents=True, exist_ok=True)
        dump(estimators[i], prefix + filename)
    dump(model.total_estimator, prefix + 'f_total.joblib') ## store total_estimator
    md = ModelMetadata(model, threshold)
    metadata_file = open(prefix + metadata_filename, 'wb')
    pickle.dump(md, metadata_file) ## store metadata
    metadata_file.close()

def load_model_from_file(prefix, metadata_filename, threshold = None) :
    metadata_file = open(prefix + metadata_filename, 'rb') ## load metadata
    md = pickle.load(metadata_file, encoding='latin1')
    model = md.get_model() ## `model` without estimators, so `len(model.estimators)` doesn't work
    for i in range(model.n_folds) : ## load estimators
        filename = 'f' + str(i+1) + '.joblib'
        model.estimators.append(load(prefix + filename))
    model.total_estimator = load(prefix + 'f_total.joblib') ## load total_estimator
    metadata_file.close()
    return model

def create_SVM_model(data, target, kernel, C, gamma) :
    m = Model(mlc.get_SVM_id(), data, target)
    m.set_estimator_param('kernel', kernel)
    m.set_estimator_param('C', C)
    m.set_estimator_param('gamma', gamma)
    return m

def create_RF_model(data, target, n_estimators, max_depth, max_features) :
    m = Model(mlc.get_RandomForest_id(), data, target)
    m.set_estimator_param('n_estimators', n_estimators)
    m.set_estimator_param('max_depth', max_depth)
    m.set_estimator_param('max_features', max_features)
    return m

def create_GNB_model(data, target, smoothing) :
    m = Model(mlc.get_NaiveBayes_id(), data, target)
    m.set_estimator_param('var_smoothing', smoothing)
    return m

def create_MLP_model(data, target, activation, hidden_layer_sizes, learning_rate) :
    m = Model(mlc.get_MLP_id(), data, target)
    m.set_estimator_param('activation', activation)
    m.set_estimator_param('hidden_layer_sizes', hidden_layer_sizes)
    m.set_estimator_param('learning_rate_init', learning_rate)
    return m

def save_SVM_models(data, target) :
    kernel = svm_hyperparams['kernel']
    C = svm_hyperparams['C']
    gamma = svm_hyperparams['gamma']
    SVM_thresholds = svm_hyperparams['thresholds']
    model_path = 'SVM/'
    for i in range(mlc.get_n_US_folds()) :
        m = create_SVM_model(data[under_sample_folds[i]], target[under_sample_folds[i]], kernel, C, gamma)
        m.learn()
        dump_model_to_file((path_prefix + get_version() + model_path + str(i) + '/'), m, 'metadata.pkl', SVM_thresholds[i])

def save_RF_models(data, target) :
    n_estimators = rf_hyperparams['n_estimators']
    max_depth = rf_hyperparams['max_depth']
    max_features = rf_hyperparams['max_features']
    model_path = 'RF/'
    for i in range(mlc.get_n_US_folds()) :
        m = create_RF_model(data[under_sample_folds[i]], target[under_sample_folds[i]], n_estimators, max_depth, max_features)
        m.learn()
        dump_model_to_file((path_prefix + get_version() + model_path + str(i) + '/'), m, 'metadata.pkl')

def save_GNB_models(data, target) :
    smoothing = gnb_hyperparams['smoothing']
    model_path = 'GNB/'
    for i in range(mlc.get_n_US_folds()) :
        m = create_GNB_model(data[under_sample_folds[i]], target[under_sample_folds[i]], smoothing)
        m.learn()
        dump_model_to_file((path_prefix + get_version() + model_path + str(i) + '/'), m, 'metadata.pkl')

def save_MLP_models(data, target) :
    activation = mlp_hyperparams['activation']
    hidden_layer_sizes = mlp_hyperparams['hidden_layer_sizes']
    learning_rate = mlp_hyperparams['learning_rate_init']
    model_path = 'MLP/'
    for i in range(mlc.get_n_US_folds()) :
        m = create_MLP_model(data[under_sample_folds[i]], target[under_sample_folds[i]], activation, hidden_layer_sizes, learning_rate)
        m.learn()
        dump_model_to_file((path_prefix + get_version() + model_path + str(i) + '/'), m, 'metadata.pkl')

def perform_SVM(data, target) :
    kernel = svm_hyperparams['kernel']
    C = svm_hyperparams['C']
    gamma = svm_hyperparams['gamma']
    SVM_thresholds = svm_hyperparams['thresholds']
    for i in range(mlc.get_n_US_folds()) :
        m = create_SVM_model(data[under_sample_folds[i]], target[under_sample_folds[i]], kernel, C, gamma)
        m.learn()
        print(str(i)+' CV -> ', m.predict_k_fold(SVM_thresholds[i]))
        print(str(i)+' Total -> ', m.predict_blind_without_CV(m.data, m.target, SVM_thresholds[i]))

def perform_RF(data, target) :
    n_estimators = rf_hyperparams['n_estimators']
    max_depth = rf_hyperparams['max_depth']
    max_features = rf_hyperparams['max_features']
    for i in range(mlc.get_n_US_folds()) :
        m = create_RF_model(data[under_sample_folds[i]], target[under_sample_folds[i]], n_estimators, max_depth, max_features)
        m.learn()
        print(str(i)+' CV -> ', m.predict_k_fold())
        print(str(i)+' Total -> ', m.predict_blind_without_CV(m.data, m.target))

def perform_GNB(data, target) :
    smoothing = gnb_hyperparams['smoothing']
    for i in range(mlc.get_n_US_folds()) :
        m = create_GNB_model(data[under_sample_folds[i]], target[under_sample_folds[i]], smoothing)
        m.learn()
        print(str(i)+' CV -> ', m.predict_k_fold())
        print(str(i)+' Total -> ', m.predict_blind_without_CV(m.data, m.target))

def perform_MLP(data, target) :
    activation = mlp_hyperparams['activation']
    hidden_layer_sizes = mlp_hyperparams['hidden_layer_sizes']
    learning_rate = mlp_hyperparams['learning_rate_init']
    for i in range(mlc.get_n_US_folds()) :
        m = create_MLP_model(data[under_sample_folds[i]], target[under_sample_folds[i]], activation, hidden_layer_sizes, learning_rate)
        m.learn()
        print(str(i)+' CV -> ', m.predict_k_fold())
        print(str(i)+' Total -> ', m.predict_blind_without_CV(m.data, m.target))

def check_saved_models(estimator_id) :
    if mlc.is_SVM_id(estimator_id) :
        model_path = 'SVM/'
    elif mlc.is_RandomForest_id(estimator_id) :
        model_path = 'RF/'
    elif mlc.is_NaiveBayes_id(estimator_id) :
        model_path = 'GNB/'
    elif mlc.is_MLP_id(estimator_id) :
        model_path = 'MLP/'
    else :
        raise ValueError('Invalid estimator')
    for i in range(mlc.get_n_US_folds()) :
        m = load_model_from_file((path_prefix + get_version() + model_path + str(i) + '/'), 'metadata.pkl')
        print(str(i) + ' CV -> ', m.predict_k_fold(m.optimal_threshold))
        print(str(i) + ' Total -> ', m.predict_blind_without_CV(m.data, m.target, m.optimal_threshold))

def load_saved_models(estimator_id) :
    if mlc.is_SVM_id(estimator_id) :
        model_path = 'SVM/'
    elif mlc.is_RandomForest_id(estimator_id) :
        model_path = 'RF/'
    elif mlc.is_NaiveBayes_id(estimator_id) :
        model_path = 'GNB/'
    elif mlc.is_MLP_id(estimator_id) :
        model_path = 'MLP/'
    else :
        raise ValueError('Invalid estimator')
    models = []
    for i in range(mlc.get_n_US_folds()) :
        models.append(load_model_from_file((path_prefix + get_version() + model_path + str(i) + '/'), 'metadata.pkl'))
    return models

def execute(estimator_id, save=False) :
    take_input()
    
    if mlc.is_SVM_id(estimator_id) :
        print(get_version())
        if save == True :
            save_SVM_models(res.data, res.target)
        perform_SVM(res.data, res.target)
        check_saved_models(estimator_id)
        #models = load_saved_models(estimator_id)
        #for m in models :
            #print(m.predict_k_fold(m.optimal_threshold))
            #print(m.predict_blind_without_CV(m.data, m.target, m.optimal_threshold))
    
    elif mlc.is_RandomForest_id(estimator_id) :
        print(get_version())
        if save == True :
            save_RF_models(res.data, res.target)
        perform_RF(res.data, res.target)
        check_saved_models(estimator_id)
        #models = load_saved_models(estimator_id)
        #for m in models :
            #print(m.predict_k_fold(m.optimal_threshold))
            #print(m.predict_blind_without_CV(m.data, m.target, m.optimal_threshold))
    
    elif mlc.is_NaiveBayes_id(estimator_id) :
        print(get_version())
        if save == True :
            save_GNB_models(res.data, res.target)
        perform_GNB(res.data, res.target)
        check_saved_models(estimator_id)
        #models = load_saved_models(estimator_id)
        #for m in models :
            #print(m.predict_k_fold(m.optimal_threshold))
            #print(m.predict_blind_without_CV(m.data, m.target, m.optimal_threshold))
    
    elif mlc.is_MLP_id(estimator_id) :
        print(get_version())
        if save == True :
            save_MLP_models(res.data, res.target)
        perform_MLP(res.data, res.target)
        check_saved_models(estimator_id)
        #models = load_saved_models(estimator_id)
        #for m in models :
            #print(m.predict_k_fold(m.optimal_threshold))
            #print(m.predict_blind_without_CV(m.data, m.target, m.optimal_threshold))

    #print(np.array_equal(n_model.data, m.data))
    #print(np.array_equal(n_model.target, m.target))

#execute('SVM', save=False)
#execute('RF', save=False)
#execute('GNB', save=False)
#execute('MLP', save=False)
