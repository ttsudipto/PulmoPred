from joblib import dump, load
from model import Model
from model import get_under_sampling_folds
from model_metadata import ModelMetadata
import reader
import pickle
import copy
#import numpy as np

path_prefix = '../output/models/'
filenames = ['f1.joblib', 'f2.joblib', 'f3.joblib', 'f4.joblib', 'f5.joblib']

res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=True)
b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
under_sample_folds = get_under_sampling_folds(res.target, 1, 6)

def dump_model_to_file(prefix, filenames, model, metadata_filename = None, threshold = None) :
    estimators = model.estimators;
    if len(filenames) != len(estimators) :
        raise ValueError('No. of estimators and filenames are not matching')
    for i in range(len(estimators)) :
        dump(estimators[i], prefix + filenames[i])
    md = ModelMetadata(model, threshold)
    metadata_file = open(prefix + metadata_filename, 'wb')
    pickle.dump(md, metadata_file)
    metadata_file.close()

def load_model_from_file(prefix, filenames, metadata_filename = None, threshold = None) :
    metadata_file = open(prefix + metadata_filename, 'rb')
    md = pickle.load(metadata_file)
    model = md.get_model()
    for f in filenames :
        model.estimators.append(load(prefix + f))
    return model

def create_SVM_model(data, target, kernel, C, gamma) :
    m = Model('SVM', data, target)
    m.set_estimator_param('kernel', kernel)
    m.set_estimator_param('C', C)
    m.set_estimator_param('gamma', gamma)
    return m

def save_SVM_models(data, target) :
    kernel = 'rbf'
    C = 5
    gamma = 0.0001
    SVM_thresholds = [0.1, 0.2, 0.3, -0.1, 0.2, 0.1]
    model_path = 'SVM/'
    for i in range(len(under_sample_folds)) :
        m = create_SVM_model(data[under_sample_folds[i]], target[under_sample_folds[i]], kernel, C, gamma)
        m.learn_k_fold()
        #print(i)
        dump_model_to_file((path_prefix + model_path + str(i) + '/'), filenames, m, 'metadata.pkl', SVM_thresholds[i])

def perform_SVM(data, target) :
    kernel = 'rbf'
    C = 5
    gamma = 0.0001
    SVM_thresholds = [0.1, 0.2, 0.3, -0.1, 0.2, 0.1]
    model_path = 'SVM/'
    for i in range(len(under_sample_folds)) :
        m = create_SVM_model(data[under_sample_folds[i]], target[under_sample_folds[i]], kernel, C, gamma)
        m.learn_k_fold()
        print(str(i)+' -> ', m.predict_k_fold([SVM_thresholds[i]]))

def check_saved_SVM_models() :
    model_path = 'SVM/'
    for i in range(len(under_sample_folds)) :
        m = load_model_from_file((path_prefix + model_path + str(i) + '/'), filenames, 'metadata.pkl')
        print(str(i)+' -> ', m.predict_k_fold([m.optimal_threshold]))

#save_SVM_models(res.data, res.target)
perform_SVM(res.data, res.target)
check_saved_SVM_models()

#print(np.array_equal(n_model.data, m.data))
#print(np.array_equal(n_model.target, m.target))
