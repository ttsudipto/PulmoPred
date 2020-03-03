from joblib import dump, load
from model import Model
from model_metadata import ModelMetadata
import reader
import pickle
#import numpy as np
import copy

d_data = d_target = l_data = l_target = lm_data = lm_target = nm_data = nm_target = None

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

path_prefix = 'output/models/'
filenames = ['f1.joblib', 'f2.joblib', 'f3.joblib', 'f4.joblib', 'f5.joblib']
res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=True)
b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)

m = Model('SVM', res.data, res.target)
m.learn_k_fold()

dump_model_to_file(path_prefix, filenames, m, 'metadata.pkl')
print(m.predict_k_fold())

n_model = load_model_from_file(path_prefix, filenames, 'metadata.pkl')
print(n_model.predict_k_fold())

#print(np.array_equal(n_model.data, m.data))
#print(np.array_equal(n_model.target, m.target))
