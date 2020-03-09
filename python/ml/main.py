import warnings
from . import reader
from .model import Model
from .model import get_under_sampling_folds
from . import grid_search as gs
from sklearn.model_selection import StratifiedShuffleSplit
#import density
#import threshold_tuning as tt
#import feature_selection as fs
#import performance
#import neural

#warnings.filterwarnings('ignore')

#-----------------------#
#          TCT          #
#-----------------------#

#res = reader.read_data('TCT_O_NO_main.csv', verbose=True)
#b_res = reader.read_data('TCT_O_NO_blind.csv', verbose=False)

#g = gs.GridSearch()
#g.init_SVM_params(C=[15], gamma=[0.1, 0.01, 0.001, 0.0001, 0.00001], kernel = ['rbf'])
#g.search_with_threshold_SVM(res.data, res.target)

#g = gs.GridSearch()
#g.init_RF_params(estimator_size=[1000], max_depths=[None], max_features=['auto'])
#g.search_with_threshold_RF(res.data, res.target)

#res.split_blind_data()

#-----------------------#
#          PFT          #
#-----------------------#

##res = reader.read_data('PFT_O_NO.csv', verbose=True)
#b_res = reader.read_data('PFT.csv', verbose=False)

#under_sample_folds = get_under_sampling_folds(res.target, 1, 5)

#g = gs.GridSearch()
#g.init_SVM_params(C=[20], gamma=[0.1, 0.01, 0.001, 0.0001, 0.00001], kernel = ['rbf'])
#g.search_with_under_sampling_SVM(res.data, res.target, under_sample_folds)

#g = gs.GridSearch()
#g.init_RF_params(estimator_size=[1000], max_depths=[None], max_features=['auto'])
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds)

#---------------------------#
#          TCT_A_NO         #
#---------------------------#

#res = reader.read_data('TCT_A_NO_main.csv', verbose=True)
#b_res = reader.read_data('TCT_A_NO_blind.csv', verbose=False)

#under_sample_folds = get_under_sampling_folds(res.target, 0, 2)

#g = gs.GridSearch()
#g.init_SVM_params(C=[20], gamma=[0.1, 0.01, 0.001, 0.0001, 0.00001], kernel = ['rbf'])
#g.search_with_under_sampling_SVM(res.data, res.target, under_sample_folds)
#g.search_with_threshold_SVM(res.data, res.target)

#g = gs.GridSearch()
#g.init_RF_params(estimator_size=[1000], max_depths=[None], max_features=['auto'])
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds)

#res.split_blind_data()

#-----------------------------#
#          PFT Common         #
#-----------------------------#

res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=True)
b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)

#under_sample_folds = get_under_sampling_folds(res.target, 1, 6)

#g = gs.GridSearch()
#g.init_SVM_params(C=[20], gamma=[0.1, 0.01, 0.001, 0.0001, 0.00001], kernel = ['rbf'])
#g.search_with_under_sampling_SVM(res.data, res.target, under_sample_folds)
#g = gs.GridSearch()
#g.init_SVM_params(C=[1], gamma=[0.0001], kernel = ['rbf'])
#g.search_with_under_sampling_SVM(res.data, res.target, under_sample_folds, b_res.data, b_res.target)

#g = gs.GridSearch()
#g.init_RF_params(estimator_size=[30], max_depths=[4, 8, 16, None], max_features=['auto', 0.3, 0.4, 0.5, 0.6, 0.7, None])
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, with_threshold=True)
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, with_threshold=False)
#g = gs.GridSearch()
#g.init_RF_params(estimator_size=[30], max_depths=[4, 8, 16, None], max_features=['auto', 0.3, 0.4, 0.5, 0.6, 0.7, None])
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=True)
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=False)

#g = gs.GridSearch()
#g.init_GNB_params(smoothing=[20, 15, 10, 5, 2, 1, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
#g.search_with_GNB(res.data, res.target, with_threshold=True)
#g.search_with_GNB(res.data, res.target, with_threshold=False)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, with_threshold=True)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, with_threshold=False)
#g = gs.GridSearch()
#g.init_GNB_params(smoothing=[20, 15, 10, 5, 2, 1, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
#g.search_with_GNB(res.data, res.target, b_data=b_res.data, b_target=b_res.target, with_threshold=True)
#g.search_with_GNB(res.data, res.target, b_data=b_res.data, b_target=b_res.target, with_threshold=False)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=True)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=False)

#-----------------------------#
#          TCT Common         #
#-----------------------------#

#res = reader.read_data('TCT_O_NO_uncommon.csv', verbose=True)
#b_res = reader.read_data('TCT_O_NO_common.csv', verbose=False)
#print(res.target, b_res.target)

#under_sample_folds = get_under_sampling_folds(res.target, 1, 2)

#g = gs.GridSearch()
#g.init_SVM_params(C=[20], gamma=[0.1, 0.01, 0.001, 0.0001, 0.00001], kernel = ['rbf'])
#g.search_with_threshold_SVM(res.data, res.target)
#g.search_with_under_sampling_SVM(res.data, res.target, under_sample_folds)
#g = gs.GridSearch()
#g.init_SVM_params(C=[0.5], gamma=[0.1, 0.01, 0.001, 0.0001, 0.00001], kernel = ['rbf'])
#g.search_with_threshold_SVM(res.data, res.target, b_data=b_res.data, b_target=b_res.target)
#g.search_with_under_sampling_SVM(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target)

##g = gs.GridSearch()
#g.init_RF_params(estimator_size=[10], max_depths=[None], max_features=['auto', 0.3, 0.4, 0.5, 0.6, 0.7, None])
#g.search_with_RF(res.data, res.target, with_threshold=True)
#g.search_with_RF(res.data, res.target, with_threshold=False)
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, with_threshold=True)
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, with_threshold=False)
#g = gs.GridSearch()
#g.init_RF_params(estimator_size=[30], max_depths=[4, 8, 16, None], max_features=['auto', 0.3, 0.4, 0.5, 0.6, 0.7, None])
#g.search_with_RF(res.data, res.target, b_data=b_res.data, b_target=b_res.target, with_threshold=True)
#g.search_with_RF(res.data, res.target, b_data=b_res.data, b_target=b_res.target, with_threshold=False)
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=True)
#g.search_with_under_sampling_RF(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=False)

#g = gs.GridSearch()
#g.init_GNB_params(smoothing=[20, 15, 10, 5, 2, 1, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
#g.search_with_GNB(res.data, res.target, with_threshold=False)
#g.search_with_GNB(res.data, res.target, with_threshold=True)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, with_threshold=True)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, with_threshold=False)
#g = gs.GridSearch()
#g.init_GNB_params(smoothing=[20, 15, 10, 5, 2, 1, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
#g.search_with_GNB(res.data, res.target, b_data=b_res.data, b_target=b_res.target, with_threshold=True)
#g.search_with_GNB(res.data, res.target, b_data=b_res.data, b_target=b_res.target, with_threshold=False)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=True)
#g.search_with_under_sampling_GNB(res.data, res.target, under_sample_folds, b_data=b_res.data, b_target=b_res.target, with_threshold=False)

#shuffle_split = StratifiedShuffleSplit(n_splits = 1, train_size=500, random_state=42)
#for (data_index, foo) in shuffle_split.split(res.data, res.target) :
    #data_X = res.data[data_index]
    #data_y = res.target[data_index]

#print(data_X.shape)
#print(data_y.shape)

#m = Model('RF', res.data, res.target)
#m.learn_k_fold()
#thresholds = gs.gen_thresholds(-1, 1.001, 0.1)
#for t in thresholds :
    #print(m.predict_k_fold(t))

#res = reader.read_data('input/PFT_n_blind.csv', verbose=False)
#b_res = reader.read_data('input/PFT_blind.csv', verbose=False)

#print('1. Grid search')
#print('2. Density of SVM decision scores')
#print('3. Threshold tuning')
#print('4. Feature selection')
#print('5. Threshold dependent performance')
#print('6. Neural Network')
#ch = input('Enter choice : ')
#if (ch == 1) :
    #grid_search.execute(res.data, res.target)
#elif (ch == 2) :
    #print('1. Overall density')
    #print('2. Negative samples density')
    #print('3. Positive samples density')
    #ch1 = input('Enter choice : ')
    #density.execute(res.data, res.target, ch1)
#elif (ch == 3) :
    #tt.execute(res.data, res.target, verbose=True)
#elif (ch == 4) :
    #print('1. No feature selection')
    #print('2. CFS Subset Evaluation')
    #print('3. Wrapper Subset Evaluation')
    #print('4. Correlation Attribute Evaluation')
    #print('5. Classifier Attribute Evatuation')
    #print('6. Gain Ratio Attribute Evaluation')
    #print('7. Information Gain Attribute Evaluation')
    #ch1 = input('Enter choice : ')
    #fs.execute(res, ch1)
#elif (ch == 5) :
    #print('1. Training-testing dataset with 10-fold CV')
    #print('2. Blind dataset')
    #ch1 = input('Enter choice : ')
    #print('1. No feature selection')
    #print('2. CFS Subset Evaluation')
    #print('3. Wrapper Subset Evaluation')
    #print('4. Correlation Attribute Evaluation')
    #print('5. Classifier Attribute Evatuation')
    #print('6. Gain Ratio Attribute Evaluation')
    #print('7. Information Gain Attribute Evaluation')
    #ch2 = input('Enter choice : ')
    #if (ch1 == 1) :
        #performance.performance_with_CV(res, ch2)
    #elif (ch1 == 2) :
        #performance.performance_on_blind_dataset(res, b_res, ch2)
    #else :
        #print('Wrong choice')
#elif (ch == 6) :
    #neural.execute(res.data, res.target)
#else :
    #print('Wrong choice')
