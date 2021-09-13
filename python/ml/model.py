from ..config import ml_config as mlc
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLP
from statistics import mean
import numpy as np
import csv
import copy

class Model :
    """Class encapsulating an estimator class that implements a learning model of Scikit-learn and the data.

    It contains methods for training and testing the learning model with 
    k-fold cross validation. It also contains method to predict blind dataset.
    """
    
    def __init__(self, e_id, X, y, k=5, scale=False, do_shuffle=True):
        """Constructor"""
        
        self.estimators = []
        self.total_estimator = None
        self.train_indices = []
        self.test_indices = []
        self.estimator_id = e_id
        if do_shuffle == True :
            self.data = shuffle(X, random_state=mlc.get_random_state())
            self.target = shuffle(y, random_state=mlc.get_random_state())
        else :
            self.data = X
            self.target = y
        self.n_folds = k
        self.optimal_threshold = None
        self.scale = scale
        self.scaler = None
        if self.scale == True :
            #self.scaler = StandardScaler().fit(self.data)
            self.scaler = Normalizer().fit(self.data)
            self.data = self.scaler.transform(self.data)
        self.init_estimator_params()
        self.split_CV_folds()

    def init_estimator_params(self) :
        #-----------------------#
        #       SVM params      #
        #-----------------------#
        self.SVM_params = dict()
        self.SVM_params['class_weight'] = 'balanced'
        self.SVM_params['max_iter'] = -1
        self.SVM_params['random_state'] = 42
        ### Optimazable params
        self.SVM_params['kernel'] = 'rbf'
        self.SVM_params['C'] = 5
        self.SVM_params['gamma'] = 0.00001
        self.SVM_params['degree'] = 2
        
        #-----------------------#
        #       RF params       #
        #-----------------------#
        self.RF_params = dict()
        self.RF_params['class_weight'] = 'balanced_subsample'
        self.RF_params['bootstrap'] = True
        self.RF_params['n_jobs'] = -1
        self.RF_params['random_state'] = 42
        self.RF_params['criterion'] = 'gini' # also, 'entropy' -> info. gain
        ### Speed up params
        self.RF_params['max_depth'] = None # also, int
        self.RF_params['min_impurity_decrease'] = 0. # also, float
        ### Optimizable params
        self.RF_params['n_estimators'] = 100
        self.RF_params['max_features'] = 'auto' # also, int -> # , float -> fraction, log2, None -> all features
        #self.RF_params['max_samples'] = None
        
        #------------------------#
        #       GNB params       #
        #------------------------#
        
        self.GNB_params = dict()
        self.GNB_params['var_smoothing'] = 1e-9
        
        #------------------------#
        #       MLP params       #
        #------------------------#
        
        self.MLP_params = dict()
        self.MLP_params['learning_rate'] = 'constant' # constant, adaptive
        self.MLP_params['solver'] = 'adam'
        self.MLP_params['random_state'] = 42
        self.MLP_params['max_iter'] = 10000
        self.MLP_params['alpha'] = 0.0001
        #self.MLP_params['batch_size'] = 128
        #self.MLP_params['early_stopping'] = True
        #self.MLP_params['activation'] = a # logistic, tanh, relu
        #self['hidden_layer_sizes'] = hls
        #self.MLP_params['learning_rate_init'] = lri
        #self.MLP_params['tol'] = 1e-5

    def set_estimator_param(self, param, value) :
        if mlc.is_SVM_id(self.estimator_id) : ## SVM
            self.SVM_params[param] = value
        elif mlc.is_RandomForest_id(self.estimator_id) : ## RF
            self.RF_params[param] = value
        elif mlc.is_NaiveBayes_id(self.estimator_id) : ## GNB
            self.GNB_params[param] = value
        elif mlc.is_MLP_id(self.estimator_id) : ## MLP
            self.MLP_params[param] = value

    def create_estimator(self) :
        """Method that instantiates an estimator"""
        
        estimator = None
        if mlc.is_SVM_id(self.estimator_id) : ## SVM
            estimator = SVC()
            estimator.set_params(**self.SVM_params)
        elif mlc.is_RandomForest_id(self.estimator_id) : ## RF
            estimator = RF()
            estimator.set_params(**self.RF_params)
        elif mlc.is_NaiveBayes_id(self.estimator_id) : ## GNB
            estimator = GNB()
            estimator.set_params(**self.GNB_params)
        if mlc.is_MLP_id(self.estimator_id) : ## MLP
            estimator = MLP()
            estimator.set_params(**self.MLP_params)
        return estimator

    def get_decision_function(self, estimator) :
        if mlc.is_SVM_id(self.estimator_id) : ## SVM
            return estimator.decision_function
        elif mlc.is_RandomForest_id(self.estimator_id) : ## RF
            return estimator.predict_proba
        elif mlc.is_NaiveBayes_id(self.estimator_id) : ## GNB
            return estimator.predict_proba
        elif mlc.is_MLP_id(self.estimator_id) : ## MLP
            return estimator.predict_proba

    def split_CV_folds(self) :
        if self.n_folds == 1 :
            self.train_indices = [range(self.data.shape[0])]
            self.test_indices = [range(self.data.shape[0])]
        else :
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=mlc.get_random_state())
            for train_index, test_index in skf.split(self.data, self.target) :
                self.train_indices.append(train_index)
                self.test_indices.append(test_index)

    def get_decision_score(self, estimator, X_test) :
        decision_function = self.get_decision_function(estimator)
        return decision_function(X_test)

    def predict(self, estimator, X_test, threshold=None) :
        if threshold==None :
            y_pred = estimator.predict(X_test)
        else :
            #des = estimator.predict_proba(X_test)
            #des = estimator.decision_function(X_test)
            decision_function = self.get_decision_function(estimator)
            des = decision_function(X_test)
            if mlc.is_SVM_id(self.estimator_id) : ## SVM
                #print(des)
                y_pred = []
                for val in des :
                    if val > threshold :
                        y_pred.append(1)
                    else :
                        y_pred.append(0)
            elif mlc.is_RandomForest_id(self.estimator_id) : ## RF
                #print(des)
                y_pred = []
                for val in des :
                    if val[1] > threshold :
                        y_pred.append(1)
                    else :
                        y_pred.append(0)
            elif mlc.is_NaiveBayes_id(self.estimator_id) : ## GNB
                #print(des)
                y_pred = []
                for val in des :
                    if val[1] > threshold :
                        y_pred.append(1)
                    else :
                        y_pred.append(0)
            elif mlc.is_MLP_id(self.estimator_id) : ## MLP
                #print(des)
                y_pred = []
                for val in des :
                    if val[1] > threshold :
                        y_pred.append(1)
                    else :
                        y_pred.append(0)
        return y_pred

    def learn_without_CV(self) :
        estimator = self.create_estimator()
        estimator.fit(self.data, self.target)
        self.total_estimator = copy.deepcopy(estimator)

    def learn_k_fold(self) :
        self.estimators = []
        #skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=mlc.get_random_state())
        for f in range(self.n_folds) :
            estimator = self.create_estimator()
            estimator.fit(self.data[self.train_indices[f]], self.target[self.train_indices[f]])
            self.estimators.append(copy.deepcopy(estimator))
    
    def learn(self) :
        self.learn_without_CV()
        self.learn_k_fold()
    
    def predict_k_fold(self, threshold=None) :
        sensitivities = []
        specificities = []
        accuracies = []
        for f in range(self.n_folds) :
            y_test = self.target[self.test_indices[f]]
            y_pred = self.predict(self.estimators[f], self.data[self.test_indices[f]], threshold)
            sensitivities.append(recall_score(y_test, y_pred))
            accuracies.append(accuracy_score(y_test, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificities.append((tn/1.0) / (tn+fp))
        #print((mean(accuracies), mean(sensitivities), mean(specificities)))
        return (mean(accuracies), mean(sensitivities), mean(specificities))
    
    def predict_blind_data(self, b_data, b_target, threshold=None) :
        """Method to perform prediction of blind dataset"""
        
        if self.scale == True :
            b_data = self.scaler.transform(b_data)
        accuracies = []
        sensitivities = []
        specificities = []
        for f in range(self.n_folds) :
            y_pred = self.predict(self.estimators[f], b_data, threshold)
            sensitivities.append(recall_score(b_target, y_pred))
            accuracies.append(accuracy_score(b_target, y_pred))
            tn, fp, fn, tp = confusion_matrix(b_target, y_pred).ravel()
            specificities.append((tn/1.0) / (tn+fp))
        #print((mean(accuracies), mean(sensitivities), mean(specificities)))
        return (mean(accuracies), mean(sensitivities), mean(specificities))

    def predict_blind_without_CV(self, b_data, b_target, threshold=None) :
        if self.scale == True :
            b_data = self.scaler.transform(b_data)
        y_pred = self.predict(self.total_estimator, b_data, threshold)
        sensitivity = recall_score(b_target, y_pred)
        accuracy = accuracy_score(b_target, y_pred)
        tn, fp, fn, tp = confusion_matrix(b_target, y_pred).ravel()
        specificity = (tn/1.0) / (tn+fp)
        return accuracy, sensitivity, specificity

    def write_to_csv(self, filename, thresholds, accuracies, sensitivities, specificities, PPVs) :
        """Method to write the results into a CSV file"""
        
        fields = ['Threshold', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV']
        csvfile = open(filename, 'w')
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in range(len(thresholds)) :
            row = {fields[0]:thresholds[i], fields[1]:accuracies[i], fields[2]:sensitivities[i], fields[3]:specificities[i], fields[4]:PPVs[i]}
            writer.writerow(row)
        csvfile.close()

def get_under_sampling_folds(y, sampling_class, n_folds) :
    sampling_class_folds = []
    sampling_class_indices = []
    other_indices = []
    for i in range(y.shape[0]) :
        if y[i] == sampling_class :
            sampling_class_indices.append(i)
        else :
            other_indices.append(i)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=mlc.get_random_state())
    for (foo, test_index) in kf.split(sampling_class_indices) :
        one_fold = []
        for i in test_index :
            one_fold.append(sampling_class_indices[i])
        sampling_class_folds.append(one_fold)
    for i in range(n_folds) :
        sampling_class_folds[i] = sampling_class_folds[i] + other_indices
    #print(len(sampling_class_indices))
    
    return sampling_class_folds
    
