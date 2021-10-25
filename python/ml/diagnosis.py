from .model import Model, get_under_sampling_folds
from .resource import path_prefix
from ..config import ml_config as mlc
from . import reader
from typing import List
from statistics import mean, stdev
from sklearn.utils import shuffle
import numpy as np
from csv import DictReader

name = 'PFT_O_NO_uncommon.csv'
bName = 'PFT_O_NO_common.csv'
#modelId = 'SVM'
#diagnosis = 'DPLD'

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

def readDiagnoses(filename) :
    csvfile = open(path_prefix + filename)
    reader = DictReader(csvfile)
    diagnoses = []
    for row in reader :
        diagnoses.append(row['Diagnosis'])
    csvfile.close()
    return diagnoses

class DiagnosisPredictor :
    
    def __init__(self, m:Model, d:List[str], bd:List[str]=None) :
        self.model = m
        self.diagnoses = shuffle(d, random_state=mlc.get_random_state())
        self.blindDiagnoses = bd
        self.diagnosisToTargetMap = {'A':1, 'C':1, 'DPLD':0}
    
    def countElementsInList(self, l:List[str], diagnosis:str) :
        count = 0
        for item in l :
            if item == diagnosis :
                count = count + 1
        return count
    
    def predictDiagnosis(self, currDiagnoses:List[str], y_pred:List[int], diagnosis:str) :
        #print(y_pred)
        #currDiagnoses = [self.diagnoses[x] for x in indices]
        correctDiagnosisCount = 0
        for i in range(len(y_pred)) :
            if currDiagnoses[i] == diagnosis :
                #print(currDiagnoses[i], self.diagnosisToTargetMap[currDiagnoses[i]], y_pred[i])
                if y_pred[i] == self.diagnosisToTargetMap[currDiagnoses[i]] :
                    correctDiagnosisCount = correctDiagnosisCount + 1
        #print(currDiagnoses)
        #print(correctDiagnosisCount, self.countElementsInList(currDiagnoses, diagnosis))
        
        return (float(correctDiagnosisCount)/self.countElementsInList(currDiagnoses, diagnosis))
    
    def predictDiagnosisKFold(self, diagnosis:str, threshold=None) :
        #for i in range(len(self.diagnoses)) :
            #print(self.diagnoses[i], self.model.target[i])
        #print('---------------------------------------------')
        accuracies = []
        for f in range(self.model.n_folds) :
            y_test = self.model.target[self.model.test_indices[f]]
            y_pred = self.model.predict(self.model.estimators[f], self.model.data[self.model.test_indices[f]], threshold)
            #print(y_pred)
            currDiagnoses = [self.diagnoses[x] for x in self.model.test_indices[f]]
            accuracies.append(self.predictDiagnosis(currDiagnoses, y_pred, diagnosis))
            #for i in range(len(y_pred)) :
                #print([self.diagnoses[x] for x in self.model.test_indices[f]][i], y_test[i], y_pred[i])
            #print('---------------------------------------------')
        return mean(accuracies)
    
    def predictDiagnosisBlind(self, bData, bTarget, diagnosis:str, threshold=None) :
        accuracies = []
        for f in range(self.model.n_folds) :
            y_test = self.model.target[self.model.test_indices[f]]
            y_pred = self.model.predict(self.model.estimators[f], bData, threshold)
            #print(y_pred)
            accuracies.append(self.predictDiagnosis(self.blindDiagnoses, y_pred, diagnosis))
        return mean(accuracies)

def execute_no_us(modelId:str, diagnosis:str, checkBlind=False) :
    res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=False)
    b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
    diagnoses = readDiagnoses(name)
    blindDiagnoses = readDiagnoses(bName)
    
    if mlc.is_SVM_id(modelId) :
        m = Model(modelId, res.data, res.target)
        for p, v in no_us_SVM_params.items() :
            m.set_estimator_param(p, v)
    elif mlc.is_RandomForest_id(modelId) :
        m = Model(modelId, res.data, res.target)
        for p, v in no_us_RF_params.items() :
            m.set_estimator_param(p, v)
    elif mlc.is_NaiveBayes_id(modelId) :
        m = Model(modelId, res.data, res.target)
        for p, v in no_us_GNB_params.items() :
            m.set_estimator_param(p, v)
    elif mlc.is_MLP_id(modelId) :
        m = Model(modelId, res.data, res.target)
        for p, v in no_us_MLP_params.items() :
            m.set_estimator_param(p, v)
        
    m.learn()
    dp = DiagnosisPredictor(m, diagnoses, blindDiagnoses)
    #for x in range(len(under_sample_diagnoses)) :
        #print('+', under_sample_diagnoses[x], res.target[under_sample_folds[i]][x])
    #print('+++++++++++++++++++++++++++++++++++++++++++')
    
    if checkBlind == False :  ## Cross validation
        if mlc.is_SVM_id(modelId) :
            acc = dp.predictDiagnosisKFold(diagnosis, no_us_SVM_threshold)
        else :
            acc = dp.predictDiagnosisKFold(diagnosis)
        #print(str(acc)+'*'+str(dp.countElementsInList(dp.diagnoses, diagnosis))+' = '+str(acc*dp.countElementsInList(dp.diagnoses, diagnosis)))
    else :  ## Blind
        if mlc.is_SVM_id(modelId) :
            acc = dp.predictDiagnosisBlind(b_res.data, b_res.target, diagnosis, no_us_SVM_threshold)
        else :
            acc = dp.predictDiagnosisBlind(b_res.data, b_res.target, diagnosis)
        #print(str(acc)+'*'+str(dp.countElementsInList(dp.blindDiagnoses, diagnosis))+' = '+str(acc*dp.countElementsInList(dp.blindDiagnoses, diagnosis)))
    print(diagnosis, acc)

def execute_us(modelId:str, diagnosis:str, checkBlind=False) :
    res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=False)
    b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
    under_sample_folds = get_under_sampling_folds(res.target, 1, mlc.get_n_US_folds())
    diagnoses = readDiagnoses(name)
    blindDiagnoses = readDiagnoses(bName)
    accuracies = []
    
    for i in range(len(under_sample_folds)) :
        if mlc.is_SVM_id(modelId) :
            m = Model(modelId, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_SVM_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_RandomForest_id(modelId) :
            m = Model(modelId, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_RF_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_NaiveBayes_id(modelId) :
            m = Model(modelId, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_GNB_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_MLP_id(modelId) :
            m = Model(modelId, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in us_MLP_params.items() :
                m.set_estimator_param(p, v)
        
        m.learn()
        under_sample_diagnoses = [diagnoses[x] for x in under_sample_folds[i]]
        #print(under_sample_diagnoses)
        dp = DiagnosisPredictor(m, under_sample_diagnoses, blindDiagnoses)
        #for x in range(len(under_sample_diagnoses)) :
            #print('+', under_sample_diagnoses[x], res.target[under_sample_folds[i]][x])
        #print('+++++++++++++++++++++++++++++++++++++++++++')
        
        if checkBlind == False :  ## Cross validation
            if mlc.is_SVM_id(modelId) :
                acc = dp.predictDiagnosisKFold(diagnosis, us_SVM_thresholds[i])
            else :
                acc = dp.predictDiagnosisKFold(diagnosis)
            #print(str(acc)+'*'+str(dp.countElementsInList(dp.diagnoses, diagnosis))+' = '+str(acc*dp.countElementsInList(dp.diagnoses, diagnosis)))
        else :  ## Blind
            if mlc.is_SVM_id(modelId) :
                acc = dp.predictDiagnosisBlind(b_res.data, b_res.target, diagnosis, us_SVM_thresholds[i])
            else :
                acc = dp.predictDiagnosisBlind(b_res.data, b_res.target, diagnosis)
            #print(str(acc)+'*'+str(dp.countElementsInList(dp.blindDiagnoses, diagnosis))+' = '+str(acc*dp.countElementsInList(dp.blindDiagnoses, diagnosis)))
        accuracies.append(acc)
    #print(accuracies)
    print(diagnosis, mean(accuracies), stdev(accuracies))

print('######################')
print('#         US         #')
print('######################')

print('--------SVM---------')
execute_us('SVM', 'A')
execute_us('SVM', 'C')
execute_us('SVM', 'DPLD')
print('--------RF---------')
execute_us('RF', 'A')
execute_us('RF', 'C')
execute_us('RF', 'DPLD')
print('--------GNB---------')
execute_us('GNB', 'A')
execute_us('GNB', 'C')
execute_us('GNB', 'DPLD')
print('--------MLP---------')
execute_us('MLP', 'A')
execute_us('MLP', 'C')
execute_us('MLP', 'DPLD')
print('--------SVM Blind---------')
execute_us('SVM', 'A', checkBlind=True)
execute_us('SVM', 'C', checkBlind=True)
execute_us('SVM', 'DPLD', checkBlind=True)
print('--------RF Blind---------')
execute_us('RF', 'A', checkBlind=True)
execute_us('RF', 'C', checkBlind=True)
execute_us('RF', 'DPLD', checkBlind=True)
print('--------GNB Blind---------')
execute_us('GNB', 'A', checkBlind=True)
execute_us('GNB', 'C', checkBlind=True)
execute_us('GNB', 'DPLD', checkBlind=True)
print('--------MLP Blind---------')
execute_us('MLP', 'A', checkBlind=True)
execute_us('MLP', 'C', checkBlind=True)
execute_us('MLP', 'DPLD', checkBlind=True)

print('#####################')
print('#       No-US       #')
print('#####################')

print('--------SVM---------')
execute_no_us('SVM', 'A')
execute_no_us('SVM', 'C')
execute_no_us('SVM', 'DPLD')
print('--------RF---------')
execute_no_us('RF', 'A')
execute_no_us('RF', 'C')
execute_no_us('RF', 'DPLD')
print('--------GNB---------')
execute_no_us('GNB', 'A')
execute_no_us('GNB', 'C')
execute_no_us('GNB', 'DPLD')
print('--------MLP---------')
execute_no_us('MLP', 'A')
execute_no_us('MLP', 'C')
execute_no_us('MLP', 'DPLD')
print('--------SVM Blind---------')
execute_no_us('SVM', 'A', checkBlind=True)
execute_no_us('SVM', 'C', checkBlind=True)
execute_no_us('SVM', 'DPLD', checkBlind=True)
print('--------RF Blind---------')
execute_no_us('RF', 'A', checkBlind=True)
execute_no_us('RF', 'C', checkBlind=True)
execute_no_us('RF', 'DPLD', checkBlind=True)
print('--------GNB Blind---------')
execute_no_us('GNB', 'A', checkBlind=True)
execute_no_us('GNB', 'C', checkBlind=True)
execute_no_us('GNB', 'DPLD', checkBlind=True)
print('--------MLP Blind---------')
execute_no_us('MLP', 'A', checkBlind=True)
execute_no_us('MLP', 'C', checkBlind=True)
execute_no_us('MLP', 'DPLD', checkBlind=True)
