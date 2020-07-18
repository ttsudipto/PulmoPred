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
pft_RF_params = {
    'n_estimators' : pft_rf_hyperparameters['n_estimators'],
    'max_depth' : pft_rf_hyperparameters['max_depth'],
    'max_features' : pft_rf_hyperparameters['max_features']
    }
pft_GNB_params = {
    'var_smoothing' : pft_nb_hyperparameters['smoothing']
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

def execute(modelId:str, diagnosis:str, checkBlind=False) :
    res = reader.read_data('PFT_O_NO_uncommon.csv', verbose=False)
    b_res = reader.read_data('PFT_O_NO_common.csv', verbose=False)
    under_sample_folds = get_under_sampling_folds(res.target, 1, mlc.get_n_US_folds('PFT'))
    diagnoses = readDiagnoses(name)
    blindDiagnoses = readDiagnoses(bName)
    accuracies = []
    
    for i in range(len(under_sample_folds)) :
        if mlc.is_SVM_id(modelId) :
            m = Model(modelId, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_SVM_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_RandomForest_id(modelId) :
            m = Model(modelId, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_RF_params.items() :
                m.set_estimator_param(p, v)
        elif mlc.is_NaiveBayes_id(modelId) :
            m = Model(modelId, res.data[under_sample_folds[i]], res.target[under_sample_folds[i]])
            for p, v in pft_GNB_params.items() :
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
                acc = dp.predictDiagnosisKFold(diagnosis, pft_SVM_thresholds[i])
            else :
                acc = dp.predictDiagnosisKFold(diagnosis)
            #print(str(acc)+'*'+str(dp.countElementsInList(dp.diagnoses, diagnosis))+' = '+str(acc*dp.countElementsInList(dp.diagnoses, diagnosis)))
        else :  ## Blind
            if mlc.is_SVM_id(modelId) :
                acc = dp.predictDiagnosisBlind(b_res.data, b_res.target, diagnosis, pft_SVM_thresholds[i])
            else :
                acc = dp.predictDiagnosisBlind(b_res.data, b_res.target, diagnosis)
            #print(str(acc)+'*'+str(dp.countElementsInList(dp.blindDiagnoses, diagnosis))+' = '+str(acc*dp.countElementsInList(dp.blindDiagnoses, diagnosis)))
        accuracies.append(acc)
    print(accuracies)
    print(diagnosis, mean(accuracies), stdev(accuracies))

#print('--------SVM---------')
#execute('SVM', 'A')
#execute('SVM', 'C')
#execute('SVM', 'DPLD')
#print('--------RF---------')
#execute('RF', 'A')
#execute('RF', 'C')
#execute('RF', 'DPLD')
#print('--------GNB---------')
#execute('GNB', 'A')
#execute('GNB', 'C')
#execute('GNB', 'DPLD')
#print('--------SVM Blind---------')
#execute('SVM', 'A', checkBlind=True)
#execute('SVM', 'C', checkBlind=True)
#execute('SVM', 'DPLD', checkBlind=True)
#print('--------RF Blind---------')
#execute('RF', 'A', checkBlind=True)
#execute('RF', 'C', checkBlind=True)
#execute('RF', 'DPLD', checkBlind=True)
#print('--------GNB Blind---------')
#execute('GNB', 'A', checkBlind=True)
#execute('GNB', 'C', checkBlind=True)
#execute('GNB', 'DPLD', checkBlind=True)
