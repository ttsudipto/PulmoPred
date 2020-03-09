from sklearn.model_selection import GridSearchCV
from .model import Model
from statistics import mean, stdev
import numpy as np
import copy

def gen_thresholds(start, stop, step) :
    t = start
    thresholds = []
    while t<=stop :
        thresholds.append(t)
        t = t + step
    return thresholds

class GridSearch :
    """
    Class that encapsulates all properties and methods for 
    hyperparameter optimization by searching a grid of hyperparameters.
    """
    
    def __init__(self) :
        """Constructor"""
        
        self.best_C = None
        self.best_gamma = None
        self.best_kernel = None
        self.best_estimator_size = None
    
    def init_SVM_params(self, C=[1], gamma=[0.01], kernel=['rbf']) :
        self.Cs = C
        self.gammas = gamma
        self.kernels = kernel
    
    def init_RF_params(self, estimator_size=[100], max_depths = [None], max_features=['auto']) :
        self.estimator_sizes = estimator_size
        self.max_depths = max_depths
        self.max_features = max_features
    
    def init_GNB_params(self, smoothing=[1e-9]) :
        self.smoothings = smoothing
        
    #def search(self, model) :
        #estimator = model.create_estimator()
        #data = model.data
        #target = model.target
        #parameters = {'kernel' : self.kernels, 'C' : self.Cs, 'gamma' : self.gammas}
        #gs = GridSearchCV(estimator, parameters, cv=10, verbose=10, return_train_score=True, n_jobs=-1)
        #gs.fit(data, target)
        #best_params = gs.best_params_
        #best_score = gs.best_score_
        #print(gs.cv_results_)
        #print(best_params)
        #print(best_score)
        
    def predict_blind_data(self, model, b_data, b_target, with_threshold=True, thresholds=[0.1]) :
        if with_threshold == True :
            for t in thresholds :
                result = model.predict_blind_data(b_data, b_target, t)
                print('blind', t, result[0], result[1], result[2])
        else :
            result = model.predict_blind_data(b_data,b_target)
            print('blind', result[0], result[1], result[2])
    
    def predict_under_sampling_blind_data(self, models, b_data, b_target, with_threshold=True, thresholds=[0.1]) : # Same threshold to all models
        if with_threshold == True :
            for t in thresholds :
                b_accuracy_sum = b_sensitivity_sum = b_specificity_sum = 0.
                for m in models :
                    b_result = m.predict_blind_data(b_data, b_target, t)
                    b_accuracy_sum = b_accuracy_sum + b_result[0]
                    b_sensitivity_sum = b_sensitivity_sum + b_result[1]
                    b_specificity_sum = b_specificity_sum + b_result[2]
                print('blind', round(t,2), b_accuracy_sum/len(models), b_sensitivity_sum/len(models), b_specificity_sum/len(models))
        else :
            b_accuracy_sum = b_sensitivity_sum = b_specificity_sum = 0.
            for m in models :
                b_result = m.predict_blind_data(b_data, b_target)
                b_accuracy_sum = b_accuracy_sum + b_result[0]
                b_sensitivity_sum = b_sensitivity_sum + b_result[1]
                b_specificity_sum = b_specificity_sum + b_result[2]
            print('blind', b_accuracy_sum/len(models), b_sensitivity_sum/len(models), b_specificity_sum/len(models))

    def predict_under_sampling_blind_data_LT(self, models, b_data, b_target, thresholds) :  # LT -> Learned Threshold : Different thresholds for different models
        b_accuracy_sum = b_sensitivity_sum = b_specificity_sum = 0.
        for i in range(len(models)) :
            m = models[i]
            b_result = m.predict_blind_data(b_data, b_target, thresholds[i])
            b_accuracy_sum = b_accuracy_sum + b_result[0]
            b_sensitivity_sum = b_sensitivity_sum + b_result[1]
            b_specificity_sum = b_specificity_sum + b_result[2]
        print('blind', 'Learned', b_accuracy_sum/len(models), b_sensitivity_sum/len(models), b_specificity_sum/len(models))
        
    def search_with_threshold_SVM(self, data, target, b_data = None, b_target = None) :
        """Method to perform grid search"""
        
        thresholds = gen_thresholds(-1, 1.001, 0.1)
        m = Model('SVM', data, target)
        print('Kernel', 'C', 'Gamma', 'Threshold', 'Accuracy', 'Sensitivity', 'Specificity')
        for k in self.kernels :
            m.set_estimator_param('kernel', k)
            for c in self.Cs :
                m.set_estimator_param('C', c)
                for g in self.gammas :
                    m.set_estimator_param('gamma', g)
                    m.learn_k_fold()
                    accuracies = []
                    sensitivities = []
                    specificities = []
                    for t in thresholds :
                        result = m.predict_k_fold(t)
                        #else :
                            #result = m.predict_blind_data(b_data, b_target, t)
                        accuracies.append(result[0])
                        sensitivities.append(result[1])
                        specificities.append(result[2])
                        #print(k, c, g, t, result[0], result[1], result[2])
                    diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                    max_index = diff.tolist().index(diff.min())
                    print(k, c, g, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                    if b_data is not None :
                        self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=[thresholds[max_index]])
                        #self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=thresholds)

    def search_with_under_sampling_SVM(self, data, target, under_sample_folds, b_data = None, b_target = None) :
        thresholds = gen_thresholds(-1, 1.001, 0.1)
        print('Kernel', 'C', 'Gamma', 'Accuracy', 'Sensitivity', 'Specificity')
        for k in self.kernels :
            for c in self.Cs :
                for g in self.gammas :
                    accuracy_sum = sensitivity_sum = specificity_sum = 0.
                    under_sample_models = []
                    under_sample_thresholds = []
                    for i in range(len(under_sample_folds)) :
                        m = Model('SVM', data[under_sample_folds[i]], target[under_sample_folds[i]])
                        m.set_estimator_param('kernel', k)
                        m.set_estimator_param('C', c)
                        m.set_estimator_param('gamma', g)
                        m.learn_k_fold()
                        under_sample_models.append(copy.deepcopy(m))
                        accuracies = []
                        sensitivities = []
                        specificities = []
                        for t in thresholds :
                            result = m.predict_k_fold(t)
                            accuracies.append(result[0])
                            sensitivities.append(result[1])
                            specificities.append(result[2])
                            #print(k, c, g, t, result[0], result[1], result[2])
                        diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                        max_index = diff.tolist().index(diff.min())
                        #print(k, c, g, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                        accuracy_sum = accuracy_sum + accuracies[max_index]
                        sensitivity_sum = sensitivity_sum + sensitivities[max_index]
                        specificity_sum = specificity_sum + specificities[max_index]
                        under_sample_thresholds.append(thresholds[max_index])
                    print(k, c, g, accuracy_sum/len(under_sample_folds), sensitivity_sum/len(under_sample_folds), specificity_sum/len(under_sample_folds))
                    print(mean(under_sample_thresholds), stdev(under_sample_thresholds), under_sample_thresholds)
                    if b_data is not None :
                        self.predict_under_sampling_blind_data_LT(under_sample_models, b_data, b_target, thresholds=under_sample_thresholds)
                        self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(mean(under_sample_thresholds), mean(under_sample_thresholds), 0.1))
                        self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(min(under_sample_thresholds), max(under_sample_thresholds), 0.1))

    def search_with_RF(self, data, target, with_threshold=True, b_data = None, b_target = None) :
        thresholds = gen_thresholds(0, 1.001, 0.05)
        m = Model('RF', data, target)
        print('Max_depth', 'Max_features', '#_estimators', 'Threshold', 'Accuracy', 'Sensitivity', 'Specificity')
        for md in self.max_depths :
            m.set_estimator_param('max_depth', md)
            for mf in self.max_features :
                m.set_estimator_param('max_features', mf)
                for ne in self.estimator_sizes :
                    m.set_estimator_param('n_estimators', ne)
                    m.learn_k_fold()
                    if with_threshold == True :
                        accuracies = []
                        sensitivities = []
                        specificities = []
                        for t in thresholds :
                            result = m.predict_k_fold(t)
                            accuracies.append(result[0])
                            sensitivities.append(result[1])
                            specificities.append(result[2])
                            #print(ne, t, result[0], result[1], result[2])
                        diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                        max_index = diff.tolist().index(diff.min())
                        print(md, mf, ne, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                        if b_data is not None :
                            self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=[thresholds[max_index]])
                            #self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=thresholds)
                    else :
                        result = m.predict_k_fold()
                        print(md, mf, ne,  result[0], result[1], result[2])
                        if b_data is not None :
                            self.predict_blind_data(m, b_data, b_target, with_threshold=False)

    def search_with_under_sampling_RF(self, data, target, under_sample_folds, with_threshold=True, b_data = None, b_target = None) :
        thresholds = gen_thresholds(0, 1.001, 0.05)
        print('Max_depth', 'Max_features', '#_estimators', 'Threshold', 'Accuracy', 'Sensitivity', 'Specificity')
        for md in self.max_depths :
            for mf in self.max_features :
                for ne in self.estimator_sizes :
                    accuracy_sum = sensitivity_sum = specificity_sum = 0.
                    under_sample_models = []
                    under_sample_thresholds = []
                    for i in range(len(under_sample_folds)) :
                        m = Model('RF', data[under_sample_folds[i]], target[under_sample_folds[i]])
                        m.set_estimator_param('max_depth', md)
                        m.set_estimator_param('max_features', mf)
                        m.set_estimator_param('n_estimators', ne)
                        m.learn_k_fold()
                        under_sample_models.append(copy.deepcopy(m))
                        if with_threshold == True :
                            accuracies = []
                            sensitivities = []
                            specificities = []
                            for t in thresholds :
                                result = m.predict_k_fold(t)
                                accuracies.append(result[0])
                                sensitivities.append(result[1])
                                specificities.append(result[2])
                                #print(ne, t, result[0], result[1], result[2])
                            diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                            max_index = diff.tolist().index(diff.min())
                            #print(md, mf, ne, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                            accuracy_sum = accuracy_sum + accuracies[max_index]
                            sensitivity_sum = sensitivity_sum + sensitivities[max_index]
                            specificity_sum = specificity_sum + specificities[max_index]
                            under_sample_thresholds.append(thresholds[max_index])
                        else :
                            result = m.predict_k_fold()
                            accuracy_sum = accuracy_sum + result[0]
                            sensitivity_sum = sensitivity_sum + result[1]
                            specificity_sum = specificity_sum + result[2]
                    print(md, mf, ne, accuracy_sum/len(under_sample_folds), sensitivity_sum/len(under_sample_folds), specificity_sum/len(under_sample_folds))
                    if with_threshold == True :
                        print(mean(under_sample_thresholds), stdev(under_sample_thresholds), under_sample_thresholds)
                    if b_data is not None :
                        if with_threshold == True :
                            self.predict_under_sampling_blind_data_LT(under_sample_models, b_data, b_target, thresholds=under_sample_thresholds)
                            self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(mean(under_sample_thresholds), mean(under_sample_thresholds), 0.05))
                            self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(min(under_sample_thresholds), max(under_sample_thresholds), 0.05))
                        else :
                            self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=False)

    def search_with_GNB(self, data, target, with_threshold=True, b_data = None, b_target = None) :
        thresholds = gen_thresholds(0, 1.001, 0.01)
        m = Model('GNB', data, target)
        print('Smoothing', 'Threshold', 'Accuracy', 'Sensitivity', 'Specificity')
        for vs in self.smoothings :
            m.set_estimator_param('var_smoothing', vs)
            ### Learning
            m.learn_k_fold()
            ### Prediction 
            if with_threshold == True :
                accuracies = []
                sensitivities = []
                specificities = []
                for t in thresholds :
                    result = m.predict_k_fold(t)
                    accuracies.append(result[0])
                    sensitivities.append(result[1])
                    specificities.append(result[2])
                    #print(ne, t, result[0], result[1], result[2])
                diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                max_index = diff.tolist().index(diff.min())
                print(vs, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                if b_data is not None :
                    self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=[thresholds[max_index]])
                    #self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=thresholds)
            else :
                result = m.predict_k_fold()
                print(vs, result[0], result[1], result[2])
                if b_data is not None :
                    self.predict_blind_data(m, b_data, b_target, with_threshold=False)
    
    def search_with_under_sampling_GNB(self, data, target, under_sample_folds, with_threshold=True, b_data = None, b_target = None) :
        thresholds = gen_thresholds(0, 1.001, 0.01)
        print('Smoothing', 'Threshold', 'Accuracy', 'Sensitivity', 'Specificity')
        for vs in self.smoothings :
            accuracy_sum = sensitivity_sum = specificity_sum = 0.
            under_sample_models = []
            under_sample_thresholds = []
            for i in range(len(under_sample_folds)) :
                m = Model('GNB', data[under_sample_folds[i]], target[under_sample_folds[i]])
                m.set_estimator_param('var_smoothing', vs)
                m.learn_k_fold()
                under_sample_models.append(copy.deepcopy(m))
                if with_threshold == True :
                    accuracies = []
                    sensitivities = []
                    specificities = []
                    for t in thresholds :
                        result = m.predict_k_fold(t)
                        accuracies.append(result[0])
                        sensitivities.append(result[1])
                        specificities.append(result[2])
                        #print(ne, t, result[0], result[1], result[2])
                    diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                    max_index = diff.tolist().index(diff.min())
                    #print(md, mf, ne, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                    accuracy_sum = accuracy_sum + accuracies[max_index]
                    sensitivity_sum = sensitivity_sum + sensitivities[max_index]
                    specificity_sum = specificity_sum + specificities[max_index]
                    under_sample_thresholds.append(thresholds[max_index])
                else :
                    result = m.predict_k_fold()
                    accuracy_sum = accuracy_sum + result[0]
                    sensitivity_sum = sensitivity_sum + result[1]
                    specificity_sum = specificity_sum + result[2]
            print(vs, accuracy_sum/len(under_sample_folds), sensitivity_sum/len(under_sample_folds), specificity_sum/len(under_sample_folds))
            if with_threshold == True :
                print(mean(under_sample_thresholds), stdev(under_sample_thresholds), under_sample_thresholds)
            if b_data is not None :
                if with_threshold == True :
                    self.predict_under_sampling_blind_data_LT(under_sample_models, b_data, b_target, thresholds=under_sample_thresholds)
                    self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(mean(under_sample_thresholds), mean(under_sample_thresholds), 0.01))
                    self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(min(under_sample_thresholds), max(under_sample_thresholds), 0.01))
                else :
                    self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=False)
