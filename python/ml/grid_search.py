from sklearn.model_selection import GridSearchCV
from threading import Thread
from .model import Model
from statistics import mean, stdev
import numpy as np
import copy
from ..config import ml_config as mlc

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
    
    #def __init__(self) :
        #"""Constructor"""
        
        #self.best_C = None
        #self.best_gamma = None
        #self.best_kernel = None
        #self.best_estimator_size = None
    
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
    
    def init_MLP_params(self, activations=['relu'], hidden_layer_sizes=[(100,)], learning_rate_inits=[0.001]) :
        self.activations = activations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_inits = learning_rate_inits
        
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
                print('blind', t, result[0], result[1], result[2], result[3], result[4])
        else :
            result = model.predict_blind_data(b_data,b_target)
            print('blind', result[0], result[1], result[2], result[3], result[4])
    
    def predict_under_sampling_blind_data(self, models, b_data, b_target, with_threshold=True, thresholds=[0.1]) : # Same threshold to all models
        if with_threshold == True :
            for t in thresholds :
                b_accuracy_sum = b_sensitivity_sum = b_specificity_sum = b_f1score_sum = b_mcc_sum = 0.
                for m in models :
                    b_result = m.predict_blind_data(b_data, b_target, t)
                    b_accuracy_sum = b_accuracy_sum + b_result[0]
                    b_sensitivity_sum = b_sensitivity_sum + b_result[1]
                    b_specificity_sum = b_specificity_sum + b_result[2]
                    b_f1score_sum = b_f1score_sum + b_result[3]
                    b_mcc_sum = b_mcc_sum + b_result[4]
                print('blind', round(t,2), b_accuracy_sum/len(models), b_sensitivity_sum/len(models), b_specificity_sum/len(models), b_f1score_sum/len(models), b_mcc_sum/len(models))
        else :
            b_accuracy_sum = b_sensitivity_sum = b_specificity_sum = b_f1score_sum = b_mcc_sum = 0.
            for m in models :
                b_result = m.predict_blind_data(b_data, b_target)
                b_accuracy_sum = b_accuracy_sum + b_result[0]
                b_sensitivity_sum = b_sensitivity_sum + b_result[1]
                b_specificity_sum = b_specificity_sum + b_result[2]
                b_f1score_sum = b_f1score_sum + b_result[3]
                b_mcc_sum = b_mcc_sum + b_result[4]
            print('blind', b_accuracy_sum/len(models), b_sensitivity_sum/len(models), b_specificity_sum/len(models), b_f1score_sum/len(models), b_mcc_sum/len(models))

    def predict_under_sampling_blind_data_LT(self, models, b_data, b_target, thresholds) :  # LT -> Learned Threshold : Different thresholds for different models
        b_accuracy_sum = b_sensitivity_sum = b_specificity_sum = b_f1score_sum = b_mcc_sum = 0.
        for i in range(len(models)) :
            m = models[i]
            b_result = m.predict_blind_data(b_data, b_target, thresholds[i])
            b_accuracy_sum = b_accuracy_sum + b_result[0]
            b_sensitivity_sum = b_sensitivity_sum + b_result[1]
            b_specificity_sum = b_specificity_sum + b_result[2]
            b_f1score_sum = b_f1score_sum + b_result[3]
            b_mcc_sum = b_mcc_sum + b_result[4]
        print('blind', 'Learned', b_accuracy_sum/len(models), b_sensitivity_sum/len(models), b_specificity_sum/len(models), b_f1score_sum/len(models), b_mcc_sum/len(models))
        
    def search_with_threshold_SVM(self, data, target, b_data = None, b_target = None, scale = False) :
        """Method to perform grid search"""
        
        thresholds = gen_thresholds(-1, 1.001, 0.1)
        #m = Model(mlc.get_SVM_id(), data, target, scale=scale)
        print('Kernel', 'C', 'Gamma', 'Threshold', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        for k in self.kernels :
            for c in self.Cs :
                for g in self.gammas :
                    m = Model(mlc.get_SVM_id(), data, target, scale=scale)
                    m.set_estimator_param('kernel', k)
                    m.set_estimator_param('C', c)
                    m.set_estimator_param('gamma', g)
                    m.learn()
                    accuracies = []
                    sensitivities = []
                    specificities = []
                    f1_scores = []
                    mccs = []
                    for t in thresholds :
                        result = m.predict_k_fold(t)
                        accuracies.append(result[0])
                        sensitivities.append(result[1])
                        specificities.append(result[2])
                        f1_scores.append(result[3])
                        mccs.append(result[4])
                        #print(k, c, g, t, result[0], result[1], result[2], result[3], result[4])
                    diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                    max_index = diff.tolist().index(diff.min())
                    print(k, c, g, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index], f1_scores[max_index], mccs[max_index])
                    if b_data is not None :
                        self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=[thresholds[max_index]])
                        #self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=thresholds)

    def search_with_under_sampling_SVM(self, data, target, under_sample_folds, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(-1, 1.001, 0.1)
        print('Kernel', 'C', 'Gamma', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        for k in self.kernels :
            for c in self.Cs :
                for g in self.gammas :
                    accuracy_sum = sensitivity_sum = specificity_sum = f1score_sum = mcc_sum =  0.
                    under_sample_models = []
                    under_sample_thresholds = []
                    for i in range(len(under_sample_folds)) :
                        m = Model(mlc.get_SVM_id(), data[under_sample_folds[i]], target[under_sample_folds[i]], scale=scale)
                        m.set_estimator_param('kernel', k)
                        m.set_estimator_param('C', c)
                        m.set_estimator_param('gamma', g)
                        m.learn()
                        under_sample_models.append(copy.deepcopy(m))
                        accuracies = []
                        sensitivities = []
                        specificities = []
                        f1_scores = []
                        mccs = []
                        for t in thresholds :
                            result = m.predict_k_fold(t)
                            accuracies.append(result[0])
                            sensitivities.append(result[1])
                            specificities.append(result[2])
                            f1_scores.append(result[3])
                            mccs.append(result[4])
                            #print(k, c, g, t, result[0], result[1], result[2], result[3], result[4])
                        diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                        max_index = diff.tolist().index(diff.min())
                        #print(k, c, g, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                        accuracy_sum = accuracy_sum + accuracies[max_index]
                        sensitivity_sum = sensitivity_sum + sensitivities[max_index]
                        specificity_sum = specificity_sum + specificities[max_index]
                        f1score_sum = f1score_sum + f1_scores[max_index]
                        mcc_sum = mcc_sum + mccs[max_index]
                        under_sample_thresholds.append(thresholds[max_index])
                    print(k, c, g, accuracy_sum/len(under_sample_folds), sensitivity_sum/len(under_sample_folds), specificity_sum/len(under_sample_folds), f1score_sum/len(under_sample_folds), mcc_sum/len(under_sample_folds))
                    #print(mean(under_sample_thresholds), stdev(under_sample_thresholds), under_sample_thresholds)
                    if b_data is not None :
                        self.predict_under_sampling_blind_data_LT(under_sample_models, b_data, b_target, thresholds=under_sample_thresholds)
                        self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(mean(under_sample_thresholds), mean(under_sample_thresholds), 0.1))
                        self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(min(under_sample_thresholds), max(under_sample_thresholds), 0.1))

    def search_with_RF(self, data, target, with_threshold=True, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(0, 1.001, 0.05)
        #m = Model(mlc.get_RandomForest_id(), data, target, scale=scale)
        print('Max_depth', 'Max_features', '#_estimators', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        for md in self.max_depths :
            for mf in self.max_features :
                for ne in self.estimator_sizes :
                    m = Model(mlc.get_RandomForest_id(), data, target, scale=scale)
                    m.set_estimator_param('max_depth', md)
                    m.set_estimator_param('max_features', mf)
                    m.set_estimator_param('n_estimators', ne)
                    m.learn()
                    if with_threshold == True :
                        accuracies = []
                        sensitivities = []
                        specificities = []
                        f1_scores = []
                        mccs = []
                        for t in thresholds :
                            result = m.predict_k_fold(t)
                            accuracies.append(result[0])
                            sensitivities.append(result[1])
                            specificities.append(result[2])
                            f1_scores.append(result[3])
                            mccs.append(result[4])
                            #print(ne, t, result[0], result[1], result[2], result[3], result[4])
                        diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                        max_index = diff.tolist().index(diff.min())
                        print(md, mf, ne, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index], f1_scores[max_index], mccs[max_index])
                        if b_data is not None :
                            self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=[thresholds[max_index]])
                            #self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=thresholds)
                    else :
                        result = m.predict_k_fold()
                        print(md, mf, ne,  result[0], result[1], result[2], result[3], result[4])
                        if b_data is not None :
                            self.predict_blind_data(m, b_data, b_target, with_threshold=False)

    def search_with_under_sampling_RF(self, data, target, under_sample_folds, with_threshold=True, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(0, 1.001, 0.05)
        print('Max_depth', 'Max_features', '#_estimators', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        for md in self.max_depths :
            for mf in self.max_features :
                for ne in self.estimator_sizes :
                    accuracy_sum = sensitivity_sum = specificity_sum = f1score_sum = mcc_sum = 0.
                    under_sample_models = []
                    under_sample_thresholds = []
                    for i in range(len(under_sample_folds)) :
                        m = Model(mlc.get_RandomForest_id(), data[under_sample_folds[i]], target[under_sample_folds[i]], scale=scale)
                        m.set_estimator_param('max_depth', md)
                        m.set_estimator_param('max_features', mf)
                        m.set_estimator_param('n_estimators', ne)
                        m.learn()
                        under_sample_models.append(copy.deepcopy(m))
                        if with_threshold == True :
                            accuracies = []
                            sensitivities = []
                            specificities = []
                            f1_scores = []
                            mccs = []
                            for t in thresholds :
                                result = m.predict_k_fold(t)
                                accuracies.append(result[0])
                                sensitivities.append(result[1])
                                specificities.append(result[2])
                                f1_scores.append(result[3])
                                mccs.append(result[4])
                                #print(ne, t, result[0], result[1], result[2], result[3], result[4])
                            diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                            max_index = diff.tolist().index(diff.min())
                            #print(md, mf, ne, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index], f1_scores[max_index], mccs[max_index])
                            accuracy_sum = accuracy_sum + accuracies[max_index]
                            sensitivity_sum = sensitivity_sum + sensitivities[max_index]
                            specificity_sum = specificity_sum + specificities[max_index]
                            f1score_sum = f1score_sum + f1_scores[max_index]
                            mcc_sum = mcc_sum + mccs[max_index]
                            under_sample_thresholds.append(thresholds[max_index])
                        else :
                            result = m.predict_k_fold()
                            accuracy_sum = accuracy_sum + result[0]
                            sensitivity_sum = sensitivity_sum + result[1]
                            specificity_sum = specificity_sum + result[2]
                            f1score_sum = f1score_sum + result[3]
                            mcc_sum = mcc_sum + result[4]
                    print(md, mf, ne, accuracy_sum/len(under_sample_folds), sensitivity_sum/len(under_sample_folds), specificity_sum/len(under_sample_folds), f1score_sum/len(under_sample_folds), mcc_sum/len(under_sample_folds))
                    #if with_threshold == True :
                        #print(mean(under_sample_thresholds), stdev(under_sample_thresholds), under_sample_thresholds)
                    if b_data is not None :
                        if with_threshold == True :
                            self.predict_under_sampling_blind_data_LT(under_sample_models, b_data, b_target, thresholds=under_sample_thresholds)
                            self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(mean(under_sample_thresholds), mean(under_sample_thresholds), 0.05))
                            self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(min(under_sample_thresholds), max(under_sample_thresholds), 0.05))
                        else :
                            self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=False)

    def search_with_GNB(self, data, target, with_threshold=True, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(0, 1.001, 0.01)
        #m = Model(mlc.get_NaiveBayes_id(), data, target, scale=scale)
        print('Smoothing', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        for vs in self.smoothings :
            m = Model(mlc.get_NaiveBayes_id(), data, target, scale=scale)
            m.set_estimator_param('var_smoothing', vs)
            m.learn()
            if with_threshold == True :
                accuracies = []
                sensitivities = []
                specificities = []
                f1_scores = []
                mccs = []
                for t in thresholds :
                    result = m.predict_k_fold(t)
                    accuracies.append(result[0])
                    sensitivities.append(result[1])
                    specificities.append(result[2])
                    f1_scores.append(result[3])
                    mccs.append(result[4])
                    #print(ne, t, result[0], result[1], result[2])
                diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                max_index = diff.tolist().index(diff.min())
                print(vs, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index], f1_scores[max_index], mccs[max_index])
                if b_data is not None :
                    self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=[thresholds[max_index]])
                    #self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=thresholds)
            else :
                result = m.predict_k_fold()
                print(vs, result[0], result[1], result[2], result[3], result[4])
                if b_data is not None :
                    self.predict_blind_data(m, b_data, b_target, with_threshold=False)
    
    def search_with_under_sampling_GNB(self, data, target, under_sample_folds, with_threshold=True, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(0, 1.001, 0.01)
        print('Smoothing', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        for vs in self.smoothings :
            accuracy_sum = sensitivity_sum = specificity_sum = f1score_sum = mcc_sum = 0.
            under_sample_models = []
            under_sample_thresholds = []
            for i in range(len(under_sample_folds)) :
                m = Model(mlc.get_NaiveBayes_id(), data[under_sample_folds[i]], target[under_sample_folds[i]], scale=scale)
                m.set_estimator_param('var_smoothing', vs)
                m.learn()
                under_sample_models.append(copy.deepcopy(m))
                if with_threshold == True :
                    accuracies = []
                    sensitivities = []
                    specificities = []
                    f1_scores = []
                    mccs = []
                    for t in thresholds :
                        result = m.predict_k_fold(t)
                        accuracies.append(result[0])
                        sensitivities.append(result[1])
                        specificities.append(result[2])
                        f1_scores.append(result[3])
                        mccs.append(result[4])
                        #print(ne, t, result[0], result[1], result[2], result[3], result[4])
                    diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                    max_index = diff.tolist().index(diff.min())
                    #print(md, mf, ne, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index])
                    accuracy_sum = accuracy_sum + accuracies[max_index]
                    sensitivity_sum = sensitivity_sum + sensitivities[max_index]
                    specificity_sum = specificity_sum + specificities[max_index]
                    f1score_sum = f1score_sum + f1_scores[max_index]
                    mcc_sum = mcc_sum + mccs[max_index]
                    under_sample_thresholds.append(thresholds[max_index])
                else :
                    result = m.predict_k_fold()
                    accuracy_sum = accuracy_sum + result[0]
                    sensitivity_sum = sensitivity_sum + result[1]
                    specificity_sum = specificity_sum + result[2]
                    f1score_sum = f1score_sum + result[3]
                    mcc_sum = mcc_sum + result[4]
            print(vs, accuracy_sum/len(under_sample_folds), sensitivity_sum/len(under_sample_folds), specificity_sum/len(under_sample_folds), f1score_sum/len(under_sample_folds), mcc_sum/len(under_sample_folds))
            #if with_threshold == True :
                #print(mean(under_sample_thresholds), stdev(under_sample_thresholds), under_sample_thresholds)
            if b_data is not None :
                if with_threshold == True :
                    self.predict_under_sampling_blind_data_LT(under_sample_models, b_data, b_target, thresholds=under_sample_thresholds)
                    self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(mean(under_sample_thresholds), mean(under_sample_thresholds), 0.01))
                    self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(min(under_sample_thresholds), max(under_sample_thresholds), 0.01))
                else :
                    self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=False)
    
    def search_with_MLP(self, data, target, with_threshold=True, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(0, 1.001, 0.05)
        #m = Model(mlc.get_MLP_id(), data, target, scale=scale)
        print('Activation', 'Layers', 'Learning_rate', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        for a in self.activations :
            for hls in self.hidden_layer_sizes :
                for lri in self.learning_rate_inits :
                    m = Model(mlc.get_MLP_id(), data, target, scale=scale)
                    m.set_estimator_param('activation', a)
                    m.set_estimator_param('hidden_layer_sizes', hls)
                    m.set_estimator_param('learning_rate_init', lri)
                    m.learn()
                    if with_threshold == True :
                        accuracies = []
                        sensitivities = []
                        specificities = []
                        f1_scores = []
                        mccs = []
                        for t in thresholds :
                            result = m.predict_k_fold(t)
                            accuracies.append(result[0])
                            sensitivities.append(result[1])
                            specificities.append(result[2])
                            f1_scores.append(result[3])
                            mccs.append(result[4])
                            #print(ne, t, result[0], result[1], result[2], result[3], result[4])
                        diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                        max_index = diff.tolist().index(diff.min())
                        print(a, hls, lri, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index], f1_scores[max_index], mccs[max_index])
                        if b_data is not None :
                            self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=[thresholds[max_index]])
                            #self.predict_blind_data(m, b_data, b_target, with_threshold=True, thresholds=thresholds)
                    else :
                        result = m.predict_k_fold()
                        print(a, hls, lri,  result[0], result[1], result[2], result[3], result[4])
                        if b_data is not None :
                            self.predict_blind_data(m, b_data, b_target, with_threshold=False)

    def search_with_under_sampling_MLP(self, data, target, under_sample_folds, with_threshold=True, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(0, 1.001, 0.05)
        print('Activation', 'Layers', 'Learning_rate', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'MCC')
        threads = []
        for a in self.activations :
            for hls in self.hidden_layer_sizes :
                for lri in self.learning_rate_inits :
                    th = Thread(target = self.check_one_MLP_model, args=(a, hls, lri, data, target, under_sample_folds, with_threshold, b_data, b_target, scale))
                    threads.append(th)
                    th.start()
        for th in threads :
            th.join()

    def check_one_MLP_model(self, a, hls, lri, data, target, under_sample_folds, with_threshold=True, b_data = None, b_target = None, scale = False) :
        thresholds = gen_thresholds(0, 1.001, 0.05)
        accuracy_sum = sensitivity_sum = specificity_sum = f1score_sum = mcc_sum = 0.
        under_sample_models = []
        under_sample_thresholds = []
        for i in range(len(under_sample_folds)) :
            m = Model(mlc.get_MLP_id(), data[under_sample_folds[i]], target[under_sample_folds[i]], scale=scale)
            m.set_estimator_param('activation', a)
            m.set_estimator_param('hidden_layer_sizes', hls)
            m.set_estimator_param('learning_rate_init', lri)
            m.learn()
            under_sample_models.append(copy.deepcopy(m))
            if with_threshold == True :
                accuracies = []
                sensitivities = []
                specificities = []
                f1_scores = []
                mccs = []
                for t in thresholds :
                    result = m.predict_k_fold(t)
                    accuracies.append(result[0])
                    sensitivities.append(result[1])
                    specificities.append(result[2])
                    f1_scores.append(result[3])
                    mccs.append(result[4])
                    #print(ne, t, result[0], result[1], result[2], result[3], result[4])
                diff = np.absolute(np.array(sensitivities)-np.array(specificities))
                max_index = diff.tolist().index(diff.min())
                #print(a, hls, lri, round(thresholds[max_index], 5), accuracies[max_index], sensitivities[max_index], specificities[max_index], f1_scores[max_index], mccs[max_index])
                accuracy_sum = accuracy_sum + accuracies[max_index]
                sensitivity_sum = sensitivity_sum + sensitivities[max_index]
                specificity_sum = specificity_sum + specificities[max_index]
                f1score_sum = f1score_sum + f1_scores[max_index]
                mcc_sum = mcc_sum + mccs[max_index]
                under_sample_thresholds.append(thresholds[max_index])
            else :
                result = m.predict_k_fold()
                accuracy_sum = accuracy_sum + result[0]
                sensitivity_sum = sensitivity_sum + result[1]
                specificity_sum = specificity_sum + result[2]
                f1score_sum = f1score_sum + result[3]
                mcc_sum = mcc_sum + result[4]
        print(a, hls, lri, accuracy_sum/len(under_sample_folds), sensitivity_sum/len(under_sample_folds), specificity_sum/len(under_sample_folds), f1score_sum/len(under_sample_folds), mcc_sum/len(under_sample_folds))
        #if with_threshold == True :
            #print(mean(under_sample_thresholds), stdev(under_sample_thresholds), under_sample_thresholds)
        if b_data is not None :
            if with_threshold == True :
                self.predict_under_sampling_blind_data_LT(under_sample_models, b_data, b_target, thresholds=under_sample_thresholds)
                self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(mean(under_sample_thresholds), mean(under_sample_thresholds), 0.05))
                self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=True, thresholds=gen_thresholds(min(under_sample_thresholds), max(under_sample_thresholds), 0.05))
            else :
                self.predict_under_sampling_blind_data(under_sample_models, b_data, b_target, with_threshold=False)
