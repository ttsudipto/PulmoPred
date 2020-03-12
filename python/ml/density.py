from sklearn.svm import SVC
from scipy.stats import gaussian_kde
import collections
import numpy as np
from .model import Model

min = 0
max = 0

def print_gausian(data) :
    """Method to print the density of SVM scores"""
    
    global min
    global max
    density = gaussian_kde(data)
    xs = np.linspace(min, max, 1000)
    density.covariance_factor = lambda : 0.25
    density._compute_covariance()
    densities = density(xs)
    print('Threshold', 'Density')
    for i in range(len(xs)) :
        print(xs[i], densities[i])
    return xs, densities

def print_histogram(data) :
    density, xs = np.histogram(data, bins='scott', density=True)
    return xs[1:], density

def split_pos_neg(target, pos_class=1, neg_class=0) :
    """Method to split positive and negative samples"""
    
    pos_indices = []
    neg_indices = []
    
    for i in range(target.shape[0]) :
        if target[i] == pos_class :
            pos_indices.append(i)
        elif target[i] == neg_class :
            neg_indices.append(i)
        else :
            raise ValueError('Invalid value present')
    
    return pos_indices, neg_indices

def execute(data, target, ch) :
    """Driver function"""
    
    global min
    global max
    
    m = Model('SVM', data, target)
    #m.set_estimator_param('C', 5)
    #m.set_estimator_param('gamma', 0.0001)
    m.learn_without_CV()

    pos_indices, neg_indices = split_pos_neg(target)
    
    des = m.estimators[0].decision_function(data)
    min = des.min()
    max = des.max()
    neg_des = m.estimators[0].decision_function(data[neg_indices])
    pos_des = m.estimators[0].decision_function(data[pos_indices])
    if (ch == 1) :
        xs, densities = print_gausian(des)
        #xs, densities = print_histogram(des)
    elif (ch == 2) :
        xs, densities = print_gausian(neg_des)
        #xs, densities = print_histogram(neg_des)
    elif (ch == 3) :
        xs, densities = print_gausian(pos_des)
        #xs, densities = print_histogram(pos_des)
    else :
        print('Wrong choice')
    return xs, densities
    

