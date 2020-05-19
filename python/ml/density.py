from sklearn.svm import SVC
from scipy.stats import gaussian_kde
import collections
import numpy as np
from scipy.integrate import simps
from .model import Model
from ..config import config
from ..config import ml_config as mlc

ROOT_PATH = config.get_ROOT_PATH()
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

def load_density(model_index, pos_or_neg) :
    filename = ROOT_PATH+'output/densities/'+str(model_index)+'.csv'
    csvfile = open(filename)
    if pos_or_neg == 'pos' :
        cols = [0,1]
    elif pos_or_neg == 'neg' :
        cols = [2,3]
    else :
        raise ValueError('Invalid value in \'pos_or_neg\'')
    density_function = np.genfromtxt(csvfile, delimiter=',', skip_header=1, usecols=cols)
    return density_function

def compute_positiveness(model_index, score) :
    pos_func = load_density(model_index, 'pos')
    if score < pos_func[0,0] :
        return 0.0
    if score > pos_func[pos_func.shape[0]-1,0] :
        return 1.0
    masked_data = np.ma.masked_where((pos_func[:,0] > score), pos_func[:,0])
    score_index = np.argmax(masked_data)
    #print(masked_data)
    #print(score_index)
    #print(pos_func[:score_index+1,0])
    #print(pos_func[:score_index+1,1])
    return simps(pos_func[:score_index+1,1], pos_func[:score_index+1,0])

def compute_negativeness(model_index, score) :
    neg_func = load_density(model_index, 'neg')
    if score > neg_func[neg_func.shape[0]-1,0] :
        return 0.0
    if score < neg_func[0,0] :
        return 1.0
    masked_data = np.ma.masked_where((neg_func[:,0] < score), neg_func[:,0])
    score_index = np.argmin(masked_data)
    #print(masked_data)
    #print(score_index)
    #print(neg_func[score_index:,0])
    #print(neg_func[score_index:,1])
    return simps(neg_func[score_index:,1], neg_func[score_index:,0])

def execute(data, target, ch) :
    """Driver function"""
    
    global min
    global max
    
    hyperparameters = mlc.get_optimal_hyperparameters('PFT', mlc.get_SVM_id())
    m = Model(mlc.get_SVM_id(), data, target)
    m.set_estimator_param('C', hyperparameters['C'])
    m.set_estimator_param('gamma', hyperparameters['gamma'])
    m.learn_without_CV()

    pos_indices, neg_indices = split_pos_neg(target)
    
    des = m.total_estimator.decision_function(data)
    min = des.min()
    max = des.max()
    neg_des = m.total_estimator.decision_function(data[neg_indices])
    pos_des = m.total_estimator.decision_function(data[pos_indices])
    if (ch == 1) : ## all density
        xs, densities = print_gausian(des)
        #xs, densities = print_histogram(des)
    elif (ch == 2) : ## negative density
        xs, densities = print_gausian(neg_des)
        #xs, densities = print_histogram(neg_des)
    elif (ch == 3) : ## positive density
        xs, densities = print_gausian(pos_des)
        #xs, densities = print_histogram(pos_des)
    else :
        print('Wrong choice')
    return xs, densities
    

