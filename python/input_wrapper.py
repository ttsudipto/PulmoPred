#import sys
#sys.path.append('/home/vbhandare/.local/lib/python2.7/site-packages')
import numpy as np
from .config import ml_config as mlc

class Input :
    def __init__(self) :
        self.input_params = ['fev1_pre_value','fev1_pre_percent','fev1_post_value','fev1_post_percent','fvc_pre_value','fvc_pre_percent','fvc_post_value','fvc_post_percent','fef_pre_value','fef_pre_percent','fef_post_value','fef_post_percent']
        self.input_values = []
        self.param_length = len(self.input_params)
        self.estimator_id = mlc.get_optimal_estimator()

    def add_param(self, param) :
        self.input_params.append(param)

    def add_value(self, value) :
        self.input_values.append(value)
    
    def set_estimator_id(self, e_id) :
        self.estimator_id = e_id;
    
    def get_estimator_id(self) :
        return self.estimator_id

    def get_all_params(self) :
        return self.input_params
    
    def get_all_values(self) :
        return self.input_values
    
    def get_value(self, index) :
        return self.input_values[index]
        
    def get_param(self, index) :
        return self.input_params[index]
    
    def get_ndarray(self) :
        return np.array(self.input_values).reshape(1, -1)
