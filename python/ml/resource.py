from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sys import maxsize
import csv
import numpy as np
import sys
from ..config import config
from ..config import input_config as ic

np.set_printoptions(threshold=maxsize)

ROOT_PATH = config.get_ROOT_PATH()
#path_prefix = ROOT_PATH + 'input/'
#path_prefix = ROOT_PATH + 'input/independent_blind_split/'
#path_prefix = ROOT_PATH + 'input/A_NO_split/'
path_prefix = ROOT_PATH + 'input/common_pft_tct/'

class DataResource :
    """Class representing the data that is used for analysis.

    It provides methods to read data from CSV files and store
    it into n-dimensional arrays. It also provides methods
    for null value imputation.
    """
    
    def __init__(self, name):
        """ Constructor"""
       
        self.filename = name
        self.attributes = self._read_attributes()
        self.labels = {ic.get_positive_label():1, ic.get_negative_label():0}
        self._null_indices = []
        self._set_dataset_names()
        self._set_column_limits()

    def _set_dataset_names(self) :
        if self.filename[:3] == 'TCT' :
            #print(self.filename[:3])
            self.dataset_name = 'TCT'
        else :
            self.dataset_name = 'PFT'

    def _set_column_limits(self) :
        self.target_col = ic.get_target_column(self.dataset_name)
        self.data_col_start = ic.get_data_column_start(self.dataset_name)
        self.data_col_end = ic.get_data_column_end(self.dataset_name)
    
    def _target_converter(self, x):
        """Method that maps the target labels to integers"""
        if int(sys.version[0]) > 2 :
            x = str(x)[2:-1]
        if x in self.labels :
            return self.labels[x]
        else :
            return -1;
    
    def _read_attributes(self):
        """Method to read the first line of the file that contains the attribute names"""
        
        csvfile = open(path_prefix + self.filename)
        reader = csv.DictReader(csvfile, delimiter=',')
        attributes = reader.fieldnames
        csvfile.close()
        return attributes

    def read_data_from_csv(self):
        """Method to read the values of all the features from a CSV file"""
        
        csvfile = open(path_prefix + self.filename)
        self.data = np.genfromtxt(csvfile, delimiter=',', skip_header=1, usecols=range(self.data_col_start, self.data_col_end))
        csvfile.close()

    def read_target_from_csv(self):
        """Method to read the last column of the file that contains the target labels"""
        
        csvfile = open(path_prefix + self.filename)
        self.target = np.genfromtxt(csvfile, delimiter=',', skip_header=1, usecols=(self.target_col), converters={self.target_col:self._target_converter})
        csvfile.close()
        
    def replace_missing_values(self):
        """Method to perform null value imputation"""
        
        col_mean = np.nanmean(self.data, axis=0)
        x,y = np.where(np.isnan(self.data))
        self._null_indices = [(x[i], y[i]) for i in range(len(x))]
        for i in range(len(x)):
            self.data[x[i],y[i]] = col_mean[y[i]]
            
    def split_blind_data(self) :
        """Method to split the blind dataset. 
        
        It uses the split() method of sklearn.model_selection.StratifiedShuffleSplit class.
        """
        
        skf = StratifiedShuffleSplit(n_splits=1, test_size=ic.get_blind_data_size(), random_state = ic.get_blind_split_random_state())
        for train_index, test_index in skf.split(self.data, self.target) :
            #print(sorted(train_index))
            #print(sorted(test_index))
            print(sorted([x+2 for x in train_index]))
            print(sorted([x+2 for x in test_index]))
            print(len(test_index))
            X_train = self.data[train_index].copy()
            X_test = self.data[test_index].copy()
            y_train = self.target[train_index].copy()
            y_test = self.target[test_index].copy()
            self.data = X_train
            self.target = y_train
            self.hold_out_data = X_test
            self.hold_out_target = y_test
            break

#def read_data(filename, verbose=False) :
    #res = DataResource(filename)
    #res.read_data_from_csv()
    #res.read_target_from_csv()
    #res.replace_missing_values()
    ##res.split_blind_data()
    #if verbose == True :
        #print(len(res.attributes))
        #print(res.attributes)
        #print('Data')
        #print(res.data.shape)
        #print(res.data[1,-1])
        #print(res.data[2,-1])
        #print('Target')
        #print(res.target.shape)
        #print(res.target[16], res.target[17])
    #return res
