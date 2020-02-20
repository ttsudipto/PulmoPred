from resource import DataResource
from sys import maxsize
import numpy as np

np.set_printoptions(threshold=maxsize)

def print_verbose(res) :
    print('\nFile Name : ')
    print(res.filename)
    print('Dataset name : ')
    print(res.dataset_name)
    print('Total no. of attributes : ')
    print(len(res.attributes))
    print('Attribute names : ')
    print(res.attributes)
    
    print('\n---------\nData\n---------')
    print('Data shape : ')
    print(res.data.shape)
    print(res.data[1,-1])
    print(res.data[2,-1])
    print('Data attributes : ')
    print(res.attributes[res.data_col_start : res.data_col_end])
    
    print('\n---------\nTarget\n---------')
    print('Target shape : ')
    print(res.target.shape)
    print(res.target[16], res.target[17])
    #for i in range(len(res.target)) :
        #print(i, res.target[i])
    print('Target attribute : ')
    print(res.attributes[res.target_col])

def read_data(filename, verbose=False) :
    """ 
    Driver function to read data from a CSV file and return 
    a DataResource
    """
    
    res = DataResource(filename)
    res.read_data_from_csv()
    res.read_target_from_csv()
    res.replace_missing_values()
    #res.split_blind_data()
    if verbose == True :
        print_verbose(res)
    return res
