import xml.etree.ElementTree as ET

config_file = 'ml_config.xml'
xml_root = ET.parse(config_file).getroot()

def get_from_xpath(path, index=None) :
    tags = xml_root.findall(path)
    if index == None :
        return tags
    if index >= len(tags) :
        return None
    return tags[index].text

def get_n_folds() :
    return get_from_xpath('./n_folds', 0)

def get_random_state() :
    return int(get_from_xpath('./random_state', 0))

def get_n_US_folds() :
    return int(get_from_xpath('./n_US_folds/pft', 0))

def get_SVM_id() :
    return get_from_xpath('./estimator_ids/svm', 0)

def get_RandomForest_id() :
    return get_from_xpath('./estimator_ids/rf', 0)

def get_NaiveBayes_id() :
    return get_from_xpath('./estimator_ids/nb', 0)

def get_MLP_id() :
    return get_from_xpath('./estimator_ids/mlp', 0)

def get_optimal_estimator() :
    return get_from_xpath('./optimal_estimator', 0)

def get_optimal_dataset() :
    return get_from_xpath('./optimal_dataset', 0)

def is_SVM_id(id) :
    return get_from_xpath('./estimator_ids/svm', 0) == id

def is_RandomForest_id(id) :
    return get_from_xpath('./estimator_ids/rf', 0) == id

def is_NaiveBayes_id(id) :
    return get_from_xpath('./estimator_ids/nb', 0) == id

def is_MLP_id(id) :
    return get_from_xpath('./estimator_ids/mlp', 0) == id

def get_optimal_hyperparameters(dataset, esimator_id) :
    hp = dict()
    if is_SVM_id(esimator_id) :
        hp['kernel'] = get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/svm/kernel', 0)
        hp['C'] = float(get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/svm/C', 0))
        hp['gamma'] = float(get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/svm/gamma', 0))
        threshold_tags = get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/svm/thresholds/threshold')
        thresholds = []
        for tt in threshold_tags :
            thresholds.append(float(tt.text))
        hp['thresholds'] = thresholds
        return hp
    elif is_RandomForest_id(esimator_id) :
        hp['n_estimators'] = int(get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/rf/n_estimators', 0))
        max_depth_str = get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/rf/max_depth', 0)
        if max_depth_str == 'None' :
            hp['max_depth'] = None
        else :
            hp['max_depth'] = int(max_depth_str)
        max_features_str = get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/rf/max_features', 0)
        if max_features_str == 'None' :
            hp['max_features'] = None
        elif max_features_str == 'auto' :
            hp['max_features'] == max_features_str
        else :
            hp['max_features'] = float(max_features_str)
    elif is_NaiveBayes_id(esimator_id) :
        hp['smoothing'] = float(get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/nb/smoothing', 0))
    elif is_MLP_id(esimator_id) :
        hp['activation'] = get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/mlp/activation', 0)
        hp['learning_rate_init'] = float(get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/mlp/learning_rate', 0))
        arch_list = get_from_xpath('./optimal_hyperparameters/' + dataset.lower() + '/mlp/hidden_layer_sizes', 0).split(',')
        arch_list = [int(x) for x in arch_list]
        hp['hidden_layer_sizes'] = tuple(arch_list)
    else :
        raise ValueError('Invalid estimator ID')
    return hp

def check() :
    print('No. of CV folds : ' + str(get_n_folds()))
    print('Random State : ' + str(get_random_state()))
    print('Under sampling fold size : ' + str(get_n_US_folds()))
    print('SVM ID : ' + get_SVM_id())
    print('Random Forest ID : ' + get_RandomForest_id())
    print('Naive Bayes ID : ' + get_NaiveBayes_id())
    print('MLP ID : ' + get_MLP_id())
    print('Optimal estimator : ' + get_optimal_estimator())
    print('Optimal dataset : ' + get_optimal_dataset())
    print('Optimal hyperparameters (no-us, SVM) : ' + str(get_optimal_hyperparameters('no-us', 'SVM')))
    print('Optimal hyperparameters (no-us, RF) : ' + str(get_optimal_hyperparameters('no-us', 'RF')))
    print('Optimal hyperparameters (no-us, GNB) : ' + str(get_optimal_hyperparameters('no-us', 'GNB')))
    print('Optimal hyperparameters (no-us, MLP) : ' + str(get_optimal_hyperparameters('no-us', 'MLP')))
    print('Optimal hyperparameters (us, SVM) : ' + str(get_optimal_hyperparameters('us', 'SVM')))
    print('Optimal hyperparameters (us, RF) : ' + str(get_optimal_hyperparameters('us', 'RF')))
    print('Optimal hyperparameters (us, GNB) : ' + str(get_optimal_hyperparameters('us', 'GNB')))
    print('Optimal hyperparameters (us, MLP) : ' + str(get_optimal_hyperparameters('us', 'MLP')))

#check()
