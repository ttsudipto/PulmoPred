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

def get_n_US_folds(dataset) :
    return int(get_from_xpath('./n_US_folds/' + dataset.lower(), 0))

def get_SVM_id() :
    return get_from_xpath('./estimator_ids/svm', 0)

def get_RandomForest_id() :
    return get_from_xpath('./estimator_ids/rf', 0)

def get_NaiveBayes_id() :
    return get_from_xpath('./estimator_ids/nb', 0)

def is_SVM_id(id) :
    return get_from_xpath('./estimator_ids/svm', 0) == id

def is_RandomForest_id(id) :
    return get_from_xpath('./estimator_ids/rf', 0) == id

def is_NaiveBayes_id(id) :
    return get_from_xpath('./estimator_ids/nb', 0) == id

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
    else :
        raise ValueError('Invalid estimator ID')
    return hp

def check() :
    print('No. of CV folds : ' + str(get_n_folds()))
    print('Random State : ' + str(get_random_state()))
    print('Under sampling fold size (PFT) : ' + str(get_n_US_folds('PFT')))
    print('Under sampling fold size (TCT) : ' + str(get_n_US_folds('TCT')))
    print('SVM ID : ' + get_SVM_id())
    print('Random Forest ID : ' + get_RandomForest_id())
    print('Naive Bayes ID : ' + get_NaiveBayes_id())
    print('Optimal hyperparameters (PFT, SVM) : ' + str(get_optimal_hyperparameters('PFT', 'SVM')))
    print('Optimal hyperparameters (PFT, RF) : ' + str(get_optimal_hyperparameters('PFT', 'RF')))
    print('Optimal hyperparameters (PFT, GNB) : ' + str(get_optimal_hyperparameters('PFT', 'GNB')))
    print('Optimal hyperparameters (TCT, SVM) : ' + str(get_optimal_hyperparameters('TCT', 'SVM')))
    print('Optimal hyperparameters (TCT, RF) : ' + str(get_optimal_hyperparameters('TCT', 'RF')))
    print('Optimal hyperparameters (TCT, GNB) : ' + str(get_optimal_hyperparameters('TCT', 'GNB')))
