import xml.etree.ElementTree as ET

config_file = 'input_config.xml'
xml_root = ET.parse(config_file).getroot()

def get_from_xpath(path, index=None) :
    tags = xml_root.findall(path)
    if index == None :
        return tags
    if index >= len(tags) :
        return None
    return tags[index].text

def get_blind_split_random_state() :
    return int(get_from_xpath('./random_state_blind_split', 0))

def get_blind_data_size() :
    return float(get_from_xpath('./blind_data_size', 0))

def get_positive_label() :
    return get_from_xpath('./labels/label[@positive=\'1\']', 0)

def get_negative_label() :
    return get_from_xpath('./labels/label[@negative=\'1\']', 0)

def get_labels() :
    label_tags = get_from_xpath('./labels/label')
    labels = []
    for lt in label_tags :
        labels.append(lt.text)
    return labels

def get_target_column(dataset) :
    return int(get_from_xpath('./column_labels/' + dataset.lower() + '/target', 0))

def get_data_column_start(dataset) :
    return int(get_from_xpath('./column_labels/' + dataset.lower() + '/data_start', 0))

def get_data_column_end(dataset) :
    return int(get_from_xpath('./column_labels/' + dataset.lower() + '/data_end', 0))

def check() :
    print('Random State : ' + str(get_blind_split_random_state()))
    print('Blind data size : ' + str(get_blind_data_size()))
    print('Positive label : ' + get_positive_label())
    print('Negative label : ' + get_negative_label())
    print('All labels : ' + str(get_labels()))
    print('PFT Data column start : ' + str(get_data_column_start('PFT')))
    print('PFT Data column end : ' + str(get_data_column_end('PFT')))
    print('PFT Target column : ' + str(get_target_column('PFT')))
    print('TCT Data column start : ' + str(get_data_column_start('TCT')))
    print('TCT Data column end : ' + str(get_data_column_end('TCT')))
    print('TCT Target column : ' + str(get_target_column('TCT')))
