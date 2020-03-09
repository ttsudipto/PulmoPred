import xml.etree.ElementTree as ET

config_file = 'config.xml'

def get_ROOT_PATH() :
    xml_root = ET.parse(config_file).getroot()
    return xml_root.findall('./project-path')[0].text

def get_OUTPUT_PATH() :
    xml_root = ET.parse(config_file).getroot()
    ROOT_PATH = xml_root.findall('./project-path')[0].text
    OP_SUFFIX = xml_root.findall('./output-path-suffix')[0].text
    return ROOT_PATH + OP_SUFFIX
