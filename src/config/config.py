import xml.etree.ElementTree as ElementTree

def read_config_file(file_path):
    """Reads a config file which is formatted as an XML document, and returns
    the resulting dictionary of parameters. The xml should be formatted as
    follows:
        <?xml version="1.0"?>
        <config>
            <param1 type="type_conversion_function1">value1</param1>
            ...
            <paramN type="type_conversion_functionN">valueN</paramN>
        </config>
    The parameter names param1, ..., paramN must be distinct. Each type must be
    one of "int", "float", or "str". If a parameter element does not have a type
    attribute, then the type attribute is assumed to be "str". 
    
    Arguments:
    file_path -- the location of the config file. 
    """
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    
    type_cast_functions = {
        'int': int,
        'float': float,
        'str': str
    }
    
    config_params = {}
    for child in root:
        if 'type' in child.attrib.keys():
            if child.attrib['type'] not in type_cast_functions.keys():
                param_type = 'str'
            else:
                param_type = child.attrib['type']
            type_cast = type_cast_functions[param_type]
            value = type_cast(child.text)
        else:
            value = child.text
        config_params[child.tag] = value
    return config_params
