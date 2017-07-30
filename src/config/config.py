import collections
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
    attribute, or the type attribute is neither "int" nor "float" nor "str",
    then the type attribute is assumed to be "str".
    
    The resulting dictionary of parameters is defined by calling the appropriate
    type casting function on the text within the param element. It is equivalent
    to the below definition--except for invalid type_conversion_function values,
    which are handled as described above. 
        {
            "param1": type_conversion_function1("value1"),
            ...,
            "paramN": type_conversion_functionN("valueN")
        }
    
    Arguments:
    file_path -- the location of the config file (relative or absolute). 
    """
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    
    type_cast_functions = collections.defaultdict(
        lambda: str,
        {'int': int, 'float': float})
    
    config_params = {}
    for child in root:
        param_type = child.attrib.get('type') or 'str'
        type_cast = type_cast_functions[param_type]
        config_params[child.tag] = type_cast(child.text)
    return config_params
