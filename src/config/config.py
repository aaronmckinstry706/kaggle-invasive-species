import collections
import xml.etree.ElementTree as ElementTree

def read_param_xml_file(file_path):
    """Reads a parameter file which is formatted as an XML document, and returns
    the resulting dictionary of parameters. The xml should be formatted as
    follows (or should be empty/not exist):
        <?xml version="1.0"?>
        <params>
            <param1 type="type_conversion_function1">value1</param1>
            ...
            <paramN type="type_conversion_functionN">valueN</paramN>
        </params>
    The parameter names param1, ..., paramN must be distinct. Each type must be
    one of "int", "float", or "str". If a parameter element does not have a type
    attribute, or the type attribute is neither "int" nor "float" nor "str",
    then the type attribute is assumed to be "str".
    
    The resulting dictionary of parameters is defined by calling the appropriate
    type casting function on the text within the param element. It is equivalent
    to the below definition--except for invalid type_conversion_function values,
    which are handled as described above. 
        collections.defaultdict(
            lambda: None,
            {"param1": type_conversion_function1("value1"),
             ...,
             "paramN": type_conversion_functionN("valueN")})
    
    Arguments:
    file_path -- the location of the param file (relative or absolute). 
    """
    with open(file_path, 'r+') as fh:
        if fh.read() == '':
            fh.write('<?xml version="1.0"><params></params>')
    
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    
    type_cast_functions = collections.defaultdict(
        lambda: str,
        {'int': int, 'float': float})
    
    params = {}
    for child in root:
        param_type = child.attrib.get('type') or 'str'
        type_cast = type_cast_functions[param_type]
        params[child.tag] = type_cast(child.text)
    return collections.defaultdict(lambda: None, params)