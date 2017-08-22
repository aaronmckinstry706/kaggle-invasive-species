import collections
import xml.etree.ElementTree as ElementTree

def read_param_xml_file(file_path):
    """Reads a parameter file which is formatted as an XML document, and returns
    the resulting dictionary of parameters. The xml should be formatted as
    follows (or should be empty/not exist):
        <?xml version="1.0"?>
        <params>
            <param1>expr1</param1>
            ...
            <paramN>exprN</paramN>
        </params>
    The parameter names param1, ..., paramN must be distinct. Each expression expr1, ..., exprN
    must be a Python expression that results in some value. 
    
    The resulting dictionary of parameters is defined as follows:
        collections.defaultdict(
            lambda: None,
            {"param1": eval("value1"),
             ...,
             "paramN": eval("valueN")})
    
    Arguments:
    file_path -- the location of the param file (relative or absolute). 
    """
    with open(file_path, 'r+') as fh:
        if fh.read() == '':
            fh.write('<?xml version="1.0"><params></params>')
    
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    
    params = {child.tag: eval(child.text) for child in root}
    return collections.defaultdict(lambda: None, params)
