import os
import unittest

import config
import definitions as defs

class TestConfig(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.xml_test_file = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/test_config_xml'
        test_file = open(cls.xml_test_file, 'w')
        test_file.write('<?xml version="1.0"?>'
                   '<config>'
                   '<int_param>12</int_param>'
                   '<float_param>12.25</float_param>'
                   '<str_param>"string"</str_param>'
                   '<complicated>{"dict": "dict"}</complicated>'
                   '</config>')
        test_file.close()
    
    @classmethod
    def tearDownClass(cls):
        os.remove(TestConfig.xml_test_file)

    def testReadParamXmlFile(self):
        config_params = config.read_param_xml_file(self.xml_test_file)
        expected_dict = {
            'int_param': 12,
            'float_param': 12.25,
            'str_param': 'string',
            'complicated': {'dict': 'dict'}
        }
        self.assertDictEqual(expected_dict, config_params)
        self.assertEqual(None, config_params['nonexistant_param'])

if __name__ == "__main__":
    unittest.main()