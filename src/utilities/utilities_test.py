import unittest

import definitions as defs
import utilities as utils

class UtilitiesTest_PathReaders(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/train'
    
    def test_get_labels(self):
        labels = utils.get_labels(self.training_directory)
        labels.sort()
        self.assertListEqual(
            labels,
            ["DOL", "LAG"])
    
    def test_get_relative_paths(self):
        relative_paths = utils.get_relative_paths(
            self.training_directory)
        self.assertEqual(184, len(relative_paths))
        relative_paths.sort()
        self.assertTrue('img_00165.jpg' in relative_paths[0])
        self.assertTrue('img_00325.jpg' in relative_paths[1])
        self.assertTrue('img_00348.jpg' in relative_paths[2])
    
    def test_get_learning_rate(self):
        with open('learning_rate.input', 'w') as lri:
            lri.write('')
        self.assertEqual(1.0, utils.get_learning_rate(None, None, 1.0))
        with open('learning_rate.input', 'w') as lri:
            lri.write(str(2.0))
        self.assertEqual(2.0, utils.get_learning_rate(None, None, 1.0))

class UtilitiesTest_FileMovers(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/train'
        self.validation_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/validation'
    
    def test00_separate_validation_set_splitOne(self):
        utils.separate_validation_set(self.training_directory,
                                          self.validation_directory,
                                          1.0)
        self.assertEqual(
            0,
            len(utils.get_relative_paths(self.training_directory)))
        self.assertEqual(
            184,
            len(
                utils.get_relative_paths(
                    self.validation_directory)))
    
    def test01_recombine_validation_and_training_splitOne(self):
        utils.recombine_validation_and_training(
            self.validation_directory,
            self.training_directory)
        self.assertEqual(
            184,
            len(utils.get_relative_paths(self.training_directory)))
        self.assertEqual(
            0,
            len(
                utils.get_relative_paths(
                    self.validation_directory)))
    
    def test10_separate_validation_set_splitQuarter(self):
        utils.separate_validation_set(self.training_directory,
                                          self.validation_directory,
                                          split=0.25)
        self.assertEqual(
            139,
            len(utils.get_relative_paths(self.training_directory)))
        self.assertEqual(
            45,
            len(
                utils.get_relative_paths(
                    self.validation_directory)))
    
    def test11_recombine_validation_and_training_splitQuarter(self):
        utils.recombine_validation_and_training(
            self.validation_directory,
            self.training_directory)
        self.assertEqual(
            184,
            len(utils.get_relative_paths(self.training_directory)))
        self.assertEqual(
            0,
            len(
                utils.get_relative_paths(
                    self.validation_directory)))
