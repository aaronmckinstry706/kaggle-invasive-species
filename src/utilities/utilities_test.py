import os
import os.path as path
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
        relative_paths = utils.get_absolute_paths(
            self.training_directory)
        self.assertEqual(184, len(relative_paths))
        relative_paths.sort()
        self.assertTrue('img_00165.jpg' in relative_paths[0])
        self.assertTrue('img_00325.jpg' in relative_paths[1])
        self.assertTrue('img_00348.jpg' in relative_paths[2])
    
class UtilitiesTest_FileMovers(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/train'
        self.validation_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/validation'
        self.pretrain_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/pretrain'
        self.pretrain_src_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/pretrain_src'
    
    def undo_pretraining_division(self):
        source_paths = utils.get_absolute_paths(self.pretrain_directory)
        for source_path in source_paths:
            destination_path = self.pretrain_src_directory + "/" + path.basename(source_path)
            os.system("mv " + source_path + " " + destination_path)
    
    def test_randomly_divide_pretraining_data(self):
        self.undo_pretraining_division()
        original_file_names = set(
            [path.basename(p) for p in utils.get_absolute_paths(self.pretrain_src_directory)])
        utils.randomly_divide_pretraining_data(self.pretrain_src_directory, self.pretrain_directory, 3)
        actual_file_names = set(
            [path.basename(p) for p in utils.get_absolute_paths(self.pretrain_directory)])
        self.assertSetEqual(original_file_names, actual_file_names)
        self.undo_pretraining_division()
    
    def test00_separate_validation_set_splitOne(self):
        utils.separate_validation_set(self.training_directory,
                                          self.validation_directory,
                                          1.0)
        self.assertEqual(
            0,
            len(utils.get_absolute_paths(self.training_directory)))
        self.assertEqual(
            184,
            len(
                utils.get_absolute_paths(
                    self.validation_directory)))
    
    def test01_recombine_validation_and_training_splitOne(self):
        utils.recombine_validation_and_training(
            self.validation_directory,
            self.training_directory)
        self.assertEqual(
            184,
            len(utils.get_absolute_paths(self.training_directory)))
        self.assertEqual(
            0,
            len(
                utils.get_absolute_paths(
                    self.validation_directory)))
    
    def test10_separate_validation_set_splitQuarter(self):
        utils.separate_validation_set(self.training_directory,
                                          self.validation_directory,
                                          split=0.25)
        self.assertEqual(
            139,
            len(utils.get_absolute_paths(self.training_directory)))
        self.assertEqual(
            45,
            len(
                utils.get_absolute_paths(
                    self.validation_directory)))
    
    def test11_recombine_validation_and_training_splitQuarter(self):
        utils.recombine_validation_and_training(
            self.validation_directory,
            self.training_directory)
        self.assertEqual(
            184,
            len(utils.get_absolute_paths(self.training_directory)))
        self.assertEqual(
            0,
            len(
                utils.get_absolute_paths(
                    self.validation_directory)))
