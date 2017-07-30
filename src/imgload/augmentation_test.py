import unittest

import scipy.misc as misc

import augmentation as aug
import definitions as defs
import utilities.utilities as utils

class PreprocessingTest(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/train'
        self.test_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/test'
        self.image_width = 182
        self.batch_size = 4
        self.num_labels = len(utils.get_labels(self.training_directory))
    
    def test_resize_and_random_crop(self):
        resize = aug.resize_and_random_crop(self.image_width)
        resized_image = resize(
            misc.imread(self.training_directory + '/DOL/img_00165.jpg'))
        self.assertTrue(max(resized_image.shape[:2]) >= self.image_width)
        self.assertEqual(self.image_width, min(resized_image.shape[:2]))
    
    def test_resize_and_center_crop(self):
        resize = aug.resize_and_center_crop(self.image_width)
        resized_image = resize(
            misc.imread(self.training_directory + '/DOL/img_00165.jpg'))
        self.assertTrue(max(resized_image.shape[:2]) >= self.image_width)
        self.assertEqual(self.image_width, min(resized_image.shape[:2]))
