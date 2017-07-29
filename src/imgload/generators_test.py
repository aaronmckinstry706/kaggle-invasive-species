import unittest

import definitions as defs
import generators as gen
import utilities.utilities as utils

class GeneratorsTest(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/train'
        self.test_directory = defs.UNIT_TEST_RESOURCE_ROOT_DIR + '/data/test'
        self.image_width = 512
        self.batch_size = 4
        self.num_labels = len(utils.get_labels(self.training_directory))
    
    def test_get_training_generator(self):
        training_generator = gen.get_training_generator(
            self.training_directory, self.image_width, self.batch_size)
        self.assertTrue(
            any(['img_00165.jpg' in path
                 for path in training_generator.filenames]))
        batches = 0
        for images, labels in training_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)
            batches += 1
            if batches == 30:
                break
    
    def test_get_validation_generator(self):
        validation_generator = gen.get_validation_generator(
            self.training_directory, self.image_width, self.batch_size)
        self.assertTrue(
            any(['img_00165.jpg' in path
                 for path in validation_generator.filenames]))
        batches = 0
        for images, labels in validation_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)
            batches += 1
            if batches == 30:
                break
    
    def test_get_test_generator(self):
        test_generator = gen.get_test_generator(
            self.test_directory, self.image_width, self.batch_size)
        self.assertTrue(
            any(['img_00120.jpg' in path
                 for path in test_generator.filenames]))
        batches = 0
        for images in test_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            batches += 1
            if batches == 30:
                break
    
    def test_get_threaded_generator__one_thread(self):
        training_generator = gen.get_generator(
            self.training_directory,
            self.image_width,
            self.batch_size,
            gen_type='training')
        threaded_generator = gen.get_threaded_generator(
            training_generator, len(training_generator.filenames),
            num_threads=1)
        for images, labels in threaded_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)

    def test_get_threaded_generator__eight_threads(self):
        training_generator = gen.get_generator(
            self.training_directory,
            self.image_width,
            self.batch_size,
            gen_type='training')
        threaded_generator = gen.get_threaded_generator(
            training_generator, len(training_generator.filenames),
            num_threads=1)
        for images, labels in threaded_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)
