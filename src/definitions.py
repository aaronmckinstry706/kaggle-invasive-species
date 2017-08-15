import os.path as path

SRC_ROOT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT_DIR = path.abspath(SRC_ROOT_DIR + '/..')
UNIT_TEST_RESOURCE_ROOT_DIR = path.abspath(
    PROJECT_ROOT_DIR + '/unit_test_resources')
DATA_DIR = PROJECT_ROOT_DIR + '/data'
TRAINING_DATA_DIR = DATA_DIR + '/training'
VALIDATION_DATA_DIR = DATA_DIR + '/validation'
PRETRAINING_DATA_DIR = DATA_DIR + '/pretraining'
PRETRAINING_VALIDATION_DATA_DIR = DATA_DIR + '/pretraining_validation'
TESTING_DATA_DIR = DATA_DIR + '/testing'
