import cPickle
import logging
import os
import os.path as path
import sys

import numpy
import scipy.misc as misc

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# From https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(filepath):
    with open(filepath, 'rb') as fo:
        d = cPickle.load(fo)
    return d

def main():
    if len(sys.argv) != 4:
        print("Usage: python unpack_cifar.py cifar_directory training_directory testing_directory\n" +
              "    cifar_directory -- the directory containing the unzipped/un-tar-ed files" +
              " downloaded from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
            + "    training_directory -- the directory to which it will unpack training images. "
            + "Each image will be put under a subdirectory named after its correct label name.\n"
            + "    testing_directory -- the directory to which it will unpack testing images. Each "
            + "image will be in its label's subdirectory, as with the training directory.\n")
        exit(0)
    
    cifar_directory = path.abspath(sys.argv[1])
    
    training_directory = sys.argv[2]
    if not path.exists(training_directory):
        os.makedirs(training_directory)
    training_directory = path.abspath(training_directory)
    
    testing_directory = sys.argv[3]
    if not path.exists(testing_directory):
        os.makedirs(testing_directory)
    testing_directory = path.abspath(testing_directory)
    
    NUM_IMAGES_PER_FILE = 10000
    NUM_TRAINING_FILES = 5
    NUM_TRAINING_IMAGES = 5*NUM_IMAGES_PER_FILE
    NUM_TESTING_IMAGES = NUM_IMAGES_PER_FILE
    IMAGE_DIMENSIONS = (3, 32, 32)
    
    # Get training data. 
    
    training_images = numpy.ndarray(
        shape=(NUM_TRAINING_IMAGES,) + IMAGE_DIMENSIONS,
        dtype="float32")
    training_labels = []
    for i in range(NUM_TRAINING_FILES):
        filename = 'data_batch_' + str(i + 1)
        data = unpickle(cifar_directory + '/' + filename)
        training_images[i*NUM_IMAGES_PER_FILE : (i+1)*NUM_IMAGES_PER_FILE] = (
            numpy.reshape(data['data'], (10000, 3, 32, 32)))
        training_labels.extend(data['labels'])
    
    # Get testing data.
    
    testing_images = numpy.ndarray(
        shape=(NUM_TESTING_IMAGES,) + IMAGE_DIMENSIONS,
        dtype="float32")
    data = unpickle(cifar_directory + '/test_batch')
    testing_images[:] = numpy.reshape(data['data'], (NUM_TESTING_IMAGES,) + IMAGE_DIMENSIONS)
    testing_labels = data['labels']
    
    # Get meta data.
    
    label_names = unpickle(cifar_directory + '/batches.meta')['label_names']
    
    # Create training/testing label directories. 
    
    for label_name in label_names:
        training_label_path = training_directory + '/' + label_name
        testing_label_path = testing_directory + '/' + label_name
        if not path.exists(training_label_path):
            os.makedirs(training_label_path)
        if not path.exists(testing_label_path):
            os.makedirs(testing_label_path)
    
    # Save all images as files.
    
    for i in range(NUM_TRAINING_IMAGES):
        image = numpy.transpose(training_images[i], (1, 2, 0))
        label = training_labels[i]
        label_name = label_names[label]
        image_destination = training_directory + '/' + label_name + '/' + str(i) + '.jpg'
        misc.imsave(image_destination, image)
    
    for i in range(NUM_TESTING_IMAGES):
        image = numpy.transpose(testing_images[i], (1, 2, 0))
        label = testing_labels[i]
        label_name = label_names[label]
        image_destination = testing_directory + '/' + label_name + '/' + str(i) + '.jpg'
        misc.imsave(image_destination, image)

if __name__ == '__main__':
    main()
