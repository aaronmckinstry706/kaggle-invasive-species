import math
import Queue
import threading
import time

import keras.preprocessing.image as image

import augmentation as aug

def get_training_generator(directory, desired_width, batch_size):
    """Convenience function for returning the training data generator G. G
    yields randomly-augmented images from the training dataset, as well as their
    labels. 
    """
    training_generator = image.ImageDataGenerator(
        rotation_range=15.0,
        zoom_range=[1.0, math.sqrt(2)*math.cos(math.pi/4 - 15.0/180.0*math.pi)],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=aug.resize_and_random_crop(desired_width))
    return training_generator.flow_from_directory(
        directory, batch_size=batch_size,
        target_size=(desired_width, desired_width), 
        shuffle=True)

def get_validation_generator(directory, desired_width, batch_size):
    """Convenience function for returning the validation data generator G. G
    yields resized and center-cropped images from the validation dataset, as
    well as their labels. 
    """
    validation_generator = image.ImageDataGenerator(
        preprocessing_function=aug.resize_and_center_crop(desired_width))
    return validation_generator.flow_from_directory(
        directory, batch_size=batch_size,
        target_size=(desired_width, desired_width), shuffle=False)

def get_test_generator(directory, desired_width, batch_size):
    """Convenience function for returning the testing data generator G. G yields
    resized and center-cropped images from the testing dataset. 
    """
    test_generator = image.ImageDataGenerator(
        preprocessing_function=aug.resize_and_center_crop(desired_width))
    return test_generator.flow_from_directory(
        directory, batch_size=batch_size, class_mode=None,
        target_size=(desired_width, desired_width), shuffle=False)

def get_threaded_generator(image_data_generator, num_data_points, num_threads=1):
    """Produces a multi-threaded generator wrapped around the given
    image_data_generator. 
    
    Arguments:
    image_data_generator -- A generator produced by calling flow_from_directory()
                            on a keras.preprocessing.image.ImageDataGenerator
                            object.
    num_data_points -- The total number of images in the data set. 
    
    Keywork Arguments:
    num_threads -- The number of threads concurrently drawing on
                   image_data_generator. 
    
    Returns:
    A multi-threaded generator. 
    """
    # Credit for this code goes to f0K's comment in the link below.
    # https://github.com/Lasagne/Lasagne/issues/12#issuecomment-58896851
    
    queue = Queue.Queue(maxsize=50)
    sentinel = object()
    stop_event = threading.Event()
    
    # Producer puts items in the Queue.
    def producer():
        while not stop_event.is_set():
            item = next(image_data_generator)
            success = False
            while not success and not stop_event.is_set():
                try:
                    queue.put_nowait(item)
                    success = True
                except Queue.Full:
                    time.sleep(0.05)

    # Start producers in background threads.
    threads = [threading.Thread(target=producer)
               for i in range(0, num_threads)]
    for thread in threads:
        thread.daemon = True
        thread.start()
    
    def get_image_count(item):
        if type(item) == tuple:
            # Then image_data_generator is yielding a (labels, images) tuple. 
            return item[0].shape[0]
        else:
            # Then image_data_generator is yielding just an images numpy array.
            return item.shape[0]
    
    # Run as consumer: read items from queue, in current thread.
    count = 0
    item = queue.get()
    while type(item) != type(sentinel):
        yield item
        count += get_image_count(item)
        if count >= num_data_points:
            stop_event.set()
            break
        queue.task_done()
        item = queue.get()
