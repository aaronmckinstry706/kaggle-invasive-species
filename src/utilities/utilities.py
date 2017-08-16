import os
import os.path as path
import random

import matplotlib.pyplot as pyplot
import numpy

def display_history(training_loss_history, validation_loss_history,
                    gradient_history, variance_window=20, recent_window=100):
    NUM_PLOT_ROWS = 3
    NUM_PLOT_COLUMNS = 2
    
    recent_iterations = range(len(training_loss_history))[-1*recent_window:]
    
    pyplot.clf()
    pyplot.gcf().set_size_inches(12, 8)
    
    # Training/validation losses.
    pyplot.subplot(NUM_PLOT_ROWS, NUM_PLOT_COLUMNS, 1)
    
    pyplot.plot(training_loss_history, label="Training")
    pyplot.title('Loss')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss at iteration')
    
    validation_loss_x = [validation_loss_history[i][0]
                         for i in range(len(validation_loss_history))]
    validation_loss_y = [validation_loss_history[i][1]
                         for i in range(len(validation_loss_history))]
    pyplot.plot(validation_loss_x, validation_loss_y, label="Validation")
    

    pyplot.xlim(xmin=0)
    pyplot.ylim(ymin=0)
    
    pyplot.legend()
    
    # Recent training/validation loss.
    pyplot.subplot(NUM_PLOT_ROWS, NUM_PLOT_COLUMNS, 2)
    
    pyplot.plot(recent_iterations,
                training_loss_history[-1*recent_window:],
                label="Training")
    pyplot.title('Recent Loss')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss at iteration')
    
    validation_loss_x = [validation_loss_history[i][0]
                         for i in range(len(validation_loss_history))
                         if validation_loss_history[i][0] > recent_iterations[0]]
    validation_loss_y = [validation_loss_history[i][1]
                         for i in range(len(validation_loss_history))
                         if validation_loss_history[i][0] > recent_iterations[0]]
    pyplot.plot(validation_loss_x, validation_loss_y, label="Validation")
    
    pyplot.ylim(ymin=0)
    
    pyplot.legend()
    
    # Gradient norm.
    pyplot.subplot(NUM_PLOT_ROWS, NUM_PLOT_COLUMNS, 3)
    pyplot.plot(gradient_history)
    pyplot.title('Gradient L2 Norm')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Gradient L2 norm at iteration')
    pyplot.xlim(xmin=0)
    pyplot.ylim(ymin=0)

    # Recent gradient norm.
    pyplot.subplot(NUM_PLOT_ROWS, NUM_PLOT_COLUMNS, 4)
    pyplot.plot(recent_iterations, gradient_history[-1*recent_window:])
    pyplot.title('Recent Gradient L2 Norm')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Gradient L2 norm at iteration')
    pyplot.ylim(ymin=0)
    
    if len(gradient_history) > variance_window:
        pyplot.subplot(NUM_PLOT_ROWS, NUM_PLOT_COLUMNS, 5)
        variances = []
        for i in range(len(gradient_history) - variance_window):
            variances.append(
                numpy.var(gradient_history[i : i + variance_window]))
        x_values = [i + variance_window - 1
                    for i in range(len(gradient_history) - variance_window)]
        pyplot.plot(x_values, variances)
        pyplot.title('Gradient Variance')
        pyplot.xlabel('Iteration')
        pyplot.ylabel('Gradient variance of ' + str(variance_window) + ' previous iterations')
        pyplot.xlim(xmin=0)
        pyplot.ylim(ymin=0)
    
    pyplot.gcf().subplots_adjust(hspace=1.0)
    
    pyplot.pause(0.0001)

def get_labels(train_directory):
    """Gets a list of all directory names in the training directory. Each
    directory name is a label.
    
    Returns:
        A list of strings, each of which is a label.
    """
    labels = []
    for directory in os.listdir(train_directory):
        labels.append(directory)
    labels.sort() # Just for deterministic behavior in unit tests.
    return labels

def get_absolute_paths(root_directory):
    """Gets a list of all path names in root_directory that have extension
    '.jpg'. All returned paths are absolute. 
    
    Args:
        root_directory -- The path for the directory that will be searched.
        file_extension -- A string containing a file extension used to
                          determine which files to include.
    Returns:
        A list of file paths, each relative to the current directory ('.').
        Every file in root_directory (or its subdirectories) that ends in
        file_extension is included in this list; no other files are
        included.
    """
    file_extension = '.jpg'
    file_paths = []
    for directory, _, file_names in os.walk(root_directory):
        valid_file_names = (
                name for name in file_names
                if name[-len(file_extension):] == file_extension)
        valid_file_paths = (path.abspath(directory + '/' + name)
                            for name in valid_file_names)
        file_paths.extend(valid_file_paths)
    return file_paths

def randomly_divide_pretraining_data(src_dir, destination_dir):
    """Takes the pretraining data and copies it into two different directories:
    project_root/data/pretraining/0, and project_root/data/pretraining/1. Each
    image is randomly put into one of the two directories. 
    """
    src_dir = path.abspath(src_dir)
    destination_dir = path.abspath(destination_dir)
    for subdir in ['0', '1']:
        subdir_path = destination_dir + '/' + subdir
        if not path.exists(subdir_path):
            os.makedirs(subdir_path)
    for image_path in get_absolute_paths(src_dir):
        image_name = path.basename(image_path)
        destination_image_path = (destination_dir + "/"
                                  + str(random.randint(0, 1)) + "/"
                                  + image_name)
        os.system("mv " + image_path + " " + destination_image_path)

def separate_validation_set(training_dir, validation_dir, split=0.1):
    training_dir = path.abspath(training_dir)
    validation_dir = path.abspath(validation_dir)
    labels = get_labels(training_dir)
    for label in labels:
        validation_label_path = validation_dir + '/' + label
        if not path.exists(validation_label_path):
            os.makedirs(validation_label_path)
    label_paths = [training_dir + '/' + label for label in labels]
    for label_path in label_paths:
        image_paths = get_absolute_paths(label_path)
        validation_image_paths = random.sample(
            image_paths, int(split*len(image_paths)))
        for validation_image_path in validation_image_paths:
            image_destination_path = validation_image_path.replace(
                training_dir, validation_dir)
            os.system(
                "mv " + validation_image_path + " " + image_destination_path)

def recombine_validation_and_training(validation_dir, training_dir):
    training_dir = path.abspath(training_dir)
    validation_dir = path.abspath(validation_dir)
    image_paths = get_absolute_paths(validation_dir)
    for image_path in image_paths:
        os.system("mv " + image_path + " "
                  + image_path.replace(validation_dir, training_dir))

if __name__ == '__main__':
    recombine_validation_and_training('data/validation', 'data/train')
