import datetime
import logging
import time

import matplotlib.pyplot as pyplot
import numpy

import definitions as defs
import imgload.generators as generators
import utilities as utils

# TODO: Add (...network, best_network) as parameters; store best network params.
def train_network(metaparams, train, validate, validate_accuracy, pretrain, learning_rate):
    """Takes in a parameter dict, a training function, and a validation function
    and then trains the network. 
    
    Args:
        metaparams -- A dictionary of parameters. Must have the following:
                          'patience' (nonnegative int),
                          'epochs' (positive int),
                          'image_width' (positive int),
                          'batch_size' (positive int), and
                          'threads' (positive int). 
        
        train -- A callable that takes as inputs:
                     1) a numpy array of 3-color-channel images, and
                     2) a numpy array of 1-dimensional image labels. 
                 This callable is what actually trains the network. 
        
        validate -- A callable that takes the same inputs as the train function. 
        
        validate_accuracy -- A callable that takes the same inputs as the train function.
        
        pretrain -- If this is meant to pretrain a network, True. Otherwise, False. 
        
        learning_rate -- A Theano symbolic scalar (must be shared variable), which controls the
                         learning rate used in the train function's weight updates.
    """
    if pretrain:
        training_directory = defs.PRETRAINING_DATA_DIR
        validation_directory = defs.PRETRAINING_VALIDATION_DATA_DIR
    else:
        training_directory = defs.TRAINING_DATA_DIR
        validation_directory = defs.VALIDATION_DATA_DIR
    
    # Keep record of various metrics as the network is training. 
    
    previous_validation_losses = []
    previous_training_losses = []
    previous_gradient_norms = []
    previous_validation_accuracies = []
    
    # Use the Early Stopping technique for determining when to end training.
    
    best_validation_loss = float("inf")
    remaining_patience = metaparams['patience']
    
    # Train the network, while graphing the metrics in real time.
    
    logging.info(str(metaparams))
    if pretrain:
        logging.info('Pretraining network.')
    else:
        logging.info('Training network.')
    logging.info('Beginning run at ' + str(datetime.datetime.now()) + '.')
    
    pyplot.ion()
    pyplot.close('all')
    pyplot.figure(figsize=(6, 8))
    
    learning_rate.set_value(numpy.float32(metaparams['learning_rate']))
    
    iteration = 0
    epoch = 0
    while epoch < metaparams['epochs']:
        # Taking from https://github.com/Lasagne/Recipes/blob/master/papers/densenet/train_test.py,
        # we will divide learning rate by 10 at 50% and 75% through the total number of epochs.
        if epoch == int(0.5*metaparams['epochs']):
            learning_rate.set_value(numpy.float32(learning_rate.get_value()/10.0))
        elif epoch == int(0.75*metaparams['epochs']):
            learning_rate.set_value(numpy.float32(learning_rate.get_value()/10.0))
        
        epoch_start_time = time.time()
        
        # Train.
        
        training_generator = generators.get_training_generator(
            training_directory,
            metaparams['image_width'],
            metaparams['batch_size'])
        
        threaded_training_generator = generators.get_threaded_generator(
            training_generator,
            len(training_generator.filenames),
            metaparams['threads'])
        
        for images_labels in threaded_training_generator:
            outputs, current_training_loss, current_gradient_norm = train(
                numpy.moveaxis(images_labels[0], 3, 1),
                images_labels[1])
            previous_training_losses.append(current_training_loss + 0.0)
            previous_gradient_norms.append(numpy.asscalar(current_gradient_norm))
            iteration += 1
        
        validation_generator = generators.get_validation_generator(
            validation_directory,
            metaparams['image_width'],
            metaparams['batch_size'])
        
        threaded_validation_generator = generators.get_threaded_generator(
            validation_generator,
            len(validation_generator.filenames),
            metaparams['threads'])
        
        epoch_end_time = time.time()
        
        # Validate.
        
        current_validation_loss = 0.0
        current_validation_accuracy = 0.0
        example_count = 0.0
        for images_labels in threaded_validation_generator:
            outputs, validation_accuracy = validate_accuracy(numpy.moveaxis(images_labels[0], 3, 1),
                                                             images_labels[1])
            current_validation_accuracy += validation_accuracy
            outputs, validation_loss = validate(numpy.moveaxis(images_labels[0], 3, 1),
                                                images_labels[1])
            current_validation_loss += numpy.sum(validation_loss)
            example_count += images_labels[0].shape[0]
        current_validation_loss = current_validation_loss / example_count
        previous_validation_losses.append((iteration, current_validation_loss))
        current_validation_accuracy = current_validation_accuracy / example_count
        previous_validation_accuracies.append((iteration, current_validation_accuracy))
        
        
        # Update best validation loss, best parameters, and patience. 
        
        if not pretrain:
            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                #current_network_params = layers.get_all_params(network)
                #best_network_params = layers.get_all_params(best_network)
                # Set best_network to current network parameters
                #for j in range(len(best_network_params)):
                #    best_network_params[j].set_value(
                #        current_network_params[j].get_value())
                remaining_patience = metaparams['patience']
            else:
                remaining_patience = remaining_patience - 1
        
        # Update graph of various metrics, and log metrics for current epoch.
    
        utils.display_history(
            previous_training_losses,
            previous_validation_losses,
            previous_validation_accuracies,
            previous_gradient_norms,
            variance_window=25,
            recent_window=1000)
        
        logging.info('Epoch ' + str(epoch) + ': '
                    + 'Crossentropy loss over validation set = '
                    + str(current_validation_loss) + ', '
                    + 'Most recent batch loss over training set = '
                    + str(current_training_loss) + ', '
                    + 'Time = '
                    + str(int(round(epoch_end_time - epoch_start_time)))
                    + ' seconds, '
                    + 'Accuracy over validation set = '
                    + str(current_validation_accuracy)
                    + '.')
        
        epoch += 1
        if remaining_patience == 0:
            logging.info('Best validation loss: ' + str(best_validation_loss) + '.')
            break
    
    logging.info('Batch training loss per iteration: '
                + str(previous_training_losses))
    logging.info('Validation loss after each epoch, indexed by iteration: '
                + str(previous_validation_losses))
    logging.info('Gradient norms per iteration: ' + str(previous_gradient_norms))
    logging.info('Validation accuracy after each epoch, indexed by iteration: '
                 + str(previous_validation_accuracies))
    logging.info('Ending run at ' + str(datetime.datetime.now()) + '.')
    
    pyplot.ioff()
