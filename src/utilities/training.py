import datetime
import logging
import time

import matplotlib.pyplot as pyplot
import numpy

import definitions as defs
import imgload.generators as generators
import utilities as utils

# TODO: Add (...network, best_network) as parameters; store best network params.
def train_network(metaparams, train, validate, pretrain):
    """Takes in a parameter dict, a training function, and a validation function
    and then trains the network. 
    
    Args:
        metaparams -- A dictionary of parameters. Must have the following:
                          'patience' (nonnegative int),
                          'iterations' (positive int),
                          'image_width' (positive int),
                          'batch_size' (positive int), and
                          'threads' (positive int). 
        train -- A callable that takes as inputs:
                     1) a numpy array of 3-color-channel images, and
                     2) a numpy array of 1-dimensional image labels. 
                 This callable is what actually trains the network. 
    """
    
    # Keep record of various metrics as the network is training. 
    
    previous_validation_losses = []
    previous_training_losses = []
    previous_gradient_norms = []
    
    # Use the Early Stopping technique for determining when to end training.
    
    best_validation_loss = float("inf")
    remaining_patience = metaparams['patience']
    
    # Train the network, while graphing the metrics in real time.
    
    logging.info(str(metaparams))
    logging.info('Training network.')
    logging.info('Beginning run at ' + str(datetime.datetime.now()) + '.')
    
    pyplot.ion()
    pyplot.figure(figsize=(6, 8))
    
    iteration = 0
    epoch = 0
    while iteration < metaparams['iterations']:
        epoch_start_time = time.time()
        
        # Train.
        
        training_generator = generators.get_training_generator(
            defs.TRAINING_DATA_DIR,
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
            defs.VALIDATION_DATA_DIR,
            metaparams['image_width'],
            metaparams['batch_size'])
        
        threaded_validation_generator = generators.get_threaded_generator(
            validation_generator,
            len(validation_generator.filenames),
            metaparams['threads'])
        
        epoch_end_time = time.time()
        
        # Validate.
        
        current_validation_loss = 0
        example_count = 0
        for images_labels in threaded_validation_generator:
            outputs, validation_loss = validate(numpy.moveaxis(images_labels[0],
                                                               3, 1),
                                                images_labels[1])
            current_validation_loss += numpy.sum(validation_loss)
            example_count += images_labels[0].shape[0]
        current_validation_loss = current_validation_loss / example_count
        previous_validation_losses.append((iteration, current_validation_loss))
        
        # Update best validation loss, best parameters, and patience. 
        
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
            previous_gradient_norms,
            variance_window=25,
            recent_window=50)
        
        logging.info('Epoch ' + str(epoch) + ': '
                    + 'Crossentropy loss over validation set = '
                    + str(current_validation_loss) + ', '
                    + 'Most recent batch loss over training set = '
                    + str(current_training_loss) + ', '
                    + 'Time = '
                    + str(int(round(epoch_end_time - epoch_start_time)))
                    + ' seconds.')
        
        epoch += 1
        if remaining_patience == 0:
            logging.info('Best validation loss: ' + str(best_validation_loss) + '.')
            break
    
    logging.info('Batch training loss per iteration: '
                + str(previous_training_losses))
    logging.info('Validation loss after each epoch, indexed by iteration: '
                + str(previous_validation_losses))
    logging.info('Gradient norms per iteration: ' + str(previous_gradient_norms))
    logging.info('Ending run at ' + str(datetime.datetime.now()) + '.')
    
    pyplot.ioff()
    pyplot.close('all')

# TODO: Add pretraining function. 