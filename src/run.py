import logging
import time

import lasagne
import lasagne.layers as layers
import lasagne.objectives as objectives
import lasagne.regularization as regularization
import lasagne.updates as updates
import matplotlib.pyplot as pyplot
import numpy
import theano
import theano.tensor as tensor

import config.config as config
import definitions as defs
import imgload.generators as generators
import utilities.utilities as utils
import datetime

logging.getLogger().setLevel(logging.INFO)

metaparams = config.read_param_xml_file(defs.PROJECT_ROOT_DIR + "/config.xml")

# Create a network, labels, outputs, and parameters with symbolic variables.

input_batch = tensor.tensor4(name="image_batch")

network = layers.InputLayer(
    input_var=input_batch,
    shape=(None, 3, metaparams['image_width'], metaparams['image_width']),
    name="input")
network = layers.Conv2DLayer(
    incoming=network,
    num_filters=2,
    filter_size=(3,3))
network = layers.GlobalPoolLayer(
    incoming=network,
    pool_function=tensor.max)
network = layers.NonlinearityLayer(
    incoming=network,
    nonlinearity=lasagne.nonlinearities.softmax)

training_output_batch = layers.get_output(network, deterministic=False)
testing_output_batch = layers.get_output(network, deterministic=True)
network_parameters = layers.get_all_params(network, trainable=True)
label_batch = tensor.matrix(name="label_batch")

# Create objective functions, gradient norm, learning rate, network updates. 

unregularized_training_loss_batch = tensor.sum(
    objectives.categorical_crossentropy(
        training_output_batch,
        label_batch)) / metaparams['batch_size']
training_loss_batch = (unregularized_training_loss_batch
    + regularization.regularize_network_params(network, regularization.l2))

validation_loss_batch = tensor.sum(
    objectives.categorical_crossentropy(testing_output_batch, label_batch))

gradients = tensor.concatenate(
    [tensor.flatten(tensor.grad(training_loss_batch, params))
     for params in layers.get_all_params(network, trainable=True)])
gradient_norm = gradients.norm(2)

learning_rate = theano.shared(numpy.float32(metaparams['learning_rate']))

network_updates = updates.nesterov_momentum(
    loss_or_grads=training_loss_batch,
    params=network_parameters,
    learning_rate=learning_rate,
    momentum=metaparams['momentum_term'])

# Compile training/validation/testing functions. 

logging.info('Compiling training/validation functions.')
train = theano.function(
    inputs=[input_batch, label_batch],
    outputs=[training_output_batch, training_loss_batch, gradient_norm],
    updates=network_updates)
validate = theano.function(
    inputs=[input_batch, label_batch],
    outputs=[testing_output_batch, validation_loss_batch])

# Reset leftover stuff from earlier runs. 

logging.info('Cleaning up training/validation split from previous runs.')
utils.recombine_validation_and_training(defs.VALIDATION_DATA_DIR,
                                        defs.TRAINING_DATA_DIR)

logging.info('Splitting training and validation images.')
utils.separate_validation_set(defs.TRAINING_DATA_DIR,
                              defs.VALIDATION_DATA_DIR,
                              split=metaparams['validation_split'])

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
        iteration += 1
        previous_gradient_norms.append(numpy.asscalar(current_gradient_norm))
    
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
