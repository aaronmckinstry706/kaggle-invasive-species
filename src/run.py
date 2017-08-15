import logging

import lasagne
import lasagne.layers as layers
import lasagne.objectives as objectives
import lasagne.regularization as regularization
import lasagne.updates as updates
import numpy
import theano
import theano.tensor as tensor

import config.config as config
import definitions as defs
import utilities.utilities as utils

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

training_loss_batch = tensor.sum(
        objectives.categorical_crossentropy(training_output_batch,
                                            label_batch)) / metaparams['batch_size'] \
    + regularization.regularize_network_params(network, regularization.l2)

validation_loss_batch = \
    tensor.sum(
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
    outputs=[training_output_batch, training_loss_batch, gradient_norm])
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
gradient_norms = []

# Use the Early Stopping technique for determining when to end training.

best_validation_loss = float("inf")
remaining_patience = metaparams['patience']

# And now we train. 


