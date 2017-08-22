import logging

import lasagne.layers as layers
import lasagne.objectives as objectives
import lasagne.regularization as regularization
import lasagne.updates as updates
import numpy
import theano
import theano.tensor as tensor

import config.config as config
import definitions as defs
import utilities.training as training
import utilities.utilities as utils
import utilities.architectures as architectures

if __name__ == '__main__':
    
    logging.getLogger().setLevel(logging.INFO)
    
    metaparams = config.read_param_xml_file(
        defs.PROJECT_ROOT_DIR + "/config.xml")
    
    # Create a network, labels, outputs, and parameters with symbolic variables.
    
    input_batch = tensor.tensor4(name="image_batch")
    
    network = architectures.all_cnn_c_model(input_var=input_batch)
    
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
        + 0.001*regularization.regularize_network_params(network, regularization.l2))
    
    # Technically not accuracy; just number correct out of the batch.
    validation_accuracy_batch = tensor.sum(
        tensor.eq(tensor.argmax(testing_output_batch, axis=1), tensor.argmax(label_batch, axis=1)))
    
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
        outputs=[testing_output_batch, validation_loss_batch, validation_accuracy_batch])
    
    # Reset leftover stuff from earlier runs. 
    
    logging.info('Assigning random labels to pretraining data.')
    utils.randomly_divide_pretraining_data(defs.PRETRAINING_SOURCE_DIR, defs.PRETRAINING_DATA_DIR)
    
    logging.info('Cleaning up training/validation splits from previous runs.')
    utils.recombine_validation_and_training(defs.VALIDATION_DATA_DIR,
                                            defs.TRAINING_DATA_DIR)
    utils.recombine_validation_and_training(defs.PRETRAINING_VALIDATION_DATA_DIR,
                                            defs.PRETRAINING_DATA_DIR)
    
    logging.info('Splitting training and validation images for training and pretraining.')
    utils.separate_validation_set(defs.TRAINING_DATA_DIR,
                                  defs.VALIDATION_DATA_DIR,
                                  split=metaparams['validation_split'])
    utils.separate_validation_set(defs.PRETRAINING_DATA_DIR,
                                  defs.PRETRAINING_VALIDATION_DATA_DIR,
                                  split=metaparams['pretraining_validation_split'])
    
    training.train_network(metaparams=metaparams,
                           train=train,
                           validate=validate,
                           pretrain=True,
                           learning_rate=learning_rate)
    
    training.train_network(metaparams=metaparams,
                           train=train,
                           validate=validate,
                           pretrain=False,
                           learning_rate=learning_rate)
