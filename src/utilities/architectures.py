"""Implements All-CNN-C model from All Convolutional Networks by Springenberg et al. Url is at 
[https://arxiv.org/pdf/1412.6806.pdf]. 
"""

import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import theano.tensor as tensor

def all_cnn_c_model(input_var):
    network = layers.InputLayer(input_var=input_var, shape=(None, 3, 32, 32))
    
    network = layers.Conv2DLayer(incoming=network, num_filters=96, filter_size=3, pad='same')
    network = layers.Conv2DLayer(incoming=network, num_filters=96, filter_size=3, pad='same')
    network = layers.Conv2DLayer(incoming=network, num_filters=96, filter_size=3, pad='same',
                                 stride=(2, 2))
    
    network = layers.Conv2DLayer(incoming=network, num_filters=192, filter_size=3, pad='same')
    network = layers.Conv2DLayer(incoming=network, num_filters=192, filter_size=3, pad='same')
    network = layers.Conv2DLayer(incoming=network, num_filters=192, filter_size=3, pad='same',
                                 stride=(2, 2))
    
    network = layers.Conv2DLayer(incoming=network, num_filters=192, filter_size=3, pad='same')
    network = layers.Conv2DLayer(incoming=network, num_filters=192, filter_size=1, pad='same')
    network = layers.Conv2DLayer(incoming=network, num_filters=10, filter_size=1, pad='same')
    
    network = layers.GlobalPoolLayer(incoming=network, pool_function=tensor.mean)
    network = layers.NonlinearityLayer(incoming=network, nonlinearity=nonlinearities.softmax)
    
    return network
