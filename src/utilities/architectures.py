"""Implements All-CNN-C model from All Convolutional Networks by Springenberg et al. Url is at 
[https://arxiv.org/pdf/1412.6806.pdf]. The main difference is the addition of batch normalization
in each layer, which should eliminate the need for contrastive normalization (and, hopefully, image
whitening). 
"""

import lasagne.init as init
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import theano.tensor as tensor

def all_cnn_c_model(input_var):
    he_normal = init.HeNormal(gain='relu')
    
    network = layers.InputLayer(input_var=input_var, shape=(None, 3, 32, 32))
    
    network = layers.batch_norm(
        layers.Conv2DLayer(
            incoming=network, num_filters=96, filter_size=3, pad='same', W=he_normal))
    network = layers.batch_norm(
        layers.Conv2DLayer(
            incoming=network, num_filters=96, filter_size=3, pad='same', W=he_normal))
    network = layers.batch_norm(
        layers.Conv2DLayer(incoming=network, num_filters=96, filter_size=3, pad='same',
                           stride=(2, 2), W=he_normal))
    
    network = layers.batch_norm(
        layers.Conv2DLayer(
            incoming=network, num_filters=192, filter_size=3, pad='same', W=he_normal))
    network = layers.batch_norm(
        layers.Conv2DLayer(
            incoming=network, num_filters=192, filter_size=3, pad='same', W=he_normal))
    network = layers.batch_norm(
        layers.Conv2DLayer(
            incoming=network, num_filters=192, filter_size=3, pad='same', stride=(2, 2),
            W=he_normal))
    
    network = layers.batch_norm(
        layers.Conv2DLayer(incoming=network, num_filters=192, filter_size=3, pad='same', W=he_normal))
    network = layers.batch_norm(
        layers.Conv2DLayer(incoming=network, num_filters=192, filter_size=1, pad='same', W=he_normal))
    network = layers.Conv2DLayer(incoming=network, num_filters=10, filter_size=1, pad='same', W=he_normal)
    
    network = layers.GlobalPoolLayer(incoming=network, pool_function=tensor.mean)
    network = layers.NonlinearityLayer(incoming=network, nonlinearity=nonlinearities.softmax)
    
    return network
