import lasagne
import lasagne.layers as layers
import theano
import theano.tensor as tensor

# Create a dummy network.
input = tensor.tensor4(name="image_batch")
network = layers.InputLayer(
    input_var=input, shape=(None, 3, 224, 224), name="input")
network = layers.Conv2DLayer(
    incoming=network, num_filters=2, filter_size=(3,3))
network = layers.GlobalPoolLayer(incoming=network, pool_function=tensor.max)
network = layers.NonlinearityLayer(incoming=network, nonlinearity=lasagne.nonlinearities.softmax)
output = layers.get_output(layer_or_layers=network)

network_function = theano.function([input], output)
print(network_function([[[[0 for k in range(224)] for j in range(224)] for i in range(3)]]))