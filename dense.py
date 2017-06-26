from __future__ import print_function
import sys
import time
import theano
import theano.tensor as T
import theano.tensor.extra_ops as ex
import lasagne
import load
import mini_batch
import numpy
import fisher_info as info
import save_param as files
import preprocessing
from scipy.misc import toimage
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

numpy.random.seed(25)
def build_net(input_var):


    l_in = lasagne.layers.InputLayer(shape=(None,784),
                                     input_var=input_var)

    l_hid_0  =lasagne.layers.DenseLayer(l_in,num_units=512,nonlinearity=lasagne.nonlinearities.rectify,
                                       b = None)

    l_hid_1 = lasagne.layers.DenseLayer(
            l_hid_0,num_units = 256,
            nonlinearity=lasagne.nonlinearities.rectify,b = None)

    l_out = lasagne.layers.DenseLayer(

            l_hid_1, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,b = None)

    return l_out,l_hid_0,l_hid_1



print("Loading data...")
X_train, y_train, X_val, y_val,X_test,y_test= load.load_minst()
#X_train, y_train, X_val, y_val = load.load_cifar10()
pass_weight = None

input_var = T.fmatrix('inputs')
target_var = T.ivector('targets')

print("Building model and...")

network,ly0,ly1 = build_net(input_var)
prediction = lasagne.layers.get_output(network)
ly0_l1_penalty = regularize_layer_params(ly0, l1) * 1e-4
ly1_l1_penalty = regularize_layer_params(ly1, l1) * 1e-4
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()#+ly0_l1_penalty+ly1_l1_penalty

# Get network params, with specifications of manually updated ones
params = lasagne.layers.get_all_params(network, trainable=True)
gradient = theano.grad(loss, params)
updates = lasagne.updates.sgd(loss,params,learning_rate=0.01)
#updates = lasagne.updates.adam(loss,params)
#updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss,test_acc, test_prediction])
gradient_fn = theano.function([input_var, target_var], gradient)



# The training loop
print("Starting training...")
num_epochs = 20
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:

    train_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in mini_batch.iterate_minibatches(X_train, y_train, 500, shuffle=True):

        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1


    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0

    for batch in mini_batch.iterate_minibatches(X_val, y_val, 500, shuffle=False):

        inputs, targets = batch
        err, acc,pre = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1


        # Then we print the results for this epoch:

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))


val_err,acc,val_predict = val_fn(X_val,y_val)
print(val_predict.shape)
information = info.evaluate_fisher_info(gradient_fn,X_val,val_predict,y_val,params)
files.save_information(information,0)
files.save_param(params)

import matplotlib.pyplot as plt
plt.hist(information[1].flatten(), bins='auto')  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
"""
print("Evaluating Fisher Information")
information = Info.evaluate_fisher_info(gradient_fn,X_val[::10],y_val[::10],params)
print(numpy.max(information[1]))

"""
