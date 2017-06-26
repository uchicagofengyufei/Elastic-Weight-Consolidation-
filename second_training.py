from __future__ import print_function
import sys
import time
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Ex
import lasagne
import load
import mini_batch
import numpy
import fisher_info as info
import save_param as files
import preprocessing
from scipy.misc import toimage






numpy.random.seed(25)
def build_net(input_var,pass_weight):

    l_in = lasagne.layers.InputLayer(shape=(None,784),
                                     input_var=input_var)

    l_hid_0  =lasagne.layers.DenseLayer(l_in,num_units=512,nonlinearity=lasagne.nonlinearities.rectify,
                                        W = pass_weight[0],
                                       b = None)

    l_hid_1 = lasagne.layers.DenseLayer(
            l_hid_0,num_units = 256,
            W = pass_weight[1],
            nonlinearity=lasagne.nonlinearities.rectify,b = None)

    l_out = lasagne.layers.DenseLayer(

            l_hid_1, num_units=10,
            W = pass_weight[2],
            nonlinearity=lasagne.nonlinearities.softmax,b = None)

    return l_out, l_hid_0.W, l_hid_1.W, l_out.W



print("Loading data...")
X_train, y_train, X_val, y_val,X_test,y_test= load.load_minst()
X_train_1 = preprocessing.digit_rotate(X_train,90)
X_val_1 = preprocessing.digit_rotate(X_val,90)

#X_train, y_train, X_val, y_val = load.load_cifar10()
pass_weight = files.load_param_tensor()
p_param = files.load_param_raw()
information = files.load_information(0)

input_var = T.fmatrix('inputs')
target_var = T.ivector('targets')
W0 = T.fmatrix("W0")
W1 = T.fmatrix("W1")
W2= T.fmatrix("W2")
I0 = T.fmatrix("I0")
I1 = T.fmatrix("I1")
I2 = T.fmatrix("I2")

print("Building model and...")

network, W_n0, W_n1, W_n2 = build_net(input_var,pass_weight)
#network = build_net(input_var,pass_weight)
prediction = lasagne.layers.get_output(network)
loss_class = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
loss_remember = (I0*lasagne.objectives.squared_error(W0,W_n0)).mean()+ \
                (I1*lasagne.objectives.squared_error(W1,W_n1)).mean()+ \
                (I2*lasagne.objectives.squared_error(W2,W_n2)).mean()
loss = loss_class + 100*loss_remember

# Get network params, with specifications of manually updated ones
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss,params,learning_rate=0.0001)
#updates = lasagne.updates.adam(loss,params,learning_rate=0.00001)
#updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var, target_var,W0,W1,W2,I0,I1,I2], [loss,loss_class, loss_remember], updates=updates)
#train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
#gradient_fn = theano.function([input_var, target_var], gradient)



# The training loop
print("Starting training...")
num_epochs = 250
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:

    train_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in mini_batch.iterate_minibatches(X_train_1, y_train, 500, shuffle=True):

        inputs, targets = batch
        err, cls,rem = train_fn(inputs, targets,p_param[0],p_param[1],p_param[2],
                              information[0],information[1],information[2])
        train_err+=err
        #print(err,cls,rem)
        #train_err += train_fn(inputs, targets)
        train_batches += 1


    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0

    for batch in mini_batch.iterate_minibatches(X_val_1, y_val, 500, shuffle=False):

        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1


        # Then we print the results for this epoch:

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    val_err = 0
    val_acc = 0
    val_batches = 0

    for batch in mini_batch.iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
"""
print("Evaluating Fisher Information")
information = info.evaluate_fisher_info(gradient_fn,X_val[::10],y_val[::10],params)
print(numpy.max(information[1]))
import matplotlib.pyplot as plt
plt.hist(information[1], bins='auto')  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
"""