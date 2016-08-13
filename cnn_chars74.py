from __future__ import print_function

import sys
import os
import time

import theano
from theano import tensor as T
import numpy as np
from load_data import load_dataset
import lasagne
import matplotlib.pyplot as plt

# print("Loading data...")
# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)
# print(X_test.shape)
# print(y_test.shape)

def getClassNumbers(an_class):
        # dictionary = ['1','2','3','4','5','6','7','8','9','0','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        # 'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
        # 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        n_class = []
        dictionary_ = {'1':'0','2':'1','3':'2','4':'3','5':'4','6':'5','7':'6','8':'7','9':'8','0':'9','a':'10', 'b':'11', 'c':'12', 'd':'13', 'e':'14', 'f':'15', 'g':'16', 'h':'17', 'i':'18', 'j':'19', 'k':'20', 'l':'21', 
        'm':'22', 'n':'23', 'o':'24', 'p':'25', 'q':'26','r':'27', 's':'28', 't':'29', 'u':'30', 'v':'31', 'w':'32', 'x':'33', 'y':'34', 'z':'35','A':'36', 'B':'37', 'C':'38', 'D':'39', 'E':'40', 'F':'41', 'G':'42', 'H':'43', 'I':'44', 'J':'45', 'K':'46', 'L':'47', 
        'M':'48', 'N':'49', 'O':'50', 'P':'51', 'Q':'52','R':'53', 'S':'54', 'T':'55', 'U':'56', 'V':'57', 'W':'58', 'X':'59', 'Y':'60', 'Z':'61'}

        for each_entry in an_class:
            n_class.append(dictionary_[each_entry.decode("utf-8")])
        # print(type(n_class))
        n_class = np.asarray(n_class,dtype=np.uint8)
        # print(type(n_class))
        return n_class



# print(dictionary_[arr[0].decode("utf-8") ])

def two_layer_model(input_var = None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 20, 20),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)

    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=62,
            nonlinearity=lasagne.nonlinearities.softmax)
    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# def main(model='mlp', num_epochs=500):
    # Load the dataset
print("Loading data...")
X_train, y_train, X_val, y_val, X_test = load_dataset()
y_train = getClassNumbers(y_train)
y_val = getClassNumbers(y_val)
# print(X_train.shape)
# print(y_train.shape)





input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
print("Building model and compiling functions...")
network = two_layer_model(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)

test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

get_output = theano.function([input_var], test_prediction)


print("Starting training...")
num_epochs = 300
tr_loss = []
val_loss = []

    # We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    tr_loss.append(train_err / train_batches)
    val_loss.append(val_err / val_batches)

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))



# outputValue = lasagne.layers.get_output(network, X_test
res = np.argmax(get_output(X_test), axis=1)
# print(res)
a = []
dictionary = ['1','2','3','4','5','6','7','8','9','0','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
        'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for ind, res_ in enumerate(res):
    # print([ind , dictionary[res_]])
    a.append([ind+6284, dictionary[res_]])

np.savetxt("foo.csv", a, delimiter=",",fmt="%s",header=('ID,Class'),comments='') 










