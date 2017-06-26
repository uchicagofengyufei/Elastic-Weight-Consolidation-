import numpy as np
import os
import  cPickle

def load_minst():
    #from urllib import urlretrieve
    #def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
       # print("Downloading %s" % filename)
       # urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        #if not os.path.exists(filename):
            #download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        print(data.shape)
        data = data.reshape(-1, 784)

        # The inputs come as bytes, we convert them to float32 in range [0,1].

        # (Actually to range [0, 255/256], for compatibility to the version

        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)

        return data / np.float32(256)

    def load_mnist_labels(filename):

        #if not os.path.exists(filename):
             #download(filename)

        # Read the labels in Yann LeCun's binary format.

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        # The labels are vectors of integers now, that's exactly what we want.

        return data



            # We can now download and read the training and test set images and labels.

    X_train = load_mnist_images('C:\Users\zjufe\PycharmProjects\\neural\mnist\\train-images-idx3-ubyte.gz')

    y_train = load_mnist_labels('C:\Users\zjufe\PycharmProjects\\neural\mnist\\train-labels-idx1-ubyte.gz')

    X_test = load_mnist_images('C:\Users\zjufe\PycharmProjects\\neural\mnist\\t10k-images-idx3-ubyte.gz')

    y_test = load_mnist_labels('C:\Users\zjufe\PycharmProjects\\neural\mnist\\t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.

    X_train, X_val = X_train[:-10000], X_train[-10000:]

    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().

    # (It doesn't matter how we do this as long as we can read them again.)

    return X_train, y_train, X_val, y_val, X_test, y_test



def load_cifar10():
    # Load train
    f_train = open('.\cifar10\data_batch_1', 'rb')
    train_dict = cPickle.load(f_train)
    f_train.close()
    X_train = train_dict['data']
    y_train = train_dict['labels']
    f_train = open('.\cifar10\data_batch_2', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train,train_dict['data'],axis = 0)
    y_train = y_train + train_dict['labels']
    f_train = open('.\cifar10\data_batch_3', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train, train_dict['data'], axis=0)
    y_train = y_train + train_dict['labels']
    f_train = open('.\cifar10\data_batch_4', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train, train_dict['data'], axis=0)
    y_train = y_train + train_dict['labels']
    f_train = open('.\cifar10\data_batch_5', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train, train_dict['data'], axis=0)
    y_train = y_train + train_dict['labels']
    f_train.close()
    # Load test
    f_test = open('.\cifar10\\test_batch', 'rb')
    test_dict = cPickle.load(f_test)
    f_test.close()

    X_test = test_dict['data']
    y_test = test_dict['labels']
    X_train = X_train/np.float32(256)
    X_test = X_test/np.float32(256)
    #X_train = X_train.reshape(-1, 3, 32, 32)
    #X_test = X_test.reshape(-1, 3, 32, 32)
    return np.asarray(X_train),np.asarray(y_train),np.asarray(X_test),np.asarray(y_test)
    #return X_train,y_train,X_test,y_test


def loadcifar10():
    # Load train
    f_train = open('.\cifar10\data_batch_1', 'rb')
    train_dict = cPickle.load(f_train)
    f_train.close()
    X_train = train_dict['data']
    y_train = train_dict['labels']
    f_train = open('.\cifar10\data_batch_2', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train,train_dict['data'],axis = 0)
    y_train = y_train + train_dict['labels']
    f_train = open('.\cifar10\data_batch_3', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train, train_dict['data'], axis=0)
    y_train = y_train + train_dict['labels']
    f_train = open('.\cifar10\data_batch_4', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train, train_dict['data'], axis=0)
    y_train = y_train + train_dict['labels']
    f_train = open('.\cifar10\data_batch_5', 'rb')
    train_dict = cPickle.load(f_train)
    X_train = np.append(X_train, train_dict['data'], axis=0)
    y_train = y_train + train_dict['labels']
    f_train.close()
    # Load test
    f_test = open('.\cifar10\\test_batch', 'rb')
    test_dict = cPickle.load(f_test)
    f_test.close()

    X_test = test_dict['data']
    y_test = test_dict['labels']
    X_train = X_train/np.float32(256)
    X_test = X_test/np.float32(256)
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)
    return np.asarray(X_train),np.asarray(y_train),np.asarray(X_test),np.asarray(y_test)
    #return X_train,y_train,X_test,y_test