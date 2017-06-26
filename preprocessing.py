import numpy
import load
from  PIL import Image
from scipy.misc import toimage
from copy import deepcopy



def digit_rotate(digit_array, degree):
    new_digit_array = numpy.zeros(digit_array.shape)
    n  = digit_array.shape[0]
    for i in range(0,n):
        arr = (digit_array[i].reshape(28,28)*255.).astype(numpy.uint8)
        img = Image.fromarray(arr)
        img = img.rotate(degree)
        new_digit_array[i] = numpy.asarray(img).reshape(1,784)/255.
    return new_digit_array.astype(numpy.float32)



def digit_permute(digit_array,ind):
    new_digit_array = numpy.zeros(digit_array.shape)
    n  = digit_array.shape[0]

    for i in range(0,784):
        new_digit_array[:,i] = digit_array[:,ind[i]]
    return new_digit_array.astype(numpy.float32)

def permute_mnist(train,val):
    perm_inds = range(0,784)
    numpy.random.shuffle(perm_inds)
    train2 = numpy.transpose(numpy.asarray([train[:,c] for c in perm_inds]))
    val2 = numpy.transpose(numpy.asarray([val[:, c] for c in perm_inds]))
    return train2.astype(numpy.float32),val2.astype(numpy.float32)




"""
data = numpy.load("MNIST_0.npy")
print(len(data))
data = data[0]
digit = digit_rotate(data,90)
a = data[0].reshape(28,28)
b = digit[0].reshape(28,28)
print("gg")
#print(toimage(digit[0].reshape(28,28)).show())
#print(toimage(digit[1].reshape(28,28)).show())
#print(toimage(digit[2].reshape(28,28)).show())
"""
