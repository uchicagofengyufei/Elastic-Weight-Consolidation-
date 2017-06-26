import numpy
import theano
import load


def save_param(param):
    array = []
    for tensor in param:
        array.append(tensor.get_value())
    numpy.save(".\Save\weight_param",numpy.asarray(array))


def load_param_raw():
    raw_param = []
    param = numpy.load(".\Save\weight_param.npy")
    n = param.shape[0]
    for i in range(0,n):
        raw_param.append(param[i].astype(numpy.float32))
    return raw_param

def save_information(info ,index):
    numpy.save(".\Save\information_{}.npy".format(index), numpy.asarray(info))

def load_information(index):
    raw_info = []
    info = numpy.load(".\Save\information_{}.npy".format(index))
    n = info.shape[0]
    for i in range(0, n):
        raw_info.append(info[i].astype(numpy.float32))
    return raw_info


def load_param_tensor():
    param = numpy.load(".\Save\weight_param.npy").tolist()
    tensor = []
    for arr in param:
        tensor.append(theano.shared(arr.astype(numpy.float32)))
    return tensor


"""
A = numpy.zeros((2,3))
B = numpy.zeros((1,2))+1
C = numpy.zeros((3,5))+2
Arr = numpy.array([A, B, C])
#print(Arr)
save_param(Arr)
Brr = numpy.load(".\Save\weight_param.npy")
print(Brr.tolist())
Crr = load_param_tensor()
g = Crr[1].get_value()
print(Crr[1].get_value())

"""


