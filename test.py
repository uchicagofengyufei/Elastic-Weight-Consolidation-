#import theano
#import theano.tensor as T
#import lasagne
import numpy as np
#import save_param as files

"""
p = np.zeros([10,10])
for i in range(0,10):
    p[i] = np.asarray([np.exp(1),np.exp(0),np.exp(2),np.exp(3),np.exp(4),np.exp(5),np.exp(6),np.exp(7),np.exp(8),np.exp(9)])
p = p.astype(np.float32)
y = np.asarray([1,0,2,3,4,5,6,7,8,9]).astype(np.int32)
#print(p)
#print(y)
X = T.fmatrix("X")
Y = T.ivector("y")
loss = lasagne.objectives.categorical_crossentropy(X,Y)
f = theano.function(inputs=[X,Y],outputs=loss)

"""
ind = range(0,20)
np.random.shuffle(ind)
print(ind)