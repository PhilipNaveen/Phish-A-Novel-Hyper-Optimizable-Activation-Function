# MXNet Implementation of Phish Activation Function.
#Phish is defined with f(x) = xTanH(GELU(x))
# Import Necessary Modules.
import mxnet as mx
import mxnet.ndarray as F

#define class for Phish activation function
class Phish(mx.gluon.HybridBlock):

    def __init__(self):
        super(Phish, self).__init__()

    def hybrid_forward(self, x):        
        return x * F.tanh(mxnet.gluon.nn.GELU(x))
