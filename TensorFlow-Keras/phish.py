#Phish implementation in TensorFlow-Keras

#imports
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class Phish(Activation):
    
    #initialization function
    def __init__(self, activation, **kwargs):
        super(Phish, self).__init__(activation, **kwargs)
        self.__name__ = 'Phish'

#returns the activation f(x) = xTanH(GELU(x))
def Phish(inputs):
    return inputs * tf.math.tanh(tf.nn.gelu(inputs))

get_custom_objects().update({'Phish': Phish(phish)})
