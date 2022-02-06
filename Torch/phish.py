#applies the Phish activation function 
#f(x) = xTanH(GELU(x))

#pytorch dependency
from torch import nn

#import activation functions
import Phish.Torch.functional as Func

class Phish(nn.Module):
    #input shape is (N, *) 
    #* denotes the number of additional dimensions
    #output shape is the same as the input
 
    #initialization
    def __init__(self):
        super().__init__()

    #forward pass through the activation function
    def forward(self, input):
        return Func.phish(input)
