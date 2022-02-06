'''
Phish activation functional
'''

#importing pytorch
import torch
import torch.nn.functional as F

@torch.jit.script
def phish(input):
    '''
    Applies the Phish activation function of
    f(x) = xTanH(GELU(x))
    '''
    return input * torch.tanh(F.gelu(input))
