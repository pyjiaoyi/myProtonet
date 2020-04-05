import torch 
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

class myFlatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(x):
        return x.reshape(x.size(0),-1)

class myProtonet(nn.Module):
    def __init__():

