import torch
from torch import nn
from torch.nn import functional as F



def cnn1dblock(num_channels, out_channels, kernel_size, stride = 2,  activation = True, padding = 2):

    layers = []
    layers.append(nn.Conv1d(num_channels, out_channels, kernel_size, stride=stride, padding = 2))
    layers.append(nn.BatchNorm1d(out_channels))
    if activation:
        layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)


def linear_layer(in_features = 300, out_features = 300 , batch_norm = False, drop_out = 0.5):
    layer = []
    if batch_norm:
        layer.append(nn.BatchNorm1d(num_features=in_features))
    if drop_out != 0 :
        layer.append(nn.Dropout(p=drop_out))
    layer.append( nn.Linear(in_features= in_features, out_features=out_features))
    return layer


def cnn1d(channels, kernels):
    conv = []

    for kernel in kernel_sizes:
        conv.append(cnn1dblock(channels, channels, kernel))
    
    conv = nn.Sequential(*conv)

    return conv