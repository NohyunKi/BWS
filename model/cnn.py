import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """ A simple CNN architecture
    
    Attribute:
        input_dims (array, int): input dimensions of data
        output_dims (int): output dimensions of data
        conv_dims (array, int): filter dimension of each convolution layers
        kernel_dims (array, int): kernel size of each convolution layers
        pooling_layers (array, bool): pooling layer between each convolution layers
        mlp_dims (array, int): dimension of each fully connected layers
        drop_prob (int, optional): probability of an element to be zeroed

    Example:
        model = CNN([32, 32, 3], 10, [32, 64], [3, 3], [True, True], [128, 256])
        --> dimension of CNN
            n * n * 3 -conv1(3, 3, 32)-> 
            n * n * 32 -pool1-> 
            n/2 * n/2 * 32 -conv2(3, 3, 64)->
            n/2 * n/2 * 64 -pool2-> 
            n/4 * n/4 * 64 -fc1(-1, 128)->
            128 -fc2(128, 256)->
            256 -linear(256, 10)->
            10
    """
    def __init__(self, input_dims, output_dims, conv_dims, kernel_dims, 
                 pooling_layers, mlp_dims, drop_prob):
        super(CNN, self).__init__()

        self.input_dims = input_dims
        self.conv_dims = conv_dims
        self.kernel_dims = kernel_dims
        self.pooling_layers = pooling_layers
        self.mlp_dims = mlp_dims
        self.drop_prob = drop_prob

        self.middle_len = self.mlp_dims[1]

        # Network structure layers 
        rowcols = np.array([input_dims[0], input_dims[1]])
        self.conv_layers = nn.ModuleList()
        for layer_idx, channel in enumerate(self.conv_dims):
            in_channels = self.input_dims[-1] if layer_idx == 0 else self.conv_dims[layer_idx-1]
            kernel_size = self.kernel_dims[layer_idx]
            padding = int(kernel_size/2) 

            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channel,
                            kernel_size=kernel_size, stride=1, dilation=1, padding=padding),
                nn.BatchNorm2d(channel, track_running_stats=False),
                nn.ReLU(),
                nn.Dropout(p=drop_prob)
                ))
            
            if pooling_layers[layer_idx]:
                self.conv_layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
                rowcols = np.ceil(rowcols/2)

        self.view_dims = int(conv_dims[-1] * rowcols[0] * rowcols[1])

        self.mlp_layers = nn.ModuleList()
        for layer_idx, out_channel in enumerate(self.mlp_dims):
            in_channels = self.view_dims if layer_idx == 0 else self.mlp_dims[layer_idx-1]

            self.mlp_layers.append(nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=out_channel),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(),
                nn.Dropout(p=drop_prob)
            ))
                
        self.mlp_layers.append(nn.Linear(in_features=self.mlp_dims[-1], out_features=output_dims))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, self.view_dims)
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
        return x

""" Function of models
Attribute:
    input_dims (array, int): input dimensions of data
    output_dims (int): output dimension of model
"""
def BasicCNN(input_dims, output_dims):
    return CNN(input_dims, output_dims, [64, 128, 256], [3, 3, 3], [True, True, False], [256, 512], 0.5)
