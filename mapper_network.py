import torch
from torchvision import models
from torch import nn


class VGG16_Perceptual(nn.Module):

    def __init__(self, requires_grad=False, n_layers=[2, 4, 14, 21]):
        super(VGG16_Perceptual, self).__init__()
        vgg_features = models.vgg16(pretrained=True).features

        self.features0 = nn.Sequential()
        self.features1 = nn.Sequential()
        self.features2 = nn.Sequential()
        self.features3 = nn.Sequential()

        for module in range(n_layers[0]):
            self.features0.add_module(str(module), vgg_features[module])
        for module in range(n_layers[0], n_layers[1]):
            self.features1.add_module(str(module), vgg_features[module])
        for module in range(n_layers[1], n_layers[2]):
            self.features2.add_module(str(module), vgg_features[module])
        for module in range(n_layers[2], n_layers[3]):
            self.features3.add_module(str(module), vgg_features[module])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h0 = self.features0(x)
        h1 = self.features1(h0)
        h2 = self.features2(h1)
        h3 = self.features3(h2)
        return h0, h1, h2, h3
