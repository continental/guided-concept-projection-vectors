'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from torchvision.models.resnet import BasicBlock
from torch import nn
from typing import Tuple, List
from functools import reduce


def get_all_resnet_layers(net: nn.Module
                          ) -> Tuple[List[str], List[str]]:
    layers = []
    names = []
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock):
            ns, ls = get_all_resnet_layers(layer)
            layers.extend(ls)
            names.extend(ns)
        else:
            layers.append(layer)
            names.append(name)
    return names, layers


def get_module_layer_by_name(model: nn.Module,
                             layer: str
                             ) -> nn.Module:
    """
    Get a layer of network by string name

    Args:
        model: nn.Module instance
        layer: layer to get

    Returns:
        desired layer (if exists): nn.Module instance
    """
    return reduce(getattr, layer.split("."), model)


def disable_resnet_inplace_activations(net: nn.Module
                                       ) -> nn.Module:
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock):
            disable_resnet_inplace_activations(layer)
        else:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False


def get_network_module_names(network: nn.Module,
                             name_prefix: str = ''
                             ) -> List[str]:
    """
    Recursively retrieve a list of full names on network: nn.Module submodules, which are kept in network._modules

    Args:
        network: network module - nn.Module
    
    Kwarg:
        name_prefix: prefix of the network name, if network=NETWORK_VARIABLE.backbone.features was passed to the function, pass name_prefix="backbone.features" to keep the prefix for all extracted names

    Returns:
        a list of full module names: List[str] - e.g. [backbone.features.0.conv1, backbone.features.0.relu1, ...]
    """
    res_list = []

    def module_recursion(net, net_name=''):
        for module_name, module in net._modules.items():

            if net_name == '':
                full_name = module_name
            else:
                full_name = f'{net_name}.{module_name}'

            if len(module._modules) == 0:
                res_list.append(full_name)
            else:
                module_recursion(module, full_name)

    module_recursion(network, name_prefix)

    return res_list
