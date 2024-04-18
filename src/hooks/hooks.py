'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from torch.nn import Module
from torch import Tensor
import torch
from typing import List, Union
from xai_utils.logging import log_assert


class ForwardHook():
    """
    Wrapper for forward hooks registration (activation maps).
    """

    def __init__(self,
                 layer_name: str,
                 module: Module
                 ) -> None:
        """
        Args:
            layer_name: hook's layer name
            module: module to register hook for
        """
        self.storage: List[Tensor] = []  # values are stored here
        self.layer_name = layer_name
        self.hook = module.register_forward_hook(self.hook_fn_fwd)

    def hook_fn_fwd(self,
                    module: Module,
                    input: Tensor,
                    output: Tensor
                    ) -> None:
        """
        Forward hook function.

        Args:
            module: torch.nn.Module instance
            input: module input Tensor
            output: module output Tensor
        """
        self.storage.append(output)

    def get_activation(self
                       ) -> List[Tensor]:
        """
        Get activations of registered module

        Returns:
            list of activation tensors: List[Tensor[...]]
        """
        return [fh.detach().cpu() for fh in self.storage]

    def get_stacked_activtion(self
                              ) -> Tensor:
        """
        Get activations of registered module (list stacked to tensor)

        Returns:
            activations tensor: Tensor[N_SAMPLES, ...]
        """
        return torch.vstack(self.get_activation())

    def remove(self
               ) -> None:
        """
        Remove hook
        """
        self.hook.remove()

    def clean_storage(self
                      ) -> None:
        """
        Clean storage with registered activations
        """
        self.storage = []


class BackwardHook():
    """
    Wrapper for backward hooks registration (gradients).
    """

    def __init__(self,
                 layer_name: str,
                 module: Module
                 ) -> None:
        """
        Args:
            layer_name: hook's layer name
            module: module to register hook for
        """
        self.storage: List[Tensor] = []  # values are stored here
        self.layer_name = layer_name
        # register_backward_hook can miss some input tensors, but only outputs are important
        self.hook = module.register_backward_hook(self.hook_fn_bwd)
        # 'register_full_backward_hook' doesn't work properly in case of inplace calculations (e.g., BatchNorm)
        # self.hook = module.register_full_backward_hook(self.hook_fn_bwd)

    def hook_fn_bwd(self,
                    module: Module,
                    input: Tensor,
                    output: Tensor
                    ) -> None:
        """
        Backward hook function.

        Args:
            module: torch.nn.Module instance
            input: module input Tensor
            output: module output Tensor
        """
        self.storage.append(output)

    def get_grad(self
                 ) -> Union[List[Tensor], None]:
        """
        Get gradients of registered module

        Returns:
            list of gradients tensors: List[Tensor[...]]
        """
        if len(self.storage) > 0:
            return [bh[0].detach().cpu() for bh in self.storage]
        else:
            return None

    def get_stacked_grad(self
                         ) -> Tensor:
        """
        Get gradients of registered module (list stacked to tensor)

        Returns:
            gradients tensor: Tensor[N_SAMPLES, ...]
        """
        sg = self.get_grad()
        if sg is None:
            return 
        else:
            return torch.vstack(sg)

    def remove(self
               ) -> None:
        """
        Remove hook
        """
        self.hook.remove()

    def clean_storage(self
                      ) -> None:
        """
        Clean storage with registered gradients
        """
        self.storage = []


class ForwardInsertHook():
    """
    Wrapper for forward insert hooks registration.
    """

    def __init__(self,
                 layer_name: str,
                 module: Module
                 ) -> None:
        """
        Args:
            layer_name: hook's layer name
            module: module to register hook for
        """
        self.layer_name = layer_name
        self.hook = module.register_forward_hook(self.hook_fn_in_fwd)
        self.insert_tensor: Tensor = None

    def hook_fn_in_fwd(self,
                       module: Module,
                       input: Tensor,
                       output: Tensor
                       ) -> None:
        """
        Forward hook function. Replaces network's activations at given layer with self.insert_tensor

        Args:
            module: torch.nn.Module instance
            input: module input Tensor
            output: module output Tensor
        """
        log_assert(self.insert_tensor is not None, "value of 'insert_tensor' shall be provided")

        return self.insert_tensor

    def set_insert_tensor(self,
                          insert_tensor: Tensor
                          ) -> None:
        """
        Sets the insert tensor

        Args:
            insert_tensor: Tensor to insert, replaces activations of the module
        """
        self.insert_tensor = insert_tensor

    def remove(self
               ) -> None:
        """
        Remove hook
        """
        self.hook.remove()
