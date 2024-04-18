'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from hooks import PropagatorTorchDetector
from typing import Union, Iterable
from torchvision.models.detection.retinanet import RetinaNet
import torch


class PropagatorTorchRetinaNet(PropagatorTorchDetector):
    
    def __init__(self,
                 model: RetinaNet,
                 layers: Union[str, Iterable[str]],
                 batch_size: int = 16,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ) -> None:
        """
        Args:
            model: model
            layers: list of layers for activations and/or gradients registration
            batch_size: butch size for sampling from AbstractDataset instances
            
        Kwargs:
            batch_size: batch size for conversion of AbstractDataset to DataLoader, default value is 32
            device: torch device
        """
        super().__init__(model, layers, batch_size, device)
