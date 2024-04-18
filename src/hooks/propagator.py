'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from .hooks import ForwardHook, BackwardHook, ForwardInsertHook
from typing import Iterable, Dict, Tuple, Union, Any
from torch import nn, Tensor
import torch
from xai_utils.nn import get_module_layer_by_name
from xai_utils.files import apply_smoothgrad_noise
from torch.utils.data import DataLoader
from data_structures import AbstractDataset
from xai_utils.logging import log_error, log_assert


SG_SAMPLES = 50  # number of SmoothGrad samples
SG_NOISE = 0.1  # SmoothGrad noise


class Propagator():

    """
    Propagates input tensors through the model and registers activations and/or gradients using ForwardHook and/or BackwardHook
    """

    def __init__(self,
                 model: nn.Module,
                 layers: Union[str, Iterable[str]],
                 batch_size: int = 64,
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
        if isinstance(layers, str):
            layers = [layers]

        self.device = device

        self.model = model.to(self.device)
        self.layers = layers
        self.batch_size = batch_size
        self.modules = {layer: get_module_layer_by_name(
            self.model, layer) for layer in self.layers}

    def get_predictions(self,
                        input: Union[Tensor, DataLoader, AbstractDataset]
                        ) -> Tensor:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input batch tensor or dataloader or dataset

        Returns:
            predictions: Tensor[...]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_predictions(input)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_predictions(input)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_predictions(DataLoader(input, self.batch_size))

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_predictions(self,
                               input: Tensor
                               ) -> Tensor:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input batch tensor

        Returns:
            predictions: Tensor[...]
        """
        with torch.no_grad():
            self.model.eval()

            device_input = input.to(self.device)

            pred = self.model(device_input)

            return pred.detach().cpu()

    def dataloader_get_predictions(self,
                                   input: DataLoader
                                   ) -> Tensor:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input dataloader

        Returns:
            predictions: Tensor[...]
        """

        temp = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.append(self.tensor_get_predictions(batch))

        return torch.vstack(temp)

    def get_activations(self,
                        input: Union[Tensor, DataLoader, AbstractDataset]
                        ) -> Dict[str, Tensor]:
        """
        Propagate forward and get activations

        Args:
            input: input batch tensor or dataloader or dataset

        Returns:
            dictionary - {layer: activations}: Dict[str, Tensor[...]]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_activations(input)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_activations(input)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_activations(DataLoader(input, self.batch_size))

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_activations(self,
                               input: Tensor
                               ) -> Dict[str, Tensor]:
        """
        Propagate forward and get activations

        Args:
            input: input batch tensor

        Returns:
            dictionary - {layer: activations}: Dict[str, Tensor[...]]
        """
        with torch.no_grad():
            self.model.eval()

            fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]

            device_input = input.to(self.device)

            _ = self.model(device_input)

            activations = {fh.layer_name: fh.get_stacked_activtion()
                        for fh in fhooks}

            fhooks = [fh.remove() for fh in fhooks]

            return activations

    def dataloader_get_activations(self,
                                   input: DataLoader
                                   ) -> Dict[str, Tensor]:
        """
        Propagate forward and get activations

        Args:
            input: input dataloader

        Returns:
            dictionary - {layer: activations}: Dict[str, Tensor[...]]
        """

        temp = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.append(self.tensor_get_activations(batch))

        acts = {}

        for layer in self.layers:
            acts[layer] = torch.vstack([act[layer] for act in temp])

        return acts

    def get_gradients(self,
                      input: Union[Tensor, DataLoader, AbstractDataset],
                      target_classes: Iterable[int]
                      ) -> Dict[str, Tensor]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor or dataloader or dataset
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """
        
        if isinstance(input, Tensor):
            return self.tensor_get_gradients(input, target_classes, )

        elif isinstance(input, DataLoader):
            return self.dataloader_get_gradients(input, target_classes)

        elif isinstance(input, AbstractDataset):
            dl = DataLoader(input, self.batch_size)
            return self.dataloader_get_gradients(dl, target_classes)

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_gradients(self,
                             input: Tensor,
                             target_classes: Iterable[int]
                             ) -> Dict[str, Tensor]:
        """
        NUMERICALLY STABLE. See self.tensor_get_gradients_fast for unstable stable version
        Propagate forward and backward and get gradients.        
        Propagates inputs one by one (across dim=0) to avoid minor gradient value fluctuations due to batch-wise operations (e.g. BN).
        torch.autograd.grad - doesn't accumulate gradients.

        Args:
            input: input batch tensor
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """
        self.model.eval()

        log_assert(len(input) == len(target_classes), "length of 'target_classes' must be equal to number of samples in batch ('input')")

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        for j, i in enumerate(input):
            ins = i.detach().to(self.device)
            ins = ins.unsqueeze(0)
            ins.requires_grad_()
            pred = self.model(ins)
            # autograd.grad does not accumulate gradients
            o = pred[:, target_classes[j]]  # scalar for backprop
            torch.autograd.grad(outputs=o, inputs=ins)

        grads = {bh.layer_name: bh.get_stacked_grad() for bh in bhooks}

        bhooks = [bh.remove() for bh in bhooks]

        return grads

    def tensor_get_gradients_fast(self,
                                  input: Tensor,
                                  target_classes: Iterable[int]
                                  ) -> Dict[str, Tensor]:
        """
        NUMERICALLY INSTABLE. FAST. See self.tensor_get_gradients for stable version
        Propagate forward and backward and get gradients.
        Propagates input as batch. Small fluctuations (~1e-10) in gradient values are possible due to batch-wise operations (e.g. BN)
        torch.autograd.grad - doesn't accumulate gradients.

        Args:
            input: input batch tensor
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """
        self.model.eval()

        log_assert(len(input) == len(target_classes), "length of 'target_classes' must be equal to number of samples in batch ('input')")

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        ins = input.detach().to(self.device)

        ins.requires_grad_()
        pred = self.model(ins)

        outs = [pred[i, t] for i, t in enumerate(target_classes)]
        torch.autograd.grad(outs, ins)

        grads = {bh.layer_name: bh.get_stacked_grad() for bh in bhooks}

        bhooks = [bh.remove() for bh in bhooks]

        return grads

    def tensor_get_gradients_smoothgrad(self,
                                        input: Tensor,
                                        target_classes: Iterable[int]
                                        ) -> Dict[str, Tensor]:
        """
        Propagate forward and backward and get gradients. SmoothGrad rules are applied to achieve the stability of gradient.

        Args:
            input: input batch tensor
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """
        self.model.eval()

        log_assert(len(input) == len(target_classes), "length of 'target_classes' must be equal to number of samples in batch ('input')")

        res_list = []

        # for each sample generate SG_SAMPLES noisy samples
        for i, tc in zip(input, target_classes):            
            
            ins = i.detach().to(self.device)
            ins_sg = apply_smoothgrad_noise(ins, SG_NOISE, SG_SAMPLES)
            ins.unsqueeze_(0)

            # stack -> SG_SAMPLES + 1 sample
            ins_sg = torch.vstack([ins, ins_sg])

            # get gradients
            grads_ins_sg = self.tensor_get_gradients_fast(ins_sg, [tc] * (SG_SAMPLES + 1))

            # calculate mean (SmoothGrad) for 1 sample
            grads_ins_sg = {k: torch.mean(v, 0, keepdim=True) for k, v in grads_ins_sg.items()}
            
            res_list.append(grads_ins_sg)

        #  init final dictionary for output (all samples)
        res_dict = {l: [] for l in self.layers}

        #  fill final dictionary for output (all samples)
        for l in self.layers:
            for entry in res_list:
                layer_entry = entry[l]
                res_dict[l].extend(layer_entry)

        #  format final dictionary for output (all samples)
        res_dict = {k: torch.stack(v) for k, v in res_dict.items()}

        return res_dict

    def dataloader_get_gradients(self,
                                 input: DataLoader,
                                 target_classes: Iterable[int]
                                 ) -> Dict[str, Tensor]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input dataloader
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """
        log_assert(len(input.dataset) == len(target_classes), "length of 'target_classes' must be equal to number of samples in dataloader.dataset ('input')")

        temp = []

        low_idx, high_idx = 0, 0

        for batch in input:

            high_idx += len(batch)

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            g = self.tensor_get_gradients(batch, target_classes[low_idx:high_idx])

            temp.append(g)

            low_idx = high_idx

        grads = {}

        for layer in self.layers:
            grads[layer] = torch.vstack([grad[layer] for grad in temp])

        return grads

    def get_activations_and_gradients(self,
                                      input: Union[Tensor, DataLoader, AbstractDataset],
                                      target_classes: Iterable[int]
                                      ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Propagate forward and backward and get activations and gradients

        Args:
            input: input batch tensor or dataloader or dataset
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary {layer: activations}: Dict[str, Tensor[...]]
            dictionary {layer: gradients}: Dict[str, Tensor[...]]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_activations_and_gradients(input, target_classes)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_activations_and_gradients(input, target_classes)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_activations_and_gradients(DataLoader(input, self.batch_size), target_classes)

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_activations_and_gradients(self,
                                             input: Tensor,
                                             target_classes: Iterable[int]
                                             ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Propagate forward and backward and get activations and gradients

        Args:
            input: input batch tensor
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary {layer: activations}: Dict[str, Tensor[...]]
            dictionary {layer: gradients}: Dict[str, Tensor[...]]
        """
        self.model.eval()

        log_assert(len(input) == len(target_classes), "length of 'target_classes' must be equal to number of samples in batch ('input')")

        fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]

        grads = self.get_gradients(input, target_classes)

        activations = {fh.layer_name: fh.get_stacked_activtion()
                       for fh in fhooks}

        fhooks = [fh.remove() for fh in fhooks]

        return activations, grads

    def dataloader_get_activations_and_gradients(self,
                                                 input: DataLoader,
                                                 target_classes: Iterable[int]
                                                 ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Propagate forward and backward and get activations and gradients

        Args:
            input: input dataloader
            target_classes: target classes for gradients computation for each input sample - Iterable[int]

        Returns:
            dictionary {layer: activations}: Dict[str, Tensor[...]]
            dictionary {layer: gradients}: Dict[str, Tensor[...]]
        """
        self.model.eval()

        log_assert(len(input.dataset) == len(target_classes), "length of 'target_classes' must be equal to number of samples in dataloader.dataset ('input')")

        temp = []

        low_idx, high_idx = 0, 0

        for batch in input:

            high_idx += len(batch)

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.append(self.tensor_get_activations_and_gradients(
                batch, target_classes[low_idx:high_idx]))

            low_idx = high_idx

        activations = {}
        gradients = {}

        for layer in self.layers:
            activations[layer] = torch.vstack([act[layer] for act, _ in temp])
            gradients[layer] = torch.vstack([grd[layer] for _, grd in temp])

        return activations, gradients

    def get_predictions_from_activations(self,
                                         activations: Union[Tensor, DataLoader, AbstractDataset],
                                         model_input_shape: Tuple[int, int, int],
                                         layer: str = None,
                                         ) -> Tensor:
        """
        Propagates the intermediate activations forward starting from :layer: X to get predictions. Dummy input is used until layer X.

        Args:
            activations: input activations tensor batch or dataloader
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]
        
        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: Tensor[...]
        """
        if isinstance(activations, Tensor):
            return self.tensor_get_predictions_from_activations(activations, model_input_shape, layer)

        elif isinstance(activations, DataLoader):
            return self.dataloader_get_predictions_from_activations(activations, model_input_shape, layer)

        elif isinstance(activations, AbstractDataset):
            return self.dataloader_get_predictions_from_activations(DataLoader(activations, self.batch_size), model_input_shape, layer)

        else:
            err_msg = f"wrong type of 'activations', current type: {type(activations)}"
            log_error(TypeError, err_msg)

    def tensor_get_predictions_from_activations(self,
                                                input: Tensor,
                                                model_input_shape: Tuple[int, int],
                                                layer: str = None,
                                                ) -> Tensor:
        """
        Propagates the intermediate activations forward starting from :layer: to get predictions. Dummy input is used until layer X.

        Args:
            input: input tensor (B, C, H, W)
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]
        
        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: Tensor[...]
        """
        with torch.no_grad():
            if layer is None:
                layer = self.layers[0]

            self.model.eval()

            hook = ForwardInsertHook(layer, self.modules[layer])
            device_input = input.to(self.device)
            hook.set_insert_tensor(device_input)  # set injection data

            # batch dimension is added to :model_input_shape:
            batch_shape = (len(device_input), ) + model_input_shape
            # dummy data is used until the injection at layer X
            dummy = torch.zeros(batch_shape)

            pred = self.model(dummy.to(self.device))

            hook.remove()

            return pred.detach().cpu()

    def dataloader_get_predictions_from_activations(self,
                                                    input: DataLoader,
                                                    model_input_shape: Tuple[int],
                                                    layer: str = None,
                                                    ) -> Tensor:
        """
        Propagates the intermediate activations forward starting from :layer: X to get predictions. Dummy input is used until layer X.

        Args:
            input: input dataloader of tensors (B, C, H, W)
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]
        
        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: Tensor[...]
        """
        if layer is None:
            layer = self.layers[0]

        temp = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.append(self.tensor_get_predictions_from_activations(
                batch, model_input_shape, layer))

        return torch.vstack(temp)

    def get_activations_and_predictions(self,
                                        input: Union[Tensor, DataLoader, AbstractDataset]
                                        ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Propagate input forward to get activations and predictions

        Args:
            input: input tensor batch or dataloader or dataset

        Returns:
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            predictions: Tensor[...]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_activations_and_predictions(input)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_activations_and_predictions(input)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_activations_and_predictions(DataLoader(input, self.batch_size))

        else:
            err_msg = f"wrong type of 'input', current type: {type(input)}"
            log_error(TypeError, err_msg)

    def tensor_get_activations_and_predictions(self,
                                               input: Tensor
                                               ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Propagate input forward to get activations and predictions

        Args:
            input: input tensor batch

        Returns:
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            predictions: Tensor[...]
        """
        with torch.no_grad():
            self.model.eval()

            fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]

            device_input = input.to(self.device)

            predictions = self.model(device_input)

            activations = {fh.layer_name: fh.get_stacked_activtion()
                        for fh in fhooks}

            fhooks = [fh.remove() for fh in fhooks]

            return activations, predictions.detach().cpu()

    def dataloader_get_activations_and_predictions(self,
                                                   input: DataLoader
                                                   ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Propagate input forward to get activations and predictions

        Args:
            input: input ataloader

        Returns:
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            predictions: Tensor[...]
        """

        temp_acts = []
        temp_preds = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            a, p = self.tensor_get_activations_and_predictions(batch)
            temp_acts.append(a)
            temp_preds.append(p)

        acts = {}

        for layer in self.layers:
            acts[layer] = torch.vstack([act[layer] for act in temp_acts])

        preds = torch.vstack(temp_preds)

        return acts, preds
