'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from typing import Any, Dict, List, Tuple, Union, Iterable
import torch
from torch import Tensor
from hooks import BackwardHook, ForwardHook, ForwardInsertHook, Propagator
from xai_utils.nn import get_module_layer_by_name
from xai_utils.files import apply_smoothgrad_noise
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from data_structures import AbstractDataset
from xai_utils.logging import log_assert, log_error


SG_SAMPLES = 50  # number of SmoothGrad samples
SG_NOISE = 0.1  # SmoothGrad noise


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou_yolo3(boxes1: Tensor, boxes2: Tensor):
    """
    Calculate IoU of 2 bbox tensors

    Arguments:
        box1 (Tensor)
        box2 (Tensor)

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    boxes1_area = box_area(boxes1.T)
    boxes2_area = box_area(boxes2.T)

    intersection = (torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])).clamp(0).prod(2)
    union = boxes1_area[:, None] + boxes2_area - intersection + 0.0000001
    return intersection / union


def box_iou_yolo5(boxes1: Tensor, boxes2: Tensor):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Calculate IoU of 2 bbox tensors

    Arguments:
        box1 (Tensor)
        box2 (Tensor)

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    (a1, a2), (b1, b2) = boxes1[:, None].chunk(2, 2), boxes2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    union = box_area(boxes1.T)[:, None] + box_area(boxes2.T) - intersection + 0.000001
    return intersection / union


def non_max_suppression(prediction, conf_thres=0.7, iou_thres=0.45, agnostic=False, multi_label=False):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Modified code from https://github.com/ultralytics/yolov3

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates


    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    output_idxs = []

    for xi, x in enumerate(prediction):  # image index, image inference

        x_idxs = torch.arange(0, len(x), 1)[xc[xi]]
        x = x[xc[xi]]  # confident preds of img xi

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # best class only
        conf, cls_idxs = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, cls_idxs.float()), 1)[
            conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            sorted_idxs = x[:, 4].argsort(descending=True)[
                :max_nms]  # sort by confidence
            x = x[sorted_idxs]
            x_idxs = x_idxs[sorted_idxs]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes = x[:, :4] + c 
        scores = x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        output[xi] = x[i]
        output_idxs.append(x_idxs[i])

    return output, output_idxs


def convert_results(results, model):
    """
    Formats predictions.

    Modified code from https://github.com/ultralytics/yolov3

    Returns:
        stacked predictions (B, -1, 5 + num_classes)
    """
    def _make_grid(det, nx=20, ny=20, i=0):
        d = det.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(
            d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, det.na, ny, nx, 2)).float()
        anchor_grid = (det.anchors[i].clone() * det.stride[i]) \
            .view((1, det.na, 1, 1, 2)).expand((1, det.na, ny, nx, 2)).float()
        return grid, anchor_grid

    det = model.model[-1]  # detector with parameters

    z = []
    for i in range(det.nl):

        x = results[i].detach()
        bs, _, ny, nx, _, = x.shape

        det.grid[i], det.anchor_grid[i] = _make_grid(det, nx, ny, i)

        y = x.sigmoid()
        if det.inplace:
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 +
                           det.grid[i]) * det.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * det.anchor_grid[i]  # wh
        else:  # for  on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
            xy = (y[..., 0:2] * 2 - 0.5 + det.grid[i]) * det.stride[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * det.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, y[..., 4:]), -1)
        z.append(y.view(bs, -1, det.no))

    return torch.cat(z, 1)


class PropagatorUltralyticsYOLOv3Old(Propagator):

    def __init__(self,
                 wrapped_model: Any,
                 layers: Union[str, Iterable[str]],
                 batch_size: int = 16,
                 device: torch.device = torch.device(
                     "cuda" if torch.cuda.is_available() else "cpu")
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
        # originally downloaded, wrapped for data input and output
        self.wrapped_model = wrapped_model
        self.model = self.wrapped_model.model  # model itself
        self.torch_sequential_model = self.model.model  # instance of nn.Sequential

        if isinstance(layers, str):
            layers = [layers]
        self.layers = layers

        self.device = device
        self.model = self.model.to(self.device)
        self.batch_size = batch_size

        self.modules = {layer: get_module_layer_by_name(
            self.torch_sequential_model, layer) for layer in self.layers}

    def tensor_get_predictions(self,
                               input: Tensor
                               ) -> List[Tensor]:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input batch tensor

        Returns:
            predictions (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
        """
        with torch.no_grad():
            self.model.eval()

            device_input = input.to(self.device)

            # list of predictions - dict with keys: "boxes", "labels", "scores"
            pred, pred_raw = self.model(device_input)

            pred, idxs = non_max_suppression(pred)

            pred = [p.detach().cpu() for p in pred]

            # Tensor[N_pred, 6]: 0:3 - bbox, 4 - probability, 5 - class
            return pred

    def dataloader_get_predictions(self,
                                   input: DataLoader
                                   ) -> List[Tensor]:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input dataloader

        Returns:
            predictions: List[Tensor]
        """

        temp = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.extend(self.tensor_get_predictions(batch))

        return temp

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

            self.model(device_input)

            activations = {fh.layer_name: fh.get_stacked_activtion()
                           for fh in fhooks}

            fhooks = [fh.remove() for fh in fhooks]

            return activations

    def get_gradients(self,
                      input: Union[Tensor, DataLoader, AbstractDataset],
                      args: Any,
                      grad_type: str = 'cls'
                      ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor or dataloader or dataset
            args: placeholder for compatibility

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_gradients(input, args, grad_type)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_gradients(input, args, grad_type)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_gradients(DataLoader(input, self.batch_size), args, grad_type)

        else:
            log_error(TypeError, f"wrong type of 'input', current type: {type(input)}")

    def dataloader_get_gradients(self,
                                 input: DataLoader,
                                 *args: Any,
                                 grad_type: str = 'cls'
                                 ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input dataloader
            args: placeholder for compatibility

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """

        grads = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            grads.extend(self.tensor_get_gradients(batch, args, grad_type))

        return grads

    def tensor_get_gradients(self,
                             input: Tensor,
                             *args: Any,
                             grad_type: str = 'cls',
                             ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor
            args: placeholder for compatibility

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            list of dictionary for each object [{layer: gradients}]: List[Dict[str, Tensor]]
        """
        log_assert(grad_type in ['cls', 'obj'], "Wrong gradient type, must be 'cls' or 'obj' - objectness or class gradients")

        self.model.eval()

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        grads = []

        for i in input:
            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()
            i = i.to(self.device)
            # processed predictions, raw predictions
            pred, raw_preds = self.model(i)

            # _pred = convert_results(raw_preds, self.model)
            # assert ((pred-_pred) != 0).sum() == 0, "Incorrect result conversion"

            nms_preds, nms_preds_idxs = non_max_suppression(pred.detach())

            if len(nms_preds_idxs) == 0:
                grads.append({bh.layer_name: None for bh in bhooks})
                [bh.clean_storage() for bh in bhooks]
                continue

            # raw predictions shape now have the same shape (and indexes) as processed ones
            raw_pred_reshaped = self._reshape_raw_preds(raw_preds)

            # raw vectors, for which bboxes were generated
            pred_vects = self._get_prediction_vectors(
                raw_pred_reshaped, nms_preds_idxs[0])

            scalars = self._get_scalars(nms_preds, pred_vects)

            for obj_scl, cls_scl in scalars:
                if grad_type == 'cls':
                    torch.autograd.grad(cls_scl, i, retain_graph=True)
                else:
                    torch.autograd.grad(obj_scl, i, retain_graph=True)

            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]

        return grads
    
    def tensor_get_gradients_smoothgrad(self,
                                        input: Tensor,
                                        *args: Any,
                                        grad_type: str = 'cls',
                                        ) -> List[Dict[str, Tensor]]:
        raise NotImplementedError("Not implemented, use YOLOv5 or YOLOv8 instead")

    def tensor_get_gradients_for_targets(self,
                                         input: Tensor,
                                         targets: Iterable[Tensor],
                                         grad_type: str = 'top_cls',
                                         ) -> List[Dict[str, Tensor]]:
        raise NotImplementedError("Not implemented, use YOLOv5 or YOLOv8 instead")
        
    def tensor_get_gradients_for_targets_and_classes(self,
                                                     input: Tensor,
                                                     targets: Iterable[Tensor],
                                                     classes: Iterable[int]
                                                     ) -> Tuple[List[List[Dict[int, Dict[str, Tensor]]]], Dict[str, List[List[Tensor]]]]:
        raise NotImplementedError("Not implemented, use YOLOv5 or YOLOv8 instead")

    @staticmethod
    def _reshape_raw_preds(raw_preds: Iterable[Tensor]
                           ) -> Tensor:

        result = []
        for rp in raw_preds:
            bs = rp.shape[0]  # batch size
            # length of vector with predictions (bbox..., objectivity, classes...)
            pred_vector_len = rp.shape[-1]
            rp_reshaped = rp.view(bs, -1, pred_vector_len)
            result.append(rp_reshaped)

        return torch.cat(result, 1)

    def _get_prediction_vectors(self,
                                raw_pred_reshaped: Tensor,
                                ids: Iterable[int]
                                ) -> List[Tensor]:
        return [raw_pred_reshaped[:, i, :].squeeze() for i in ids]

    def _get_scalars(self,
                     nms_preds: Iterable[Tensor],
                     pred_vects: Iterable[Tensor]
                     ) -> List[Tuple[Tensor, Tensor]]:
        res = []

        for i in range(len(nms_preds[0])):
            objectivness_scalar = pred_vects[i][4]
            class_scalar_idx = nms_preds[0][i].squeeze()[5].int()
            class_scalar = pred_vects[i][class_scalar_idx + 5]  # 5 is offset
            res.append((objectivness_scalar, class_scalar))

        return res

    def tensor_get_predictions_from_activations(self,
                                                input: Tensor,
                                                model_input_shape: Tuple[int, int],
                                                layer: str = None,
                                                ) -> List[Tensor]:
        """
        Propagates the intermediate activations forward starting from :layer: to get predictions. Dummy input is used until layer X.

        Args:
            input: input tensor (B, C, H, W)
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]

        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
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

            pred, pred_raw = self.model(dummy.to(self.device))

            pred, idxs = non_max_suppression(pred)

            pred = [p.detach().cpu() for p in pred]

            hook.remove()

            return pred
        
    def tensor_get_predictions_from_activations_2(self,
                                                  input: Tensor,
                                                  dummy_input: Tensor,
                                                  layer: str = None,
                                                  ) -> List[Tensor]:
        """
        Propagates the intermediate activations forward starting from :layer: to get predictions. Dummy input is used until layer X.

        Args:
            input: input tensor (B, C, H, W)
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]

        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
        """
        with torch.no_grad():
            if layer is None:
                layer = self.layers[0]

            self.model.eval()

            hook = ForwardInsertHook(layer, self.modules[layer])
            device_input = input.to(self.device)
            hook.set_insert_tensor(device_input)  # set injection data

            # dummy data is used until the injection at layer X
            dummy = dummy_input.to(self.device)

            pred, pred_raw = self.model(dummy.to(self.device))

            pred, idxs = non_max_suppression(pred)

            pred = [p.detach().cpu() for p in pred]

            hook.remove()

            return pred

    def dataloader_get_predictions_from_activations(self,
                                                    input: DataLoader,
                                                    model_input_shape: Tuple[int],
                                                    layer: str = None,
                                                    ) -> List[Tensor]:
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

            temp.extend(self.tensor_get_predictions_from_activations(
                batch, model_input_shape, layer))

        return temp

    def tensor_get_preds_acts_grads(self,
                                    input: Tensor,
                                    grad_type: str = 'cls',
                                    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            predictions: List[Tensor]
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            gradients {layer:gradients}: Dict[str, Tensor[...]]asser
        """
        log_assert(grad_type in ['cls', 'obj'], "Wrong gradient type, must be 'cls' or 'obj' - objectness or class gradients")

        self.model.eval()

        fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]
        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        preds = []
        acts = []
        grads = []

        for i in input:
            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()

            # processed predictions, raw predictions
            pred, raw_preds = self.model(i)

            # _pred = convert_results(raw_preds, self.model)
            # assert ((pred-_pred) != 0).sum() == 0, "Incorrect result conversion"

            nms_preds, nms_preds_idxs = non_max_suppression(pred.detach())

            # raw predictions shape now have the same shape (and indexes) as processed ones
            raw_pred_reshaped = self._reshape_raw_preds(raw_preds)

            # raw vectors, for which bboxes were generated
            pred_vects = self._get_prediction_vectors(
                raw_pred_reshaped, nms_preds_idxs[0])

            scalars = self._get_scalars(nms_preds, pred_vects)

            for obj_scl, cls_scl in scalars:
                if grad_type == 'cls':
                    torch.autograd.grad(cls_scl, i, retain_graph=True)
                else:
                    torch.autograd.grad(obj_scl, i, retain_graph=True)

            preds.append(pred)
            acts.append({fh.layer_name: fh.get_stacked_activtion()
                         for fh in fhooks})
            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [fh.clean_storage() for fh in fhooks]
            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]
        fhooks = [fh.remove() for fh in fhooks]

        return preds, acts, grads


class PropagatorUltralyticsYOLOv5Old(PropagatorUltralyticsYOLOv3Old):

    def __init__(self,
                 wrapped_model: Any,
                 layers: Union[str, Iterable[str]],
                 batch_size: int = 16,
                 device: torch.device = torch.device(
                     "cuda" if torch.cuda.is_available() else "cpu")
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
        # originally downloaded, wrapped for data input and output
        self.wrapped_model = wrapped_model
        self.model = self.wrapped_model.model  # model itself
        self.torch_sequential_model = self.model.model.model  # instance of nn.Sequential

        if isinstance(layers, str):
            layers = [layers]
        self.layers = layers

        self.device = device
        self.model = self.model.to(self.device)
        self.batch_size = batch_size

        self.modules = {layer: get_module_layer_by_name(
            self.torch_sequential_model, layer) for layer in self.layers}

    def tensor_get_predictions(self,
                               input: Tensor
                               ) -> List[Tensor]:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input batch tensor

        Returns:
            predictions (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
        """
        with torch.no_grad():
            self.model.eval()

            device_input = input.to(self.device)

            # list of predictions - dict with keys: "boxes", "labels", "scores"
            pred = self.model(device_input)

            pred, idxs = non_max_suppression_yolov5(pred)

            pred = [p.detach().cpu() for p in pred]

            # Tensor[N_pred, 6]: 0:3 - bbox, 4 - probability, 5 - class
            return pred

    def dataloader_get_predictions(self,
                                   input: DataLoader
                                   ) -> List[Tensor]:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input dataloader

        Returns:
            predictions: List[Tensor]
        """

        temp = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.extend(self.tensor_get_predictions(batch))

        return temp

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

            self.model(device_input)

            activations = {fh.layer_name: fh.get_stacked_activtion()
                           for fh in fhooks}

            fhooks = [fh.remove() for fh in fhooks]

            return activations

    def get_gradients(self,
                      input: Union[Tensor, DataLoader, AbstractDataset],
                      args: Any,
                      grad_type: str = 'cls'
                      ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor or dataloader or dataset
            args: placeholder for compatibility

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """
        if isinstance(input, Tensor):
            return self.tensor_get_gradients(input, args, grad_type)

        elif isinstance(input, DataLoader):
            return self.dataloader_get_gradients(input, args, grad_type)

        elif isinstance(input, AbstractDataset):
            return self.dataloader_get_gradients(DataLoader(input, self.batch_size), args, grad_type)

        else:
            log_error(TypeError, f"wrong type of 'input', current type: {type(input)}")

    def dataloader_get_gradients(self,
                                 input: DataLoader,
                                 *args: Any,
                                 grad_type: str = 'cls'
                                 ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input dataloader
            args: placeholder for compatibility

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """

        grads = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            grads.extend(self.tensor_get_gradients(batch, args, grad_type))

        return grads

    def tensor_get_gradients(self,
                             input: Tensor,
                             *args: Any,
                             grad_type: str = 'cls',
                             ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor
            args: placeholder for compatibility

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            list of dictionary for each object [{layer: gradients}]: List[Dict[str, Tensor]]
        """
        log_assert(grad_type in ['cls', 'obj'], "Wrong gradient type, must be 'cls' or 'obj' - objectness or class gradients")

        self.model.eval()

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        grads = []

        for i in input:
            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()
            i = i.to(self.device)
            # processed predictions, raw predictions
            pred = self.model(i)

            # _pred = convert_results(raw_preds, self.model)
            # assert ((pred-_pred) != 0).sum() == 0, "Incorrect result conversion"

            nms_preds, nms_preds_idxs = non_max_suppression_yolov5(pred.detach())

            # check if objects were detected
            if len(nms_preds_idxs) == 0:
                grads.append({bh.layer_name: None for bh in bhooks})
                [bh.clean_storage() for bh in bhooks]
                continue

            # raw vectors, for which bboxes were generated
            pred_vects = self._get_prediction_vectors(pred, nms_preds_idxs[0])
            # neurons to backprop from
            scalars = self._get_scalars(nms_preds, pred_vects)

            for obj_scl, cls_scl in scalars:
                if grad_type == 'cls':
                    torch.autograd.grad(cls_scl, i, retain_graph=True)
                else:
                    torch.autograd.grad(obj_scl, i, retain_graph=True)

            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]

        return grads

    def tensor_get_gradients_smoothgrad(self,
                                        input: Tensor,
                                        *args: Any,
                                        grad_type: str = 'cls',
                                        ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients using SmoothGrad

        Args:
            input: input batch tensor
            args: placeholder for compatibility

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            list of dictionary for each object [{layer: gradients}]: List[Dict[str, Tensor]]
        """
        log_assert(grad_type in ['cls', 'obj'], "Wrong gradient type, must be 'cls' or 'obj' - objectness or class gradients")

        self.model.eval()

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        grads = []

        for i in input:

            grads_sg = []

            ins = i.detach().to(self.device)  # orig
            ins_sg = apply_smoothgrad_noise(ins, SG_NOISE, SG_SAMPLES)  # noisy
            ins.unsqueeze_(0).requires_grad_()

            # prediction for original image
            pred = self.model(ins)
            nms_preds, nms_preds_idxs = non_max_suppression_yolov5(pred.detach())

            # check if objects were detected
            if len(nms_preds_idxs) == 0:
                grads.append({bh.layer_name: None for bh in bhooks})
                [bh.clean_storage() for bh in bhooks]
                continue

            # raw vectors, for which bboxes were generated
            pred_vects = self._get_prediction_vectors(pred, nms_preds_idxs[0])
            # neurons to backprop from
            scalars = self._get_scalars(nms_preds, pred_vects)

            for obj_scl, cls_scl in scalars:
                if grad_type == 'cls':
                    torch.autograd.grad(cls_scl, ins, retain_graph=True)
                else:
                    torch.autograd.grad(obj_scl, ins, retain_graph=True)

            grads_sg.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [bh.clean_storage() for bh in bhooks]

            # iterate noisy samples
            for i_sg in ins_sg:
                i_sg.unsqueeze_(0).requires_grad_()
                pred_sg = self.model(i_sg)

                pred_vects_sg = self._get_prediction_vectors(pred_sg, nms_preds_idxs[0])
                scalars_sg = self._get_scalars(nms_preds, pred_vects_sg)

                for obj_scl, cls_scl in scalars_sg:
                    if grad_type == 'cls':
                        torch.autograd.grad(cls_scl, i_sg, retain_graph=True)
                    else:
                        torch.autograd.grad(obj_scl, i_sg, retain_graph=True)

                grads_sg.append({bh.layer_name: bh.get_stacked_grad()
                            for bh in bhooks})

                [bh.clean_storage() for bh in bhooks]

            grads_mean = {l: torch.mean(torch.stack([g[l] for g in grads_sg]), 0) for l in self.layers}

            grads.append(grads_mean)

        bhooks = [bh.remove() for bh in bhooks]

        return grads

    def tensor_get_gradients_for_targets(self,
                                         input: Tensor,
                                         targets: Iterable[Tensor],
                                         grad_type: str = 'top_cls',
                                         ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients for target bounding boxes and classes.
        For backpropagation, the bounding box with highest IoU between target bbox and proposed by detector bbox is selected.

        Args:
            input: input batch tensor
            targets: list of target Tensors with bboxes and classes for each image - List[Tensor[n_obj, 5]], where tensor's 0th dim is n_obj long and 1st dim is [x1, y1, x2, y2, class]

        Kwargs:
            grad_type: 'tgt_cls', 'top_cls' or 'obj' - objectness or class gradients (top class or target class)

        Returns:
            list of dictionary for each object [{layer: gradients}]: List[Dict[str, Tensor]]
        """
        log_assert(grad_type in ['tgt_cls', 'top_cls', 'obj'], "Wrong gradient type, must be 'tgt_cls', 'top_cls' or 'obj' - objectness or class gradients")

        self.model.eval()

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        grads = []

        iou_val_list = []

        for i, tgts in zip(input, targets):

            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()
            i = i.to(self.device)

            # processed predictions, raw predictions
            pred = self.model(i)
            # prediction bboxes
            pred_boxes = xywh2xyxy(pred.detach()[0, :, :4])

            # get prediction bboxes with best IoU with target boxes
            tgt_bboxes = tgts[:, :4].to(self.device)
            boxes_iou = box_iou_yolo5(tgt_bboxes, pred_boxes)
            iou_max_vals, iou_max_idxs = torch.max(boxes_iou, dim=1)
            iou_val_list.append(iou_max_vals)

            # target bbox classes
            tgt_classes = tgts[:, 4].to(self.device).int()

            tgt_pred_vects = self._get_prediction_vectors(pred, iou_max_idxs)

            # list[tuple(objectness_scalar, tgt_cls_scalar, top_cls_scalar)]
            tgt_scalars = [(v[4], v[5+tc], torch.amax(v[5:])) for v, tc in zip(tgt_pred_vects, tgt_classes)]

            for obj_scl, tgt_cls_scl, top_cls_scl in tgt_scalars:
                if grad_type == 'top_cls':
                    torch.autograd.grad(top_cls_scl, i, retain_graph=True)
                elif grad_type == 'tgt_cls':
                    torch.autograd.grad(tgt_cls_scl, i, retain_graph=True)
                else:
                    torch.autograd.grad(obj_scl, i, retain_graph=True)

            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]

        return grads, iou_val_list
    
    def tensor_get_gradients_for_targets_and_classes(self,
                                                     input: Tensor,
                                                     targets: Iterable[Tensor],
                                                     classes: Iterable[int]
                                                     ) -> Tuple[List[List[Dict[int, Dict[str, Tensor]]]], Dict[str, List[List[Tensor]]]]:
        """
        Propagate forward and backward and get gradients for target bounding boxes and classes.
        For backpropagation, the bounding box with highest IoU between target bbox and proposed by detector bbox is selected.

        Args:
            input: input batch tensor
            targets: list of target Tensors with bboxes and classes for each image - List[Tensor[n_obj, 5]], where tensor's 0th dim is n_obj long and 1st dim is [x1, y1, x2, y2, class]
            classes: class ids for backpropagation

        Returns:
            tuple: (gradients, predictions)
                gradients: samples[bbox[{cls:{layer:Tensor[1, N_FILTERS, H, W] of gradients}}]]
                predictions: {'pred_vec': samples[bbox[Tensor[N_CLS + 4]]] with prediction, samples[bbox['pred_iou': Tensor[1]]] with IoU of prediction with target}
        """

        self.model.eval()

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        grads = []

        predictions = {'pred_vec': [],
                       'pred_iou': []}

        for i, tgts in zip(input, targets):

            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()
            i = i.to(self.device)

            # processed predictions, raw predictions
            pred = self.model(i)
            # prediction bboxes
            pred_boxes = xywh2xyxy(pred.detach()[0, :, :4])

            # get prediction bboxes with best IoU with target boxes
            tgt_bboxes = tgts[:, :4].to(self.device)
            boxes_iou = box_iou_yolo5(tgt_bboxes, pred_boxes)
            iou_max_vals, iou_max_idxs = torch.max(boxes_iou, dim=1)

            tgt_pred_vects = self._get_prediction_vectors(pred, iou_max_idxs)

            # scalars (neurons) of desired classes for target predictions
            # first 4 neurons - bbox coords, 5th neuron - objectness
            tgt_scalars = [[v[5+c] for c in classes] for v in tgt_pred_vects]

            tgt_bbox_grads = []
            # bboxes (targets) loop
            for bbox_tgt_scalars in tgt_scalars:
                tgt_cls_grads = {}
                # target classes loop
                for bbox_tgt_scalar, tgt_cls in zip(bbox_tgt_scalars, classes):
                    # backprop for given bbox and class
                    torch.autograd.grad(bbox_tgt_scalar, i, retain_graph=True)
                    # accumulate gradients for each class 
                    tgt_cls_grads[tgt_cls] = {bh.layer_name: bh.get_stacked_grad() for bh in bhooks}
                    [bh.clean_storage() for bh in bhooks]
                # accumulate gradients for each bbox
                tgt_bbox_grads.append(tgt_cls_grads)

            # accumulate gradients and predictions
            # torch.hstack([v[:4], v[5:]]) removes objectness neuron with id 4
            predictions['pred_vec'].append([torch.hstack([v[:4], v[5:]]).detach().cpu() for v in tgt_pred_vects])
            predictions['pred_iou'].append(iou_max_vals.detach().cpu())
            grads.append(tgt_bbox_grads)

        bhooks = [bh.remove() for bh in bhooks]

        return grads, predictions

    def _get_prediction_vectors(self,
                                raw_pred_reshaped: Tensor,
                                ids: Iterable[int]
                                ) -> List[Tensor]:
        return [raw_pred_reshaped[:, i, :].squeeze() for i in ids]

    def _get_scalars(self,
                     nms_preds: Iterable[Tensor],
                     pred_vects: Iterable[Tensor]
                     ) -> List[Tuple[Tensor, Tensor]]:
        res = []

        for i in range(len(nms_preds[0])):
            objectness_scalar = pred_vects[i][4]
            class_scalar_idx = nms_preds[0][i].squeeze()[5].int()
            class_scalar = pred_vects[i][class_scalar_idx + 5]  # 5 is offset
            res.append((objectness_scalar, class_scalar))

        return res

    def tensor_get_predictions_from_activations(self,
                                                input: Tensor,
                                                model_input_shape: Tuple[int, int],
                                                layer: str = None,
                                                ) -> List[Tensor]:
        """
        Propagates the intermediate activations forward starting from :layer: to get predictions. Dummy input is used until layer X.

        Args:
            input: input tensor (B, C, H, W)
            model_input_shape: input shape of model input (without batch dimension: e.g., (3, 224, 224)), required for dummy data pass - Tuple[int, int, int]

        Kwargs:
            layer: layer to inject activations and propagate further, if None - self.layers[0]

        Return:
            predictions: (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
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

            pred, idxs = non_max_suppression_yolov5(pred)

            pred = [p.detach().cpu() for p in pred]

            hook.remove()

            return pred
        
    def tensor_get_predictions_from_activations_multilayer(self,
                                                           activation_inputs: Dict[str, Tensor],
                                                           model_input: Tensor,
                                                           conf_thres: float = 0.25,
                                                           iou_thres: float = 0.45
                                                           ) -> List[Tensor]:
        """
        Propagates the intermediate activations forward starting from :layer: to get predictions. Dummy input is used until layer X.

        Args:
            activation_inputs (Dict[str, Tensor[B, C, H, W]]): dict of input tensors (B, C, H, W) for each layer
            model_input [Tensor]: dummy model input

        Kwargs:
            conf_thres (float): confidence threshold for NMS
            iou_thres (float): IoU threshold for NMS
            
        Return:
            predictions: (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
        """
        with torch.no_grad():

            self.model.eval()

            hooks = []

            for layer, activation_input in activation_inputs.items():
                hook = ForwardInsertHook(layer, self.modules[layer])
                device_input = activation_input.to(self.device)
                hook.set_insert_tensor(device_input)  # set injection data
                hooks.append(hook)

            # dummy data is used until the injection at layer X
            dummy = model_input.to(self.device)

            pred = self.model(dummy.to(self.device))

            pred, idxs = non_max_suppression_yolov5(pred, conf_thres=conf_thres, iou_thres=iou_thres)

            pred = [p.detach().cpu() for p in pred]

            [hook.remove() for hook in hooks]

            return pred

    def dataloader_get_predictions_from_activations(self,
                                                    input: DataLoader,
                                                    model_input_shape: Tuple[int],
                                                    layer: str = None,
                                                    ) -> List[Tensor]:
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

            temp.extend(self.tensor_get_predictions_from_activations(
                batch, model_input_shape, layer))

        return temp

    def tensor_get_preds_acts_grads(self,
                                    input: Tensor,
                                    grad_type: str = 'cls',
                                    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor

        Kwargs:
            grad_type: 'cls' or 'obj' - objectness or class gradients

        Returns:
            predictions: List[Tensor]
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            gradients {layer:gradients}: Dict[str, Tensor[...]]
        """
        log_assert(grad_type in ['cls', 'obj'], "Wrong gradient type, must be 'cls' or 'obj' - objectness or class gradients")

        self.model.eval()

        fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]
        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        preds = []
        acts = []
        grads = []

        for i in input:
            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()

            # processed predictions, raw predictions
            pred, raw_preds = self.model(i)

            # _pred = convert_results(raw_preds, self.model)
            # assert ((pred-_pred) != 0).sum() == 0, "Incorrect result conversion"

            nms_preds, nms_preds_idxs = non_max_suppression_yolov5(pred.detach())

            # raw predictions shape now have the same shape (and indexes) as processed ones
            raw_pred_reshaped = self._reshape_raw_preds(raw_preds)

            # raw vectors, for which bboxes were generated
            pred_vects = self._get_prediction_vectors(
                raw_pred_reshaped, nms_preds_idxs[0])

            scalars = self._get_scalars(nms_preds, pred_vects)

            for obj_scl, cls_scl in scalars:
                if grad_type == 'cls':
                    torch.autograd.grad(cls_scl, i, retain_graph=True)
                else:
                    torch.autograd.grad(obj_scl, i, retain_graph=True)

            preds.append(pred)
            acts.append({fh.layer_name: fh.get_stacked_activtion()
                         for fh in fhooks})
            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [fh.clean_storage() for fh in fhooks]
            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]
        fhooks = [fh.remove() for fh in fhooks]

        return preds, acts, grads


def non_max_suppression_yolov5(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=1000,
        nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Modified code from https://github.com/ultralytics/yolov5

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # YOLOv5 model in validation model, output = (inference_out, loss_out)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    batch_size = prediction.shape[0] 
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_box_area = 7680  # in pixels
    max_boxes = 10000

    mask_start_idx = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * batch_size
    output_idxs = []

    for xi, x in enumerate(prediction):  # image index, image inference

        x_idxs = torch.arange(0, len(x), 1).to(xc.device)[xc[xi]]
        x = x[xc[xi]]  # confidence

        if not x.shape[0]:
            continue

        # confidences
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mask_start_idx:]  # zero columns if no masks

        conf, j = x[:, 5:mask_start_idx].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_boxes:  # excess boxes
            sorted_idxs = x[:, 4].argsort(descending=True)[:max_boxes]
            x = x[sorted_idxs]  # sort by confidence
            x_idxs = x_idxs[sorted_idxs]
        else:
            sorted_idxs = x[:, 4].argsort(descending=True)[:max_boxes]
            x = x[sorted_idxs]  # sort by confidence
            x_idxs = x_idxs[sorted_idxs]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_box_area)  # classes
        boxes = x[:, :4] + c
        scores = x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        output[xi] = x[i]
        output_idxs.append(x_idxs[i])

    return output, output_idxs
