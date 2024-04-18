'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from typing import Any, Dict, List, Tuple, Union, Iterable
import torch
from torch import Tensor
from hooks import BackwardHook, ForwardHook, ForwardInsertHook, Propagator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torch.utils.data import DataLoader
from xai_utils.logging import log_assert
from xai_utils.bbox import box_iou_yolo5


class PropagatorTorchDetector(Propagator):

    def __init__(self,
                 model: torch.nn.Module,
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

    def tensor_get_predictions(self,
                               input: Tensor
                               ) -> List[Tensor]:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input batch tensor

        Returns:
            predictions: (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
        """
        with torch.no_grad():
            self.model.eval()

            device_input = input.to(self.device)

            # list of predictions - dict with keys: "boxes", "labels", "scores"
            pred = self.model(device_input)

            return self._process_pred(pred)

    @staticmethod
    def _process_pred(pred: Dict[str, Tensor]
                      ) -> List[Tensor]:
        """
        process the prediction of RCNN, convert to YOLO's format

        Args:
            prediction: list of dictionaries, dicts with keys: "boxes", "labels", "scores"

        Returns:
            predictions: (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
        """

        # predictions to YOLO format
        res = []
        for p in pred:
            boxes = p['boxes']
            scores = p['scores'].unsqueeze(1)
            labels = p['labels'].unsqueeze(1)
            # Tensor[N_pred, 6]: 0:3 - bbox, 4 - probability, 5 - class
            res.append(torch.hstack([boxes, scores, labels]).detach().cpu())

        return res

    def dataloader_get_predictions(self,
                                   input: DataLoader
                                   ) -> List[Tensor]:
        """
        Propagates the input through the network to get predictions

        Args:
            input: input dataloader

        Returns:
            predictions: (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
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

    def tensor_get_gradients(self,
                             input: Tensor,
                             *args: Any
                             ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input batch tensor
            args: placeholder for compatibility

        Returns:
            list of dictionary for each object [{layer: gradients}]: List[Dict[str, Tensor]]
        """
        self.model.eval()

        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        grads = []

        for i in input:
            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()
            i = i.to(self.device)
            # predictions dict with keys: "boxes", "labels", "scores"
            predictions = self.model(i)

            scores = predictions[0]["scores"]

            for score in scores:
                torch.autograd.grad(score, i, retain_graph=True)

            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]

        return grads

    def tensor_get_gradients_for_targets(self,
                                         input: Tensor,
                                         targets: Iterable[Tensor]
                                         ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients for target bounding boxes and classes.
        For backpropagation, the bounding box with highest IoU between target bbox and proposed by detector bbox is selected.

        Args:
            input: input batch tensor
            targets: list of target Tensors with bboxes and classes for each image - List[Tensor[n_obj, 5]], where tensor's 0th dim is n_obj long and 1st dim is [x1, y1, x2, y2, class]

        Returns:
            list of dictionary for each object [{layer: gradients}]: List[Dict[str, Tensor]]
        """
        # change model-specific parameters here, e.g., IoU or NMS thresholds

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
            pred_boxes = pred[0]['boxes'].detach()

            # get prediction bboxes with best IoU with target boxes
            tgt_bboxes = tgts[:, :4].to(self.device)
            boxes_iou = box_iou_yolo5(tgt_bboxes, pred_boxes)
            iou_max_vals, iou_max_idxs = torch.max(boxes_iou, dim=1)
            iou_val_list.append(iou_max_vals)

            for iou_idx in iou_max_idxs:
                score = pred[0]['scores'][iou_idx]
                torch.autograd.grad(score, i, retain_graph=True)

            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]

        # restore model-specific parameters here, e.g., IoU or NMS thresholds

        return grads, iou_val_list
    
    def tensor_get_gradients_smoothgrad(self,
                                        input: Tensor,
                                        *args: Any,
                                        grad_type: str = 'cls',
                                        ) -> List[Dict[str, Tensor]]:
        raise NotImplementedError("Not implemented.")
    
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
            pred_boxes = pred[0]['boxes'].detach()

            # get prediction bboxes with best IoU with target boxes
            tgt_bboxes = tgts[:, :4].to(self.device)
            boxes_iou = box_iou_yolo5(tgt_bboxes, pred_boxes)
            iou_max_vals, iou_max_idxs = torch.max(boxes_iou, dim=1)

            #for iou_idx in iou_max_idxs:
            #    score = pred[0]['scores'][iou_idx]
            #    torch.autograd.grad(score, i, retain_graph=True)

            #grads.append({bh.layer_name: bh.get_stacked_grad() for bh in bhooks})

            #[bh.clean_storage() for bh in bhooks]

            tgt_pred_scores = pred[0]['all_scores'][iou_max_idxs]
            tgt_pred_boxes = pred[0]['boxes'][iou_max_idxs]

            tgt_scalars = [[v[c] for c in classes] for v in tgt_pred_scores]

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
            # torch.hstack([tpb, tps]) combines bboxes and scores
            predictions['pred_vec'].append([torch.hstack([tpb.squeeze(), tps.squeeze()]).detach().cpu() for tpb, tps in zip(tgt_pred_boxes, tgt_pred_scores)])
            predictions['pred_iou'].append(iou_max_vals.detach().cpu())
            grads.append(tgt_bbox_grads)

        bhooks = [bh.remove() for bh in bhooks]

        # restore model-specific parameters here, e.g., IoU or NMS thresholds

        return grads, predictions

    def dataloader_get_gradients(self,
                                 input: DataLoader,
                                 *args: Any
                                 ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients

        Args:
            input: input dataloader
            args: placeholder for compatibility

        Returns:
            dictionary - {layer: gradients}: Dict[str, Tensor[...]]
        """

        grads = []

        for batch in input:

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            grads.extend(self.tensor_get_gradients(batch, args))

        return grads

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

            hook.remove()

            return self._process_pred(pred)

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

        for batch in input, 'Predictions from Activations':

            err_msg = f"'input' dataloader must return tensors, current type: {type(batch)}"
            log_assert(isinstance(batch, Tensor), err_msg)

            temp.extend(self.tensor_get_predictions_from_activations(
                batch, model_input_shape, layer))

        return temp

    def tensor_get_preds_acts_grads(self,
                                    input: Tensor
                                    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
        """
        Propagate forward and backward and get predictions, activations, gradients

        Args:
            input: input tensor batch

        Returns:
            predictions: List[Tensor]
            dictionary {layer:activations}: Dict[str, Tensor[...]]
            gradients {layer:gradients}: Dict[str, Tensor[...]]
        """

        self.model.eval()

        fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]
        bhooks = [BackwardHook(l, m) for l, m in self.modules.items()]

        preds = []
        acts = []
        grads = []

        for i in input:
            # always pass 1 img
            i.unsqueeze_(0).requires_grad_()

            # predictions dict with keys: "boxes", "labels", "scores"
            predictions = self.model(i)

            scores = predictions[0]["scores"]

            for score in scores:
                torch.autograd.grad(score, i)

            preds.append(predictions)
            acts.append({fh.layer_name: fh.get_stacked_activtion()
                         for fh in fhooks})
            grads.append({bh.layer_name: bh.get_stacked_grad()
                         for bh in bhooks})

            [fh.clean_storage() for fh in fhooks]
            [bh.clean_storage() for bh in bhooks]

        bhooks = [bh.remove() for bh in bhooks]
        fhooks = [fh.remove() for fh in fhooks]

        return preds, acts, grads
