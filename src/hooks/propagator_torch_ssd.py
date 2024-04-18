'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from hooks import PropagatorTorchDetector
from typing import Dict, List, Union, Iterable, Tuple
from torchvision.models.detection.ssd import SSD
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops


class PropagatorTorchSSD(PropagatorTorchDetector):

    def __init__(self,
                 model: SSD,
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

        # override model's postpocessing
        self.model.postprocess_detections = postprocess_detections_new.__get__(self.model, SSD)

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

        # change model parameters to get additional bounding boxes
        temp_score_thresh = self.model.score_thresh
        temp_nms_thresh = self.model.nms_thresh
        temp_detections_per_img = self.model.detections_per_img
        temp_topk_candidates = self.model.topk_candidates

        # set parameters, to soften NMS and aquire more bboxes
        self.model.score_thresh = 0.001
        self.model.nms_thresh = 0.9
        self.model.detections_per_img = 100_000
        self.model.topk_candidates = 100_000

        res = super(PropagatorTorchSSD, self).tensor_get_gradients_for_targets(input, targets)

        # restore model parameters
        self.model.score_thresh = temp_score_thresh
        self.model.nms_thresh = temp_nms_thresh
        self.model.detections_per_img = temp_detections_per_img
        self.model.topk_candidates = temp_topk_candidates

        return res

    def tensor_get_gradients_for_targets_and_classes(self,
                                                     input: Tensor,
                                                     targets: Iterable[Tensor],
                                                     classes: Iterable[int]
                                                     ) -> Tuple[List[List[Dict[int, Dict[str, Tensor]]]], Dict[str, List[List[Tensor]]]]:
        
        # change model parameters to get additional bounding boxes
        temp_score_thresh = self.model.score_thresh
        temp_nms_thresh = self.model.nms_thresh
        temp_detections_per_img = self.model.detections_per_img
        temp_topk_candidates = self.model.topk_candidates

        # set parameters, to soften NMS and aquire more bboxes
        self.model.score_thresh = 0.001
        self.model.nms_thresh = 0.9
        self.model.detections_per_img = 100_000
        self.model.topk_candidates = 100_000

        res = super(PropagatorTorchSSD, self).tensor_get_gradients_for_targets_and_classes(input, targets, classes)

        # restore model parameters
        self.model.score_thresh = temp_score_thresh
        self.model.nms_thresh = temp_nms_thresh
        self.model.detections_per_img = temp_detections_per_img
        self.model.topk_candidates = temp_topk_candidates

        return res

def postprocess_detections_new(self, 
                               head_outputs: Dict[str, Tensor],
                               img_anchors: List[Tensor],
                               img_shapes: List[Tuple[int, int]]
                               ) -> List[Dict[str, Tensor]]:
    """
    overrides SSD.postprocess_detections
    returns dict with all class logits "all_scores" instead of only of class logint for top-class
    """
    bbox_regression = head_outputs["bbox_regression"]
    cls_logits = F.softmax(head_outputs["cls_logits"], dim=-1)

    num_classes = cls_logits.size(-1)
    device = cls_logits.device

    detections: List[Dict[str, Tensor]] = []

    for bboxes, scores, anchors, img_shape in zip(bbox_regression, cls_logits, img_anchors, img_shapes):
        bboxes = self.box_coder.decode_single(bboxes, anchors)
        bboxes = box_ops.clip_boxes_to_image(bboxes, img_shape)

        img_boxes = []
        img_scores = []
        img_labels = []
        img_all_scores = []
        for label in range(1, num_classes):
            score = scores[:, label]
            all_scores = scores[:, 1:]  # all scores without background class - 0th

            score_keep_idxs = score > self.score_thresh
            score = score[score_keep_idxs]
            bbox = bboxes[score_keep_idxs]
            all_scores = all_scores[score_keep_idxs]

            # topk scoring predictions
            num_topk = min(len(score), self.topk_candidates)
            score, idxs = score.topk(num_topk)
            bbox = bbox[idxs]
            all_scores = all_scores[idxs]

            img_boxes.append(bbox)
            img_scores.append(score)
            img_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))
            img_all_scores.append(all_scores)

        img_boxes = torch.cat(img_boxes, dim=0)
        img_scores = torch.cat(img_scores, dim=0)
        img_labels = torch.cat(img_labels, dim=0)
        img_all_scores = torch.cat(img_all_scores, dim=0)

        # nms
        nmsed = box_ops.batched_nms(img_boxes, img_scores, img_labels, self.nms_thresh)

        # uncomment to keep less: nmsed = nmsed[: self.detections_per_img]

        detections.append(
            {
                "boxes": img_boxes[nmsed],
                "scores": img_scores[nmsed],
                "labels": img_labels[nmsed],
                "all_scores": img_all_scores[nmsed]  # all class logits
            }
        )
    return detections
