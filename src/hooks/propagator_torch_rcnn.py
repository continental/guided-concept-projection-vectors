'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from typing import Dict, List, Union, Iterable, Tuple, Optional
import torch
from torch import Tensor
from hooks import PropagatorTorchDetector
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
import torchvision.ops.boxes as box_ops
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import keypointrcnn_loss, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference, keypointrcnn_inference, RoIHeads


class PropagatorTorchRCNN(PropagatorTorchDetector):

    def __init__(self,
                 model: GeneralizedRCNN,
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

        # override model's RoI heads' methods
        self.model.roi_heads.postprocess_detections = postprocess_detections_new.__get__(self.model.roi_heads, RoIHeads)
        self.model.roi_heads.forward = forward_new.__get__(self.model.roi_heads, RoIHeads)

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
        # roi params
        temp_roi_score_thresh = self.model.roi_heads.score_thresh
        temp_roi_nms_thresh = self.model.roi_heads.nms_thresh
        temp_roi_detections_per_img = self.model.roi_heads.detections_per_img
        self.model.roi_heads.score_thresh = 0.0
        self.model.roi_heads.nms_thresh = 0.0
        self.model.roi_heads.detections_per_img = 100_000
        # rpn params
        # temp_rpn_score_thresh = self.model.rpn.score_thresh
        # temp_rpn_nms_thresh = self.model.rpn.nms_thresh
        temp_rpn_pre_nms_top_n = self.model.rpn._pre_nms_top_n
        temp_rpn_post_nms_top_n = self.model.rpn._post_nms_top_n
        # self.model.rpn.score_thresh = 0.0
        # self.model.rpn.nms_thresh = 0.0
        self.model.rpn._pre_nms_top_n = {'training': 2000, 'testing': 10_000}
        self.model.rpn._post_nms_top_n = {'training': 2000, 'testing': 10_000}

        res = super(PropagatorTorchRCNN, self).tensor_get_gradients_for_targets(input, targets)

        # restore model parameters
        self.model.roi_heads.score_thresh = temp_roi_score_thresh
        self.model.roi_heads.nms_thresh = temp_roi_nms_thresh
        self.model.roi_heads.detections_per_img = temp_roi_detections_per_img
        self.model.rpn._pre_nms_top_n = temp_rpn_pre_nms_top_n
        self.model.rpn._post_nms_top_n = temp_rpn_post_nms_top_n

        return res


def postprocess_detections_new(self,
                               class_logits: Tensor,
                               box_regression: Tensor,
                               proposals: List[Tensor],
                               img_shapes: List[Tuple[int, int]],
                               ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    overrides torchvision.models.detection.roi_heads.postprocess_detections
    returns dict with all class logits "all_scores" instead of only of class logint for top-class
    """

    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    bboxes_all = []
    scores_all = []
    labels_all = []
    scores_unbatched_all = []
    for bboxes, pred_logits, img_shape in zip(pred_boxes_list, pred_scores_list, img_shapes):
        bboxes = box_ops.clip_boxes_to_image(bboxes, img_shape)

        # prediction labels
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(pred_logits)

        # remove bg predictions
        scores_unbatched = pred_logits[:, 1:]  # save all scores
        bboxes = bboxes[:, 1:]
        pred_logits = pred_logits[:, 1:]
        labels = labels[:, 1:]

        # batch
        bboxes = bboxes.reshape(-1, 4)
        pred_logits = pred_logits.reshape(-1)
        labels = labels.reshape(-1)

        # keeps ids of scores
        scores_ids = torch.arange(len(pred_logits)).to(pred_logits.device)

        # remove low conf boxes
        inds = torch.where(pred_logits > self.score_thresh)[0]
        bboxes, pred_logits, labels = bboxes[inds], pred_logits[inds], labels[inds]
        scores_ids = scores_ids[inds]

        # remove small boxes
        keep = box_ops.remove_small_boxes(bboxes, min_size=1e-2)
        bboxes, pred_logits, labels = bboxes[keep], pred_logits[keep], labels[keep]
        scores_ids = scores_ids[keep]

        # nms
        nmsed = box_ops.batched_nms(bboxes, pred_logits, labels, self.nms_thresh)
        # uncomment to keep less: nmsed = nmsed[: self.detections_per_img]
        bboxes, pred_logits, labels = bboxes[nmsed], pred_logits[nmsed], labels[nmsed]
        scores_ids = scores_ids[nmsed]

        bboxes_all.append(bboxes)
        scores_all.append(pred_logits)
        labels_all.append(labels)

        # all scores from unbatched scores
        scores_ids = scores_ids // (num_classes - 1)  # get the row of scores from ids
        scores_unbatched = scores_unbatched[scores_ids]  # for each resulting bbox get all scores
        scores_unbatched_all.append(scores_unbatched)

    return bboxes_all, scores_all, labels_all, scores_unbatched_all


def forward_new(self,
                features: Dict[str, Tensor],
                proposals: List[Tensor],
                img_shapes: List[Tuple[int, int]],
                targets: Optional[List[Dict[str, Tensor]]] = None,
                ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
    """
    overrides torchvision.models.detection.roi_heads.forward
    returns dict with all class logits "all_scores" instead of only of class logint for top-class
    """

    if self.training:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    bbox_features = self.box_roi_pool(features, proposals, img_shapes)
    bbox_features = self.box_head(bbox_features)
    class_logits, box_regression = self.box_predictor(bbox_features)

    result: List[Dict[str, torch.Tensor]] = []
    losses = {}
    if self.training:
        loss_classif, loss_bbox_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classif, "loss_box_reg": loss_bbox_reg}
    else:
        bboxes, pred_logits, labels, pred_logits_unbatched = self.postprocess_detections(class_logits, box_regression, proposals, img_shapes)
        n_imgs = len(bboxes)
        for i in range(n_imgs):
            result.append(
                {
                    "boxes": bboxes[i],
                    "labels": labels[i],
                    "scores": pred_logits[i],
                    "all_scores": pred_logits_unbatched[i]
                }
            )

    # segmentation and keypoint detection removed

    return result, losses
