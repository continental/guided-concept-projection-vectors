'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from typing import Dict, Any, Tuple, Iterable
import torch
from PIL import Image, ImageDraw


def norm_bbox_coord(bbox_xywh: Tuple[float, float, float, float],
                    img_w: int,
                    img_h: int
                    ) -> Tuple[float, float, float, float]:
    """
    Normalize absolute bounding box coords to normalized ones from MSCOCO annotations (from 0 to 1)

    Args:
        bbox_xywh: bbox tuple with absolute coordinates (x, y, w, h) (top-left)
        img_w: image width
        img_h: image height

    Returns:
        bbox tuple with normalized coordinates (x, y, w, h)
    """
    x, y, w, h = bbox_xywh
    return x / img_w, y / img_h, w / img_w, h / img_h


def xywh2xyxy_bbox_coord(bbox_xywh: Tuple[float, float, float, float],
                         ) -> Tuple[float, float, float, float]:
    """
    Convert MSCOCO bounding box coords from xywh (top-left) to xyxy format

    Args:
        bbox_xywh: bbox tuple with absolute coordinates (x, y, w, h)

    Returns:
        bbox tuple with normalized coordinates (x, y, w, h) (top-left)
    """
    x, y, w, h = bbox_xywh
    return x, y, x + w, y + h


def scale_bbox_coord(norm_bbox_xywh: Tuple[float, float, float, float],
                     img_w: int,
                     img_h: int
                     ) -> Tuple[float, float, float, float]:
    """
    Scale normalized bounding box coords to absolute ones from MSCOCO annotations

    Args:
        norm_bbox_xywh: bbox tuple with normalized coordinates (x, y, w, h) (top-left)
        img_w: image width
        img_h: image height

    Returns:
        bbox tuple with absolute coordinates (x, y, w, h) (top-left)
    """
    x, y, w, h = norm_bbox_xywh
    return x * img_w, y * img_h, w * img_w, h * img_h


def box_iou_yolo5(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Calculate IoU of 2 bbox tensors

    Arguments:
        box1 (Tensor)
        box2 (Tensor)

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def area_bbox(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    a1, a2 = boxes1[:, None].chunk(2, 2)
    b1, b2 = boxes2.chunk(2, 1)

    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    union = area_bbox(boxes1.T)[:, None] + area_bbox(boxes2.T) - intersection + 0.000001
    return intersection / union


def add_bboxes_to_img(img: Image,
                      pred_bboxes_xyxy: Iterable[Iterable[int]],
                      pred_w_h: Tuple[int, int] = (640, 480),
                      bbox_color: str = 'red'
                      ) -> Image:
    """
    converts grayscale binary mask to RGB mask

    Args:
        img (Image): PIL image
        pred_bboxes_xyxy (Iterable[Iterable[int]]): xyxy bboxes

    Kwargs:
        pred_w_h (Tuple[int, int]): W, H dimensions of original network input image
        bbox_color (str): color of bounding box

    Returns:
        (Image) RGB image with bboxes
    """
    #bboxes_xyxy = bbox[:, :4].tolist()

    bboxes_xyxy_norm = [norm_bbox_coord(bbox, pred_w_h[0], pred_w_h[1]) for bbox in pred_bboxes_xyxy]
    bboxes_xyxy_scaled = [[int(c) for c in scale_bbox_coord(bbox, img.size[0], img.size[1])] for bbox in bboxes_xyxy_norm]

    draw = ImageDraw.Draw(img)

    for x1, y1, x2, y2 in bboxes_xyxy_scaled:
        draw.rectangle([(x1, y1), (x2, y2)], outline=bbox_color, width=5)

    return img