'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

from PIL import Image, ImageDraw
import numpy as np

from data_structures import get_polygons_from_rle


class AbstractSemanticSegmenter(ABC):

    @abstractmethod
    def __init__(self,
                 obj_category_id: int
                 ) -> None:
        self.obj_category_id = obj_category_id # keeps category id of segmented object
        pass
    
    @abstractmethod
    def segment_sample(self,
                       image_name: str
                       ) -> np.ndarray:
        pass


class MSCOCOSemanticSegmentationLoader(AbstractSemanticSegmenter):

    def __init__(self,
                 coco_annotations: Dict[str, Any],
                 obj_category_id: int
                 ) -> None:
        """
        Args:
            coco_annotations (Dict[str, Any]): MS COCO annotations
            obj_category_id (int): MS COCO category to draw polygons for
        """
        self.coco_json = coco_annotations
        self.obj_category_id = obj_category_id

    def segment_sample(self,
                       img_name: str,
                       ) -> np.ndarray:
        """
        Generate segmentation mask for given category of MS COCO sample

        Args:
            img_name (str): MS COCO image file name

        Returns:
            (np.ndarray) numpy ndarray with binary mask
        """
        # get sample metadata
        coco_img_metadata = [i for i in self.coco_json['images'] if i['file_name'] == img_name][0]
        coco_img_id = coco_img_metadata['id']
        coco_img_annots = [a for a in self.coco_json['annotations'] if a['image_id'] == coco_img_id]

        # get segmentations for given category
        coco_img_annot_category = [a for a in coco_img_annots if a['category_id'] == self.obj_category_id]
        object_segmentations = []
        for a in coco_img_annot_category:
            polygons = a['segmentation']
            if not isinstance(polygons, list):
                polygons = get_polygons_from_rle(a)
                
            object_segmentations.append(polygons)

        segmentations_xy = self._process_segmentations(object_segmentations)
        w, h = coco_img_metadata['width'], coco_img_metadata['height']

        coco_mask = self._get_segmentations_map(w, h, segmentations_xy)

        return coco_mask
    
    @staticmethod
    def _process_segmentations(coco_segmentation: List[List[float]]) -> List[List[Tuple[int, int]]]:
        """
        list of COCO segmentations lists of length 2*N -> list of segmentation xy-points tuples lists of length N
        [[x1, y1, x2, y2, x3, y3, ...]] -> [[(x1, y1), (x2, y2), (x3, y3), ...]]

        Args:
            coco_segmentation (List[List[float]]): list of coco segmentations - [[x1, y1, x2, y2, x3, y3, ...]]

        Returns:
            (List[List[Tuple[int, int]]]) list of segmentations suitable for polygon drawing in PIL: [[(x1, y1), (x2, y2), (x3, y3), ...]]
        """
        segmentations_xy = []
        for object_polygons in coco_segmentation:
            object_polygons_xy = []
            for p in object_polygons:
                plygon_list_list = np.array(p).reshape(-1, 2).astype(np.int32).tolist()
                polygon_list_tuple = [(x, y) for x, y in plygon_list_list]
                object_polygons_xy.append(polygon_list_tuple)
            segmentations_xy.append(object_polygons_xy)

        # one-line hard-to-read solution
        # segmentations_xy = [[[(x, y) for x, y in np.array(p).reshape(-1, 2).astype(np.int32).tolist()] for p in polygons] for polygons in object_segmentations]
        return segmentations_xy
    
    @staticmethod
    def _get_segmentations_map(width: int,
                               height: int,
                               segmentations_xy: List[List[Tuple[int, int]]]
                               ) -> np.ndarray:
        """
        Create image from scratch and draw segmentation polygons on it

        Args:
            width (int): image width
            height (int): image height

        Returns:
            (np.ndarray) numpy ndarray with binary mask
        """
        coco_masks = []

        for object_polygons_xy in segmentations_xy:
            coco_mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(coco_mask)
            # object mask can contain several polygons
            for p in object_polygons_xy:
                draw.polygon(p, outline=1, fill=1)

            coco_mask_array = np.array(coco_mask)
            coco_masks.append(coco_mask_array  > 0)

        coco_masks = np.array(coco_masks)

        joint_coco_mask = coco_masks.sum(axis=0) > 0

        #plot_binary_mask(joint_coco_mask)
        #print("coco masks shape", coco_masks.shape)
        #print("Joint-mask:", joint_coco_mask.shape)

        return joint_coco_mask


class MSCOCORectangleSegmenter(AbstractSemanticSegmenter):

    def __init__(self,
                 coco_annotations_json: Dict[str, Any],
                 obj_category_id: int
                 ) -> None:
        """
        Args:
            coco_annotations_json (Dict[str, Any]): JSON-file with MS COCO annotations
            obj_category_id (int): MS COCO category to draw polygons for
        """
        self.coco_json = coco_annotations_json
        self.obj_category_id = obj_category_id

    def segment_sample(self,
                       img_name: str                       
                       ) -> np.ndarray:
        """
        Generate segmentation mask for given category of MS COCO sample

        Args:
            img_name (str): MS COCO image file name            

        Returns:
            (np.ndarray) numpy ndarray with binary mask
        """
        # get sample metadata
        coco_img_metadata = [i for i in self.coco_json['images'] if i['file_name'] == img_name][0]
        coco_img_id = coco_img_metadata['id']
        coco_img_annots = [a for a in self.coco_json['annotations'] if a['image_id'] == coco_img_id]

        # get segmentations for given category
        coco_img_annot_category = [a for a in coco_img_annots if a['category_id'] == self.obj_category_id]

        object_bboxes = [[int(coord) for coord in a['bbox']] for a in coco_img_annot_category if isinstance(a['segmentation'], list)]
        w, h = coco_img_metadata['width'], coco_img_metadata['height']

        coco_mask = self._get_segmentations_map(h, w, object_bboxes)

        return coco_mask
    
    
    @staticmethod
    def _get_segmentations_map(width: int,
                               height: int,
                               object_bboxes: List[List[int]]
                               ) -> np.ndarray:
        """
        Create image from scratch and draw segmentation polygons on it

        Args:
            width (int): image width
            height (int): image height
            object_bboxes (List[List[int]]): list of bbox coordinates

        Returns:
            (np.ndarray) numpy ndarray with binary mask
        """
        coco_masks = []

        for bbox_xywh in object_bboxes:
            bbox = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]

            # Create a new image with a black background
            background = Image.new('L', (height, width), 0)

            # Create a new image with a white ellipse
            draw = ImageDraw.Draw(background)

            # Draw the ellipse
            draw.rectangle(bbox, fill=255)

            coco_mask_array = np.array(background)
            coco_masks.append(coco_mask_array  > 0)

        coco_masks = np.array(coco_masks)

        joint_coco_mask = coco_masks.sum(axis=0) > 0

        #plot_binary_mask(joint_coco_mask)
        #print("coco masks shape", coco_masks.shape)
        #print("Joint-mask:", joint_coco_mask.shape)

        return joint_coco_mask


class MSCOCOEllipseSegmenter(AbstractSemanticSegmenter):

    def __init__(self,
                 coco_annotations_json: Dict[str, Any],
                 obj_category_id: int
                 ) -> None:
        """
        Args:
            coco_annotations_json (Dict[str, Any]): JSON-file with MS COCO annotations
            obj_category_id (int): MS COCO category to draw polygons for
        """
        self.coco_json = coco_annotations_json
        self.obj_category_id = obj_category_id

    def segment_sample(self,
                       img_name: str                       
                       ) -> np.ndarray:
        """
        Generate segmentation mask for given category of MS COCO sample

        Args:
            img_name (str): MS COCO image file name            

        Returns:
            (np.ndarray) numpy ndarray with binary mask
        """
        # get sample metadata
        coco_img_metadata = [i for i in self.coco_json['images'] if i['file_name'] == img_name][0]
        coco_img_id = coco_img_metadata['id']
        coco_img_annots = [a for a in self.coco_json['annotations'] if a['image_id'] == coco_img_id]

        # get segmentations for given category
        coco_img_annot_category = [a for a in coco_img_annots if a['category_id'] == self.obj_category_id]

        object_bboxes = [[int(coord) for coord in a['bbox']] for a in coco_img_annot_category if isinstance(a['segmentation'], list)]
        w, h = coco_img_metadata['width'], coco_img_metadata['height']

        coco_mask = self._get_segmentations_map(h, w, object_bboxes)

        return coco_mask
    
    
    @staticmethod
    def _get_segmentations_map(width: int,
                               height: int,
                               object_bboxes: List[List[int]]
                               ) -> np.ndarray:
        """
        Create image from scratch and draw segmentation polygons on it

        Args:
            width (int): image width
            height (int): image height
            object_bboxes (List[List[int]]): list of bbox coordinates

        Returns:
            (np.ndarray) numpy ndarray with binary mask
        """
        coco_masks = []

        for bbox_xywh in object_bboxes:
            bbox = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]

            # Create a new image with a black background
            background = Image.new('L', (height, width), 0)

            # Create a new image with a white ellipse
            ellipse_image = Image.new('L', (bbox[2] - bbox[0], bbox[3] - bbox[1]), 0)
            draw = ImageDraw.Draw(ellipse_image)

            # Calculate the radii of the ellipse
            radius_x = (bbox[2] - bbox[0]) // 2
            radius_y = (bbox[3] - bbox[1]) // 2

            # Draw the ellipse
            draw.ellipse([(0, 0), (radius_x * 2, radius_y * 2)], fill=255)

            # Paste the ellipse onto the black background using the bounding box coordinates
            background.paste(ellipse_image, (bbox[0], bbox[1]))

            coco_mask_array = np.array(background)
            coco_masks.append(coco_mask_array  > 0)

        coco_masks = np.array(coco_masks)

        joint_coco_mask = coco_masks.sum(axis=0) > 0

        #plot_binary_mask(joint_coco_mask)
        #print("coco masks shape", coco_masks.shape)
        #print("Joint-mask:", joint_coco_mask.shape)

        return joint_coco_mask
