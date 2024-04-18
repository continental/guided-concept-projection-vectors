'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from typing import Union, Iterable, List, Dict, Any, Tuple
from xai_utils.files import read_json, write_json, mkdir, load_image_to_tensor, img_to_numpy_axes_order, apply_bbox, save_np_uint8_arr_to_img, write_pickle
import os
from .datasets import AbstractDataset
import random
import torchvision.transforms.functional as TF
from torch import Tensor, vstack, stack
from xai_utils.bbox import norm_bbox_coord, xywh2xyxy_bbox_coord, scale_bbox_coord
import numpy as np
from xai_utils.logging import log_error
import torch
import cv2
from pycocotools import mask as cocomask


def get_polygons_from_rle(annotation: dict) -> list[dict]:
    """
    Modified answer:
    https://stackoverflow.com/questions/75326066/coco-annotations-convert-rle-to-polygon-segmentation
    """
    coco_seg = cocomask.frPyObjects(
        annotation["segmentation"],
        annotation["segmentation"]["size"][0],
        annotation["segmentation"]["size"][1],
    )

    maskedArr = cocomask.decode(coco_seg)
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour)

    if len(segmentation) == 0:
        polygons = []

    else:
        polygons = list()
        for i, seg in enumerate(segmentation):
            polygons.append(seg.astype(float).flatten().tolist())

    return polygons


class MSCOCOAnnotationsProcessor:

    def __init__(self,
                 coco_images_folder: str,
                 annotations_json_path: str,
                 output_path: str = "./data/mscoco2017val/processed/"
                 ) -> None:
        """
        Args:
            coco_images_folder: path to MS COCO images
            annotations_json_path: path to JSON file with MS COCO annotations

        Kwargs:
            output_path: output path
        """
        self.coco_images_folder = coco_images_folder
        self.annotations_json_path = annotations_json_path
        self.output_path = output_path

        mkdir(output_path)

    def _load_annotations(self) -> Dict[str, Any]:
        """
        Load MS COCO Annotations

        Returns:
            dictionary with annotations
        """
        return read_json(self.annotations_json_path)

    def _save_annotations(self,
                          obj: Dict[str, Any],
                          out_path: str,
                          ) -> None:
        """
        Save processed MS COCO Annotations

        Args:
            obj: dictionary to write as JSON file
            out_path: path to save annotations
        """
        write_json(obj, out_path, "\t")

    def select_relevant_annotations_by_categtory(self,
                                                 categories: Union[int, Iterable[int]],
                                                 json_save_path: str = None
                                                 ) -> Dict[str, Any]:
        """
        Selects only given categories from JSON

        Args:
            categories: MS COCO categories to select

        Kwargs:
            json_save_path: save path for the resulting file

        Returns:
            dictionary with MS COCO annotations for given categories
        """
        if isinstance(categories, int):
            categories = [categories]

        annots_json = self._load_annotations()

        # relevant_imgs
        relevant_annots = []
        relevant_img_ids = set()
        for annot in annots_json['annotations']:
            if annot['category_id'] in categories:
                relevant_annots.append(annot)
                relevant_img_ids.add(annot['image_id'])
        annots_json['annotations'] = relevant_annots

        relevan_imgs = []
        for image in annots_json['images']:
            if image['id'] in relevant_img_ids:
                relevan_imgs.append(image)
        annots_json['images'] = relevan_imgs

        if json_save_path is not None:
            out_path = os.path.join(self.output_path, json_save_path)
            self._save_annotations(annots_json, out_path)

        return annots_json

    def get_person_annotations(self):
        return self.select_relevant_annotations_by_categtory(1, "person_annotations_val2017.json")
    
    def get_person_annotations_va(self):
        pa = self.select_relevant_annotations_by_categtory(1, "person_annotations_val2017.coco.json")
        return self._convert_coco_to_va(pa, "person_annotations_val2017.va.json")

    def _convert_coco_to_va(self, annot: Dict[str, Any], json_save_path: str = None) -> Dict[str, Any]:
        """
        Converts MS COCO to VA format

        Args:
            annot : MS COCO annotations
            json_save_path: save path for the resulting file

        Returns:
            VA annotations
        """
        va_dict_list = {}

        for img in annot['images']:
            img_annots = [a for a in annot['annotations'] if a['image_id'] == img['id']]

            img_path = os.path.join(self.coco_images_folder, img['file_name'])

            va_dict =  {
                "boxes": [],
                "labels": [],
                "boxesVisRatio": [],
                "boxesHeight": [],
                "original_labels": []
            }

            for ia in img_annots:
                va_dict['boxes'].append([int(ia['bbox'][0]), int(ia['bbox'][1]), int(ia['bbox'][0] + ia['bbox'][2]),int(ia['bbox'][1] + ia['bbox'][3])])
                va_dict['labels'].append(ia['category_id'])
                va_dict['boxesVisRatio'].append(1.0)
                va_dict['boxesHeight'].append(int(ia['bbox'][3]))
                va_dict['original_labels'].append(ia['category_id'])

            va_dict_list[img_path] = va_dict

        if json_save_path is not None:
            out_path = os.path.join(self.output_path, json_save_path)
            self._save_annotations(va_dict_list, out_path)

        return va_dict_list


class MSCOCOImgOnlyDataset(AbstractDataset):

    def __init__(self,
                 data_path: str,
                 annotations_json_path: str,
                 num_samples: int = None,
                 resize_shape: Tuple[int, int] = None,
                 seed: int = None,
                 ) -> None:
        """
        Samples all image files from the folder.
        Returns only images.

        Args:
            data_path: path to the data
            annotations_json_path: path to JSON with annotations

        Kwargs:
            num_samples: number of samples to use
            resize_shape: (W, H)
            seed: seed for random sampling
        """

        random.seed(seed)

        self.data_path = data_path
        self.annotations = read_json(annotations_json_path)

        self.num_samples = num_samples
        self.resize_shape = resize_shape

        random.shuffle(self.annotations['images'])

    def __getitem__(self,
                    idx: int
                    ) -> Tensor:
        img_name = self.annotations['images'][idx]['file_name']
        img = load_image_to_tensor(self.data_path, img_name)

        if self.resize_shape is not None:
            img = TF.resize(img, self.resize_shape)

        return img

    def __len__(self) -> int:
        if self.num_samples:
            return min([self.num_samples, len(self.annotations['images'])])
        else:
            return len(self.annotations['images'])


class MSCOCOCropDataset(AbstractDataset):

    def __init__(self,
                 data_path: str,
                 annotations_json_path: str,
                 num_samples: int = None,
                 min_crop_area: int = 20000,
                 resize_shape: Tuple[int, int] = None,
                 seed: int = None,
                 ignore_crowd: bool = True,
                 ) -> None:
        """
        Samples all objects, which have an area larger than min_crop_area
        Returns only images (cropped objects).

        Args:
            data_path: path to the data
            annotations_json_path: path to JSON with annotations

        Kwargs:
            num_samples: number of samples to use
            min_crop_area: minimal area size (in pixels) to crop
            seed: seed for random sampling
            ignore_crowd: ignore 'iscrowd' objects
        """

        random.seed(seed)

        self.data_path = data_path
        self.annotations = read_json(annotations_json_path)
        self.min_crop_area = min_crop_area

        self.num_samples = num_samples
        self.resize_shape = resize_shape

        self.ignore_crowd = ignore_crowd

        self.items = self._get_cropable_objects()
        random.shuffle(self.items)

    def __getitem__(self,
                    idx: int
                    ) -> Tensor:
        img = self._get_item(idx)

        if self.resize_shape is not None:
            img = TF.resize(img, self.resize_shape)

        return img

    def get_item_no_reshape(self,
                            idx: int
                            ) -> Tensor:        
        return self._get_item(idx)

    def _get_item(self,
                  idx: int
                  ) -> Tensor:
        annot_i, img_i = self.items[idx]

        img_name = img_i['file_name']            
        img_tensor = load_image_to_tensor(self.data_path, img_name)

        x, y, w, h = annot_i['bbox']
        img = TF.crop(img_tensor, int(y), int(x), int(h), int(w))
        return img

    def __len__(self) -> int:
        if self.num_samples:
            return min([self.num_samples, len(self.items)])
        else:
            return len(self.items)            

    def _get_cropable_objects(self) -> List[Dict[str, Any]]:
        """
        Get list of cropable objects (where object area is larger than self.min_crop_area)

        Returns:
            List of tuples (obj_id, img_annot_dict)
        """
        cropable_list = []

        for i in range(len(self.annotations['annotations'])):
            # annotation
            annot_i = self.annotations['annotations'][i]

            # iscrowd check
            if self.ignore_crowd and (annot_i['iscrowd'] == 1):
                continue

            # area check
            _, _, w, h = annot_i['bbox']            
            if (w * h) < self.min_crop_area: 
                continue
            
            # corresponding image
            img_i = [img for img in self.annotations['images'] if annot_i['image_id'] == img['id']][0]

            cropable_list.append((annot_i, img_i))

        return cropable_list


class MSCOCOAnnotatedDataSampler(AbstractDataset):

    def __init__(self,
                 data_path: str,
                 annotations_json_path: str,
                 num_samples: int = None,
                 resize_shape: Tuple[int, int] = None,
                 seed: int = None,
                 items_categories: Iterable[int] = [1],  # person
                 item_bbox_min_area: int = 20000,
                 ignore_crowd: bool = True
                 ) -> None:
        """
        Samples MSCOCO images with annotations.

        Args:
            data_path: path to the data
            annotations_json_path: path to JSON with annotations

        Kwargs:
            num_samples: number of samples to use
            resize_shape: size to reshape the input images (W, H)
            seed: seed for random sampling
            items_categories: sample only items of given categories, by default - only 'person'
            item_min_area: min area of bbox to sample, by default - min area is 20000 pixels
            ignore_crowd: ignore crowds of objects
        """

        random.seed(seed)
        self.seed = seed

        self.data_path = data_path
        self.annotations = read_json(annotations_json_path)

        self.num_samples = num_samples
        self.resize_shape = resize_shape[::-1]

        self.items_categories = items_categories
        self.item_bbox_min_area = item_bbox_min_area
        self.ignore_crowd = ignore_crowd

        self.items = self._get_items()
        random.shuffle(self.items)

    def __getitem__(self,
                    idx: int
                    ) -> Tensor:

        img_tensor, norm_annot_list = self._item_to_tensor_and_norm_annot(self.items[idx])

        if self.resize_shape is not None:
            img_tensor = TF.resize(img_tensor, self.resize_shape)

        return img_tensor.squeeze()

    def __len__(self) -> int:
        if self.num_samples:
            return min([self.num_samples, len(self.items)])
        else:
            return len(self.items)            

    def _get_items(self) -> List[Tuple[Dict, List[Dict]]]:
        """
        Get list of items: tuple of image and objects list (bbox + class)

        Returns:
            List of tuples (img_annot_dict, list(obj_annot_dict))
        """
        items_list = []

        for i in range(len(self.annotations['images'])):
            # image
            img_i = self.annotations['images'][i]
            
            # corresponding image
            annots_i = [annot for annot in self.annotations['annotations'] if annot['image_id'] == img_i['id']]

            # filter by size and category
            annots_i_filtered = []
            for a in annots_i:
                # class check
                if a['category_id'] not in self.items_categories:
                    continue
                
                # area check
                if a['area'] < self.item_bbox_min_area: 
                    continue

                # iscrowd check
                if self.ignore_crowd and (a['iscrowd'] == 1):
                    continue

                annots_i_filtered.append(a)
            
            # continue if annots were filtered out
            if len(annots_i_filtered) == 0:
                continue

            items_list.append((img_i, annots_i_filtered))

        return items_list

    def _item_to_tensor_and_norm_annot(self, 
                                       item: Tuple[Dict, List[Dict]]
                                       ) -> Tuple[Tensor, Tensor]:
        """
        Convert single item to image tensor and list of object normalized tuples (obj_bbox, obj_class)
        Object bounding boxes (x, y, w, h) obj_bbox are normalized

        Args:
            item: item tuple (img_annot_dict, list(obj_annot_dict))

        Returns:
            tuple(images_tensor, annot_tensor) - (Tensor[1, C, H, W], Tensor[n_obj, 5]), where annot_tensor_list 0th dim is n_obj long and 1st dim is [x1, y1, x2, y2, class]
        """
        img_i, annots_i = item

        img_name = img_i['file_name']
        img_w = img_i['width']
        img_h = img_i['height']

        img_tensor = load_image_to_tensor(self.data_path, img_name, add_batch_dim=True)

        if self.resize_shape is not None:
            img_tensor = TF.resize(img_tensor, self.resize_shape)

        tensor_w = img_tensor.shape[-1]
        tensor_h = img_tensor.shape[-2]

        annot_list = []

        for a in annots_i:
            xywh_bbox_coord = a['bbox']
            xywh_bbox_coord = norm_bbox_coord(xywh_bbox_coord, img_w, img_h)
            xyxy_norm_bbox_coord = xywh2xyxy_bbox_coord(xywh_bbox_coord)
            xyxy_bbox_coord = scale_bbox_coord(xyxy_norm_bbox_coord, tensor_w, tensor_h)
            category_id = a['category_id']  # id
            category_class = category_id - 1  # class
            annot_list.append(Tensor(xyxy_bbox_coord + (category_class,)))

        return img_tensor, stack(annot_list)

    def sample(self, 
               n_samples: int
               ) -> Tuple[Tensor, List[Tensor]]:
        """
        Sample data items

        Args:
           n_samples: number of samples to sample

        Returns:
            tuple(images_tensor, annot_tensor_list) - (Tensor[B, C, H, W], list(Tensor[n_obj, 5])), where annot_tensor_list consists of n_obj tensors, where [x1, y1, x2, y2, class]
        """
        tensors = []
        annots = []

        for i in range(n_samples):

            img_tensor, norm_annot_list = self._item_to_tensor_and_norm_annot(self.items[i])
            tensors.append(img_tensor)
            annots.append(norm_annot_list)

        return vstack(tensors), annots

    def save_items(self, 
                   n_samples: int,
                   save_folder: str,
                   concept_labels: Iterable[str] = None
                   ) -> None:
        """
        Save n_samples items to given folder
        Files are saved as: './save_folder/{i}_{img_original_filename}.jpg'
        Items JSON is saved to './save_folder/_{n_samples}_mscoco_items_seed_{self.seed}.jpg'

        Args:
            n_samples: number of samples
            save_folder: saving folder

        Kwargs:
            concept_labels: insert dummies for concept labels in annotations if Not none
        """
        mkdir(save_folder)
        imgs, targets = self.sample(n_samples)
        items = [self.items[i] for i in range(n_samples)]

        for img, annots in items:
            for a in annots:
                if "segmentation" in a:
                    a.pop("segmentation")
                if concept_labels is not None:
                    if "concepts" in a:
                        for cl in concept_labels:
                            if cl not in a["concepts"]:
                                a["concepts"][cl] = None
                    else:
                        a["concepts"] = {cl: None for cl in concept_labels}

        json_path = os.path.join(save_folder, f'_{n_samples}_mscoco_items_seed_{self.seed}.json')
        write_json(items, json_path, "\t")

        for i, (img, tgts, (img_item, bbox_items)) in enumerate(zip(imgs, targets, items)):

            img_fn = img_item['file_name'].split(".")[0]

            img = np.uint8(img_to_numpy_axes_order(img.numpy()) * 255)
            
            tgts = tgts[:,:4].int().numpy()

            for j, tgt in enumerate(tgts):
                img = apply_bbox(img, tgt, str(j))
            
            save_np_uint8_arr_to_img(img, os.path.join(save_folder, f'{i}_{img_fn}.jpg'))

    def load_items(self,
                   path: str
                   ) -> None:
        """
        Load JSON with items

        Args:
            path: path to items JSON
        """
        self.items = read_json(path)

    def save_baseline_score_dict_from_items(self,
                                            path: str,
                                            concept_names: List[str]
                                            ) -> None:
        """
        Save baseline score dict in the format similar to TCAVDetectionResultDict (use only 'concept_names' and 'scores' fields)

        Args:
            path: path to save pickle
            concept_names: concepts (concept scores) to save (order is kept)
        """
        score_dict = {'concept_names': concept_names, 'detector_type': 'manual'}

        concept_scores = []

        for img, bboxs in self.items:

            img_concept_scores = []

            for bbox in bboxs:

                bbox_concept_scores = []

                if "concepts" not in bbox:
                    log_error(Exception, "Concepts not labeled")

                concepts = bbox["concepts"]

                for cn in concept_names:
                    if cn not in concepts:
                        log_error(Exception, f"Concept '{cn}' not labeled")

                    bbox_concept_scores.append(concepts[cn])

                img_concept_scores.append(bbox_concept_scores)

            scores_tensor = Tensor(img_concept_scores).T
            scores_tensor = torch.where(scores_tensor > 0, 1., -1.)
            
            concept_scores.append({"proxy_layer": scores_tensor})

        score_dict['scores'] = concept_scores

        write_pickle(score_dict, path)


class AnnotatedDataSamplerVA(AbstractDataset):

    def __init__(self,
                 annotations_json_path: str,
                 resize_shape: Tuple[int, int] = None,
                 #seed: int = None,
                 ) -> None:
        """
        Samples MSCOCO images with annotations.

        Args:
            annotations_json_path: path to JSON with annotations

        Kwargs:
            num_samples: number of samples to use
            resize_shape: size to reshape the input images (W, H)
            items_categories: sample only items of given categories, by default - only 'person'
            item_min_area: min area of bbox to sample, by default - min area is 20000 pixels
            ignore_crowd: ignore crowds of objects
        """

        #random.seed(seed)
        #self.seed = seed

        self.annotations = read_json(annotations_json_path)

        self.resize_shape = resize_shape[::-1]

        self.items = list(self.annotations.keys())
        #random.shuffle(self.items)

    def _item_to_tensor_and_norm_annot(self, 
                                       item: Tuple[Dict, List[Dict]]
                                       ) -> Tuple[Tensor, Tensor]:
        """
        Convert single item to image tensor and list of object normalized tuples (obj_bbox, obj_class)
        Object bounding boxes (x, y, w, h) obj_bbox are normalized

        Args:
            item: item tuple (img_annot_dict, list(obj_annot_dict))

        Returns:
            tuple(images_tensor, annot_tensor) - (Tensor[1, C, H, W], Tensor[n_obj, 5]), where annot_tensor_list 0th dim is n_obj long and 1st dim is [x1, y1, x2, y2, class]
        """
        img_tensor = load_image_to_tensor(*os.path.split(item), add_batch_dim=True)

        img_w = img_tensor.shape[-1]
        img_h = img_tensor.shape[-2]

        if self.resize_shape is not None:
            img_tensor = TF.resize(img_tensor, self.resize_shape)

        tensor_w = img_tensor.shape[-1]
        tensor_h = img_tensor.shape[-2]

        annot_list = []

        annots_i = self.annotations[item]

        for b, l in zip(annots_i['boxes'], annots_i['labels']):
            xyxy_bbox_coord = b
            xyxy_norm_bbox_coord = norm_bbox_coord(xyxy_bbox_coord, img_w, img_h)
            xyxy_bbox_coord_scaled = scale_bbox_coord(xyxy_norm_bbox_coord, tensor_w, tensor_h)
            category_class = l - 1  # class
            annot_list.append(Tensor(xyxy_bbox_coord_scaled + (category_class,)))

        return img_tensor, stack(annot_list)

    def sample(self, 
               n_samples: int
               ) -> Tuple[Tensor, List[Tensor]]:
        """
        Sample data items

        Args:
           n_samples: number of samples to sample

        Returns:
            tuple(images_tensor, annot_tensor_list) - (Tensor[B, C, H, W], list(Tensor[n_obj, 5])), where annot_tensor_list consists of n_obj tensors, where [x1, y1, x2, y2, class]
        """
        tensors = []
        annots = []
        files = []

        for i in range(n_samples):

            img_tensor, norm_annot_list = self._item_to_tensor_and_norm_annot(self.items[i])
            tensors.append(img_tensor)
            annots.append(norm_annot_list)
            files.append(self.items[i])

        return vstack(tensors), annots, files
    
    def __len__(self) -> int:
        if self.num_samples:
            return min([self.num_samples, len(self.items)])
        else:
            return len(self.items)
