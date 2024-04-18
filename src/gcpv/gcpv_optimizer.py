'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from xai_utils.logging import init_logger, log_info
init_logger()

import torch
from torch import Tensor
from torch.optim import AdamW
import numpy as np
from skimage.transform import resize

from data_structures import MSCOCOAnnotationsProcessor
from .gcpv import GCPV, GCPVMultilayerStorage, GCPVMultilayerStorageSaver
from .gcpv_utils import get_projection, get_acts_preds_dict
from .gcpv_semantic_segmenter import AbstractSemanticSegmenter, MSCOCOSemanticSegmentationLoader, MSCOCORectangleSegmenter, MSCOCOEllipseSegmenter
from xai_utils.files import read_json, mkdir
from hooks import Propagator

from abc import ABC, abstractmethod
import os
from typing import Dict, Any, Iterable


EPSILON = 0.0001


class AbstractSampleGCPVOptimizer(ABC):

    @abstractmethod
    def optimize_gcpv_for_sample(self) -> GCPVMultilayerStorage:
        pass


class TorchCustomSampleGCPVOptimizer(AbstractSampleGCPVOptimizer):

    def __init__(self,
                 propagator: Propagator,
                 semantic_segmenter: AbstractSemanticSegmenter,
                 mse_coeff: float = 1.0,
                 dice_coeff: float = 1.0
                 ) -> None:
        """
        propagator (Propagator): wrapped model
        semantic_segmenter (AbstractSemanticSegmenter): segmenter
        """
        self.propagator = propagator
        self.semantic_segmenter = semantic_segmenter

        self.mse_coeff = mse_coeff
        self.dice_coeff = dice_coeff

    def _get_gcpv_prototypes(self,
                             baseline_mask: np.ndarray,
                             acts_0to1: np.ndarray,
                             lr: float = 0.01,
                             epochs: int = 200
                             ) -> Dict[str, Any]:
        """
        Optimize GCPVs

        Args:
            baseline_mask (np.ndarray): baseline segmentation
            acts_0to1 (np.ndarray): activations of sample
        
        Kwargs:
            lr (float): learning rate of optimizer
            epochs (int): optimization epochs
        
        Returns:
            (Dict[str, Any]) of optimization results
        """

        def objective(projection_vector: Tensor, baseline_mask_tensor: Tensor):
            projection_vector_expanded = projection_vector.view(-1, 1, 1)
            
            projected_mask = (acts_0to1_tensor * projection_vector_expanded).sum(dim=0)

            projected_mask = torch.sigmoid(projected_mask)

            mse = ((baseline_mask_tensor - projected_mask) ** 2).mean()
            #msa = (torch.abs(baseline_mask_tensor - projected_mask)).mean()

            numerator = 2.0 * torch.sum(baseline_mask_tensor * projected_mask) + EPSILON
            denominator = torch.sum(baseline_mask_tensor) + torch.sum(projected_mask) + EPSILON
            dice_loss = 1 - (numerator / denominator)

            l1 = torch.norm(projection_vector, 1) / len(projection_vector)
            l2 = torch.norm(projection_vector, 2) / len(projection_vector)
            regularization = (l1 + l2)

            loss = self.mse_coeff * mse + self.dice_coeff * dice_loss + regularization
            
            return loss
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n = acts_0to1.shape[0]

        #acts_0to1 = np.clip(acts_0to1, 0, None)

        acts_0to1_tensor = torch.from_numpy(acts_0to1).to(device)
        baseline_mask_binary = torch.from_numpy(baseline_mask).to(device)
        baseline_mask_tensor = torch.from_numpy(baseline_mask).float().to(device)

        gcpv = torch.ones(n, device=device) #* 0.0001
        gcpv.requires_grad_()

        opt = AdamW([gcpv], lr=0.1)
        losses = []

        for i in range(epochs):
            loss = objective(gcpv, baseline_mask_tensor)
            losses.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            #with torch.no_grad():
                    #gcpv.clamp_(None, None)

        #print(f"\tLoss ({init_fn}):", result['fun'])
        return {'x': gcpv.detach(), 'fun': np.array(losses)}

    def optimize_gcpv_for_sample(self,
                                 img_name: str,
                                 img_folder: str,
                                 lr: float = 0.01,
                                 epochs: int = 200,
                                 min_seg_area: float = 0.025,
                                 max_seg_area: float = 0.8
                                 ) -> GCPVMultilayerStorage:
        """
        Get prototypes of GCPVs for a single sample in all given layers

        Args:
            img_name (str): name of the image
            img_folder (str): folder of the image

        Kwargs:
            lr (float): learning rate of optimizer
            epochs (int): optimization epochs
            min_seg_area (float): minimal allowed segmentation area to process
            max_seg_area (float): maximal allowed segmentation area to process

        Returns:
            (GCPVMultilayerStorage) GCPV Storage with results
        """
        #print(f"\n{img_name}")

        seg_mask = self.semantic_segmenter.segment_sample(img_name)

        seg_area = seg_mask.sum() / seg_mask.size
        if seg_area < min_seg_area:
            return None
        
        if seg_area > max_seg_area:
            return None
        
        acts_dict, preds = get_acts_preds_dict(self.propagator, img_folder, img_name)

        # init GCPV storage
        img_path = os.path.join(img_folder, img_name)
        img_preds = preds[0].numpy().astype(np.float16)
        segmentation_category_id = self.semantic_segmenter.obj_category_id

        gcpv_storage = GCPVMultilayerStorage(img_path, img_preds, seg_mask, segmentation_category_id)

        for layer, acts in acts_dict.items():

            acts_current = acts[0].numpy()  # acts of layer 10
            #print(f"{layer} activations:", acts_current.shape)

            acts_current_0_to_1 = acts_current # normalize_0_to_1(acts_current)
            seg_mask_resized = resize(seg_mask, acts_current_0_to_1.shape[1:])          

            result_ones = self._get_gcpv_prototypes(seg_mask_resized, acts_current, lr, epochs)

            # JUST DONT REPEAT THAT FOR THE SAKE OF OUR LORD JESUS CHRIST
            gcpv_np = result_ones['x'].cpu().numpy()# (result_ones['x'] / result_ones['x'].mean()).cpu().numpy()

            gcpv_loss = result_ones['fun'].tolist()[-1]
            gcpv_proj = get_projection(gcpv_np, acts_current_0_to_1)
            gcpv_storage.set_gcpv(layer, GCPV(gcpv_np, gcpv_loss, gcpv_proj))

        return gcpv_storage


class SampleSetGCPVOptimizer:

    def __init__(self,
                 optimizer: AbstractSampleGCPVOptimizer,
                 gcpv_io: GCPVMultilayerStorageSaver,
                 imgs_dir: str
                 #mae_coeff: float = 0.0,
                 #mse_coeff: float = 0.0,
                 #dice_coeff: float = 1.0,
                 #reg_coeff: float = 1.0
                 ) -> None:
        """
        Args:
            optimizer (SampleGCPVOptimizerTorch): optimizer
            gcpv_io (GCPVMultilayerStorageSaverLoader): GCPV saver
            imgs_dir (str): directory with images for optimization

        Kwargs:
            mae_coeff (float = 0.0): MAE loss term weight
            mse_coeff (float = 0.0): MSE loss term weight
            dice_coeff (float = 1.0): Dice loss term weight
            reg_coeff (float = 1.0): regularization coefficient weight
        """
        self.optimizer = optimizer
        self.imgs_dir = imgs_dir
        self.gcpv_io = gcpv_io
        
        #self.mae_coeff = mae_coeff
        #self.mse_coeff = mse_coeff
        #self.dice_coeff = dice_coeff
        #self.reg_coeff = reg_coeff

    def optimize_images(self,
                        image_names: Iterable[str],
                        category_id: int = None) -> None:
        """
        Optimize for list of images

        Args:
            image_names (Iterable[str]): iterable of image names

        Kwargs:
            category_id (int): prefix of save files with optimization results - class category id
        """
        log_info(f"Optimizing {len(image_names)} images")

        for image_name in image_names:

            out_path_pkl, out_path_err = self.gcpv_io.get_gcpv_storage_path_for_img_name(image_name, category_id)

            try:
                if not os.path.exists(out_path_pkl):
                    gcpv = self.optimizer.optimize_gcpv_for_sample(image_name, self.imgs_dir) #, self.mae_coeff, self.mse_coeff, self.dice_coeff, self.reg_coeff)
                    if gcpv is None:
                        raise ValueError
                    self.gcpv_io.save(gcpv, out_path_pkl)
            except:
                open(out_path_err, 'a').close()  # write empty file


class AllGuidesGCPVOptimizerMSCOCO:

    coco_segmenters = {
        'original': MSCOCOSemanticSegmentationLoader,
        'rectangle': MSCOCORectangleSegmenter,
        'ellipse': MSCOCOEllipseSegmenter
        }

    mscoco_tags = {
        # 1: 'person',
        # vehicles
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        # animals
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe'
        }

    mscoco_imgs_path = "../data/mscoco2017val/val2017/"
    mscoco_default_annots = "../data/mscoco2017val/annotations/instances_val2017.json"
    mscoco_processed_annots = "../data/mscoco2017val/processed/"

    def __init__(self,
                 propagator: Propagator,
                 propagator_tag: str,
                 out_base_dir: str = "../data/mscoco2017val/processed/gcpvs"
                 ) -> None:
        """
        Args:
            propagator (Propagator): instance of model wrapper
            propagator_tag: unique strin tag of model wrapper, e.g. 'yolo', 'ssd', etc.

        Kwargs:
            out_base_dir (str = "../data/mscoco2017val/processed/gcpvs"): base directory for outputs, subdirs will be created
        """
        self.propagator = propagator
        self.propagator_tag = propagator_tag
        self.out_base_dir = out_base_dir
        
        self.annots = self._load_coco_annots()

        mkdir(out_base_dir)

    def run_optimization(self):
        """
        Perform optimization. Results are saved to: self.out_base_dir/gcpv_{segmenter_tag}_{self.propagator_tag}
        """
        for segmenter_tag in self.coco_segmenters.keys():

            self.run_optimization_one_segmenter(segmenter_tag)

    def run_optimization_one_segmenter(self, segmenter_tag):
        """
        Perform optimization. Results are saved to: self.out_base_dir/gcpv_{segmenter_tag}_{self.propagator_tag}
        """
        segmenter = self.coco_segmenters[segmenter_tag]

        out_dir = os.path.join(self.out_base_dir, f'gcpv_{segmenter_tag}_{self.propagator_tag}')
        mkdir(out_dir)

        # analyze each category
        for category_id, category in self.mscoco_tags.items():
            log_info(f"Optimizing '{category}' category")
            coco_annot = self.annots[category]
            coco_imgs = sorted([a['file_name'] for a in coco_annot['images']])

            segmenter_instance = segmenter(coco_annot, category_id)
            so = TorchCustomSampleGCPVOptimizer(self.propagator, segmenter_instance)

            sso = SampleSetGCPVOptimizer(so, GCPVMultilayerStorageSaver(out_dir), self.mscoco_imgs_path)

            sso.optimize_images(coco_imgs, category_id)

    def _load_coco_annots(self) -> Dict[str, Any]:
        """
        Load and process annotations of MSCOCO with MSCOCOAnnotationsProcessor

        Returns:
            annots (Dict[str, Any]): dictionary with MSCOCO annotations
        """
        annots = {}

        for v_id, v_name in self.mscoco_tags.items():

            annot_path = f"{self.mscoco_processed_annots}{v_name}_annotations_val2017.json"

            try:            
                coco_annot = read_json(annot_path)
            except:
                mcp = MSCOCOAnnotationsProcessor(self.mscoco_imgs_path, self.mscoco_default_annots, self.mscoco_processed_annots)
                mcp.select_relevant_annotations_by_categtory(v_id, f"{v_name}_annotations_val2017.json")

                coco_annot = read_json(annot_path)

            annots[v_name] = coco_annot

        return annots
    