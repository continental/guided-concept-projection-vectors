'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from xai_utils.logging import init_logger
init_logger()

from typing import Dict, List, Iterable, Tuple
import os
import random

import numpy as np

from .gcpv_utils import get_projection

from xai_utils.files import write_pickle, read_pickle


class GCPV():
    """
    Guided Concept Projection Vector (GCPV)

    Storage for single optimized instance
    """
    def __init__(self,
                 gcpv: np.ndarray,
                 loss: float,
                 projection: np.ndarray
                 ) -> None:
        """
        Args:
            gcpv (np.ndarray): projection vector with unit importance weights
            loss (float): final loss after GCPV optimization
            projection (np.ndarray): projection of GCPV with original activations used for optimization
        """
        
        self.gcpv = gcpv
        self.loss = loss
        self.projection = projection

    def project(self, activations: np.ndarray) -> np.ndarray:
        """
        Args:
            activations (np.ndarray): activations to project with stored GCPV vector
        """
        return get_projection(self.gcpv, activations)


class GCPVWithMetaInformation(GCPV):
    """
    Standalone instance of GCPV, which contains all meta information (layer, original image, segmentation, etc.)
    """
    def __init__(self, 
                 gcpv: np.ndarray, 
                 loss: float, 
                 projection: np.ndarray,
                 layer: str = None, 
                 image_path: str = None, 
                 image_predictions: np.ndarray = None, 
                 segmentation: np.ndarray = None,
                 segmentation_category_id: int = None
                 ) -> None:
        """
        Args:
            gcpv (np.ndarray): projection vector with unit importance weights
            loss (float): final loss after GCPV optimization
            projection (np.ndarray): projection of GCPV with original activations used for optimization
        
        Kwargs:
            layer (str = None): layer at which GCPV was optimized
            image_path (str = None): path to image used for optimization
            image_predictions (np.ndarray = None): predictions made for image used for optimization
            segmentation (np.ndarray = None): original segmentation used for optimization
            segmentation_category_id (int = None): segmentation class (object class, etc.)
        """
        super().__init__(gcpv, loss, projection)

        self.layer = layer
        self.image_path = image_path
        self.image_predictions = image_predictions
        self.segmentation = segmentation
        self.segmentation_category_id = segmentation_category_id


class GCPVMultilayerStorage():
    """
    Storage for GCPVs obtained in multiple layers.

    Contains GCPV instances and meta information (layer, original image, segmentation, etc.)
    """

    def __init__(self,
                 image_path: str,
                 image_predictions: np.ndarray,
                 segmentation: np.ndarray,
                 segmentation_category_id: int
                 ) -> None:
        """        
        Args:
            image_path (str = None): path to image used for optimization
            image_predictions (np.ndarray = None): predictions made for image used for optimization
            segmentation (np.ndarray = None): original segmentation used for optimization
            segmentation_category_id (int): segmentation class (object class, etc.)
        """
        self.image_path = image_path
        self.image_predictions = image_predictions
        self.segmentation = segmentation
        self.segmentation_category_id = segmentation_category_id

        # init storage, make it instance specific
        self.gcpv_storage: Dict[str, GCPV] = {}  # {layer: gcpv}

    def set_gcpv(self,
                 layer: str,
                 gcpv: GCPV
                 ) -> None:
        """
        Add GCPV instance to storage

        Args:
            layer (str): layer of GCPV
            gcpv (GCPV): instance of GCPV
        """
        self.gcpv_storage[layer] = gcpv

    def get_gcpv(self, layer: str) -> GCPV:
        """
        Return GCPV from given layer

        Args:
            layer (str): retrieval layer

        Returns:
            gcpv (GCPV): retrieved GCPV instance
        """
        return self.gcpv_storage[layer]

    def get_storage_layers(self) -> List[str]:
        """
        Return all layer names contained in storage

        Returns:
            storage_layers (List[str]): list of storage layer names
        """
        return list(self.gcpv_storage.keys())
    
    def get_multilayer_gcpv(self, layers_to_concatenate: List[str]) -> np.ndarray:
        """
        Return multi-layer GCPV (MLGCPV)

        Args:
            layers_to_concatenate (List[str]): list of layers for GCPV

        Returns:
            multilayer_gcpv (np.ndarray): stacked GCPVs for multiple layers
        """
        gcpvs = [self.get_gcpv(l).gcpv for l in layers_to_concatenate]

        multilayer_gcpv = np.concatenate(gcpvs, axis=0)

        return multilayer_gcpv


class GCPVMultilayerClusterInfo:

    def __init__(self,
                 storages: List[GCPVMultilayerStorage],
                 ) -> None:
        """
        Args:
            storages (List[GCPVMultilayerStorage]): storages in cluster
        """
        self.layers = storages[0].get_storage_layers()

        self.accumulated_gcpvs = self._accumulate_gcpvs(storages)

        self.cluster_img_paths = [s.image_path for s in storages]
        
        self.cluster_img_true_segm_categories = [s.segmentation_category_id for s in storages]

        self.cluster_category_counts = self._get_cluster_category_counts(storages)

        self.cluster_category_probabilities = self._get_cluster_category_probs()

    def _accumulate_gcpvs(self,
                          storages: List[GCPVMultilayerStorage]
                          ) -> Dict[str, np.ndarray]:
        """
        Estimate centroids for GCPVs

        Args:
            storages (List[GCPVMultilayerStorage]): GCPVs

        Returns:
            gcpvs (Dict[str, np.ndarray]): dictionary of per-layer GCPVs
        """
        gcpvs = dict()

        for l in self.layers:
            gcpvs[l] = np.array([s.get_gcpv(l).gcpv for s in storages])
        
        return gcpvs
    
    def get_centroid_gcpv(self,
                          layer: str
                          ) -> np.ndarray:
        """
        Retrieve a centroid for single layer

        Args:
            layer (str): layer name

        Returns:
            gcpv_centroid (np.ndarray): centroid for given layer
        """
        return self.accumulated_gcpvs[layer].mean(axis=0)
    
    def get_cumulative_gcpv(self,
                            layer: str
                            ) -> np.ndarray:
        """
        Retrieve an accumulated GCPV for single layer

        Args:
            layer (str): layer name

        Returns:
            gcpv_centroid (np.ndarray): centroid for given layer
        """
        return self.accumulated_gcpvs[layer].sum(axis=0)

    def get_centroid_mlgcpv(self,
                            layers: List[str]
                            ) -> np.ndarray:
        """
        Retrieve a centroids for multiple layers

        Args:
            layers (List[str]): layers name

        Returns:
            mlgcpv_centroid (np.ndarray): concatenated centroids for given layers
        """
        mlgcpv_centroid = np.concatenate([self.get_centroid_gcpv(l) for l in layers])
        return mlgcpv_centroid

    @staticmethod
    def _get_cluster_category_counts(storages: List[GCPVMultilayerStorage]
                                     ) -> Dict[int, int]:
        """
        Get count of true segmentation categories for cluster

        Args:
            storages (List[GCPVMultilayerStorage]): GCPV storages

        Returns:
            categories_dict (Dict[int, int]): {category_id: category_counts}
        """
        categories_dict = dict()

        for storage in storages:
            storage_category = storage.segmentation_category_id

            if storage_category in categories_dict:
                categories_dict[storage_category] += 1
            else:
                categories_dict[storage_category] = 1

        return categories_dict
    
    def _get_cluster_category_probs(self) -> Dict[int, float]:
        """
        Get probabilities (count / n_samples) of true segmentation categories for cluster

        Args:
            storages (List[GCPVMultilayerStorage]): GCPV storages

        Returns:
            probabilities_dict (Dict[int, float]): {category_id: category_probability}
        """
        probabilities_dict = dict()

        categories = sorted(list(self.cluster_category_counts.keys()))
        samples_total = sum([self.cluster_category_counts[c] for c in categories])

        probabilities_dict = {c: self.cluster_category_counts[c]/ samples_total for c in categories}

        return probabilities_dict
    
    def get_cluster_top_category_prob(self) -> float:
        """
        Get top category prob of the cluster

        Returns:
            purity (float): cluster top category prob
        """
        prob = max(self.cluster_category_probabilities.values())

        return prob
    
    def get_cluster_top_category_count(self) -> int:
        """
        Get top category count of the cluster

        Returns:
            purity (float): cluster top category count
        """
        count = max(self.cluster_category_counts.values())

        return count
    
    def __len__(self):
        return len(self.cluster_img_paths)


class GCPVMultilayerStorageSaver:

    def __init__(self, working_directory: str) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of GCPVMultilayerStorages
        """
        self.working_directory = working_directory

    def get_gcpv_storage_path_for_img_name(self,
                                           image_name: str,
                                           category_id: int) -> str:
        """
        Generate image for GCPV storage from image and category_id

        Args:
            image_name (str): image name
            category_id (int): category of object used for optimization
        """
        image_name_no_ext = image_name.split(".")[0]

        image_name_no_ext_with_prefix = '_'.join([str(category_id), image_name_no_ext])

        image_path_no_ext_with_prefix = os.path.join(self.working_directory, image_name_no_ext_with_prefix)

        out_path_pkl = image_path_no_ext_with_prefix + ".pkl"  # correct file name
        out_path_err = image_path_no_ext_with_prefix + ".err"  # error file name

        return out_path_pkl, out_path_err

    def save(self, gcpv_storage: GCPVMultilayerStorage, save_path: str = None) -> None:
        """
        Saves gcpv_storage to {save_path}

        Args:
            gcpv_storage (GCPVMultilayerStorage): storage to save

        Kwargs:
            save_path (str): saving path, if not given - evaluate path with self.get_gcpv_storage_path_for_img_name()
        """
        if save_path is None:
            save_path = self.get_gcpv_storage_path_for_img_name(os.path.basename(gcpv_storage.image_path), gcpv_storage.segmentation_category_id)

        write_pickle(gcpv_storage, save_path)


class GCPVMultilayerStorageDirectoryLoader:

    def __init__(self,
                 working_directory: str,
                 seed: int = None,
                 min_seg_area: float = 0.025,
                 max_seg_area: float = 0.8
                 ) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of GCPVMultilayerStorages

        Kwargs:
            seed (int = None) seed for data sampling
            min_seg_area (float): minimal allowed segmentation area, GCPVs with smaller segmentation will be ignored
            max_seg_area (float): maximal allowed segmentation area, GCPVs with larger segmentation will be ignored
        """
        self.working_directory = working_directory
        self.seed = seed
        random.seed(seed)

        self.min_seg_area = min_seg_area
        self.max_seg_area = max_seg_area

        self.pkl_file_names = self._select_pkl_files()

        self._filter_files_by_segmentation_size()

    def _select_pkl_files(self) -> List[str]:
        """
        Find file names of pickles with GCPVs

        Returns:
            selected_files (List[str]): filtered files by extension (only .pkl)
        """
        all_files_in_dir = sorted(os.listdir(self.working_directory))

        selected_files = []

        for fn in all_files_in_dir:

            if fn.split('.')[-1] == 'pkl':
                selected_files.append(os.path.join(self.working_directory, fn))
            else:
                continue

        return selected_files
    
    def _filter_files_by_segmentation_size(self) -> None:
        """
        Additional filtering of GCPVs by min and max segmentation areas
        """
        filtered_files = []

        for fn in self.pkl_file_names:
            gcpv: GCPVMultilayerStorage = read_pickle(fn)

            seg_mask = gcpv.segmentation
            seg_area = seg_mask.sum() / seg_mask.size

            if (seg_area > self.min_seg_area) and (seg_area < self.max_seg_area):
                filtered_files.append(fn)
            
        self.pkl_file_names = filtered_files

    def _categorize_files(self,
                          ) -> Dict[int, List[str]]:

        """
        Find file names of pickles with GCPVs

        Returns:
            (Dict[int, List[str]]) dictionary with GCPV file names - {tag_id: [file_name]}
        """
        file_names = self.pkl_file_names

        random.shuffle(file_names)

        selected_files = dict()

        for fn in file_names:
            base_fn = os.path.basename(fn)
            category, name = base_fn.split('_')
            
            # init key for current category
            if int(category) not in list(selected_files.keys()):
                selected_files[int(category)] = []
            
            selected_files[int(category)].append(fn)

        return selected_files
    
    def load(self,
             allowed_categories: Iterable[int] = None
             ) -> Dict[int, List[GCPVMultilayerStorage]]:
        """
        Get dictionary of GCPVMultilayerStorages lists per category

        Kwargs:
            allowed_categories (Iterable[int]): load only GCPV storage of allowed categories ids, None to load all

        Returns:
            gcpv_storages_dict (Dict[int, List[GCPVMultilayerStorage]]) GCPV storages per category
        """
        gcpv_file_names = self._categorize_files()

        if allowed_categories is not None:
            gcpv_storages_dict = {t: [read_pickle(f) for f in gcpv_file_names[t]] for t in allowed_categories}
        else:
            gcpv_storages_dict = {t: [read_pickle(f) for f in gcpv_file_names[t]] for t in sorted(gcpv_file_names.keys())}

        return gcpv_storages_dict
    
    def load_train_test_splits(self,
                               allowed_categories: Iterable[int] = None,
                               train_size: float = 0.8
                               )  -> Tuple[Dict[int, List[GCPVMultilayerStorage]], Dict[int, List[GCPVMultilayerStorage]]]:
        """
        Get 2 dictionaries of GCPVMultilayerStorages lists per category, where one dict is 'train' part, second one is 'test' part

        Kwargs:
            allowed_categories (Iterable[int] = None): load only GCPV storage of allowed categories ids, None to load all
            test_size (float = 0.2): size of 'test' split

        Returns:
            gcpv_storages_dict_train (Dict[int, List[GCPVMultilayerStorage]]) 'train' lists of GCPV storages per category
            gcpv_storages_dict_test (Dict[int, List[GCPVMultilayerStorage]]) 'test' lists of GCPV storages per category
        """
        gcpv_storages_dict = self.load(allowed_categories)

        # init output dicts
        gcpv_storages_dict_train = dict()
        gcpv_storages_dict_test = dict()

        # split each category separately
        for k in sorted(gcpv_storages_dict.keys()):
            gcpv_storages_k = gcpv_storages_dict[k]
            split_idx = int(len(gcpv_storages_k) * train_size)
            gcpv_storages_dict_train[k] = gcpv_storages_k[:split_idx]
            gcpv_storages_dict_test[k] = gcpv_storages_k[split_idx:]

        return gcpv_storages_dict_train, gcpv_storages_dict_test


class GCPVMultilayerStorageFilesLoader(GCPVMultilayerStorageDirectoryLoader):

    def __init__(self,
                 working_directory: str,
                 files_to_load: Iterable[str],
                 seed: int = None,
                 min_seg_area: float = 0.025,
                 max_seg_area: float = 0.8
                 ) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of GCPVMultilayerStorages
            files_to_load (Iterable[str]): GCPVMultilayerStorages files to load

        Kwargs:
            seed (int = None) seed for data sampling
            min_seg_area (float): minimal allowed segmentation area, GCPVs with smaller segmentation will be ignored
            max_seg_area (float): maximal allowed segmentation area, GCPVs with larger segmentation will be ignored
        """
        self.working_directory = working_directory
        self.seed = seed
        random.seed(seed)

        self.min_seg_area = min_seg_area
        self.max_seg_area = max_seg_area

        self.pkl_file_names = files_to_load

        self._filter_files_by_segmentation_size()


class GCPVMultilayerStorageMultiDirectoryLoader(GCPVMultilayerStorageDirectoryLoader):

    def __init__(self,
                 working_directories: List[str],
                 seed: int = None,
                 min_seg_area: float = 0.025,
                 max_seg_area: float = 0.8) -> None:
        """
        Args:
            working_directories (List[str]): working directory for input-output of GCPVMultilayerStorages

        Kwargs:
            seed (int = None) seed for data sampling
            min_seg_area (float): minimal allowed segmentation area, GCPVs with smaller segmentation will be ignored
            max_seg_area (float): maximal allowed segmentation area, GCPVs with larger segmentation will be ignored
        """
        self.working_directories = working_directories
        self.seed = seed
        random.seed(seed)

        self.min_seg_area = min_seg_area
        self.max_seg_area = max_seg_area

        self.pkl_file_names = self._select_pkl_files()

        self._filter_files_by_segmentation_size()

    def _select_pkl_files(self) -> List[str]:
        """
        Find file names of pickles with GCPVs

        Returns:
            selected_files (List[str]): filtered files by extension (only .pkl)
        """
        selected_files = []

        for wd in self.working_directories:
            all_files_in_dir = sorted(os.listdir(wd))            

            for fn in all_files_in_dir:

                if fn.split('.')[-1] == 'pkl':
                    selected_files.append(os.path.join(wd, fn))
                else:
                    continue

        return selected_files