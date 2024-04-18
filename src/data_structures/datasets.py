'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

import os
from typing import Iterable, List, Tuple, Union
import random
from abc import abstractmethod

from torch import Tensor
import torchvision.transforms.functional as TF
from PIL import Image

import numpy as np
from torch.utils.data import Dataset

from xai_utils.files import load_image_to_tensor, filter_strings_by_head, filter_strings_by_tail, set_random_seeds
from xai_utils.logging import log_assert


class AbstractDataset(Dataset):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self,
                    index: int
                    ) -> Tensor:
        pass

    @abstractmethod
    def __len__(self
                ) -> int:
        pass

    def get_item_no_reshape(self,
                            idx: int
                            ) -> Tensor:
        """
        Implement only in case if returning of single non-reshaped image is required.

        For instance: 
            ./ice/ice_detection.py -> _get_prots()
            ./concept_analysis/ice_concept_similarity.py -> _plot_sample_concepts()
        
        Otherwise use self[i]
        """
        raise NotImplementedError("Not implemented")
    
    def get_item_as_pil_image(self,
                              idx: int
                              ) -> Image.Image:
        raise NotImplementedError("Not implemented")


class FolderDataset(AbstractDataset):

    def __init__(self,
                 data_path: str,
                 img_shape: Tuple[int, int] = None,
                 seed: int = None,
                 num_samples: int = None
                 ) -> None:
        """
        Samples all files from the folder

        Args:
            data_path: path to the data

        Kwargs:
            img_shape: image shape (w, h)
            seed: shuffling seed
            num_samples: number of items to sample (samples all, if None)
        """
        super(FolderDataset, self).__init__()        

        self.data_path = data_path
        self.items = os.listdir(data_path)
        self.img_shape = img_shape
        self.num_samples = num_samples

        if seed is not None:
            set_random_seeds(seed)
            random.shuffle(self.items)

        if num_samples is not None:
            self.items = self.items[:num_samples]

    def __getitem__(self,
                    idx: int
                    ) -> Tensor:
        img = load_image_to_tensor(self.data_path, self.items[idx], shape=self.img_shape)
        return img
    
    def get_item_no_reshape(self,
                            idx: int
                            ) -> Tensor:        
        return load_image_to_tensor(self.data_path, self.items[idx])

    def __len__(self) -> int:
        return len(self.items)
    
    def get_item_as_pil_image(self,
                              idx: int
                              ) -> Image.Image:
        tensor = self.__getitem__(idx)
        image = TF.to_pil_image(tensor, "RGB")
        return image


class SuperpixelDataset(FolderDataset):

    def __init__(self,
                 data_path: str
                 ) -> None:
        """
        Samples all superpixels from the folder, extracts mettadata from file names

        Args:
            data_path: path to the data
        """
        super(SuperpixelDataset, self).__init__(data_path)

        # metadata: (img_id, segmentation_setting_id, segment_id)
        self.metadata = self._split_superpixel_name_string()

    def _split_superpixel_name_string(self) -> List[Tuple[int, int, int]]:
        metadata = []
        for item in self.items:
            name, ext = item.split(".")
            data = name.split("_")
            metadata_strs = data[1:]
            metadata_tuple = tuple(int(m) for m in metadata_strs)
            metadata.append(metadata_tuple)
        return metadata


class LabeledSuperpixelDataset(FolderDataset):

    def __init__(self,
                 data_path: str,
                 items: List[str],
                 metadata: np.ndarray,
                 label: int,
                 distances: np.ndarray = None
                 ) -> None:
        """
        Dataset for the superpixels of certain label 

        Args:
            data_path: path to the data
            items: data items from the path to be included into dataset
            metadata: superpixels metadata (image id, segmentation setting id, segment id)
            label: superpixels label

        Kwargs:
            distances: distance of each superpixel to cluster centroid
        """
        self.data_path = data_path
        self.items = items
        self.metadata = metadata
        self.label = label
        self.distances = distances


class BaseConceptDataset(AbstractDataset):

    def __init__(self,
                 data_path: str,
                 num_samples: int = None,
                 file_ext: Tuple[str] = None,
                 seed: int = None,
                 shape: Tuple[int, int] = None
                 ) -> None:
        """
        Base Dataset for the concept. Samples all files (or all files of given extensions if provided) from the folder

        Args:
            data_path: path to the dataset

        Kwargs:
            num_samples: number of items to sample (samples all, if None)
            file_ext: permitted concept file types, None to ignore
            seed: seed for random sampling
            shape: image shape (w, h) - Tuple[int, int]
        """
        super(BaseConceptDataset, self).__init__()

        set_random_seeds(seed)

        self.data_path = data_path
        self.file_ext = file_ext

        self.shape = shape

        self.concept_files = os.listdir(data_path)

        random.shuffle(self.concept_files)

        # filter files by the extension
        if file_ext:
            self.concept_files = filter_strings_by_tail(
                self.concept_files, file_ext)

        self.num_samples = num_samples

    def __getitem__(self,
                    idx: int
                    ) -> Tensor:
        img = load_image_to_tensor(self.data_path, self.concept_files[idx], shape=self.shape)
        # return idx, img
        return img
    
    def get_item_no_reshape(self,
                            idx: int
                            ) -> Tensor:        
        return load_image_to_tensor(self.data_path, self.concept_files[idx])

    def __len__(self
                ) -> int:
        if self.num_samples:
            return min([self.num_samples, len(self.concept_files)])
        else:
            return len(self.concept_files)
    
    def get_item_as_pil_image(self,
                              idx: int
                              ) -> Image.Image:
        tensor = self.__getitem__(idx)
        image = TF.to_pil_image(tensor, "RGB")
        return image


class NamedConceptDataset(BaseConceptDataset):

    def __init__(self,
                 data_path: str,
                 concept_name: str,
                 num_samples: int = None,
                 file_ext: Tuple[str] = None,
                 seed: int = None,
                 shape: Tuple[int, int] = None
                 ) -> None:
        """
        Dataset for the named concept. Samples 'num_samples' files, name of which starts with 'concept_name', from the folder

        Args:
            data_path: path to the dataset
            concept_name: unique name of concept, the start of file name

        Kwargs:
            num_samples: number of items to sample (samples all, if None)
            file_ext: permitted concept images file types, None to ignore
            seed: seed for random sampling
            shape: image shape (w, h) - Tuple[int, int]
        """
        super(NamedConceptDataset, self).__init__(
            data_path, num_samples, file_ext, seed, shape)

        self.concept = concept_name

        # filter concept files by the filename beginning
        self.concept_files = filter_strings_by_head(
            self.concept_files, concept_name)


class RandomConceptDataset(BaseConceptDataset):

    def __init__(self,
                 data_path: str,
                 num_samples: int = None,
                 excluded_concepts: Tuple[str] = None,
                 file_ext: Tuple[str] = None,
                 seed: int = None,
                 shape: Tuple[int, int] = None
                 ) -> None:
        """
        Dataset for the concept.

        Args:
            data_path: path to the dataset

        Kwargs:
            num_samples: number of items to sample (samples all, if None)
            excluded_concepts: avoids files of listed concepts
            file_ext: permitted concept images file types, None to ignore
            seed: seed for random sampling
            shape: image shape (w, h) - Tuple[int, int]
        """
        super(RandomConceptDataset, self).__init__(
            data_path, num_samples, file_ext, seed, shape)

        self.excluded_concepts = excluded_concepts

        # remove files with filenames starting from 'excluded_concepts'
        if excluded_concepts:
            self.concept_files = filter_strings_by_head(
                self.concept_files, excluded_concepts, invert=True)


class DatasetFromTensor(AbstractDataset):

    def __init__(self,
                 tensor: Tensor
                 ) -> None:
        """
        Dataset from n-dim tensor. Tensor must have at least 2 dimensions.
        0th (batch) dimension is a sampling dimension

        Args:
            tensor: tensor to query samples from
        """
        super(DatasetFromTensor, self).__init__()
        log_assert(len(tensor.shape) > 1, "Tensor must have at least 2 dimensions")
        self.tensor = tensor

    def __getitem__(self,
                    idx: int
                    ) -> Tensor:
        sample = self.tensor[idx]
        return sample

    def __len__(self) -> int:
        return self.tensor.shape[0]


class SKLearnDataset(Dataset):

    def __init__(self,
                 data: Tensor,
                 labels: Tensor = None
                 ) -> None:
        """
        Args:
            x: samples - Tensor[N_SAMPLES, N_DIMS]
            y: (optional) labels - Tensor[N_SAMPLES] or None
        """
        self.data = data

        if labels is not None:
            log_assert(len(data) == len(labels), "0th dimensions of data and labels shall be of equal sizes")

        self.labels = labels

    def __getitem__(self,
                    idx: int
                    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.labels is None:
            return self.data[idx]
        else:
            return self.data[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.data)


class LimitedFolderDataset(FolderDataset):

    def __init__(self,
                 data_path: str,
                 file_names_list: Iterable[str],
                 img_shape: Tuple[int, int] = None
                 ) -> None:
        """
        Samples listed files from the folder

        Args:
            data_path: path to the data
            file_names_list: iterable of file names to sample

        Kwargs:
            img_shape: image shape (w, h)
        """
        self.data_path = data_path
        self.items = file_names_list
        self.img_shape = img_shape

