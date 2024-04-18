'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from .datasets import AbstractDataset, SuperpixelDataset, LabeledSuperpixelDataset, FolderDataset, LimitedFolderDataset
from .datasets import BaseConceptDataset, NamedConceptDataset, RandomConceptDataset
from .datasets import DatasetFromTensor
from .datasets import SKLearnDataset

from .mscoco import MSCOCOAnnotationsProcessor, MSCOCOImgOnlyDataset, MSCOCOCropDataset, MSCOCOAnnotatedDataSampler, AnnotatedDataSamplerVA
from .mscoco import get_polygons_from_rle
