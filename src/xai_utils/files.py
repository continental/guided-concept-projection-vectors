'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

import urllib
import os
import pickle
from typing import Iterable, Tuple, List, Dict, Union, Any
from PIL import Image
import torch
import numpy as np
import pathlib
from skimage.transform import rescale, resize
import cv2
import random
import json
from .logging import log_assert, log_info, log_debug


EPSILON = 1e-10


def mkdir(path: str) -> None:
    """
    Creates directory if not exists

    Args:
        path: directory path
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path: str) -> None:
    """
    Recursively delete a directory with all its contents
    
    Args:
        path: directory path
    """
    directory = pathlib.Path(path)
    if directory.is_dir():
        for item in directory.iterdir():
            if item.is_dir():
                rmdir(item)
            else:
                item.unlink()
        directory.rmdir()


def resetdir(path: str) -> None:
    """
    reset directory (clean contents)

    Args:
        path: directory path
    """
    rmdir(path)
    mkdir(path)


def load_imagenet_classes() -> Dict[int, str]:
    # download pickle from gist if not yet
    path = 'data/imagenet_classes.pkl'

    if not os.path.exists(path):
        response = urllib.request.urlopen(
            'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl')
        classes = pickle.load(response)

        write_pickle(classes, path)

        return classes

    # or open from the local file
    return read_pickle(path)


def filter_strings_by_tail(files: Iterable[str],
                           tail: Tuple[str],
                           invert: bool = False
                           ) -> List[str]:
    """
    Filters file name strings from the folder by the tail characters.

    Args:
        files: iterable of strings
        tail: tail value, the selection criterion

    Kwargs:
        invert: if True - return files that end with 'tail', False - return rest fails

    Returns:
        list of filtered files by the file name tail
    """
    res = []
    inv_res = []
    for f in files:
        if f.endswith(tail):
            res.append(f)
        else:
            inv_res.append(f)

    if invert:
        return inv_res
    else:
        return res


def filter_strings_by_head(files: Iterable[str],
                           head: Tuple[str],
                           invert: bool = False
                           ) -> List[str]:
    """
    Filters file name strings from the folder by the head characters.

    Args:
        files: iterable of strings
        tail: head value, the selection criterion

    Kwargs:
        invert: if True - return files that start with 'head', False - return rest fails

    Returns:
        list of filtered files by the file name head
    """
    res = []
    inv_res = []
    for f in files:
        if f.startswith(head):
            res.append(f)
        else:
            inv_res.append(f)

    if invert:
        return inv_res
    else:
        return res


def load_images_to_numpy(img_dir: str,
                         n: int = None,
                         shape: Tuple[int, int] = None,
                         float_rescale: bool = True
                         ) -> List[np.ndarray]:
    """
    Load single image to np.ndarray

    Args:
        img_dir: images directory to load

    Kwargs:
        n: number of images to load
        shape: shape to respahe, tuple like (224, 224)
        float_rescale: concert to float and rescale to range from 0. to 1.

    Return:
        list of np.ndarray image: List[np.ndarray[H, W, C]]
    """

    imgs = os.listdir(img_dir)

    if n:
        imgs = imgs[:n]

    images = [load_image_to_numpy(img_dir, img, shape=shape,
                                  float_rescale=float_rescale) for img in imgs]

    return images


def load_image_to_numpy(img_path: str,
                        img_name: str,
                        shape: Tuple[int, int] = None,
                        float_rescale: bool = True
                        ) -> np.ndarray:
    """
    Load single image to np.ndarray

    Args:
        img_path: image path
        img_name: image name

    Kwargs:
        shape: shape to respahe, tuple like (224, 224)
        float_rescale: concert to float and rescale to range from 0. to 1.

    Return:
        image: np.ndarray[H, W, C]
    """
    img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
    if shape:
        img = img.resize(shape)

    if float_rescale:
        img = np.array(img).astype(np.float32)
        img = np.array(img) / 255.
    else:
        img = np.array(img)

    return img


def save_image_from_numpy(array: np.ndarray,
                          file_path: Union[Iterable[str], str]
                          ) -> None:
    """
    Save image from numpy array to file

    Args:
        array: numpy array image
        file_path: path to save image
    """
    full_path = _check_path(file_path)

    path, file = os.path.split(full_path)

    mkdir(path)

    img = Image.fromarray(array)
    img.save(full_path)


def img_to_torch_axes_order(img: np.ndarray
                            ) -> np.ndarray:
    """
    Args:
        img: image with dimensions (h, w, c) - np.ndarray[H, W, C]

    Return:
        image with dimensions (c, h, w): np.ndarray[C, H, W]
    """
    return np.moveaxis(img, [2], [0])


def img_to_numpy_axes_order(img: np.ndarray
                            ) -> np.ndarray:
    """
    Args:
        img: image with dimensions (c, h, w) - np.ndarray[C, H, W]

    Return:
        image with dimensions (h, w, c): np.ndarray[H, W, C]
    """
    return np.moveaxis(img, [0], [2])


def load_image_to_tensor(img_path: str,
                         img_name: str,
                         add_batch_dim: bool = False,
                         shape: Tuple[int, int] = None
                         ) -> torch.Tensor:
    """
    Load single image to torch.Tensor

    Args:
        img_path: image path
        img_name: image name

    Kwargs:
        add_batch_dim: if True add 0th dimension for batch: (channel, width, height)  -> (1, channel, width, height)
        shape: image shape (w, h)

    Returns:
        image tensor: Tensor[C, H, W] or Tensor[1, C, H, W]
    """
    img = load_image_to_numpy(img_path, img_name, shape)

    img = img_to_torch_axes_order(img)
    img_tensor = torch.from_numpy(img)

    if add_batch_dim:
        img_tensor.unsqueeze_(0)
    return img_tensor


def save_image_from_tensor(tensor: torch.Tensor,
                           file_path: Union[Iterable[str], str],
                           convert_to_uint8: bool = True
                           ) -> None:
    """
    Save image from Tensor to file

    Args:
        tensor: image Tensor
        file_path: path to save image

    Kwargs:
        convert_to_uint8: convert to uint8 flag
    """
    array = tensor.detach().cpu().numpy()
    array = img_to_numpy_axes_order(array)

    if convert_to_uint8:
        array = (array * 255).astype(np.uint8)

    save_image_from_numpy(array, file_path)


def load_images_to_tensor(img_dir: str,
                          n: int = None,
                          shape: Tuple[int, int] = None,
                          seed: int = None
                          ) -> torch.Tensor:
    """
    Load images from dir to torch.Tensor

    Args:
        img_dir: images path

    Kwargs:
        n: number of images to load
        shape: image shape (w, h)
        seed: seed for shuffling, no shuffling if None

    Returns:
        images tensor (batch, channel, width, height): Tensor[n, C, H, W]
    """

    imgs = os.listdir(img_dir)

    random.seed(seed)
    random.shuffle(imgs)

    if n:
        imgs = imgs[:n]

    img_tensors = [load_image_to_tensor(
        img_dir, img, shape=shape) for img in imgs]

    img_tensor = torch.stack(img_tensors)
    return img_tensor


def get_concept_files_dict(file_names: Iterable[str]
                           ) -> Dict[str, Iterable[str]]:
    """
    Converts file names from iterable to dict - concept_name:[concept_file_name]

    Args:
        file_names: iterable with file names, concept file name is <concept>_<id>.<extension>

    Returns:
        dictionaty of concept file lists - {concept_name:[concept_file_name]}
    """
    concepts = {}

    for f in file_names:
        concept = f.split('_')[0]
        if concept not in concepts:
            concepts[concept] = []
        concepts[concept].append(f)

    return concepts


def _check_path(path: Union[Iterable[str], str]
                ) -> str:
    """
    Gets final path to the file from string iterable or string

    Args:
        path: string path or iterable of strings, which will be concatenated into path

    Return:
        full path string
    """
    log_assert(isinstance(path, str) or len(path) > 0, "'path_args' be a string or iterable of strings")

    if isinstance(path, str):
        return path
    else:
        return os.path.join(*path)


def write_pickle(obj: object,
                 file_path: Union[Iterable[str], str]
                 ) -> None:
    """
    Writes pickle to the given path

    Args:
        obj: object to write as pickle
        file_path: string path or iterable of strings, which will be concatenated into path
    """

    full_path = _check_path(file_path)

    path, file = os.path.split(full_path)

    mkdir(path)

    with open(full_path, 'wb') as f:
        log_debug(f"Writing to: {full_path}")
        pickle.dump(obj, f)


def read_pickle(file_path: Union[Iterable[str], str]
                ) -> object:
    """
    Reads pickle from the given path

    Args:
        file_path: string path or iterable of strings, which will be concatenated into path

    Returns:
        loaded object
    """

    full_path = _check_path(file_path)

    with open(full_path, 'rb') as f:
        log_debug(f"Reading from: {full_path}")
        return pickle.load(f)


def apply_mask(img: np.ndarray,
               mask: np.ndarray,
               threshold: float = None,
               crop_around_mask: bool = True
               ) -> np.ndarray:
    """
    Apply mask to image

    Args:
        image: image to mask - np.ndarray[H, W, C]
        mask: mask - np.ndarray[H, W]

    Kwargs:
        threshold: values lower than threshold will be set to 0, rest - to 1
        crop_around_mask: apply (or not) cropping around the active mask pixels, i.e., remove excessive black regions

    Returns:
        masked image: np.ndarray[H, W, C]
    """
    mask = normalize_0_to_1(mask)

    img_mask = resize(mask, (img.shape[0], img.shape[1]))

    if threshold:
        img_mask = img_mask >= threshold

    if crop_around_mask:
        a0min, a0max, a1min, a1max = get_mask_bbox(img_mask)  # bbox for non-zero mask values 

    img_mask = np.expand_dims(img_mask, 2)  # 3d-mask
    masked_img = img * img_mask

    if crop_around_mask:
        masked_img = masked_img[a1min:a1max+1, a0min:a0max+1]  # remove excessive blackness

    return masked_img


def add_countours_around_non_black_pixels(image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
    """
   Draw countours around non-black image

    Args:
        image: image to mask - np.ndarray[H, W, C]
        mask_image: mask - np.ndarray[H, W]

    Returns:
        image with countours: np.ndarray[H, W, C]
    """
    gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    return image


def add_countours_around_mask(image: np.ndarray, binary_mask_image: np.ndarray) -> np.ndarray:
    """
   Draw countours around non-black image

    Args:
        image: image to mask - np.ndarray[H, W, C]
        binary_mask_image: mask - np.ndarray[H, W]

    Returns:
        image with countours: np.ndarray[H, W, C]
    """
    contours, _ = cv2.findContours(binary_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return image


def get_mask_bbox(mask: np.ndarray
                  ) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of mask's content

    Args:
        mask: mask np.ndarray[H, W]

    Returns:
        bounding box coordinates: Tuple[int, int, int, int]
    """

    def get_first_nonzero_arg(arr: Iterable[float]
                              ) -> int:
        """
        Get the argument of the first non-zero element in array

        Args:
            arr: array - Iterable[float]

        Returns:
            index of the first non-zero element of array: int
        """
        arg = 0
        for i, a in enumerate(arr):
            if a:
                arg = i
                break
        return arg

    a0 = mask.sum(axis=0) > 0
    a1 = mask.sum(axis=1) > 0

    a0min = get_first_nonzero_arg(a0)
    a0max = len(a0) - get_first_nonzero_arg(a0[::-1]) - 1
    a1min = get_first_nonzero_arg(a1)
    a1max = len(a1) - get_first_nonzero_arg(a1[::-1]) - 1

    return a0min, a0max, a1min, a1max


def apply_heatmap(img: np.ndarray,
                  heatmap: np.ndarray,
                  cmap: int = 2  # cv2.COLORMAP_JET
                  ) -> np.ndarray:
    """
    Apply heatmap to image

    Args:
        image: image (np.uint8) - np.ndarray[H, W, C]
        heatmap: float-valued heatmap - np.ndarray[H, W]

    Kwargs:
        cmap: int of cv2 colormap, defaults to 2 == cv2.COLORMAP_JET

    Returns:
        heatmapped image: np.ndarray[H, W, C]
    """
    heatmap = normalize_0_to_1(heatmap)

    heatmap_img = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_img = np.uint8(255 * heatmap_img)
    heatmap_img = cv2.applyColorMap(heatmap_img, cmap)

    superimposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    return superimposed_img[:, :, ::-1]  # Image.fromarray(superimposed_img[:, :, ::-1])


def apply_bbox(img: np.ndarray,
               bbox: np.ndarray,
               text: str = None
               ) -> np.ndarray:
    """
    Draw bounding boxes
    
    Args:
        image: image (np.uint8) - np.ndarray[H, W, C]
        bbox: (x1, y1, x2, y2) - np.ndarray[4]

    Kwargs:
        text: bbox text placed at (x1, y1, x2, y2)

    Returns:
        image with bbox: np.ndarray[H, W, C]
    """
    img = np.ascontiguousarray(img)
    bbox = bbox.astype(np.int32)

    cv2.rectangle(img, bbox[0:2], bbox[2:4], (0, 255, 0), 2)
    if text is not None:
        cv2.putText(img, text, bbox[0:2], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img


def normalize_0_to_1(array: Union[np.ndarray, torch.Tensor]
                     ) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize np.ndarray from 0 to 1

    Args:
        array: array to normalize - np.ndarray[...]

    Returns:
        normalized array: np.ndarray[...]
    """
    array = array - array.min()
    return array / (array.max() + EPSILON)


def save_np_uint8_arr_to_img(array: np.ndarray,
                             file_path: str,
                             shape: Tuple[int, int] = None
                             ) -> None:
    """
    Saver np.uint8 array to image under file_path

    Args:
        array: array to convert to image and save
        file_path: file path for saving

    Kwargs:
        shape: shape to rezize to, e.g. (200, 200). No resize if None
    """
    img = Image.fromarray(array)

    if shape is not None:
        img = img.resize(shape, Image.NEAREST)
    
    img.save(file_path)


def apply_smoothgrad_noise(input_sample: torch.Tensor,
                           noise_level: float = 0.1,
                           n_samples: int = 50
                           ) -> torch.Tensor:
    """
    Generates n_samples with applied Gaussian noise from input_sample, according to SmoothGrad method rules

    gaussian_noise(mean=0, variance=var**2), where var = (input_sample.max - input_sample.min) * noise_level

    Args:
        input_sample: input sample to generate distorted samples from - Tensor[:]

    Kwargs:
        noise_level: noise level for gaussian noise variance, defaults to 10%
        n_samples: number of samples to generate, defaults to 50

    Returns:
        batch of tensors: Tensor[n_samples, :]
    """
    repeat_shape = (n_samples, ) + tuple([1 for _ in input_sample.shape])

    tiled_input = input_sample.unsqueeze(0).repeat(repeat_shape)

    var = (tiled_input.max() - tiled_input.min()) * noise_level

    noise = torch.randn_like(tiled_input) * var**2

    result = tiled_input + noise

    return result


def write_string_to_file(s: str,
                         file_path: Union[Iterable[str], str]
                         ) -> None:
    """
    Writes string to the given path

    Args:
        s: string to write
        file_path: string path or iterable of strings, which will be concatenated into path
    """

    full_path = _check_path(file_path)

    path, file = os.path.split(full_path)

    mkdir(path)

    log_debug(f"Writing string to: {full_path}")

    with open(full_path, 'w') as f:
        f.write(s)


def read_json(json_path: str
              ) -> Any:
    """
    Read from JSON file

    Args:
        json_path: Path to JSON file

    Returns:
        JSON contents
    """    
    log_debug(f"Reading JSON from: {json_path}")

    with open(json_path, 'r') as file:
        content = json.load(file)
        return content


def write_json(obj: Any,
               json_path: str,
               indent: Union[str, int] = None
               ) -> None:
    """
    Write to JSON file

    Args:
        obj: JSON object to write
        json_path: Path to JSON file

    Kwargs:
        indent: indent in JSON file, see the original documentation of json.dump

    Returns:
        dictionary with JSON contents
    """
    log_debug(f"Writing JSON to: {json_path}")

    with open(json_path, 'w') as file:
        json.dump(obj, file, indent=indent)


def blend_imgs(img1: Image,
               img2: Image,
               alpha: float = 0.5
               ) -> Image:
    """
    Blend 2 PIL.Image

    Args:
        img1 (Image): image 1
        img2 (Image): image 2

    Kwargs:
        alpha (float = 0.5): blending strength variable

    Returns:
        (Image) blended image
    """
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img2 = img2.resize(img1.size)
    return Image.blend(img1, img2, alpha)


def get_colored_mask(mask: np.ndarray,
                     color_channels: Iterable[int] = [1]
                     ) -> Image:
    """
    converts grayscale binary mask to RGB mask

    Args:
        mask (np.ndarray): binary grayscale mask

    Kwargs:
        color_channels (Iterable[int] = [1]): channels (RGB) to expand mask to

    Returns:
        (Image) mask RGB image
    """
    rgb_img = np.zeros((*mask.shape, 3), dtype=bool)
    for c in color_channels:
        rgb_img[:,:,c] = mask
    return Image.fromarray(rgb_img.astype(np.uint8) * 255)


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
