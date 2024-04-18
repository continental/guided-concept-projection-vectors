'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from PIL import Image, ImageDraw
import numpy as np
from hooks import Propagator, PropagatorTorchSSD, PropagatorUltralyticsYOLOv5Old
from xai_utils.files import load_image_to_tensor, mkdir, resetdir
import torch
import math
from torchvision.models.detection.ssd import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, efficientnet_b0, EfficientNet_B0_Weights, squeezenet1_1, SqueezeNet1_1_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import List, Dict, Tuple, Iterable, Any
import matplotlib.pyplot as plt


MSCOCO_CATEGORY_COLORS = {3: 'firebrick',
                          4: 'forestgreen',
                          5: 'royalblue',
                          6: 'darkorange',
                          7: 'darkmagenta',
                          8: 'slateblue',
                          9: 'yellowgreen',
                          17: 'indigo',
                          18: 'darkcyan',
                          19: 'hotpink',
                          22: 'gold',
                          23: 'darkkhaki',
                          24: 'navy',
                          25: 'tomato'}

MSCOCO_CATEGORIES = {3: 'car',
                     4: 'motorcycle',
                     5: 'airplane',
                     6: 'bus',
                     7: 'train',
                     8: 'truck',
                     9: 'boat',
                     17: 'cat',
                     18: 'dog',
                     19: 'horse',
                     22: 'elephant',
                     23: 'bear',
                     24: 'zebra',
                     25: 'giraffe'}

YOLO5_LAYERS = ['4.cv3.conv',
                '5.conv',
                '6.cv3.conv',
                '7.conv',
                '8.cv3.conv',
                '9.cv2.conv',
                '10.conv',
                '12',
                '14.conv',
                '16',
                '17.cv3.conv',
                '18.conv',
                '19',
                '20.cv3.conv',
                '21.conv',
                '22',
                '23.cv3.conv']

SSD_LAYERS = ['backbone.features.19',
              'backbone.features.21',
              'backbone.extra.0.1',
              'backbone.extra.0.3', 
              'backbone.extra.0.5',
              'backbone.extra.0.7.1',
              'backbone.extra.0.7.3',
              'backbone.extra.1.0',
              'backbone.extra.1.2',
              'backbone.extra.2.0',
              'backbone.extra.2.2',
              'backbone.extra.3.0']


MOBILENET_LAYERS = ['features.9',
                    'features.10',
                    'features.11',
                    'features.12',
                    'features.13',
                    'features.14',
                    'features.15']


EFFICIENTNET_LAYERS = ['features.4.2',
                       'features.5.0',
                       'features.5.1',
                       'features.5.2',
                       'features.6.0', 
                       'features.6.1',
                       'features.6.2',
                       'features.7.0']

EFFICIENTNETV2_LAYERS = ['features.3',
                         'features.4.1',
                         'features.4.5',
                         'features.5.3',
                         'features.6.0',
                         'features.6.4',
                         'features.6.9',
                         'features.6.14']

SQUEEZENET_LAYERS = ['features.6.expand3x3',
                     'features.7.expand3x3',
                     'features.9.expand3x3',
                     'features.10.expand3x3',
                     'features.11.expand3x3',
                     'features.12.expand3x3']


EPSILON = 0.000001


def draw_mscoco_categories_and_colors(mscoco_category_ids: Iterable[int]
                                      ) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Draw MS COCO categories and colors for plotting by id

    Args:
        mscoco_category_ids (Iterable[int]): category ids

    Returns:
        id2name (Dict[int, str]): category id to name dict
        id2color (Dict[int, str]): category id to color dict
    """
    id2name = {i: MSCOCO_CATEGORIES[i] for i in mscoco_category_ids}
    id2color = {i: MSCOCO_CATEGORY_COLORS[i] for i in mscoco_category_ids}
    return id2name, id2color


def blend_imgs(img1: Image, img2: Image, alpha: float = 0.5) -> Image:
    """
    Blend two Image instances: aplha * img1 + (1 - alpha) * img2

    Args:
        img1 (Image): image 1
        img2 (Image): image 2

    Kwargs:
        alpha (np.ndarray = 0.5): alpha for blending

    Returns:
        (Image) blended image
    """
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img2 = img2.resize(img1.size)
    return Image.blend(img1, img2, alpha)


def get_colored_mask(mask: np.ndarray, color_channels: List[int] = [1], mask_value_multiplier: int = 1) -> Image:
    """
    Expand greyscale mask to RGB dimensions.

    Args:
        mask (np.ndarray): greyscale mask

    Kwargs:
        color_channels (List[int] = [1]): channels to fill with mask values, values of other other channels stay equal to 0. default - green mask
        mask_value_multiplier (int = 1): final value multiplier, use 255 if original mask was boolean, otherwise - 1

    Returns:
        (Image) RGB mask
    """
    rgb_img = np.zeros((*mask.shape, 3), dtype=mask.dtype)
    for c in color_channels:
        rgb_img[:,:,c] = mask
    return Image.fromarray(rgb_img.astype(np.uint8) * mask_value_multiplier)


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine (and rescale) masks to get an averaged mask.

    Args:
        masks (List[np.ndarray]): list of masks (may have different sizes) to combine

    Returns:
        combined_mask (np.ndarray) rescaled and combined mask
    """
    largest_mask_id = np.argmax([m.size for m in masks])

    img_masks = [Image.fromarray(m) for m in masks]
    new_size = img_masks[largest_mask_id].size
    resized_masks = [np.array(i.resize(new_size)) for i in img_masks]

    combined_mask = np.array(resized_masks).mean(axis=0).astype(np.uint8)

    return combined_mask


def add_bboxes(img: Image,
               predictions: np.ndarray,
               predictions_img_tensor_shape: Tuple[int, int] = (640, 480),
               bbox_color: str = 'red'
               ) -> Image:
    """
    Add bboxes from predictions to the image

    Args:
        img (Image): image instance
        predictions (np.ndarray): numpy array with predictions (N, 6), N - number of bboxes, each bbox has (x1, y1, x2, y2, prob, label)
        predictions_img_tensor_shape (Tuple[int, int] = (640, 480)): (W, H) of original tensor image for which predictions were calculated
        bbox_color (str = 'red'): color of bbox
    """
    img_resized = img.resize(predictions_img_tensor_shape)

    draw = ImageDraw.Draw(img_resized)

    for x1, y1, x2, y2, prob, label in predictions:
        draw.rectangle([(x1, y1), (x2, y2)], outline=bbox_color, width=5)

    return img_resized 


def binary_to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """
    Convert binary np.ndarray to np.uint8 image array

    Args:
        arr (np.ndarray): array to convert

    Returns:
        (np.ndarray) np.uint8 image array
    """
    return (arr  * 255).astype(np.uint8)


def downscale_numpy_img(img: np.ndarray,
                        downscale_factor: float = 5.0
                        ) -> np.ndarray:
    """
    Downscale numpy image array

    Args:
        img (np.ndarray): image

    Kwargs:
        downscale_factor (float = 5.0): downscale factor

    Returns:
        (np.ndarray) downscaled image
    """
    img = Image.fromarray(img)
    return np.array(img.resize((int(c / downscale_factor) for c in img.size)))


def plot_binary_mask(mask: np.ndarray,
                     downscale_factor: float = 5.0
                     ) -> None:
    """
    Plot (downscaled) binary mask

    Args:
        mask (np.ndarray): mask

    Kwargs:
        downscale_factor (float = 5.0): downscale factor
    """
    img_arr = downscale_numpy_img(mask, downscale_factor)
    img = Image.fromarray(img_arr)
    img.show()


def gcpv_stats(gcpv: np.ndarray) -> None:
    """
    Print GCPV stats (mean, var, sparsity)

    Args:
        gcpv (np.ndarray): GCPV
    """
    print('\tmean:', gcpv.mean())
    print('\tvar:', gcpv.var())
    print(f'\tsparsity: {(gcpv == 0).sum()}/{len(gcpv)}', )


def plot_projection(gcpv: np.ndarray,
                    acts: np.ndarray,
                    proj_name: str = None
                    ) -> None:
    """
    Plot projection of GCPV and activations

    Args:
        gcpv (np.ndarray): GCPV
        acts (np.ndarray): activations

    Kwargs:
        proj_name: projection name to print
    """
    if proj_name is not None:
        print(proj_name)
    gcpv_stats(gcpv)

    projecion_uint8 = get_projection(gcpv, acts)
    plot_binary_mask(projecion_uint8, 0.1)


def get_projection(gcpv: np.ndarray,
                   acts: np.ndarray,
                   downscale_factor: float = None
                   ) -> np.ndarray:
    """
    Get projection of GCPV and activations

    Args:
        gcpv (np.ndarray): GCPV
        acts (np.ndarray): activations

    Kwargs:
        downscale_factor (float = None): downscale factor

    Returns:
        (np.ndarray) np.uint8 image array
    """
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    gcpv3d = np.expand_dims(gcpv, axis=[1,2])
    projecion = (acts * gcpv3d).sum(axis=0)
    projecion = sigmoid(projecion) #rescale_to_range(projecion, 0, 1) # 
    
    if downscale_factor is not None:
        projecion = downscale_numpy_img(projecion, downscale_factor)

    projecion_uint8 = (projecion * 255).astype(np.uint8)
    return projecion_uint8


def rescale_to_range(data, new_min, new_max):
    old_min = np.min(data)
    old_max = np.max(data)
    
    rescaled_data = ((data - old_min) / (old_max - old_min + EPSILON)) * (new_max - new_min) + new_min
    return rescaled_data


def get_rgb_binary_mask(mask: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:

    img = Image.fromarray(mask.astype(np.float32))
    if target_size:
        img = img.resize(target_size)
    
    img_np = np.array(img)

    img_np = rescale_to_range(img_np, 0, 1)

    # apply colormap
    cmap = plt.get_cmap('bwr')
    img_rgba = cmap(img_np)
    # rgba to rgb
    img_rgb = (img_rgba[:, :, :3] * 255).astype(np.uint8)

    return img_rgb

def yolo5_propagator_builder(layers: List [str] = YOLO5_LAYERS):
    yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', skip_validation=True)

    yolo5_prop = PropagatorUltralyticsYOLOv5Old(yolo5, layers)

    return yolo5_prop

def ssd_propagator_builder(layers: List [str] = SSD_LAYERS):
    ssd = ssd300_vgg16(weights=SSD300_VGG16_Weights)

    ssd_prop = PropagatorTorchSSD(ssd, layers)

    return ssd_prop

def mobilenet_propagator_builder(layers: List [str] = MOBILENET_LAYERS):
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    mobilenet_prop = Propagator(mobilenet, layers)

    return mobilenet_prop

def efficientnet_propagator_builder(layers: List [str] = EFFICIENTNET_LAYERS):
    efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    efficientnet_prop = Propagator(efficientnet, layers)

    return efficientnet_prop

def efficientnetv2_propagator_builder(layers: List [str] = EFFICIENTNETV2_LAYERS):
    efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    efficientnet_prop = Propagator(efficientnet, layers)

    return efficientnet_prop

def squeezenet_propagator_builder(layers: List [str] = SQUEEZENET_LAYERS):
    squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

    squeezenet_prop = Propagator(squeezenet, layers)

    return squeezenet_prop


def get_acts_preds_dict(propagator: Propagator,
                        img_folder: str,
                        img_name: str,
                        img_shape: Tuple[int, int] = (640, 480)
                        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get activations and predictions for image

        Args:
            propagator (Propagator): propagator instance
            img_folder (str): folder / directory with image
            img_name (str): image file name

        Kwargs:
            img_shape (Tuple[int, int] = (640, 480)): [W, H] image shape after liading, before propagation

        Returns:
            acts (Dict[str, Tensor]) dictionary with activations
            preds (Tensor) bbox predictions Tensor[N,6]
        """
        img_tensor = load_image_to_tensor(img_folder, img_name, add_batch_dim=True, shape=img_shape)
        acts = propagator.get_activations(img_tensor)
        preds = propagator.get_predictions(img_tensor)
        return acts, preds


def get_grads_dict(propagator: Propagator,
                   img_folder: str,
                   img_name: str,
                   img_shape: Tuple[int, int] = (640, 480)
                   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    img_tensor = load_image_to_tensor(img_folder, img_name, add_batch_dim=True, shape=img_shape)
    grads = propagator.tensor_get_gradients(img_tensor)
    return grads


def save_cluster_imgs_as_tiles(clustered_gcpv_storages: Iterable[Iterable[Any]],
                               selected_tag_id_names: Dict[int, str],
                               selected_tag_ids_and_colors: Dict[int, str],
                               img_out_path: str,
                               image_tile_size: Tuple[int, int] = (256, 192),
                               model_categories: Dict[int, str] = None
                               ) -> None:
    """
    Tile images from cluster and save as '{img_out_path}/cluster_{cluster_number}.jpg'

    Args:
        clustered_gcpv_storages (List[List[GCPVMultilayerStorage]]): list of lists of GCPVMultilayerStorage, where external list is clusters, internal lists are leaf GCPVMultilayerStorage
        selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding colors, e.g., {3: 'red', 4: 'blue'}
        selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding names, e.g., {3: 'car', 4: 'motorcycle'}
        img_out_path (str): output directory for tiled images

    Kwargs:
        image_tile_size (Tuple[int, int]): size of each tile
        model_categories (str = None): different models may have different id-label mappings, needed for correct visualization of bboxes
    """
    
    def find_closest_square_rootable_number(x):
        y = x
        while True:
            sqrt_y = math.sqrt(y)
            if sqrt_y.is_integer():
                return y
            y += 1

    mkdir(img_out_path)
    resetdir(img_out_path)

    # number of images per cluster
    cluster_counts = [len(cl) for cl in clustered_gcpv_storages]
        
    # evaluation of tile cells
    tile_cells = [find_closest_square_rootable_number(i) for i in cluster_counts]

    # for each cluster:
    for cluster_number, (gcpv_cluster, tc) in enumerate(zip(clustered_gcpv_storages, tile_cells)):

        # finalize number of rows and cols
        cols = rows = int(math.sqrt(tc))
        for r in range(rows, 0, -1):
            if (r * cols) >= len(gcpv_cluster):
                rows = r
            else:
                break

        # canvas
        canvas_width = image_tile_size[0] * cols
        canvas_height = image_tile_size[1] * rows
        canvas = Image.new("RGB", (canvas_width, canvas_height), color='white')

        # paste into canvas
        for row in range(rows):
            for col in range(cols):
                # offsets
                x_offset = col * image_tile_size[0]
                y_offset = row * image_tile_size[1]

                # id
                temp_idx = row * cols + col

                # leave space empty if no more images
                if temp_idx >= len(gcpv_cluster):
                    continue

                temp_gcpv = gcpv_cluster[temp_idx]
                    
                # get current image, segmentation and color
                temp_category = temp_gcpv.segmentation_category_id
                temp_category_name = selected_tag_id_names[temp_category]
                # fix mapping of model categories and MS COCO original categories
                model_category = temp_category
                if model_categories is not None:
                    for mc_idx, mc_name in model_categories.items():
                        if mc_name == temp_category_name:
                            model_category = mc_idx
                temp_image_path = temp_gcpv.image_path
                temp_seg = temp_gcpv.segmentation
                #temp_selected_predictions = temp_gcpv.image_predictions
                #temp_selected_predictions = [pred for pred in temp_gcpv.image_predictions if int(pred[5]) == model_category]
                temp_color = selected_tag_ids_and_colors[temp_category]
                # blend image end segmentation (add colored mask)
                img_with_seg = blend_imgs(get_colored_mask(temp_seg, mask_value_multiplier=255), Image.open(temp_image_path))
                img_with_seg_bbox = img_with_seg
                #img_with_seg_bbox = add_bboxes(img_with_seg, temp_selected_predictions, bbox_color=selected_tag_ids_and_colors[temp_category])
                #resize before inserting
                img_with_seg_resized = img_with_seg_bbox.resize(image_tile_size)

                frame_width = 5

                # add colored frame
                draw = ImageDraw.Draw(img_with_seg_resized)
                #top
                draw.line([(0, 0), (image_tile_size[0], 0)], fill=temp_color, width=frame_width)
                # bot
                draw.line([(0, image_tile_size[1] - 1), (image_tile_size[0], image_tile_size[1] - 1)], fill=temp_color, width=frame_width)
                # left
                draw.line([(0, 0), (0, image_tile_size[1])], fill=temp_color, width=frame_width)
                #right
                draw.line([(image_tile_size[0] - 1, 0), (image_tile_size[0] - 1, image_tile_size[1])], fill=temp_color, width=frame_width)

                canvas.paste(img_with_seg_resized, (x_offset, y_offset))
            
        canvas.save(f'{img_out_path}/cluster_{cluster_number}.jpg')


class GCPVExperimentConstants:

    NET_TAGS = ['yolo', 'ssd', 'efficientnet', 'efficientnetv2', 'mobilenet', 'squeezenet']
    CLUSTERING_SETTINGS = ['strict', 'relaxed']

    GCPV_DIR = {
        'yolo': r"../data/mscoco2017val/processed/gcpvs/gcpv_original_yolo",
        'ssd': r"../data/mscoco2017val/processed/gcpvs/gcpv_original_ssd",
        'efficientnet': r"../data/mscoco2017val/processed/gcpvs/gcpv_original_efficientnet",
        'efficientnetv2': r"../data/mscoco2017val/processed/gcpvs/gcpv_original_efficientnetv2",
        'mobilenet': r"../data/mscoco2017val/processed/gcpvs/gcpv_original_mobilenet",
        'squeezenet': r"../data/mscoco2017val/processed/gcpvs/gcpv_original_squeezenet"
    }

    MODEL_CATEGORIES = {
        'yolo': yolo5_propagator_builder().model.names,
        'ssd': None,
        'efficientnet': None,
        'efficientnetv2': None,
        'mobilenet': None,
        'squeezenet': None
    }

    MLGCPV_LAYER_SETS = {
        'yolo': [['10.conv'], ['10.conv', '20.cv3.conv'], ['10.conv', '17.cv3.conv', '20.cv3.conv']],
        'ssd': [['backbone.features.21', 'backbone.extra.0.5', 'backbone.extra.1.0']],
        'efficientnet': [['features.5.0', 'features.6.0', 'features.7.0']],
        'efficientnetv2': [['features.4.1', 'features.5.3', 'features.6.14']],                         
        'mobilenet': [['features.12', 'features.14', 'features.15']],
        'squeezenet': [['features.6.expand3x3', 'features.11.expand3x3', 'features.12.expand3x3']]
    }

    CLUSTER_IMGS_FLAG = {  # works only for ODs, because also build bboxes, code cna be fixed - need to remove/comment 2 lines, but who cares
        'yolo': True,
        'ssd': True,
        'efficientnet': False,
        'efficientnetv2': False,
        'mobilenet': False,
        'squeezenet': False
    }

    N_SAMPLES_PER_TAG = 100
    DISTANCE = 'euclidean'
    METHOD = 'ward'
    
    PURITY = {
        'strict': 0.90,
        'relaxed': 0.80
    }

    SAMPLE_THRESHOLD_COEFFICIENT = {
        'strict': 0.025,
        'relaxed': 0.05
    }

    MSCOCO_CATEGORIES = [3, 4, 5, 17, 19, 22]

    N_SEEDS = 50
    K_NEIGHBOURS = [3, 5, 7, 9, 11]
    
