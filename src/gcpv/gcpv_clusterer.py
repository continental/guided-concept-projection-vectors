'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from xai_utils.logging import init_logger, log_info
init_logger()

from typing import Dict, List, Tuple, Iterable, Callable, Union, Literal
import math
import os

from .gcpv_utils import get_colored_mask, blend_imgs, combine_masks, get_acts_preds_dict, get_projection, add_bboxes, get_grads_dict, save_cluster_imgs_as_tiles, get_rgb_binary_mask
from .gcpv import GCPVMultilayerStorage, GCPVMultilayerClusterInfo, GCPVMultilayerStorageDirectoryLoader
from xai_utils.files import mkdir, apply_heatmap, apply_mask, add_countours_around_mask, normalize_0_to_1
from hooks import Propagator

from skimage.transform import resize
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster import hierarchy


class GCPVClusterer:

    def __init__(self,
                 selected_tag_ids_and_colors: Dict[int, str],
                 selected_tag_id_names: Dict[int, str],
                 gcpv_layers: List[str],
                 max_samples_per_category: int,
                 cluster_size_threshold: int = 20,
                 cluster_purity_threshold: float = 0.8,
                 distance_metric: str = 'cosine',
                 clustering_method: str = 'complete',
                 save_dir: str = './experiment_outputs/gcpv_clustering',
                 cluster_linkage_threshold: float = 4.0
                 ) -> None:
        """
        Args:
            selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding colors, e.g., {3: 'red', 4: 'blue'}
            selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding names, e.g., {3: 'car', 4: 'motorcycle'}
            gcpv_layers (List[str]): layers for GCPV concatenation (several layers may improve the result)
            max_samples_per_category (int): max number of samples per clustered category / concept, if category has less samples provided than this threshold - all samples will be used

        Kwargs:
            cluster_size_threshold (int = 20): if size of the cluster is smaller than given value it will be saved and noth further decomposed into subclusters
            cluster_purity_threshold (float = 0.8): if purity of cluster is higher than given value it will be saved and noth further decomposed into subclusters
            distance_metric (str = 'cosine'): distance method for evaluation of distance matrix, 'cosine' is advised ('euclidean' is the only option for 'ward' clustering)
            clustering_method (str = 'complete'): hierarchial clustering linkage method, advised: 'complete', 'average' and 'ward', for other options check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage            
            save_dir (str = './experiment_outputs/gcpv_clustering'): directory to save results
            cluster_linkage_threshold (float = 3.0): cluster selection by linkage distance (similar to color_threshold of scipy.cluster.hierarchy.dendrogram) 
        """
        self.selected_tag_ids_and_colors = selected_tag_ids_and_colors
        self.selected_tag_id_names = selected_tag_id_names
        self.selected_tag_ids = list(selected_tag_ids_and_colors.keys())
        self.gcpv_layers = gcpv_layers
        self.max_samples_per_category = max_samples_per_category

        self.cluster_size_threshold = cluster_size_threshold
        self.cluster_purity_threshold = cluster_purity_threshold
        self.distance_metric = distance_metric
        self.clustering_method = clustering_method

        self.save_dir = save_dir

        self.cluster_linkage_threshold = cluster_linkage_threshold

        self.cluster_infos: List[GCPVMultilayerClusterInfo] = None  # meta info for each cluster (centroid, category counts, etc.) obtained from the last run of cluster_gcpvs()

        self.cluster_labeled_gcpv: Tuple[List[GCPVMultilayerStorage], List[int]] = None  # gcpvs storages with cluster labels for kNN, tuple - (GCPVs, cluster int labels)

    def cluster_gcpvs(self,
                      gcpv_storages_dict: Dict[int, List[GCPVMultilayerStorage]],
                      save_imgs: bool = True,
                      model_categories: Dict[int, str] = None,
                      clustering_type: Literal['adaptive', 'linkage'] = 'adaptive'
                      ) -> List[List[GCPVMultilayerStorage]]:
        """
        Perform hierarchial clustering of GCPVs

        Args:
            gcpv_storages (Dict[int, List[GCPVMultilayerStorage]]): GCPV storages per category for clustering

        Kwargs:
            save_imgs (bool = True): save or not visual results to self.save_dir/gcpv_clustered_imgs_tiles/
            model_categories (str = None): different models may have different id-label mappings, needed for correct visualization of bboxes
            clustering_type (Literal['adaptive', 'linkage'] = 'adaptive'): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, while 'linkage' uses self.cluster_linkage_threshold

        Returns:
            clustered_gcpv_storages (List[List[GCPVMultilayerStorage]]): list of lists of GCPVMultilayerStorage, where external list is clusters, internal lists are leaf GCPVMultilayerStorage
        """
        log_info("Clustering...")
        # clustering
        selected_gcpv_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids = self._cluster(gcpv_storages_dict, clustering_type)

        # reorder storages according to cluster_leaf_list
        clustered_gcpv_storages = self._reorder_selected_gcpv_storages_to_clusters(selected_gcpv_storages, cluster_leaf_list)

        # get clusters meta information (centroids, categories count, etc)
        self.cluster_infos = self._gcpv_cluster_infos(clustered_gcpv_storages)

        # estimate clustering quality metric
        n_categories = len(gcpv_storages_dict)
        clustering_state = self._estimate_clustering_state()

        # gcpvs and their cluster labels (for kNN)
        self.cluster_labeled_gcpv = (selected_gcpv_storages, self._gcpv_cluster_labels(cluster_leaf_list))
        
        if save_imgs:
            save_cluster_imgs_as_tiles(clustered_gcpv_storages, self.selected_tag_id_names, self.selected_tag_ids_and_colors, os.path.join(self.save_dir, 'gcpv_clustered_imgs_tiles'), (128,96), model_categories)

        return clustered_gcpv_storages
    
    @staticmethod
    def _reorder_selected_gcpv_storages_to_clusters(selected_gcpv_storages: Iterable[GCPVMultilayerStorage],
                                                    cluster_leaf_list: Iterable[Iterable[int]]
                                                    ) -> List[List[GCPVMultilayerStorage]]:
        """
        Wraps list of GCPV storages to double list, where external list is clusters, internal - cluster leafs (samples)

        Args:
            selected_gcpv_storages (Iterable[GCPVMultilayerStorage]): GCPV storages
            cluster_leaf_list (Iterable[Iterable[int]]): cluster leafs indices

        Retrns:
            clustered_gcpv_storages (List[List[GCPVMultilayerStorage]]): GCPV storages in cluster-like wrapping, external list is clusters, internal - cluster leafs (samples)
        """
        clustered_gcpv_storages = []
        for cluster in cluster_leaf_list:
            gcpv_storages_in_cluster = []
            for leaf in cluster:
                gcpv_storages_in_cluster.append(selected_gcpv_storages[leaf])
            clustered_gcpv_storages.append(gcpv_storages_in_cluster)

        return clustered_gcpv_storages

    def _gcpv_cluster_infos(self,
                            clustered_gcpv_storages: List[List[GCPVMultilayerStorage]],
                            ) -> List[GCPVMultilayerClusterInfo]:
        """
        Get list of centroids obtained from all clustered GCPVs

        Args:
            clustered_gcpv_storages (List[List[GCPVMultilayerStorage]]): GCPV storages per category for clustering

        Returns:
            cluster_infos (List[GCPVMultilayerClusterInfo]): list of meta information for each cluster
        """ 
        cluster_infos = [GCPVMultilayerClusterInfo(storages) for storages in clustered_gcpv_storages]

        for i, ci in enumerate(cluster_infos):
            prob_str = ', '.join([f'P({self.selected_tag_id_names[cat]}) = {prob:.2f}' for cat, prob in ci.cluster_category_probabilities.items()])
            log_info(f'Cluster {i}: {prob_str}')

        return cluster_infos
    
    def _estimate_clustering_state(self) -> Tuple[float, int]:
        """
        Estimate the state after clustering

        State is estimated:
        maximize purity -> homogenous clusters -> mean purity
        minimize number of clusters -> less confusion in the feature space -> 1 / number of clusters

        Returns:
            clustering_state (float): clustering state metric
        """
        top_cat_counts = [ci.get_cluster_top_category_count() for ci in self.cluster_infos]
        cluster_sizes = [len(ci) for ci in self.cluster_infos]
        purity = sum(top_cat_counts) / sum(cluster_sizes)
        n_clusters = len(self.cluster_infos)

        layers_string = ' & '.join(self.gcpv_layers)
        
        log_info(f'Clustering layers: {layers_string}')
        log_info(f'Clustering purity: {purity:.3f}')
        log_info(f'Number of clusters: {n_clusters}')

        return purity, n_clusters

    @staticmethod
    def _gcpv_cluster_labels(cluster_leaf_list: List[List[int]]) -> List[int]:
        """
        Get GCPV cluster labels in original order (obtained from clustered leafes)

        Args:
            cluster_leaf_list (List[List[int]]): cluster leafs indices

        Returns:
            gcpv_cluster_labels (List[int]): GCPV cluster labels in original order
        """
        n_leafs = sum([len(i) for i in cluster_leaf_list])

        gcpv_cluster_labels = []

        for i in range(n_leafs):
            for cluster_idx, cluster_leafs in enumerate(cluster_leaf_list):
                if i in cluster_leafs:
                    gcpv_cluster_labels.append(cluster_idx)
        
        return gcpv_cluster_labels
    
    def cluster_stats(self) -> None:

        gcpvs = [{l: cluster.get_centroid_gcpv(l) for l in self.gcpv_layers} for cluster in self.cluster_infos]
        fig, axs = plt.subplots(1, len(self.gcpv_layers), figsize=(20, 5))
        for j, l in enumerate(self.gcpv_layers):
            for i, gcpv in enumerate(gcpvs):            
                axs[j].plot(gcpv[l], label=f'cluster {i}')
            axs[j].legend()

        plt.savefig(os.path.join(self.save_dir, 'filter_freqs.pdf'), bbox_inches='tight', pad_inches=0)
        plt.close()

        mlgcpvs = [cluster.get_centroid_mlgcpv(self.gcpv_layers) for cluster in self.cluster_infos]

        hm1 = pairwise_distances(mlgcpvs, metric='euclidean')
        hm2 = pairwise_distances(mlgcpvs, metric='cosine')

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(hm1, ax=axs[0], cmap='vlag')
        sns.heatmap(hm2, ax=axs[1], cmap='vlag')
        #sns.clustermap(hm2, cmap='vlag')

        plt.savefig(os.path.join(self.save_dir, 'cluster_stats.pdf'), bbox_inches='tight', pad_inches=0)
        plt.close()

    def cluster_filter_frequencies_analysis(self, n=5) -> None:

        cluster_gcpvs = [{l: cluster.get_centroid_gcpv(l) for l in self.gcpv_layers} for cluster in self.cluster_infos]
        
        for cluster_id, gcpv_dict in enumerate(cluster_gcpvs):
            print(cluster_id)
            for l in self.gcpv_layers:
                gcpv = gcpv_dict[l]
                argsorted_gcpv = np.argsort(gcpv)[::-1]
                argsorted_gcpv_abs = np.argsort(np.abs(gcpv))[::-1]

                top_filters = argsorted_gcpv[:n]
                top_filters_abs = argsorted_gcpv_abs[:n]
                print('orig', l, top_filters.tolist())
                print('abs', l, top_filters_abs.tolist())
    
    def predict_gcpvs_with_centroids(self,
                                     gcpv_storages_dict: Dict[int, List[GCPVMultilayerStorage]],
                                     save_imgs: bool = True
                                     ) -> Dict[int, List[Dict[int, float]]]:
        """
        Predict new GCPVs with cluster centroids

        Args:
            gcpv_storages (Dict[int, List[GCPVMultilayerStorage]]): GCPV storages per category for assignment to new clusters

        Kwargs:
            save_imgs (bool): save or not visual results to self.save_dir/gcpv_predictions_centroids/

        Returns:
            predictions (Dict[int, List[Dict[int, float]]]): predictions for every GCPV storage per category
        """
        log_info("Predicting with centroids...")

        if self.cluster_infos is None:
            raise ValueError("Train clusterer first, run cluster_gcpvs() first")
        else:
            predictions = dict()
            predictions_cluster_idxs = dict() 
            for c in sorted(list(gcpv_storages_dict.keys())):
                category_storages = gcpv_storages_dict[c]

                category_gcpvs = self._get_mlgcpv(category_storages)

                cluster_centroids = np.array([i.get_centroid_mlgcpv(self.gcpv_layers) for i in self.cluster_infos])

                dist_matr = pairwise_distances(cluster_centroids, category_gcpvs, metric=self.distance_metric)

                predictions_cluster_idxs[c] = np.argmin(dist_matr, axis=0)
                predictions[c] = [self.cluster_infos[i].cluster_category_probabilities for i in predictions_cluster_idxs[c]]
                
            if save_imgs:
                save_dir = os.path.join(self.save_dir, 'gcpv_predictions_centroids')
                self._save_images_for_predictions(gcpv_storages_dict, predictions_cluster_idxs, save_dir, self._generate_prediction_string_centroid)

            return predictions
        
    def predict_gcpvs_with_knn(self,
                               gcpv_storages_dict: Dict[int, List[GCPVMultilayerStorage]],
                               k: int = 5,
                               save_imgs: bool = True
                               ) -> Dict[int, List[Dict[int, float]]]:
        """
        Predict new GCPVs with k-nearest neigbours

        Args:
            gcpv_storages (Dict[int, List[GCPVMultilayerStorage]]): GCPV storages per category for assignment to new clusters

        Kwargs:
            k (int): k neighbours to use
            save_imgs (bool): save or not visual results to self.save_dir/gcpv_predictions_knn/

        Returns:
            predictions (Dict[int, List[Dict[int, float]]]): predictions for every GCPV storage per category (probabilities of true labels of nearest neigbours)
        """
        log_info("Predicting with kNN...")

        if self.cluster_labeled_gcpv is None:
            raise ValueError("Train clusterer first, run cluster_gcpvs() first")
        else:
            # knn data
            neigbour_gcpvs = self._get_mlgcpv(self.cluster_labeled_gcpv[0])
            neigbour_gcpvs_true_labels = np.array([gcpv.segmentation_category_id for gcpv in self.cluster_labeled_gcpv[0]])

            knn = KNeighborsClassifier(k, metric=self.distance_metric)
            knn.fit(neigbour_gcpvs, neigbour_gcpvs_true_labels)

            predictions = dict()
            for c in sorted(list(gcpv_storages_dict.keys())):
                category_storages = gcpv_storages_dict[c]

                category_gcpvs = self._get_mlgcpv(category_storages)

                probs = knn.predict_proba(category_gcpvs)

                category_pred_dict = [{int(c): float(p) for c, p in zip(knn.classes_, prob)} for prob in probs]

                predictions[c] = category_pred_dict

            if save_imgs:
                save_dir = os.path.join(self.save_dir, 'gcpv_predictions_knn')
                self._save_images_for_predictions(gcpv_storages_dict, predictions, save_dir, self._generate_prediction_string_knn)

            return predictions

    def _save_images_for_predictions(self,
                                     gcpv_storages_dict: Dict[int, List[GCPVMultilayerStorage]],
                                     predictions: Union[Dict[int, List[int]], Dict[int, List[Dict[int, float]]]],
                                     img_out_path: str,
                                     title_string_fn: Callable[[Union[int, Dict[int, float]]], str]
                                     ) -> None:
        """
        Save images of centroid predictions of GCPVs

        Args:
            gcpv_storages (Dict[int, List[GCPVMultilayerStorage]]): GCPV storages per category for assignment to new clusters
            predictions (Union[Dict[int, List[int]], Dict[int, List[Dict[int, float]]]]): cluster or kNN predictions for every GCPV storage per category
            img_out_path (str): output directory for images
            title_string_fn (Callable[[Union[int, Dict[int, float]]], str]): function that takes single predicion as argument and produces an image title string from it
        """
        mkdir(img_out_path)

        for c in sorted(list(gcpv_storages_dict.keys())):
            category_storages = gcpv_storages_dict[c]
            category_predictions = predictions[c]

            for storage, pred in zip(category_storages, category_predictions):

                img_load_path = storage.image_path
                img_name = os.path.basename(img_load_path)
                img_name_base, ext = os.path.splitext(img_name)
                save_img_name = f"{img_name_base}_{c}{ext}"
                img_save_path = os.path.join(img_out_path, save_img_name)
                img_seg = storage.segmentation

                projections = [storage.gcpv_storage[l].projection for l in self.gcpv_layers]
                avg_projection = combine_masks(projections)

                img_str = title_string_fn(pred)

                # load image, create plot, add title and save
                img = Image.open(img_load_path).convert('RGB')
                seg_arr = np.array(blend_imgs(img, get_colored_mask(img_seg, [1], mask_value_multiplier=255)))
                proj_arr = apply_heatmap(np.array(img), avg_projection / 255.)

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(seg_arr)
                axs[0].axis('off')
                axs[0].set_title('segmentation')

                axs[1].imshow(proj_arr)
                axs[1].axis('off')
                axs[1].set_title('avg. projection')

                plt.suptitle(img_str)
                #plt.tight_layout()
                plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

    def _generate_prediction_string_centroid(self,
                                             pred: int,
                                             ) -> str:
        """
        Generate a prediction string which contains information about probabilities of true class label assignments in clusters

        Args:
            pred (int): predicted cluster label

        Returns:
            pred_str (str): prediction string
        """
        cluster_category_probs = self.cluster_infos[pred].cluster_category_probabilities
        pred_str = f'Cluster {pred}:'
        for cat, prob in cluster_category_probs.items():
            pred_str += f' {self.selected_tag_id_names[cat]} - {prob:.2f}'
        
        return pred_str

    def _generate_prediction_string_knn(self,
                                        pred: Dict[int, float]
                                        ) -> str:
        """
        Generate a prediction string which contains information about probabilities of true class label assignments in clusters

        Args:
            pred (Dict[int, float]): probabilities of labels of nearest neighbors

        Returns:
            pred_str (str): prediction string
        """
        labels = sorted(list(pred.keys()))

        labels_total = sum([pred[l] for l in labels])

        pred_str = f'kNN category probs.:'
        for l in labels:
            if pred[l] != 0.:
                pred_str += f' {self.selected_tag_id_names[l]} - {(pred[l]/ labels_total):.2f}'
        
        return pred_str
    
    def _cluster(self,
                 gcpv_storages: Dict[int, List[GCPVMultilayerStorage]],
                 clustering_type: Literal['adaptive', 'linkage']
                 ) -> Tuple[List[GCPVMultilayerStorage], np.ndarray, List[List[int]], List[List[int]], List[int]]:
        """
        Perform hierarchial clustering of GCPVs

        Args:
            gcpv_storages (Dict[int, List[GCPVMultilayerStorage]]): GCPV storages per category for clustering,
            clustering_type (Literal['adaptive', 'linkage']): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, while 'linkage' uses self.cluster_linkage_threshold
        
        Returns:
            selected_gcpv_storages (List[GCPVMultilayerStorage]) - GCPV storages used for clustering
            linkage (np.ndarray) - scipy linkage array
            cluster_leaf_list (List[List[int]]): list of lists of leaf ids, where external list is clusters, internal lists are leaf ids
            cluster_node_list (List[List[int]]): list of lists of tree node ids, where external list is clusters, internal lists are tree node ids
            cluster_node_ids (List[int]): list of ids of cluster branch nodes
        """
        # select GCPVs for clustering
        #gcpv_data = []  # storage for only 
        selected_gcpv_storages = []  # selected samples for clustering
        #labels_int = []  # true category labels
        for c in self.selected_tag_ids:
            category_storages = gcpv_storages[c]
            for gcpv_storage in category_storages[:self.max_samples_per_category]:
                selected_gcpv_storages.append(gcpv_storage)

        labels_int = [s.segmentation_category_id for s in selected_gcpv_storages]
        gcpv_data = self._get_mlgcpv(selected_gcpv_storages)

        # metric + method combinations:
        # average + cosine
        # complete + cosine
        # ward + euclidean
        # 'average' = mean of distances between all points in both connected clusters (distance between centroids)
        # 'complete' = max distance between two furthest points in both connected clusters
        # 'ward' (variance minimization) - dist = obtained state variance (sum of distances from samples to possible cluster centroids), of all possible cluster connections

        linkage = hierarchy.linkage(gcpv_data, method=self.clustering_method, metric=self.distance_metric)

        if clustering_type == 'adaptive':
            # run adaptive clustering with min_purity and min_cluster_size constraints
            cluster_leaf_list, cluster_node_list, cluster_node_ids = self._run_adaptive_clustering(hierarchy.to_tree(linkage), np.array(labels_int))
        else:
            cluster_leaf_list, cluster_node_list, cluster_node_ids = self._run_linkage_clustering(hierarchy.to_tree(linkage), np.array(labels_int))
        
        return selected_gcpv_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids
    
    def _get_mlgcpv(self, storages: List[GCPVMultilayerStorage]) -> np.ndarray:
        """
        Get ML-GCPVs for given storages, self.gcpv_layers are used

        Args:
            storages (List[GCPVMultilayerStorage]): storages for ML-GCPVs extraction

        Returns:
            mlgcpvs (np.ndarray): 2D-numpy-array of stacked and combined ML-GCPVs
        """
        mlgcpvs = np.array([s.get_multilayer_gcpv(self.gcpv_layers) for s in storages])
        return mlgcpvs
    
    @staticmethod
    def _get_subtree_labels_frequencies(subtree_leafs_ids: Iterable[int],
                                        true_labels: np.ndarray
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate frequencies of labels in the subtree (potential cluster)

        Args:
            subtree_leafs_ids (Iterable[int]): ids of subtree leafs
            true_labels (np.ndarray): all true labels

        Returns:
            unique_lables (np.ndarray): unique labels in the subtree (potential cluster)
            label_freqs (np.ndarray): frequencies of unique labels in the subtree (potential cluster)
        """
        labels_in_subtree = true_labels[subtree_leafs_ids]

        unique_lables, unique_counts = np.unique(labels_in_subtree, return_counts=True)

        label_freqs = unique_counts / len(labels_in_subtree)

        return unique_lables, label_freqs

    @staticmethod
    def _get_subtree_node_ids(linkage_tree: hierarchy.ClusterNode) -> Tuple[List[int], List[int]]:
        """
        Return linkage_tree node ids 

        Args:
            linkage_tree (hierarchy.ClusterNode): tree

        Returns:
            subtree_leaf_ids (List[int]): list with only leaf node ids
            subtree_node_ids (List[int]): list with all node ids
        """

        subtree_leaf_ids = []
        subtree_node_ids = []
        
        def traverse_and_gather_ids(subtree: hierarchy.ClusterNode):

            if subtree.is_leaf():
                subtree_leaf_ids.append(subtree.get_id())
                subtree_node_ids.append(subtree.get_id())
            else:
                subtree_node_ids.append(subtree.get_id())

                if subtree.left:
                    traverse_and_gather_ids(subtree.left)

                if subtree.right:
                    traverse_and_gather_ids(subtree.right)

        traverse_and_gather_ids(linkage_tree)

        return subtree_leaf_ids, subtree_node_ids

    def _run_adaptive_clustering(self,
                                 linkage_tree: hierarchy.ClusterNode,
                                 true_labels: np.ndarray,
                                 ) -> Tuple[np.ndarray, List[List[int]], List[List[int]], List[int]]:
        """
        Perform adaptive clustering with min_purity and min_cluster_size constraints
        
        Args:
            linkage_tree (hierarchy.ClusterNode): tree
            true_labels (np.ndarray): true labels of samples for purity estimation

        Return:
            cluster_leaf_list (List[List[int]]): list of lists of leaf ids, where external list is clusters, internal lists are leaf ids
            cluster_node_list (List[List[int]]): list of lists of tree node ids, where external list is clusters, internal lists are tree node ids
            cluster_node_ids (List[int]): list of ids of cluster branch nodes
        """
        cluster_leaf_list = []  # only leaf-nodes
        cluster_node_list = []  # all nodes
        cluster_node_ids = []  # only node ids of cluster branches

        def traverse_and_cluster(linkage_tree: hierarchy.ClusterNode):

            # if subtree is a leaf -> attach the leaf index as a single cluster
            if linkage_tree.is_leaf():
                cluster_leaf_list.append([linkage_tree.get_id()])
                cluster_node_list.append([linkage_tree.get_id()])
                cluster_node_ids.append(linkage_tree.id)

            # otherwise check size and purity thresholds
            else:
                
                #subtree_leaf_ids = linkage_tree.pre_order(lambda leaf: leaf.id)
                subtree_leaf_ids, subtree_node_ids = self._get_subtree_node_ids(linkage_tree)
                # purity check
                _, label_freqs = self._get_subtree_labels_frequencies(subtree_leaf_ids, true_labels)
                purity = max(label_freqs)

                # size check
                if linkage_tree.count <= self.cluster_size_threshold:
                    cluster_leaf_list.append(subtree_leaf_ids)
                    cluster_node_list.append(subtree_node_ids)
                    cluster_node_ids.append(linkage_tree.id)

                # if purity is high -> write to dict and exit recursion
                elif purity >= self.cluster_purity_threshold:
                    cluster_leaf_list.append(subtree_leaf_ids)
                    cluster_node_list.append(subtree_node_ids)
                    cluster_node_ids.append(linkage_tree.id)

                # if checks passed -> decompose into subclusters / branches
                else:
                    if linkage_tree.left:
                        traverse_and_cluster(linkage_tree.left)

                    if linkage_tree.right:
                        traverse_and_cluster(linkage_tree.right)

        traverse_and_cluster(linkage_tree)

        return cluster_leaf_list, cluster_node_list, cluster_node_ids
    
    def _run_linkage_clustering(self,
                                linkage_tree: hierarchy.ClusterNode,
                                true_labels: np.ndarray,
                                ) -> Tuple[np.ndarray, List[List[int]], List[List[int]], List[int]]:
        """
        Perform linkage clustering with cluster selection by linkage distance
        
        Args:
            linkage_tree (hierarchy.ClusterNode): tree
            true_labels (np.ndarray): true labels of samples for purity estimation

        Return:
            cluster_leaf_list (List[List[int]]): list of lists of leaf ids, where external list is clusters, internal lists are leaf ids
            cluster_node_list (List[List[int]]): list of lists of tree node ids, where external list is clusters, internal lists are tree node ids
            cluster_node_ids (List[int]): list of ids of cluster branch nodes
        """
        cluster_leaf_list = []  # only leaf-nodes
        cluster_node_list = []  # all nodes
        cluster_node_ids = []  # only node ids of cluster branches

        def traverse_and_cluster(linkage_tree: hierarchy.ClusterNode):

            # if subtree is a leaf -> attach the leaf index as a single cluster
            if linkage_tree.is_leaf():
                cluster_leaf_list.append([linkage_tree.get_id()])
                cluster_node_list.append([linkage_tree.get_id()])
                cluster_node_ids.append(linkage_tree.id)

            # otherwise check size and purity thresholds
            else:
                
                #subtree_leaf_ids = linkage_tree.pre_order(lambda leaf: leaf.id)
                subtree_leaf_ids, subtree_node_ids = self._get_subtree_node_ids(linkage_tree)

                # size check
                if linkage_tree.dist <= self.cluster_linkage_threshold:
                    cluster_leaf_list.append(subtree_leaf_ids)
                    cluster_node_list.append(subtree_node_ids)
                    cluster_node_ids.append(linkage_tree.id)

                # if checks passed -> decompose into subclusters / branches
                else:
                    if linkage_tree.left:
                        traverse_and_cluster(linkage_tree.left)

                    if linkage_tree.right:
                        traverse_and_cluster(linkage_tree.right)

        traverse_and_cluster(linkage_tree)

        return cluster_leaf_list, cluster_node_list, cluster_node_ids

    def gcpv_dendogram(self,
                       gcpv_storages_dict: Dict[int, List[GCPVMultilayerStorage]],
                       save_name: str = 'gcpv_dendogram.pdf',
                       clustering_type: Literal['adaptive', 'linkage'] = 'adaptive'
                       ) -> None:
        """
        Draw a dendogram of hierarchial clustering of GCPVs

        Args:
            gcpv_storages (Dict[int, List[GCPVMultilayerStorage]]): GCPV storages per category for clustering

        Kwargs:
            save_name (str): name for plot saving in self.save_dir
            clustering_type (Literal['adaptive', 'linkage'] = 'adaptive'): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, while 'linkage' uses self.cluster_linkage_threshold
        """
        log_info("Plotting dendogram...")

        def color_map_fn(idx: int,
                         cluster_node_list: List[List[int]],
                         mpl_color_strings: List[str],
                         no_clust_color: str) -> str:

            for j, c in enumerate(cluster_node_list):
                if idx in c:
                    return mpl_color_strings[j]

            return no_clust_color

        selected_gcpv_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids = self._cluster(gcpv_storages_dict, clustering_type)

        # color strings initialiaztion
        # mpl_color_strings = ['#3e3636', '#5d5555', '#716868', '#999090', '#b9b0b0'] * 50
        mpl_color_strings = ['#3e3636', '#716868', '#b9b0b0'] * 50
        no_clust_color = '#efe5e5'
        leaf_colors = []
        [leaf_colors.extend([cs] * len(cll)) for cs, cll in zip(mpl_color_strings, cluster_leaf_list)]

        labels_int = [s.segmentation_category_id for s in selected_gcpv_storages]

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 2))
        dendogram = hierarchy.dendrogram(linkage, link_color_func=lambda k: color_map_fn(k, cluster_node_list, mpl_color_strings, no_clust_color), ax=ax)

        order = dendogram['leaves']
        reordered_labels_char = np.array([self.selected_tag_ids_and_colors[i] for i in labels_int])[order]

        max_y = linkage[:, 2].max()
        underline_1_y = max_y * 0.02
        underline_2_y = max_y * 0.05
        #y_bot_lim = min(max_y * 0.07, np.ceil(underline_2_y))
        y_bot_lim = min(max_y * 0.05, np.ceil(underline_1_y))

        # cluster_labels
        #ax.scatter(np.array(range(len(labels_int))) * 10 + 10, [-underline_1_y for _ in labels_int], c=leaf_colors, marker='s')
        # true labels
        ax.scatter(np.array(range(len(labels_int))) * 10 + 10, [-underline_1_y for _ in labels_int], c=reordered_labels_char, marker='s')
        ax.set_ylim(bottom=-y_bot_lim)

        #ax.plot(0, self.cluster_linkage_threshold, len(labels_int) * 10, self.cluster_linkage_threshold, linestyle = '--', c='black')
        # remove x labels with sample ids
        ax.set_xticks([])
        ax.set_xticklabels([])

        # add cluster markers
        cluster_node_ids_changed = {(i - len(linkage) - 1): j for j, i in enumerate(cluster_node_ids)}
        # non-leafes - non-negative idxs
        ii = np.argsort(np.array(dendogram['dcoord'])[:, 1])
        for j, (icoord, dcoord) in enumerate(zip(dendogram['icoord'], dendogram['dcoord'])):
            x = 0.5 * sum(icoord[1:3])
            y = dcoord[1]
            ind = np.nonzero(ii == j)[0][0]
            if ind in cluster_node_ids_changed.keys():
                cluster_id = cluster_node_ids_changed[ind]
                ax.plot(x, y, marker='X', color='black', markersize=10)
                ax.annotate(f"{cluster_id}", (x-15, y), va='bottom', ha='right', fontsize=15)
        # leafes - negative idxs
        for idx, cluster_id in cluster_node_ids_changed.items():
            if idx < 0:
                # leaf_idx = cluster_leaf_list[cluster_id][0]
                leaf_idx = idx + len(linkage) + 1
                leaf_pos = (10 * sum([len(cluster_leaf_list[i]) for i in range(cluster_id)])) + 5
                ax.plot(leaf_pos, 0, marker='X', color='black', markersize=10)
                ax.annotate(f"{cluster_id}", (leaf_pos-15, 0), va='bottom', ha='right', fontsize=15)


        legend_dict = {self.selected_tag_ids_and_colors[i]: self.selected_tag_id_names[i]  for i in self.selected_tag_ids}

        # Create a custom legend with square markers and category labels
        legend_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=label) for color, label in legend_dict.items()]

        # Add the custom legend under the plot
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(legend_dict), frameon=False)

        mkdir(self.save_dir)
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.show()
        plt.close()

    def gcpv_clustermap(self,
                        gcpv_storages_dict: Dict[int, List[GCPVMultilayerStorage]],
                        save_name: str = 'gcpv_clustermap.pdf',
                        clustering_type: Literal['adaptive', 'linkage'] = 'adaptive'
                        ) -> None:
        """
        Draw a dendogram of hierarchial clustering of GCPVs

        Args:
            gcpv_storages (Dict[int, List[GCPVMultilayerStorage]]): GCPV storages per category for clustering

        Kwargs:
            save_name (str): name for plot saving in self.save_dir
            clustering_type (Literal['adaptive', 'linkage'] = 'adaptive'): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, while 'linkage' uses self.cluster_linkage_threshold
        """
        log_info("Plotting clustermap...")

        def color_map_fn(idx: int,
                         cluster_node_list: List[List[int]],
                         mpl_color_strings: List[str],
                         no_clust_color: str) -> str:

            for j, c in enumerate(cluster_node_list):
                if idx in c:
                    return mpl_color_strings[j]

            return no_clust_color

        selected_gcpv_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids = self._cluster(gcpv_storages_dict, clustering_type)

        # color strings initialiaztion
        mpl_color_strings = ['dimgrey', 'darkgrey'] * 50
        no_clust_color = 'lightgrey'
        labels_int = [s.segmentation_category_id for s in selected_gcpv_storages]
        labels_char = [self.selected_tag_ids_and_colors[i] for i in labels_int]

        gcpv_data = self._get_mlgcpv(selected_gcpv_storages)

        dendogram = hierarchy.dendrogram(linkage, link_color_func=lambda k: color_map_fn(k, cluster_node_list, mpl_color_strings, no_clust_color), no_plot=True)

        distance_map = pairwise_distances(gcpv_data, metric=self.distance_metric)

        sns.clustermap(distance_map, cmap='vlag', row_linkage=linkage, col_linkage=linkage, row_colors=labels_char, col_colors=labels_char, tree_kws={'colors': dendogram['color_list']})
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def gcpv_cluster_projection(self,
                                propagator: Propagator,
                                img_folder: str,
                                img_name: str,
                                selected_clusters_id: Iterable[int] = None,
                                selected_bts: Iterable[float] = [0.75],
                                ) -> None:
        """
        Get projections of cluster GCPVs for input image. Saves image as './self.save_dir/projections_{img_name}'

        Args:
            propagator (Propagator): propagator of neural network to obtain activations
            img_folder (str): folder with image
            img_name (str): name of image
            

        Kwargs:
            selected_clusters_id: (Iterable[int] = None): selected clusters to build projections for 
            selected_bts (Iterable[float] = [0.75]): binarization thresholds to plot masks
        """
        log_info("Evaluating cluster projections...")

        if self.cluster_infos is None:
            raise ValueError("Train clusterer first, run cluster_gcpvs() first")
        else:
            acts_dict, preds = get_acts_preds_dict(propagator, img_folder, img_name)

            all_projections = []

            for i, cluster_info in enumerate(self.cluster_infos):
                if selected_clusters_id is not None:
                    if i not in selected_clusters_id:
                        continue
                cluster_projections = {l: get_projection(cluster_info.get_centroid_gcpv(l), acts_dict[l][0].numpy()) for l in self.gcpv_layers}
                all_projections.append(cluster_projections)
            
            self._save_image_cluster_projections(img_folder, img_name, all_projections, selected_clusters_id, selected_bts)

    def _save_image_cluster_projections(self,
                                        img_folder: str,
                                        img_name: str,
                                        all_projections: List[Dict[str, np.ndarray]],
                                        selected_clusters_id: Iterable[int] = None,
                                        selected_bts: Iterable[float] = [0.5],
                                        ) -> None:
        """
        Save projections of cluster GCPVs for input image as './self.save_dir/projections/projections_{img_name}' and in separate images in './self.save_dir/projections/imgs/'

        Args:
            img_folder (str): folder with image
            img_name (str): name of image
            all_projections (List[Dict[str, np.ndarray]]): results of GCPVs projecting on input image

        Kwargs:
            selected_clusters: (selected_clusters = None): selected clusters to build projections for 
            selected_bts (Iterable[float] = [0.75]): binarization thresholds to plot masks
        """
        mkdir(os.path.join(self.save_dir, "projections", 'imgs'))

        img = Image.open(os.path.join(img_folder, img_name))
        img_size = img.size

        selected_layers = self.gcpv_layers
        if selected_clusters_id is not None:
            selected_clusters = [self.cluster_infos[sc] for sc in selected_clusters_id]
        else:
            selected_clusters = self.cluster_infos
        
        nrows = len(selected_clusters) + 1
        ncols = len(selected_bts) + 3

        fig, axs = plt.subplots(ncols, nrows, figsize=(nrows * 2, ncols * 2), dpi=100)

        # original image: [0,0] axis
        axs[0,0].imshow(np.array(img))
        axs[0,0].axis('off')

        # index: continious masked images
        axs[1,0].text(0.5, 0.5, 'Heatmaps', ha='center', va='center')
        axs[1,0].set_frame_on(False)
        axs[1,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # index: continious masks
        axs[2,0].text(0.5, 0.5, 'Continious masks', ha='center', va='center')
        axs[2,0].set_frame_on(False)
        axs[2,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # index: BTs
        for y, bt in enumerate(selected_bts):
            axs[y+3,0].text(0.5, 0.5, f'BT={bt}', ha='center', va='center')
            axs[y+3,0].set_frame_on(False)
            axs[y+3,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # header: cluster labels
        for x, c in enumerate(selected_clusters):
            cluster_category_probs = c.cluster_category_probabilities
            cluster_number = selected_clusters_id[x] if selected_clusters_id is not None else x
            pred_str = f'Cluster {cluster_number}:'
            for cat, prob in cluster_category_probs.items():
                pred_str += f'\n{self.selected_tag_id_names[cat]} - {prob:.2f}'
            axs[0,x+1].text(0.0, 0.0, pred_str, ha='left', va='bottom')
            axs[0,x+1].set_frame_on(False)
            axs[0,x+1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # combined projections
        for x, c in enumerate(selected_clusters):
            projections = all_projections[x]
            avg_projection = combine_masks(list(projections.values()))
            hm = apply_heatmap(np.array(img), avg_projection / 255.)
            #proj_img_arr = np.array(blend_imgs(img, get_colored_mask(avg_projection, [1], mask_value_multiplier=1)))
            axs[1,x+1].imshow(hm)
            axs[1,x+1].axis('off')
            hm_img = Image.fromarray(hm)
            tags_str = '_'.join(list(self.selected_tag_id_names.values()))
            if len(selected_clusters) == 1:
                x = 'single'
            hm_img.save(os.path.join(self.save_dir, "projections", 'imgs',  f"gcpv_heatmap_{tags_str}_{x}_{img_name}"))
            
        # combined projections
        for x, c in enumerate(selected_clusters):
            projections = all_projections[x]
            avg_projection = combine_masks(list(projections.values()))
            rgb_proj = get_rgb_binary_mask(avg_projection, img_size)
            axs[2,x+1].imshow(rgb_proj)
            axs[2,x+1].axis('off')
            hm_img = Image.fromarray(rgb_proj)
            tags_str = '_'.join(list(self.selected_tag_id_names.values()))
            if len(selected_clusters) == 1:
                x = 'single'
            hm_img.save(os.path.join(self.save_dir, "projections", 'imgs', f"gcpv_rgb_map_{tags_str}_{x}_{img_name}"))

        for y, bt in enumerate(selected_bts):
            for x, c in enumerate(selected_clusters):
                projections = all_projections[x]
                avg_projection = combine_masks(list(projections.values()))
                mask = avg_projection  / 255. 
                mask = normalize_0_to_1(mask)
                mask = resize(mask, (np.array(img).shape[0], np.array(img).shape[1]))
                binary_mask = mask >= bt
                masked_img = apply_mask(np.array(img), mask, bt, crop_around_mask=False)
                blended_img = blend_imgs(Image.fromarray(masked_img), img, 0.4)
                blended_img = add_countours_around_mask(np.array(blended_img), binary_mask.astype(np.uint8) * 255)
                #proj_img_arr = np.array(blend_imgs(img, get_colored_mask(projection, [1], mask_value_multiplier=1)))
                axs[y+3,x+1].imshow(blended_img)
                axs[y+3,x+1].axis('off')
                tags_str = '_'.join(list(self.selected_tag_id_names.values()))
                if len(selected_clusters) == 1:
                    x = 'single'
                Image.fromarray(blended_img).save(os.path.join(self.save_dir, "projections", 'imgs', f"gcpv_mask_{tags_str}_{x}_{bt}_{img_name}"))

        """# projections
        for y, l in enumerate(selected_layers):
            for x, c in enumerate(selected_clusters):
                projection = all_projections[x][l]
                hm = apply_heatmap(np.array(img), projection / 255.)
                #proj_img_arr = np.array(blend_imgs(img, get_colored_mask(projection, [1], mask_value_multiplier=1)))
                axs[y+2,x+1].imshow(hm)
                axs[y+2,x+1].axis('off')"""

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'projections_{img_name}'))
        plt.show()
        plt.close()


class GCPVClustererManyLoaders:

    def __init__(self,
                 clusterer: GCPVClusterer,
                 tagged_loaders: Dict[str, GCPVMultilayerStorageDirectoryLoader],
                 save_base_dir: str = './experiment_outputs/gcpv_clustering_many',
                 ) -> None:
        """
        Args:
            clusterer (GCPVClusterer): clusterer instance
            tagged_loaders (Dict[str, GCPVMultilayerStorageDirectoryLoader]): GCPV loaders with string tags
        
        Kwargs:
            save_base_dir (str = './experiment_outputs/gcpv_clustering_many'): base directory for saving, folders with tag names will be created for individual results
        """
        self.clusterer = clusterer
        self.tagged_loaders = tagged_loaders
        self.save_base_dir = save_base_dir

        self.common_files = self._get_common_files()

        # set loaders to have same files
        for storage in self.tagged_loaders.values():
            storage.pkl_file_names = self.common_files
        

    def _get_common_files(self) -> List[str]:
        """
        Get list of common files in all arrays for a fair comparison (due to different areas of segmentations and segmentation area thresholding, lists may be different)

        Returns:
            common_files (List[str]): common file names list across all self.tagged_loaders
        """
        files = [s.pkl_file_names for s in self.tagged_loaders.values()]

        common_files = set(files[0])

        for array in files[1:]:
            common_files.intersection_update(array)

        return sorted(list(common_files))
    
    def run_clustering(self, data_categories: List[int], train_split: float = 0.8):
        """
        Run clustering for all Loaders with given sampling parameters

        Args:
            allowed_categories (Iterable[int]): load only GCPV storage of allowed categories ids, None to load all

        Kwargs:            
            test_size (float = 0.2): size of 'test' split
        """
        for tag, loader in self.tagged_loaders.items():

            log_info(f"Loader with tag '{tag}'")

            self.clusterer.save_dir = os.path.join(self.save_base_dir, tag)

            gcpv_storages_train, gcpv_storages_test = loader.load_train_test_splits(data_categories, train_split)

            clustered_samples = self.clusterer.cluster_gcpvs(gcpv_storages_train)

            #self.clusterer.cluster_stats()
            #self.clusterer.gcpv_cluster_projection(yolo5, in_dir_imgs, '000000338986.jpg')
            #predictions_knn = self.clusterer.predict_gcpvs_with_knn(gcpv_storages_test, 11)
            #predictions_centroids = self.clusterer.predict_gcpvs_with_centroids(gcpv_storages_test)

            self.clusterer.gcpv_clustermap(gcpv_storages_train)
            self.clusterer.gcpv_dendogram(gcpv_storages_train)

