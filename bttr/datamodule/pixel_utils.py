import logging
import numpy as np
import os
import pandas as pd
import time
from multiprocessing import Pool
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from typing import Tuple


def get_effective_dims(sparse_pixels: np.ndarray) -> Tuple[int, int]:
    x_min = np.min(sparse_pixels[:, 0])
    x_max = np.max(sparse_pixels[:, 0])
    y_min = np.min(sparse_pixels[:, 1])
    y_max = np.max(sparse_pixels[:, 1])

    w = x_max - x_min
    h = y_max - y_min

    return h, w


def split_subimage_kd(subimage: np.ndarray) -> np.ndarray:
    if subimage.shape[0] == 1:
        return subimage

    h, w = get_effective_dims(subimage)
    l = subimage.shape[0]

    if h > w:                       # tall subimage, split by horizontal plane
        ind = np.argpartition(subimage[:, 1], l // 2)
    else:                           # fat subimage, split by vertical plane
        ind = np.argpartition(subimage[:, 0], l // 2)
    subimage[:] = subimage[ind]

    subimage[:l//2] = split_subimage_kd(subimage[:l//2])
    subimage[l//2:] = split_subimage_kd(subimage[l//2:])

    return subimage


def is_correct_order(l_child, r_child):
    l_centroid = np.mean(l_child, axis=0)
    r_centroid = np.mean(r_child, axis=0)

    h, w = np.abs(l_centroid - r_centroid)
    if h > w:
        return l_centroid[0] < r_centroid[0]
    else:
        return l_centroid[1] < r_centroid[1]


def split_subimage_binary_cluster(subimage: np.ndarray, pixel_count, min_pixels, linkage):
    if pixel_count == min_pixels:
        subimage = split_subimage_kd(subimage)

    else:
        clustering = AgglomerativeClustering(
            n_clusters=2,
            affinity="euclidean",
            linkage=linkage
        ).fit(subimage)
        labels = clustering.labels_

        if np.count_nonzero(labels == 1) <= 1 or np.count_nonzero(labels == 0) <= 1:
            subimage = split_subimage_kd(subimage)

        else:
            subimage = subimage[np.argsort(labels)]
            pixels_per_cluster = pixel_count // 2
            pivot_1 = np.count_nonzero(labels == -1)
            pivot_2 = pivot_1 + np.count_nonzero(labels == 0)

            l_child = subimage[pivot_1:pivot_2]
            r_child = subimage[pivot_2:]

            if not is_correct_order(l_child, r_child):
                l_child, r_child = r_child, l_child

            np.random.shuffle(l_child)
            np.random.shuffle(r_child)
            if l_child.shape[0] < pixels_per_cluster:
                l_child = np.tile(l_child, (pixels_per_cluster // len(l_child) + 1, 1))
            if r_child.shape[0] < pixels_per_cluster:
                r_child = np.tile(r_child, (pixels_per_cluster // len(r_child) + 1, 1))

            l_child = l_child[:pixels_per_cluster]
            r_child = r_child[:pixels_per_cluster]
            l_child = split_subimage_binary_cluster(l_child, pixels_per_cluster, min_pixels, linkage)
            r_child = split_subimage_binary_cluster(r_child, pixels_per_cluster, min_pixels, linkage)
            subimage = np.concatenate([l_child, r_child], axis=0)

    return subimage


def sample_and_split(sparse_pixels, max_pixels):
    pixel_count = sparse_pixels.shape[0]
    if pixel_count < max_pixels:
        sparse_pixels = np.tile(sparse_pixels, (max_pixels // len(sparse_pixels) + 1, 1))

    np.random.shuffle(sparse_pixels)
    sparse_pixels = sparse_pixels[:max_pixels]
    # sparse_pixels = split_subimage_kd(sparse_pixels)
    sparse_pixels = split_subimage_binary_cluster(sparse_pixels, max_pixels, min_pixels=32, linkage="ward")

    return sparse_pixels


def _get_n_slots(_labels, n_clusters):
    n_slots = 0
    if _labels[0] % 2 == 1:
        n_slots += 1
    for j in range(1, len(_labels)):
        if (_labels[j] - _labels[j-1]) % 2 == 0:
            n_slots += 1
    if (n_clusters - _labels[-1]) % 2 == 0:
        n_slots += 1
    return n_slots


def get_double_clusters(labels_with_counts, n_clusters, n_double):
    clusters_to_double_sample = []
    i = 0
    count = 0

    while count < n_double:
        if labels_with_counts[i][0] == -1:
            i += 1
            continue

        tmp_labels = clusters_to_double_sample.copy()
        tmp_labels.append(labels_with_counts[i][0])
        tmp_labels = sorted(tmp_labels)

        n_slots = _get_n_slots(tmp_labels, n_clusters)
        remaining_slots = n_double - (count + 1)

        i += 1

        if n_slots <= remaining_slots:
            clusters_to_double_sample = tmp_labels
            count += 1

    return clusters_to_double_sample


def sort_to_relabel(sparse_pixels, labels):
    unique, counts = np.unique(labels, return_counts=True)
    end_indices = np.cumsum(counts)
    start_indices = [0]
    start_indices.extend(end_indices[:-1])

    centroids = [np.mean(sparse_pixels[start:end], axis=0)
                 for start, end in zip(start_indices, end_indices)]
    centroids = np.asarray(centroids)

    forward_order = np.argsort(centroids[:, 1])
    reverse_order = np.argsort(forward_order)

    for start, end, order in zip(start_indices, end_indices, reverse_order):
        labels[start:end] = order

    return labels, forward_order, reverse_order


def split_subimage_hierarchical_cluster(sparse_pixels, max_pixels, start_eps=2, min_samples=16):
    for eps in range(start_eps, 10):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(sparse_pixels)
        labels = clustering.labels_
        n_clusters = max(labels) + 1
        if n_clusters <= 64 and np.count_nonzero(labels == -1) < 30:
            break
    # assert -1 not in labels, np.count_nonzero(labels == -1)

    # find the smallest 2-power
    for p in range(12):
        if 2 ** p >= n_clusters:
            break

    sample_per_clusters = max_pixels // (2 ** p)
    n_double_sample = 2 ** p - n_clusters
    noise_count = np.count_nonzero(labels == -1)

    sorted_pixels = sparse_pixels[np.argsort(labels)]
    sorted_pixels = sorted_pixels[noise_count:]

    labels = np.sort(labels)
    labels = labels[noise_count:]
    labels, forward_order, reverse_order = sort_to_relabel(sorted_pixels, labels)
    sorted_pixels = sorted_pixels[np.argsort(labels)]

    unique, counts = np.unique(labels, return_counts=True)
    labels_with_counts = sorted(list(zip(unique, counts)), key=lambda _tup: _tup[1], reverse=True)
    end_indices = np.cumsum(counts)
    start_indices = [0]
    start_indices.extend(end_indices[:-1])

    clusters_to_double_sample = get_double_clusters(labels_with_counts, n_clusters, n_double_sample)

    pixels_kd_list = []
    for label, start, end in zip(unique, start_indices, end_indices):
        if label == -1:             # drop the noise
            continue

        pixels_in_i = sorted_pixels[start:end].copy()
        # print(f"cluster {i}, len pix: {pix_i.shape}", end=",")
        if label in clusters_to_double_sample:
            pixels_kd = sample_and_split(pixels_in_i, max_pixels=sample_per_clusters * 2)
        else:
            pixels_kd = sample_and_split(pixels_in_i, max_pixels=sample_per_clusters)
        pixels_kd_list.append(pixels_kd)
        # print(pix_kd.shape)

    clustered_pixels = np.concatenate(pixels_kd_list, axis=0)
    assert clustered_pixels.shape == (max_pixels, 2), \
        f"Wrong pixel shape: {clustered_pixels.shape}\n" \
        f"labels_with_counts: {labels_with_counts}\n" \
        f"counts: {counts}\n" \
        f"start_indices: {start_indices}\n" \
        f"end_indices: {end_indices}" \
        f"shapes: {[pixels_kd.shape[0] for pixels_kd in pixels_kd_list]}"

    return clustered_pixels


def cluster_split(sparse_pixels, cluster_type="hierarchical", max_pixels=4096, min_pixels=32, linkage="ward"):
    if cluster_type == "binary":
        sparse_pixels = split_subimage_binary_cluster(sparse_pixels, max_pixels, min_pixels, linkage)

    elif cluster_type == "hierarchical":
        sparse_pixels = split_subimage_hierarchical_cluster(sparse_pixels, max_pixels, start_eps=2, min_samples=16)

    else:
        raise NotImplementedError

    return sparse_pixels


def sparsify_with_cluster_split(image: np.ndarray, max_pixels=4096):
    image = np.asarray(image)
    image = 255 - image
    assert len(image.shape) == 2

    sparse_pixels = np.asarray(np.where(image == 0), dtype=np.uint16).T
    sparse_pixels = cluster_split(
        sparse_pixels, cluster_type="hierarchical",
        max_pixels=max_pixels, min_pixels=32, linkage="ward"
    )
    sparse_pixels = sparse_pixels.astype(np.float32)

    h, w = image.shape
    sparse_pixels[:, 0] = sparse_pixels[:, 0] / h
    sparse_pixels[:, 1] = sparse_pixels[:, 1] / w

    return sparse_pixels
