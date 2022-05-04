import argparse
import numpy as np
import os
import skimage.io
import sys
import time
import torch
from multiprocessing import Pool
from typing import Tuple


def get_preprocess_parser():
    parser = argparse.ArgumentParser("sparsify")
    parser.add_argument("--num_workers", help="No. of workers", type=int, default=40)
    parser.add_argument('--max_pixels', type=int, default=4096)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--sort_by', type=str, default="kd-tree")
    parser.add_argument("--processed_data_path", type=str)

    return parser


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
    pivot = int(2 ** np.floor(np.log2(l - 1)))

    if h > w:                                   # tall subimage, split by horizontal plane
        ind = np.argpartition(subimage[:, 1], pivot)
    else:                                       # fat subimage, split by vertical plane
        ind = np.argpartition(subimage[:, 0], pivot)
    subimage[:] = subimage[ind]

    subimage[:pivot] = split_subimage_kd(subimage[:pivot])
    subimage[pivot:] = split_subimage_kd(subimage[pivot:])

    return subimage


def sparsify(image: np.ndarray, max_pixels: int, color_threshold: int = 0,
             kernel_size: int = 3, sort: str = "kd-tree"):
    if len(image.shape) == 3:
        assert np.array_equal(image[:, :, 0], image[:, :, 1])
        assert np.array_equal(image[:, :, 0], image[:, :, 2])
        image = image[:, :, 0]
    else:
        assert len(image.shape) == 2

    h, w = image.shape
    pixel_count = np.sum(image <= color_threshold)
    sparse_pixels = np.asarray(np.where(image <= color_threshold), dtype=np.uint16).T

    np.random.shuffle(sparse_pixels)
    if pixel_count > max_pixels:
        pixel_count = max_pixels
    else:
        pixel_count = pixel_count - pixel_count % 16
    sparse_pixels = sparse_pixels[:pixel_count]

    if sort == "kd-tree":
        sparse_pixels = split_subimage_kd(sparse_pixels)
    else:
        raise NotImplementedError

    features = []
    for x, y in sparse_pixels:
        feature = []
        for i in range(x - kernel_size // 2, x + kernel_size // 2 + 1):
            for j in range(y - kernel_size // 2, y + kernel_size // 2 + 1):
                if i < 0 or i >= h or j < 0 or j >= w:
                    feature.append(0.0)
                else:
                    feature.append(1 - image[i, j] / 255)

                if i == x and j == y:
                    assert image[i, j] == 0
        features.append(feature)
    features = np.array(features, dtype=np.float32)

    sparse_pixels = sparse_pixels.astype(np.float32)
    sparse_pixels = np.concatenate([sparse_pixels, features], axis=1)

    return sparse_pixels, pixel_count, image.shape


def crohme_helper(_args):
    i, fn, max_pixels, kernel_size, sort_by = _args
    if i > 0 and i % 200 == 0:
        print(f"Processing {i}th image")

    image = skimage.io.imread(fn)
    image = 255 - image             # invert black and white

    sparse_pixels, pixel_count, image_shape = sparsify(
        image, max_pixels, color_threshold=0, kernel_size=kernel_size, sort=sort_by)

    return sparse_pixels, pixel_count, image_shape


def get_crohme_features(args, folder, captions):
    print("Getting image features")
    start = time.time()

    p = Pool(args.num_workers)
    image_features = p.imap(
        crohme_helper,
        ((i, f"data/{folder}/{line.strip().split()[0]}.bmp",
          args.max_pixels, args.kernel_size, args.sort_by) for i, line in enumerate(captions))
    )

    p.close()
    p.join()

    image_features = list(image_features)
    print(f"Done graph featurization, time: {time.time() - start}. Collating and saving...")
    sparse_pixels, pixel_counts, image_shapes = zip(*image_features)

    return sparse_pixels, pixel_counts, image_shapes


def sparsify_main(args):
    os.makedirs(args.processed_data_path, exist_ok=True)

    for folder in ["train", "2014", "2016", "2019"]:
        os.makedirs(os.path.join(args.processed_data_path, folder), exist_ok=True)

        with open(f"data/{folder}/caption.txt", "r") as f:
            captions = f.readlines()
        sparse_pixels, pixel_counts, image_shapes = get_crohme_features(args, folder, captions)

        sparse_pixels = np.concatenate(sparse_pixels, axis=0)
        pixel_counts = np.array(pixel_counts, dtype=np.int32)
        image_shapes = np.stack(image_shapes, axis=0)

        print(f"sparse_pixels: {sparse_pixels.shape}")
        print(f"pixel_counts: {pixel_counts.shape}, total: {np.sum(pixel_counts)}")
        print(f"image_shapes: {image_shapes.shape}")

        output_file = os.path.join(args.processed_data_path, folder, "caption.txt")
        with open(output_file, "w") as of:
            of.writelines(captions)

        output_file = os.path.join(args.processed_data_path, folder, f"{folder}.npz")
        print(f"Saving into {output_file}")
        np.savez(
            output_file,
            sparse_pixels=sparse_pixels,
            pixel_counts=pixel_counts,
            image_shapes=image_shapes
        )


if __name__ == "__main__":
    preprocess_parser = get_preprocess_parser()
    args = preprocess_parser.parse_args()

    # set random seed
    # set_seed(args.seed)

    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")

    sparsify_main(args)
