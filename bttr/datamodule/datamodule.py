import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader

from .vocab import CROHMEVocab

vocab = CROHMEVocab()
MAX_SIZE = 32e4  # change here according to your GPU memory


@dataclass
class PixelBatch:
    img_bases: List[str]            # [b,]
    imgs: torch.Tensor              # [b, L]
    img_shapes: torch.Tensor        # [b, 2]
    pixel_counts: torch.Tensor      # [b]
    indices: List[List[int]]        # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "PixelBatch":
        return PixelBatch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            img_shapes=self.img_shapes.to(device),
            pixel_counts=self.pixel_counts.to(device),
            indices=self.indices,
        )


def sparse_data_iterator(
    data: List,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE
) -> List[PixelBatch]:
    batches = []
    biggest_image_size = 0
    fnames = []
    imgs = []
    img_shapes = []
    pixel_counts = []
    labels = []

    data.sort(key=lambda x: x[2][0] * x[2][1])

    i = 0
    for fname, img, img_shape, pixel_count, lab in data:
        size = img_shape[0] * img_shape[1]
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(f"image: {fname} size: {img_shape[0]} x {img_shape[1]} =  bigger than {maxImagesize}, ignore")
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:       # a batch is full
                max_pixels = np.max(pixel_counts)
                imgs = [np.pad(img, ((0, max_pixels - len(img)), (0, 0))) for img in imgs]
                imgs = np.stack(imgs, axis=0)
                imgs = torch.as_tensor(imgs, dtype=torch.float)

                img_shapes = np.stack(img_shapes, axis=0)
                img_shapes = torch.as_tensor(img_shapes, dtype=torch.long)
                pixel_counts = torch.as_tensor(pixel_counts, dtype=torch.long)
                batch = PixelBatch(
                    img_bases=fnames,
                    imgs=imgs,
                    img_shapes=img_shapes,
                    pixel_counts=pixel_counts,
                    indices=labels
                )
                batches.append(batch)

                i = 1
                biggest_image_size = size
                fnames = [fname]
                imgs = [img]
                img_shapes = [img_shape]
                pixel_counts = [pixel_count]
                labels = [vocab.words2indices(lab)]

            else:
                fnames.append(fname)
                imgs.append(img)
                img_shapes.append(img_shape)
                pixel_counts.append(pixel_count)
                labels.append(vocab.words2indices(lab))
                i += 1
    # lastly
    max_pixels = np.max(pixel_counts)
    imgs = [np.pad(img, ((0, max_pixels - len(img)), (0, 0))) for img in imgs]
    imgs = np.stack(imgs, axis=0)
    imgs = torch.as_tensor(imgs, dtype=torch.float)

    img_shapes = np.stack(img_shapes, axis=0)
    img_shapes = torch.as_tensor(img_shapes, dtype=torch.long)
    pixel_counts = torch.as_tensor(pixel_counts, dtype=torch.long)
    batch = PixelBatch(
        img_bases=fnames,
        imgs=imgs,
        img_shapes=img_shapes,
        pixel_counts=pixel_counts,
        indices=labels
    )
    batches.append(batch)
    print("total ", len(batches), "batch data loaded")

    return batches


def len2idx(lens) -> np.ndarray:
    end_indices = np.cumsum(lens)
    start_indices = np.concatenate([[0], end_indices[:-1]], axis=0)
    indices = np.stack([start_indices, end_indices], axis=1)

    return indices


def build_sparse_dataset(processed_data_path, folder: str, batch_size: int) -> List[PixelBatch]:
    dir_name = os.path.join(processed_data_path, folder)
    with open(os.path.join(dir_name, "caption.txt"), "r") as f:
        captions = f.readlines()

    feat = np.load(os.path.join(dir_name, f"{folder}.npz"))
    sparse_pixels = feat["sparse_pixels"].astype(np.float32)
    pixel_counts = feat["pixel_counts"].astype(np.int64)
    image_shapes = feat["image_shapes"].astype(np.int64)
    pixel_indices = len2idx(pixel_counts)

    data = []
    for i, line in enumerate(captions):
        tmp = line.strip().split()
        img_name = tmp[0]
        formula = tmp[1:]

        start, end = pixel_indices[i]
        img = sparse_pixels[start:end].copy()
        pixel_count = pixel_counts[i]
        img_shape = image_shapes[i]

        data.append((img_name, img, img_shape, pixel_count, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return sparse_data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        processed_data_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../processed",
        test_year: str = "2014",
        batch_size: int = 8,
        num_workers: int = 5,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.processed_data_path = processed_data_path
        self.test_year = test_year
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        print(f"Load data from: {self.processed_data_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = build_sparse_dataset(self.processed_data_path, "train", self.batch_size)
            self.val_dataset = build_sparse_dataset(self.processed_data_path, self.test_year, 1)
        if stage == "test" or stage is None:
            self.test_dataset = build_sparse_dataset(self.processed_data_path, self.test_year, 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda _batch: _batch[0]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda _batch: _batch[0]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda _batch: _batch[0]
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    batch_size = 2

    parser = ArgumentParser()
    parser = CROHMEDatamodule.add_argparse_args(parser)

    args = parser.parse_args(["--batch_size", f"{batch_size}"])

    dm = CROHMEDatamodule(**vars(args))
    dm.setup()

    train_loader = dm.train_dataloader()
    for img, mask, tgt, output in train_loader:
        break
