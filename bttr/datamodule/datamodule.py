import numpy as np
import os
import pytorch_lightning as pl
import torch
from dataclasses import dataclass
from multiprocessing import Pool
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from typing import List, Optional, Tuple
from zipfile import ZipFile

from .vocab import CROHMEVocab

vocab = CROHMEVocab()

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 32e5  # change here according to your GPU memory


# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    # data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:
        # size = fea.size[0] * fea.size[1]
        # fea = transforms.ToTensor()(fea)
        size = 4096
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[1]} x {fea.shape[2]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def get_effective_dims(sparse_pixels: np.ndarray) -> Tuple[int, int]:
    x_min = np.min(sparse_pixels[:, 0])
    x_max = np.max(sparse_pixels[:, 0])
    y_min = np.min(sparse_pixels[:, 1])
    y_max = np.max(sparse_pixels[:, 1])

    w = x_max - x_min
    h = y_max - y_min

    return h, w


def split_subimage_kd(subimage: np.ndarray, depth: int) -> None:           # "inplace"
    if depth == 0:
        return

    h, w = get_effective_dims(subimage)
    l = subimage.shape[0]

    if h > w:                       # tall subimage, split by horizontal plane
        ind = np.argpartition(subimage[:, 1], l // 2)
    else:                           # fat subimage, split by vertical plane
        ind = np.argpartition(subimage[:, 0], l // 2)
    subimage[:] = subimage[ind]

    split_subimage_kd(subimage[:l//2], depth - 1)
    split_subimage_kd(subimage[l//2:], depth - 1)


def sparsify(image):
    image = np.asarray(image)
    image = 255 - image
    assert len(image.shape) == 2

    pixel_count = np.sum(image == 0)
    sparse_pixels = np.asarray(np.where(image == 0), dtype=np.float32).T

    max_pixels = 4096
    if pixel_count > max_pixels:
        np.random.shuffle(sparse_pixels)
    else:
        np.random.shuffle(sparse_pixels)
        sparse_pixels = np.tile(sparse_pixels,
                                (max_pixels // len(sparse_pixels) + 1, 1))

    pixel_count = max_pixels
    sparse_pixels = sparse_pixels[:pixel_count]
    split_subimage_kd(sparse_pixels, depth=12)

    h, w = image.shape
    sparse_pixels[:, 0] = sparse_pixels[:, 0] / h
    sparse_pixels[:, 1] = sparse_pixels[:, 1] / w

    return sparse_pixels


def extract_data(archive: ZipFile, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    print(f"Loading from {dir_name}/caption.txt")

    img_names = []
    imgs = []
    formulas = []
    for line in tqdm(captions):
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"{dir_name}/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()

        img_names.append(img_name)
        imgs.append(img)
        formulas.append(formula)

    p = Pool(20)
    sparse_pixels = []
    for sparse_img in tqdm(p.imap(sparsify, imgs)):
        sparse_pixels.append(sparse_img)

    p.close()
    p.join()

    data = list(zip(img_names, sparse_pixels, formulas))
    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


@dataclass
class Batch:
    img_bases: List[str]            # [b,]
    imgs: torch.Tensor              # [b, t, 2]
    indices: List[List[int]]        # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    x = []
    for image in images_x:
        x.append(torch.as_tensor(image, dtype=torch.float32))
    x = torch.stack(x)

    return Batch(fnames, x, seqs_y)


def build_dataset(archive, folder: str, batch_size: int):
    data = extract_data(archive, folder)
    return data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        test_year: str = "2014",
        batch_size: int = 8,
        num_workers: int = 5,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = build_dataset(archive, "train", self.batch_size)
                self.val_dataset = build_dataset(archive, self.test_year, 1)
            if stage == "test" or stage is None:
                self.test_dataset = build_dataset(archive, self.test_year, 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
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
