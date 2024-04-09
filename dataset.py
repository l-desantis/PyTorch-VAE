import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import pandas as pd
from utils import pkl_loader
from sklearn.model_selection import train_test_split
import numpy as np
import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MazeDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 # transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / "maze/dataset.pkl"
        # self.transforms = transform
        mazes = pkl_loader(self.data_dir)
        mazes_split = train_test_split(mazes, test_size=0.25, random_state=42)
        self.mazes = mazes_split[0] if split == "train" else mazes_split[1]

    def __len__(self):
        return len(self.mazes)

    def __getitem__(self, idx):
        maze = self.mazes[idx]
        maze = np.concatenate((np.concatenate((maze, np.zeros((1, 63))), axis=0), np.zeros((64, 1))), axis=1).astype(dtype=np.float32)
        maze = np.expand_dims(maze, axis=0)
        maze = torch.from_numpy(maze)

        # if self.transforms is not None:
        #     img = self.transforms(img)

        return maze, torch.tensor(0.0, dtype=torch.float)  # dummy datat to prevent breaking


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            maze: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.maze = maze
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  OxfordPets Dataset  =========================

        #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.CenterCrop(self.patch_size),
        # #                                               transforms.Resize(self.patch_size),
        #                                               transforms.ToTensor(),
        #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                             transforms.CenterCrop(self.patch_size),
        # #                                             transforms.Resize(self.patch_size),
        #                                             transforms.ToTensor(),
        #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         self.train_dataset = OxfordPets(
        #             self.data_dir,
        #             split='train',
        #             transform=train_transforms,
        #         )

        #         self.val_dataset = OxfordPets(
        #             self.data_dir,
        #             split='val',
        #             transform=val_transforms,
        #         )

        #       =========================  CelebA Dataset  =========================

        if not self.maze:

            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.CenterCrop(148),
                                                  transforms.Resize(self.patch_size),
                                                  transforms.ToTensor(),])

            val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(self.patch_size),
                                                transforms.ToTensor(),])

            self.train_dataset = MyCelebA(
                self.data_dir,
                split='train',
                transform=train_transforms,
                download=False,
            )

            # Replace CelebA with your dataset
            self.val_dataset = MyCelebA(
                self.data_dir,
                split='test',
                transform=val_transforms,
                download=False,
            )

        # =========================  Mazes Dataset  =========================

        #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.CenterCrop(self.patch_size),
        # #                                               transforms.Resize(self.patch_size),
        #                                               transforms.ToTensor(),
        #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        #
        #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                             transforms.CenterCrop(self.patch_size),
        # #                                             transforms.Resize(self.patch_size),
        #                                             transforms.ToTensor(),
        #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        else:

            self.train_dataset = MazeDataset(
                self.data_dir,
                split='train',
                # transform=train_transforms,
            )

            self.val_dataset = MazeDataset(
                self.data_dir,
                split='val',
                # transform=val_transforms,
            )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True
        )
