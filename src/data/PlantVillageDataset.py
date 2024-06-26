from __future__ import print_function, division

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wget as wget
from skimage import io
from torch.utils.data import Dataset
from enum import Enum
from PIL import Image

# Ignore warnings
warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


class PlantVillageStatus(Enum):
    HEALTHY = 1
    DISEASE = 2


class PlantVillage(Dataset):
    """PlantVillage data."""

    def __init__(self,
                 csv_file,
                 root_dir,
                 transform=None,
                 status=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 2],
                                self.df.iloc[idx, 1])

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        res = {'image': image, 'label': 0 if self.df.iloc[idx, 2] == 'healthy' else 1}

        return res

    @staticmethod
    def create_csv(path):
        """
        :type path: Path to image folder
        """
        print('CSV creation...')

        data = []

        for status in next(os.walk(path))[1]:
            for img in next(os.walk(os.path.join(path, status)))[2]:
                obj = {'Name': img, 'Status': status}
                data.append(obj)

                print(os.path.join(path, status, img))

        df = pd.DataFrame(data)
        df.to_csv(path + '/data.csv')
        print('CSV successfully created')
