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


class MendeleyPlant(Enum):
    ALSTONIA_SCHOLARIS = 1
    ARJUN = 2
    BAEL = 3
    BASIL = 4
    CHINAR = 5
    GAUVA = 6
    JAMUN = 7
    JATROPHA = 8
    LEMON = 9
    MANGO = 10
    POMEGRANATE = 11
    PONGAMIA_PINNATA = 12


class MendeleyDataset(Dataset):
    """Mendeley data."""

    def __init__(self,
                 csv_file,
                 root_dir,
                 transform=None,
                 healthy_only=False,
                 plants=list(MendeleyPlant),
                 validation=False,
                 validationSplit=0):
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
        self.healthy_only = healthy_only
        self.plants = plants
        self.validation = validation
        self.validationSplit = validationSplit

        if healthy_only:
            self.df = self.df[self.df.Status == 'healthy']

        diff = list(set(list(MendeleyPlant)) - set(plants))
        for plant in diff:
            self.df = self.df[~(self.df.PlantCode == plant.value)]

        if self.validation:
            self.df = self.df.tail(int(self.validationSplit * len(self.df)))
        else:
            self.df = self.df.head(int((1 - self.validationSplit) * len(self.df)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 2],
                                self.df.iloc[idx, 4],
                                self.df.iloc[idx, 1])

        image = Image.open(img_name)

        #landmarks = self.df.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            image = self.transform(image)

        return image

    @staticmethod
    def csvExists():
        return os.path.exists('../data/mendeley/mendeley.csv')

    @staticmethod
    def create_csv(path):
        """
        :type path: Path to image folder
        """
        print('CSV creation...')

        columns = ['No.', 'Name', 'Plant', 'Status']

        data = []
        plant_code = 0
        for plant in next(os.walk(path))[1]:
            plant_code += 1
            for status in next(os.walk(os.path.join(path, plant)))[1]:
                for img in next(os.walk(os.path.join(path, plant, status)))[2]:
                    obj = {'Name': img, 'Plant': plant, 'PlantCode': plant_code, 'Status': status}
                    data.append(obj)

                    print(os.path.join(path, plant, status, img))

        df = pd.DataFrame(data)
        df.to_csv('../data/mendeley/mendeley.csv')
        print('CSV successfully created')

    @staticmethod
    def isLocallyAvailable():
        return os.path.exists('../data/mendeley')

    @staticmethod
    def download():
        wget.download('https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/hb74ynkjcn-1.zip', '../data')