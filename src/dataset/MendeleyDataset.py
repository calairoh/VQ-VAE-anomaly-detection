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

# Ignore warnings
warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


class MendeleyDataset(Dataset):
    """Face Landmarks data."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def create_csv(path):
        """
        :type path: Path to image folder
        """
        print('CSV creation...')

        columns = ['No.', 'Name', 'Plant', 'Healthy']
        df = pd.DataFrame(columns=columns)

        for plant in os.walk(path):
            for status in os.walk(os.path.join(path, plant)):
                for img in os.walk(os.path.join(path, plant, status)):
                    print(os.path.join(path, plant, status, img))

        print('CSV successfully created')

    @staticmethod
    def isLocallyAvailable():
        return os.path.exists('../data/mendeley')

    @staticmethod
    def download():
        wget.download('https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/hb74ynkjcn-1.zip', '../data')
