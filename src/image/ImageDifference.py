import torchvision.transforms as transforms
import numpy as np
import tensorflow as tf
from PIL import ImageChops


class ImageDifference:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def difference(self):
        # transformations
        PilTrans = transforms.ToPILImage()

        img1_arr = np.transpose(tf.squeeze(self.img1).numpy(), [1, 2, 0]) * 255
        img1_ui8 = img1_arr.astype(np.uint8)

        img2_arr = np.transpose(np.squeeze(self.img2).detach().numpy(), [1, 2, 0]) * 255
        img2_ui8 = img2_arr.astype(np.uint8)

        image1 = PilTrans(img1_ui8)
        image2 = PilTrans(img2_ui8)

        return ImageChops.difference(image1, image2)
