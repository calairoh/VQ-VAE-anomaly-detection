from PIL import Image

from image.ImageElaboration import ImageElaboration
from image.RGB import RGB
from utils import save_elab_image

img = Image.open('C:\\Polimi\\anomaly-detection-CVAE\\outputs\\test\\segmented1.jpg')

image = ImageElaboration(img)

image.keep_only(RGB.RED)
image.negative()

save_elab_image(image.get(), 1)
