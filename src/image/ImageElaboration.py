import PIL.ImageOps

from image.RGB import RGB


class ImageElaboration:
    def __init__(self, image):
        self.image = image

    def keep_only(self, rgb):
        r, g, b = self.image.split()

        if rgb == RGB.RED:
            self.image = r
        elif rgb == RGB.GREEN:
            self.image = g
        else:
            self.image = b

    def negative(self):
        self.image = PIL.ImageOps.invert(self.image)

    def get(self):
        return self.image
