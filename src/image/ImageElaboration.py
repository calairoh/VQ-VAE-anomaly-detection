class ImageElaboration:
    def __init__(self, image):
        self.image = image

    def keep_only(self, rgb):
        return self.image[:, :, int(rgb) - 1]
