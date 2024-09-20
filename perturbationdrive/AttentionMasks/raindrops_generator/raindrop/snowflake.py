import numpy as np
from PIL import Image, ImageDraw


class SnowFlake:
    def __init__(self, key, centerxy=None, radius=None, shape=None):
        """
        :param key: a unique key identifying a snowflake
        :param centerxy: tuple defining coordinates of snowflake center in the image
        :param radius: radius of a snowflake
        :param shape: int from 0 to 2 defining snowflake shape type
        """
        self.key = key
        self.ifcol = False
        self.col_with = []
        self.center = centerxy
        self.radius = radius
        self.shape = shape
        self.type = "default"
        self.labelmap = np.zeros((self.radius * 2, self.radius * 2))
        self.alphamap = np.zeros((self.radius * 2, self.radius * 2))
        self.background = None
        self.texture = None
        self._create_label()
        self.use_label = False

    def setCollision(self, col, col_with):
        self.ifcol = col
        self.col_with = col_with

    def updateTexture(self, bg):
        tmp = np.expand_dims(self.alphamap, axis=-1)
        white_blob = np.ones_like(tmp) * 255  # Create a white blob
        tmp = np.concatenate((white_blob, tmp), axis=2)
        self.texture = Image.fromarray(tmp.astype("uint8"), "RGBA")

    def _create_label(self):
        self._createWhiteBlob()

    def _createWhiteBlob(self):
        """
        Create a simple white blob as the snowflake Alpha Map
        """
        img = Image.fromarray(np.uint8(self.labelmap), "L")
        draw = ImageDraw.Draw(img)

        # Draw a white blob
        x0, y0 = self.radius // 2, self.radius // 2
        x1, y1 = x0 + self.radius, y0 + self.radius
        draw.ellipse((x0, y0, x1, y1), fill=255)

        self.alphamap = np.asarray(img).astype(np.float64)
        # Ensure the blob is white by setting all values to 255
        self.alphamap[self.alphamap > 0] = 255
        # Set label map
        self.labelmap[self.alphamap > 0] = 1

    def setKey(self, key):
        self.key = key

    def getLabelMap(self):
        return self.labelmap

    def getAlphaMap(self):
        return self.alphamap

    def getTexture(self):
        return self.texture

    def getCenters(self):
        return self.center

    def getRadius(self):
        return self.radius

    def getKey(self):
        return self.key

    def getIfColli(self):
        return self.ifcol

    def getCollisionList(self):
        return self.col_with

    def getUseLabel(self):
        return self.use_label
