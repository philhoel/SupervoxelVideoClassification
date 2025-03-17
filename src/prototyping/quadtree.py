import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.asarray(Image.open("131.jpg"))


class Node:

    def __init__(self, img, box, depth):
        self.box = box
        self.depth = depth
        self.children = None
        self.leaf = False

    def splits(self, img):

        l, t, r, b = self.box
        lr = l + (r - l) / 2
        tb = t + (b - t) / 2

        tl = Node(img, (l, t, lr, tb), self.depth+1)
        tr = Node(img, (lr, t, r, tb), self.depth+1)
        bl = Node(img, (l, tb, lr, b), self.depth+1)
        br = Node(img, (lr, tb, r, b), self.depth+1)

        self.children = [tl, tr, bl, br]


class Quadtree:

    def __init__(self, image, max_depth=1024):
        self.root = Node(image, )
