import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from PIL import Image
import PIL.ImageDraw as draw

COCO_ANN_IMG = r"/Users/magichao/PycharmProjects/coco2017/images/val2017"

img_file = Image.open("{}/{}".format(COCO_ANN_IMG, "000000458755.jpg"))



imgdraw = draw.ImageDraw(img_file)
imgdraw.rectangle([590.02, 91.69, 49.62, 97.07], outline='red')
img_file.show()

# plt.show(img_file)
plt.pause(1)
print("================")
