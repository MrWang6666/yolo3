COCO_BASE_PATH = r"/Users/magichao/PycharmProjects/coco2017/"
COCO_ANN_PATH = r"/Users/magichao/PycharmProjects/coco2017/annotations/"
COCO_ANN_IMG = r"/Users/magichao/PycharmProjects/coco2017/images/val2017"


CATS_NAME = ['person', 'bicycle', 'car', 'bus', 'bird', 'cat', 'dog']

IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 10

ANCHORS_GROUP = {
    # h w
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
