COCO_BASE_PATH = r"/Users/magichao/PycharmProjects/coco2017/"
COCO_ANN_PATH = r"/Users/magichao/PycharmProjects/coco2017/annotations/"
COCO_ANN_IMG = r"/Users/magichao/PycharmProjects/coco2017/images/val2017"


CATS_NAME = ['person', 'bicycle', 'car', 'bus', 'bird', 'cat', 'dog']
# CATS_NAME = ['person']

IMG_HEIGHT = 416
IMG_WIDTH = 416

ANCHOR_SIZE_13 = 13
ANCHOR_SIZE_26 = 26
ANCHOR_SIZE_52 = 52

CLASS_NUM = 10

ANCHORS_GROUP = {
    # h w
    ANCHOR_SIZE_13: [[116, 90], [156, 198], [373, 326]],
    ANCHOR_SIZE_26: [[30, 61], [62, 45], [59, 119]],
    ANCHOR_SIZE_52: [[10, 13], [16, 30], [33, 23]]
}

ANCHORS_GROUP_AREA = {
    ANCHOR_SIZE_13: [x * y for x, y in ANCHORS_GROUP[13]],
    ANCHOR_SIZE_26: [x * y for x, y in ANCHORS_GROUP[26]],
    ANCHOR_SIZE_52: [x * y for x, y in ANCHORS_GROUP[52]],
}


LOSS_ALPHA = 0.9

PARAMS_PATH = r"/Users/magichao/PycharmProjects/yolo3/commons"