from commons.config import *
from pycocotools.coco import COCO
import torch
import os
import numpy as np

dataType = 'val2017'
annFile = os.path.join(COCO_BASE_PATH, "annotations/instances_val2017.json")

coco = COCO(annFile)
print("coco.getCatIds(): ", coco.getCatIds())

# 获得COCO数据集的所有分类的ID, 格式如下
# {'supercategory': 'person', 'id': 1, 'name': 'person'}
# {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
# {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
# {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}
cats = coco.loadCats(coco.getCatIds())
print("cats: ", cats)
# catIds = coco.getCatIds(catNms=['person', 'car', 'bicycle'])

for catName in CATS_NAME:

    catIds = coco.getCatIds(catNms=[catName])

    # 获得指定分类ID下的所有图片的ID
    # 图片ID格式为：文件名称去掉全面的0和.jpg，图片文件名为 000000001296.jpg，则该图片的ID为1296
    imgIds = coco.getImgIds(catIds=catIds)

    i = 0
    ann_line = ''
    preparedAnnFile = os.path.join(COCO_ANN_PATH, "label_{}.txt".format(catName))
    file_obj = open(preparedAnnFile, 'w')

    # 遍历图片ID，取出每张图片的标签数据
    for imgId in imgIds:

        # 获得图片的信息，图片名称，高，宽，id等
        # img的类型是list，只有一个元素，元素数据类型是str，格式是json
        img = coco.loadImgs(imgIds[i])

        # 获得该图片的所有标签信息，可能有多个。类型id, bbox
        # annIds是标签ID
        annIds = coco.getAnnIds(imgIds=img[0]['id'], catIds=catIds, iscrowd=None)
        # anns是标签对象
        anns = coco.loadAnns(annIds)

        ann_line = img[0]['file_name']

        # 获得该张图的类型和bbox的标签
        for ann in anns:

            bbox = ann['bbox']
            print("prepared bbx: ", bbox)
            # coco数据集的的坐标为：左上角X，左上角Y，宽，高
            # 需要转成：中心点X，中心点Y，宽，高
            bbox[0] = round(bbox[0] + bbox[2] / 2, 2)
            bbox[1] = round(bbox[1] + bbox[3] / 2, 2)

            category_id, bbox_str = str(ann['category_id']), str(bbox)

            ann_line += ' ' + category_id + ' ' + bbox_str.replace('[', '').replace(']', '').replace(',', '')
            print("ann_line: ", ann_line)
        file_obj.write(ann_line+'\n')
        i += 1
    file_obj.flush()