import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import numpy as np
from PIL import Image
import os
from commons.config import *
import math

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([IMG_WIDTH, IMG_HEIGHT])
    ,torchvision.transforms.ToTensor()
    # ,torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LABEL_FILE = os.path.join(COCO_ANN_PATH, "label_person.txt")

def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.

    return b

class CocoDataSet(Dataset):

    def __init__(self):
        """
        初始化方法，行读取标签文件的每行，结果为list类型对象，每个元素类型为字符串
        """
        with open(LABEL_FILE) as f:
            self.dataset = f.read().splitlines()

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        labels = {}
        # 分割出数据：文件名、类型、bbox
        strs_list = self.dataset[index].split()

        # 第一位是文件名
        file_name = strs_list[0]
        img_data = Image.open(os.path.join(COCO_ANN_IMG, file_name))

        # 判断是否为单通道的灰度图，如是，需要转成RGB
        if img_data.getbands()[0] == 'L':
            print("gray to RGB, fine name: ", file_name)
            img_data = img_data.convert('RGB')

        img_date_tensor = transform(img_data)

        _boxes = np.array([float(x) for x in strs_list[1:]])
        # 每5个数据为一组，分别代表，类别、bbox(4)
        boxes = np.split(_boxes, len(_boxes) // 5)

        # 特征图尺寸（13，26， 52），建议框的尺寸
        for feature_size, anchors in ANCHORS_GROUP.items():
            # 标签的数据格式：H W 3种建议框
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + CLASS_NUM))

            for box in boxes:
                # 类别、中心点x，中心点y，宽，高
                cls_v, cx, cy, w, h = box

                # modf返回值小数在前，整数在后
                # 求中心点在小框中的偏移量、中心点在小框的索引。 因三种尺寸的建议框是同心的，所以中心点计算一次就行了
                cx_offset, cx_index = math.modf(cx * feature_size / IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / IMG_HEIGHT)

                if cx_index >= feature_size or cy_index >= feature_size:
                    continue

                # 遍历三种不同的建议框
                for i, anchor in enumerate(anchors):

                    # 建议框的面积
                    anchors_area = ANCHORS_GROUP_AREA[feature_size][i]

                    # 标签的宽高在建议框中的比例
                    p_w, p_h = w / anchor[0], h / anchor[1]

                    # 标签框的面积
                    label_area = w * h
                    # print("cy_index, cx_index: ", int(cy_index), int(cx_index))

                    #
                    iou = min(label_area, anchors_area) / max(label_area, anchors_area)

                    # print("lables shape: ", (labels[feature_size][int(cy_index), int(cx_index), i]).shape)

                    # 置信度、中心点、宽、高、类别
                    corrd_cls = np.array([
                        iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cls_num=CLASS_NUM, v=int(cls_v))
                    ])
                    # print("corrd_cls shape: ", corrd_cls.shape)

                    # corrd_cls = np.array([
                    #     iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), cls_v
                    # ])

                    # 标签形状 H W C，其中C包括置信度、中心点、宽、高、类别
                    labels[feature_size][int(cy_index), int(cx_index), i] = corrd_cls

        return labels[13], labels[26], labels[52], img_date_tensor

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":

    coco_dataset = CocoDataSet()

    dataloader = torch.utils.data.DataLoader(dataset=coco_dataset, batch_size=10, shuffle=True)

    for i, (labels_13, labels_26, labels_52, img_data) in  enumerate(dataloader):

        print(i)

