import torch
from commons.config import *
from models.DarkNet53 import DarkNet53

# 目标侦测
class Detector (torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self.net = DarkNet53()
        self.net.eval()


    def forward(self, input, thresh, anchors):
        """

        :param input: 输入图片
        :param thresh: 置信度的阀值
        :param anchors: 建议框
        :return: 三个尺寸的特征图
        """
        # 网络输出三种尺寸的特征图
        output_13, output_26, output_52 = self.net(input)

        # 过滤掉置信度过小的输出，index_13: 所在格子的索引，vector_13：bbox坐标及分类

        index_13, vector_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(index_13, vector_13, IMG_HEIGHT / ANCHOR_SIZE_13, ANCHORS_GROUP[ANCHOR_SIZE_13])

        index_26, vector_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(index_26, vector_26, IMG_HEIGHT / ANCHOR_SIZE_26, ANCHORS_GROUP[ANCHOR_SIZE_26])

        index_52, vector_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(index_52, vector_52, IMG_HEIGHT / ANCHOR_SIZE_52, ANCHORS_GROUP[ANCHOR_SIZE_52])

        # 连接三个尺寸的框
        torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        """
        过滤掉置信度小于阀值的特征
        :param output: 网络输出特征
        :param thresh: 阀值
        :return: index: 大于阀值的格子的索引， vector：输出的建议框的坐标及物体类别
        """
        pass
        print(output.size())
        output = output.permute(0, 2, 3, 1)

        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        # 最后一个维度的第0个元素是置信度， 大于阀值后取到掩码
        mask = output[..., 0] > thresh
        # 大于阀值的索引，AB结构，第0个维度表示有多少条满足的数据，第1个维度表示每个满足条件的数据的索引，索引的形状同同output的形状
        index = mask.nonzero()

        vector = output[mask]

        return index, torch.randn(vector.shape)

    def _parse(self, indexes, vector, t, anchors):
        """
        建议框还原到原图
        :param indexes: 中心点所处的格子索引
        :param vector: 位置向量
        :param t: 一张图
        :param anchors: 建议框
        :return:
        """

        anchor = torch.tensor(anchors)
        # 图片
        n = indexes[:, 0]
        # 建议框
        a = indexes[:, 3]

        cy = (indexes[:, 1].float() + vector[:, 2]) * t
        cx = (indexes[:, 2].float() + vector[:, 1]) * t
        print("vector: ", vector)

        print("vector[:, 3]: ", vector[:, 3])
        w = anchor[a, 0].float() * torch.exp(vector[:, 3].float())

        h = anchor[a, 1].float() * torch.exp(vector[:, 4].float())

        return torch.stack([n.float(), cx, cy, w, h], dim=1)

if __name__ == '__main__':

    detector = Detector()
    y = detector(torch.randn(3, 3, 416, 416), 0.3, ANCHORS_GROUP)

    print(y)
