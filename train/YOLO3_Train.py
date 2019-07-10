import torch
from torch.utils.data import DataLoader
from commons.config import *
from dataset.dataloader import CocoDataSet
from models.DarkNet53 import *
import time

def loss_fn(output, target, alpha):
    """
    置信度、建议框的损失
    :param output: 网络输出
    :param target: 建议框
    :param alpha: 损失权重，用于解决有物体的数据占比较少的问题
    :return: 损失
    """
    # NHCW >> NHWC, 1轴和3轴交换
    output = output.permute(0, 2, 3, 1)
    # N H W C >> N H W 3 15
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

    # print("target.shape in loss_fn: ", target.shape)
    # print("output.shape after reshape in loss_fn: ", output.shape)

    # [..., 0]表示取最后一个维度的第0个元素
    # label张量中shape（N H W 3 15）最后一个维度的第0个元素代表置信度
    # mask_obj则为置信度的掩码，输出形状同label. (掩码：符合条件的位置输出1，否则输出0，同数组形状一致）
    mask_obj = target[..., 0] > 0
    mask_noobj = target[..., 0] == 0

    has_obj_target = target[mask_obj].double()
    # 没有物体，只需要计算置信度
    no_obj_target = target[mask_noobj][0].double()

    has_obj_output = output[mask_obj].double()

    no_obj_output = output[mask_noobj][0].double()

    # 输出与目标的平方差
    loss_obj = torch.mean((has_obj_output - has_obj_target) ** 2)
    loss_noobj = torch.mean((no_obj_output - no_obj_target) ** 2)

    # 有物体的损失+没有物体的损失，有物体损失权重较高
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj

    return loss


if __name__ == '__main__':

    # 创建数据集对象
    coco_dataset = CocoDataSet()
    # 创建数据加载器
    dataloader = DataLoader(dataset=coco_dataset, batch_size=10, shuffle=True)

    # 网络对象
    net = DarkNet53()
    net.train()

    # 优化器
    opt = torch.optim.Adam(net.parameters())
for epoch in range(50):
    for i, (target_13, target_26, target_52, img_data) in enumerate(dataloader):

        # print("img_data.size(): ", img_data.size())
        # 网络输出3组建议框
        output_13, output_26, output_52 = net(img_data)


        # output_13 = output_13.permute(0, 2, 3, 1)
        # output_13 = output_13.reshape(output_13.size(0), output_13.size(1), output_13.size(2), 3, -1)
        #
        # output_26 = output_26.permute(0, 2, 3, 1)
        # output_26 = output_26.reshape(output_26.size(0), output_26.size(1), output_26.size(2), 3, -1)
        #
        # output_52 = output_52.permute(0, 2, 3, 1)
        # output_52 = output_52.reshape(output_52.size(0), output_52.size(1), output_52.size(2), 3, -1)


        # print("output_13.shape: ", output_13.shape)
        # print("target_13.shape: ", target_13.shape)

        loss_13 = loss_fn(output_13, target_13, LOSS_ALPHA)
        loss_26 = loss_fn(output_26, target_26, LOSS_ALPHA)
        loss_52 = loss_fn(output_52, target_52, LOSS_ALPHA)

        loss = loss_13 + loss_26 + loss_52


        opt.zero_grad()
        loss.backward()
        opt.step()


        print("{}--epoch:{},i:{},train_loss:{:.3}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, i, loss.item()))


    torch.save(net, "{0}/yolo3_valset-{1}.pth".format(PARAMS_PATH, epoch))





