import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.test import test

# 模型选择：GPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# 数据集位置
dataset_path = '../../../Datasets'
batch_size = 1
shuffle = True
download = False
# 学习率
learning_rate = 0.001
# 预训练模型位置
model_path = "../../../Pretrained_models/Model/MNISTModel_9.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Grayscale(),
])
# 扰动参数
epsilons = [0, .05, .1, .15, .2, .25, .3]

if __name__ == '__main__':
    # 1.预处理
    train_dataset = datasets.MNIST(dataset_path, train=True, download=download, transform=transform)
    val_dataset = datasets.MNIST(dataset_path, train=False, download=download, transform=transform)

    train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    model = torch.load(model_path).to(device)

    # 2.开始测试
    # 记录不同扰动下的准确度
    accuracies = []
    # 记录样本
    examples = []

    # 对每个epsilon运行测试
    for eps in epsilons:
        # 进行对抗样本攻击
        acc, ex = test(model, device, val_DataLoader, eps)
        # 将此扰动的准确度记录
        accuracies.append(acc)
        # 二维数组，行代表不同的epsilon，列代表当前epsilon生成的对抗样本
        examples.append(ex)

    # 3.绘图,用于可视化
    # 创建一个新的图形对象，图形大小设置为 5x5 英寸
    plt.figure(figsize=(5, 5))
    # 用epsilons作为x轴数据，accuracies作为y轴数据
    # *-代表数据点用*标记，点之间用直线链接
    plt.plot(epsilons, accuracies, "*-")
    # 设置y轴刻度，
    # np.arange(0, 1.1, step=0.1)生成0~1的数组，步长为0.1
    plt.yticks(np.arange(0, 1.1, step=0.1))
    # 设置x轴刻度
    # 生成0~0.3的数组，步长为0.05
    plt.xticks(np.arange(0, .35, step=0.05))
    # 将图标标题设为Accuracy vs Epsilon
    plt.title("Accuracy vs Epsilon")
    # x轴标签为Epsilon
    plt.xlabel("Epsilon")
    # y轴标签为Accuracy
    plt.ylabel("Accuracy")
    # 显示图表
    plt.show()

    cnt = 0
    plt.figure(figsize=(8, 10))
    # 行代表不同的epsilon
    for i in range(len(epsilons)):
        # 列代表同一epsilon生成的图像
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
