import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
from model import GoogLeNet, Inception
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 测试数据集加载函数
def test_data_process():

    # 导入测试图像路径
    root_test = r'data/test'

    # 归一化处理
    normalize = transforms.Normalize([0.0420662, 0.04281093, 0.04413987], [0.03315472, 0.03433457, 0.03628447])

    # 数据格式处理
    trans_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

    # 加载数据
    test_data = ImageFolder(root_test, trans_test)

    # 封装测试数据集
    test_data_loader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_data_loader

# 定义测试集处理函数
def test_model_process(model,test_data_loader):
    # 选择测试设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 将模型导入到测试设备中
    model = model.to(device)

    # 设置训练参数
    test_acc = 0.0  # 测试的准确度
    test_num = 0    # 测试的样本数

    # 不考虑优化器，只进行前向传播，不考虑反向传播和梯度计算
    with torch.no_grad():
        for test_x, test_y in test_data_loader:
            # 将应用值导入到模型中
            test_x = test_x.to(device)
            # 将对应的标签导入到模型中
            test_y = test_y.to(device)
            # 设备为验证模式
            model.eval()
            # 进行前向传播，得到结果值
            output = model(test_x)
            # 查找每一行中最大值的索引
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确值加1
            test_acc += torch.sum(pre_lab == test_y.data)
            # 将样本数量相加
            test_num += test_x.size(0)

        # 计算测试的正确率
        test_corrects = test_acc.double().item() / test_num
        print("测试的正确率为：",test_corrects)

# 进行函数测试
if __name__ == "__main__":
    # 导入模型
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('best_model.pth'))
    # # 导入数据集
    # test_data_loader = test_data_process()
    # # 进行模型测试
    # test_model_process(model, test_data_loader)


# 模型推理

    # 重建类型列表
    classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

    # 导入图像
    img = Image.open('OIP.jfif')

    # 归一化处理
    normalize = transforms.Normalize([0.0420662, 0.04281093, 0.04413987], [0.03315472, 0.03433457, 0.03628447])

    # 数据格式处理
    trans_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 图像数据处理
    img = trans_test(img)

    # 添加维度批次
    img = img.unsqueeze(0)

    # 选择处理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型实例化
    model = model.to(device)

    # 模型推理
    with torch.no_grad():
        # 图像实例化
        img = img.to(device)

        # 设置模式为验证模式
        model.eval()

        # 获取图像推理结果
        output = model(img)

        # 获取预测值
        pre_lab = torch.argmax(output, dim=1)

        # 将预测结果张量转化为数值形式
        result = pre_lab.item()

        # 打印预测结果
        print("预测值为：", classes[result])

        # 反归一化处理
        mean = np.array([0.0420662, 0.04281093, 0.04413987])
        std = np.array([0.03315472, 0.03433457, 0.03628447])

        # 显示图像
        img = img.squeeze().cpu().numpy()   # 去除批次维度，并转换为数组形式
        img = img.copy()
        # 反归一化处理
        img = img * std[:,None,None] + mean[:,None,None]
        # 保持像素维持在0-1之间
        img = np.clip(img, 0, 1)
        # 转换通道数
        img = np.transpose(img, (1, 2, 0))
        # 显示单一预测图像
        plt.imshow(img)
        # 显示标题
        plt.title(classes[result], size=20)
        # 不显示轴
        plt.axis("off")
    plt.show()












