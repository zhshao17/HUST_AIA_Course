# # 作业二
# 使用Keras（推荐）/Pytorch构建简单的CNN和LSTM网络模型各一个。其中CNN使用Cifar-10数据集进行训练和测试，
# LSTM使用德里市每日气候数据集进行训练和测试。介绍所构建的模型，并报告在测试集上的准确率。
#
# cifar-10: https://www.cs.toronto.edu/~kriz/cifar.html
#
# daily-climate: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data

# 可选内容：尝试在CNN网络中加入残差连接结构（Residual block）。
import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))])

TrainData = torchvision.datasets.CIFAR10(root='./datasets-writeable', train=True,
                                         download=True, transform=transform)
TrainLoader = torch.utils.data.DataLoader(TrainData, batch_size=64,
                                          shuffle=True, num_workers=2)
TestData = torchvision.datasets.CIFAR10(root='./datasets-writeable', train=False,
                                        download=True, transform=transform)
TestLoader = torch.utils.data.DataLoader(TestData, batch_size=64,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 3x3 卷积定义
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Resnet 的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)


def train(epochs):
    test_step = 1
    for epoch in range(epochs):
        print("Training epoch %d" % (epoch + 1))
        model.train()
        for i, datas in enumerate(TrainLoader, 0):
            data, target = datas
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch + 1,
                                                                               i * len(data), len(TrainLoader.dataset),
                                                                               100. * i * len(data) / len(TrainLoader.dataset),
                                                                               loss))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for j, datas in enumerate(TestLoader, 0):
                data, target = datas
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total = total + target.size(0)
                correct = correct + (predicted == target).sum().item()
                test_step += 1
            print('Test Accuracy: %.4f' % (correct / total * 100.0))
    print('end!')


if __name__ == '__main__':
    train(epochs=10)
