import torch
import torchvision.models
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 100


def Getdata():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    TrainData = torchvision.datasets.CIFAR10(root='./datasets-writeable', train=True, download=True,
                                             transform=transform)
    TrainLoader = torch.utils.data.DataLoader(TrainData, batch_size=batch_size, shuffle=True, num_workers=2)
    TestData = torchvision.datasets.CIFAR10(root='./datasets-writeable', train=False, download=True,
                                            transform=transform)
    TestLoader = torch.utils.data.DataLoader(TestData, batch_size=batch_size, shuffle=False, num_workers=2)
    return TrainLoader, TestLoader


def train(model, epoch, optimizer, TrainLoader):
    model.train()
    epoch_loss = 0.0
    correct = 0.0
    train_step = 0
    for i, (data, label) in enumerate(TrainLoader, 0):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        train_output = model(data)
        _, pred = torch.max(train_output, 1)
        loss = criterion(train_output, label)
        loss.backward()
        optimizer.step()

        epoch_loss = epoch_loss + loss
        correct += (pred == label.data).sum()
        train_step += 1
        if train_step % 200 == 0:
            print('epoch:', epoch, train_step, '/', len(TrainLoader), 'acc:', (pred == label.data).sum().item() / batch_size)
    print('Epoch: ', epoch,  'Train acc: ', correct.item() / (train_step * batch_size))


def test(model, TestLoader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    for i, (data, label) in enumerate(TestLoader):
        data = data.to(device)
        label = label.to(device)
        test_out = model(data)
        loss = criterion(test_out, label)
        test_loss = test_loss + loss.item()
        pred = torch.max(test_out, 1)[1]
        test_correct = (pred == label).sum()
        correct = correct + test_correct.item()
    print('Test acc: ', correct / (i + 1))


if __name__ == '__main__':
    # ALexNet
    alexnet = torchvision.models.alexnet(pretrained=True)
    # 修改最后一层全连接层输出
    num_fc = alexnet.classifier[6].in_features
    alexnet.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=10)
    alexnet.features[12] = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    for param in alexnet.parameters():
        param.requires_grad = False
    for param in alexnet.classifier[6].parameters():
        param.requires_grad = True
    alexnet = alexnet.to(device)

    # vgg
    vgg19 = torchvision.models.vgg11(pretrained=True)
    vgg19.classifier[6] = torch.nn.Linear(in_features=vgg19.classifier[6].in_features, out_features=10)
    for param in vgg19.parameters():
        param.requires_grad = False
    for param in vgg19.classifier[6].parameters():
        param.requires_grad = True
    vgg19 = vgg19.to(device)

    # resnet
    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Linear(in_features=resnet.fc.in_features, out_features=10)
    for param in resnet.parameters():
        param.requires_grad = False
    for param in resnet.fc.parameters():
        param.requires_grad = True
    resnet = resnet.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_alexnet = torch.optim.Adam(alexnet.parameters(), lr=0.0001)
    optimizer_vgg19 = torch.optim.Adam(vgg19.parameters(), lr=0.0001)
    optimizer_resnet = torch.optim.Adam(resnet.parameters(), lr=0.0001)

    TrainLoader, TestLoader = Getdata()
    epochs = 5
    for epoch in range(epochs):
        train(model=alexnet, epoch=epoch, optimizer=optimizer_alexnet, TrainLoader=TrainLoader)
    test(model=alexnet, TestLoader=TestLoader)
    print('alexnet done!')

    for epoch in range(epochs):
        train(model=vgg19, epoch=epoch, optimizer=optimizer_vgg19, TrainLoader=TrainLoader)
    test(model=vgg19, TestLoader=TestLoader)
    print('vgg19 done!')

    for epoch in range(epochs):
        train(model=resnet, epoch=epoch, optimizer=optimizer_resnet, TrainLoader=TrainLoader)
    test(model=resnet, TestLoader=TestLoader)
    print('resnet done!')



