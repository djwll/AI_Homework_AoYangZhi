import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import time
import random
from matplotlib import pyplot as plt
import os
import math
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler




class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = os.listdir(img_dir)
        self.classes.sort()
        self.imgs = [os.path.join(root, name)
                     for root, dirs, files in os.walk(img_dir)
                     for name in files]
        self.class_to_idx = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
                             '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14, 
                             '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
                             '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
        self.idx_to_class = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
                             8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
                             15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
                             22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.class_to_idx[os.path.basename(os.path.dirname(img_path))]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # return image, self.idx_to_class[label]
        return image, label

class ConvNet(nn.Module):
    def __init__(self, num_classes=26):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5))
        self.fc = nn.Linear(56*56*64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class CustomLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        super(CustomLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 10:
            return self.base_lrs
        else:
            return [base_lr * 0.5 ** (self.last_epoch - 9) for base_lr in self.base_lrs]

def train(train_iter, test_iter):
    # 训练模型
    num_epochs = 10
    loss_history = []

    #  创建模型实例
    model = ConvNet(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # 创建学习率调度器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # scheduler = CustomLR(optimizer)
    # 创建余弦退火学习率调度器
    
    
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_iter):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 5 == 0:
                loss_history.append(loss.item())
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, len(train_iter), loss.item()))
                
        # 在每个epoch结束后，更新学习率
        scheduler.step()

    plt.plot(loss_history)            
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

    # 测试模型
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    print('开始验证')
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_iter:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # 保存模型
    torch.save(model.state_dict(), 'Convresults/model.pth')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('代码运行在:',device)
    train_dir = "datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/train"
    test_dir = "datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val"

    # 将图像调整为224×224尺寸并归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # train_augs = transforms.Compose([
    #     transforms.RandomResizedCrop(size=224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = CustomImageFolder(train_dir, transform=train_augs)
    test_set = CustomImageFolder(test_dir, transform=test_augs)

    batch_size = 20
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=batch_size)

    train(train_iter, test_iter)