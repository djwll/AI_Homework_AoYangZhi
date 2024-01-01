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
from torch.cuda.amp import autocast, GradScaler # 导入自动混合精度模块
from pytorch_lamb import Lamb # 导入LAMB优化器

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







def train(train_iter, val_iter, test_iter):
    # 训练模型
    num_epochs = 10
    loss_history = []
    val_loss_history = []
    val_acc_history = []

    #  创建模型实例
    model = models.resnet18(pretrained=True) # 使用预训练的ResNet18
    for param in model.parameters():
        param.requires_grad = False # 冻结参数，不更新梯度
    model.fc = nn.Sequential( # 修改最后一层
        nn.Linear(model.fc.in_features, 26),
        nn.Softmax(dim=1) # 加上softmax层
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01) # 使用AdamW优化器
    optimizer = Lamb(model.fc.parameters(), lr=0.001, weight_decay=0.01) # 使用LAMB优化器
    scaler = GradScaler() # 创建梯度缩放器

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True) # 使用ReduceLROnPlateau

    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_iter):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast(): # 开启自动混合精度
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward() # 使用梯度缩放器
            scaler.step(optimizer) # 使用梯度缩放器
            scaler.update() # 使用梯度缩放器
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                loss_history.append(running_loss / 10)
                running_loss = 0.0
        end = time.time()
        print('Epoch %d cost %.3f seconds' % (epoch + 1, end - start))
        # 在验证集上评估模型
        val_loss, val_acc = evaluate(model, val_iter, criterion)
        print('Validation loss: %.3f, Validation accuracy: %.3f' % (val_loss, val_acc))
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        # 调整学习率
        scheduler.step(val_loss)
    print('Finished Training')
    # 在测试集上评估模型
    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print('Test loss: %.3f, Test accuracy: %.3f' % (test_loss, test_acc))
    # 绘制损失和准确率曲线
    plot_loss_and_acc(loss_history, val_loss_history, val_acc_history)
    # 保存模型
    torch.save(model.state_dict(), 'Convresults/model4.pth')
    
def evaluate(model, data_iter, criterion):
    # 评估模型
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        for images, labels in data_iter:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_acc += torch.sum(preds == labels).item()
            total_count += images.size(0)
        return total_loss / total_count, total_acc / total_count

def plot_loss_and_acc(loss_history, val_loss_history, val_acc_history):
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(loss_history, label='train loss')
    plt.plot(val_loss_history, label='validation loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(val_acc_history, label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 定义数据增强的变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 加载数据集
    train_dir = "datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/train"
    test_dir = "datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val"
    train_data = CustomImageFolder(train_dir, transform=train_transform)
    test_data = CustomImageFolder(test_dir, transform=test_transform)

    # 划分验证集
    val_size = int(len(train_data) * 0.2)
    train_size = len(train_data) - val_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    # 创建数据加载器
    train_iter = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_iter = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    test_iter = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 训练模型
    train(train_iter, val_iter, test_iter)
