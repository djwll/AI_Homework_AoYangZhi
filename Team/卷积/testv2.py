import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import time
import random
from matplotlib import pyplot as plt
import os
from PIL import Image
from conv_v2 import ConvNet



train_dir = "datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/train"
test_dir = "datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val"

# 将图像调整为224×224尺寸并归一化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

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

train_set = CustomImageFolder(train_dir, transform=train_augs)
test_set = CustomImageFolder(test_dir, transform=test_augs)

batch_size = 32
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_set, batch_size=batch_size)




# 定义模型
# model = ConvNet(num_classes=26)

# 加载模型参数
# model.load_state_dict(torch.load('Convresults/model4.pth'))
print('model successfully loaded!')

# 创建模型实例
model = models.resnet50(pretrained=True) # 使用预训练的ResNet18
for param in model.parameters():
    param.requires_grad = False # 冻结参数，不更新梯度
model.fc = nn.Sequential( # 修改最后一层
    nn.Linear(model.fc.in_features, 26),
    nn.Softmax(dim=1) # 加上softmax层
)

# 加载模型参数
model.load_state_dict(torch.load('Convresults/model3.pth'))


# 归一化
def denorm(img):
    for i in range(img.shape[0]):
        img[i] = img[i] * std[i] + mean[i]
    return img


# 显示随机选择的图片
plt.figure(figsize=(8, 8))
for i in range(9):
    img, label = test_set[random.randint(0, len(test_set) - 1)]
    # 对选择的图片进行反归一化使其显示出来
    img = denorm(img)
    # 对图片的维度进行重排
    img = img.permute(1, 2, 0)
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(img.numpy())
    # 重排维度以适应模型
    img = img.permute(2, 0, 1).unsqueeze(0)
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    ax.set_title("label=%s" % test_set.idx_to_class[predicted.item()])
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()