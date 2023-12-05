# 导入相关包
import copy
import os
import random
import numpy as np
import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator

def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = [] # 片段
    target = [] # 作者
    
    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index,line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))

# 定义Field
TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 读取数据，是由tuple组成的列表形式
path = "dataset/"
mydata = load_data(path)

# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)
# 构建中文词汇表
TEXT.build_vocab(dataset)

# 切分数据集
train, val = dataset.split(split_ratio=0.7)

# 生成可迭代的mini-batch
train_iter, val_iter = BucketIterator.splits(
    (train,val), # 数据集
    batch_sizes=(8, 8),
    device=-1, # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text), 
    sort_within_batch=False,
    repeat=False
)


# Pytorch定义模型的方式之一：
# 继承 Module 类并实现其中的forward方法
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()                
        self.lstm = torch.nn.LSTM(1,64)
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,5)

    def forward(self,x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        output,hidden = self.lstm(x.unsqueeze(2).float())
        h_n = hidden[1]
        out = self.fc2(self.fc1(h_n.view(h_n.shape[1],-1)))
        return out

# 创建模型实例
model = Net()

for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_acc_list, train_loss_list = [], []
val_acc_list, val_loss_list = [], []
for epoch in range(3):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    for idx, batch in enumerate(train_iter):
        text, label = batch.text, batch.category
        optimizer.zero_grad()
        out = model(text)
        loss = loss_fn(out,label.long())
        loss.backward( retain_graph=True)
        optimizer.step()
        accracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
        # 计算每个样本的acc和loss之和
        train_acc += accracy*len(batch)
        train_loss += loss.item()*len(batch)
        
        print("\r opech:{} loss:{}, train_acc:{}".format(epoch,loss.item(),accracy),end=" ")
        
    
    # 在验证集上预测
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text, label = batch.text, batch.category
            out = model(text)
            loss = loss_fn(out,label.long())
            accracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
            # 计算一个batch内每个样本的acc和loss之和
            val_acc += accracy*len(batch)
            val_loss += loss.item()*len(batch)
            
    train_acc /= len(train_iter.dataset)
    train_loss /= len(train_iter.dataset)
    val_acc /= len(val_iter.dataset)
    val_loss /= len(val_iter.dataset)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
        
# 保存模型
torch.save(model.state_dict(), 'results/temp.pth')

# 绘制曲线
plt.figure(figsize=(15,5.5))
plt.subplot(121)
plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title("acc")
plt.subplot(122)
plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title("loss")