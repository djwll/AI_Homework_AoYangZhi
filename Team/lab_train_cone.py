import math
import numpy as np
import os
import cv2
import random
import shutil
import time
from matplotlib import pyplot as plt
from easydict import EasyDict
from PIL import Image

import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, save_checkpoint, export
from mindspore.train.callback import Callback, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.dataset.vision import Normalize, HWC2CHW, Decode, Resize, CenterCrop
from mindspore.dataset.transforms import TypeCast
from mindspore.common.initializer import HeNormal
from mindspore import dataset
from mindspore import Tensor
from mindspore.dataset.transforms.c_transforms import Compose
import mindspore.dataset.vision.c_transforms as c_trans
from mindspore.dataset.vision import Inter
from mindspore.dataset import transforms, vision, text
from mindspore.dataset import GeneratorDataset, MnistDataset

from mindspore.train.callback import EarlyStopping

from src_mindspore.dataset import create_dataset # 数据处理脚本
from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2 # 模型定义脚本


def build_lr(total_steps, lr_init=0.0, lr_end=0.0, lr_max=0.1, warmup_steps=0, decay_type='cosine'):
    lr_init, lr_end, lr_max = float(lr_init), float(lr_end), float(lr_max)
    decay_steps = total_steps - warmup_steps
    lr_all_steps = []
    inc_per_step = (lr_max - lr_init) / warmup_steps if warmup_steps else 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + inc_per_step * (i + 1)
        else:
            if decay_type == 'cosine':
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (i - warmup_steps) / decay_steps))
                lr = (lr_max - lr_end) * cosine_decay + lr_end
            elif decay_type == 'square':
                frac = 1.0 - float(i - warmup_steps) / (total_steps - warmup_steps)
                lr = (lr_max - lr_end) * (frac * frac) + lr_end
            elif decay_type == "exponential":
                exponential_decay = (0.5)**(total_steps/decay_steps)
                lr = (lr_max - lr_end) * exponential_decay + lr_end
            else:
                lr = lr_max
        lr_all_steps.append(lr)

    return lr_all_steps


def extract_features(net, dataset_path, config):
    if not os.path.exists(config.features_path):
        os.makedirs(config.features_path)
    dataset = create_dataset(config=config)

    # random_crop = c_trans.RandomCrop([112, 112])
    # resize = c_trans.Resize(size=[224, 224])
    #竖直方向上反转和水平方向上反转
    random_horizontal_flip = c_trans.RandomHorizontalFlip(prob=0.85)
    random_vertical_flip = c_trans.RandomVerticalFlip(prob=0.7)
    # invert = c_trans.Invert()
    dataset = dataset.map(operations=random_horizontal_flip, input_columns=["image"])
    # dataset = dataset.map(operations=random_vertical_flip, input_columns=["image"])
    composed = transforms.Compose(
        [
            #vision.Rescale(1.0 / 224.0, 0),
            vision.Rescale(1.0 / 224, 0),
            # vision.Normalize(mean=(0.1307,), std=(0.3081,))
            vision.Normalize(mean=(0.01,), std=(0.3081,))
        ]
    )
    dataset = dataset.map(operations=composed, input_columns=["image"])

    #  获取数据集的大小
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")

    data_iter = dataset.create_dict_iterator()
    # 使用 for 循环遍历数据集迭代器，获取每个样本的图像和标签
    for i, data in enumerate(data_iter):
        # 构建特征保存路径 features_path 和标签保存路径 label_path
        features_path = os.path.join(config.features_path, f"feature_{i}.npy")
        label_path = os.path.join(config.features_path, f"label_{i}.npy")
        if not os.path.exists(features_path) or not os.path.exists(label_path):
            image = data["image"]
            label = data["label"]
            features = net(image)
            # 使用神经网络模型 net 对图像数据进行特征提取，得到特征 features
            np.save(features_path, features.asnumpy())
            np.save(label_path, label.asnumpy())
        print(f"Complete the batch {i+1}/{step_size}")
    return




class GlobalPooling(nn.Cell):

    def __init__(self, reduction='mean'):
        super(GlobalPooling, self).__init__()
        if reduction == 'max':
            self.mean = ms.ops.ReduceMax(keep_dims=False)
        else:
            self.mean = ms.ops.ReduceMean(keep_dims=False)
    # 池化
    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class MobileNetV2Head(nn.Cell):

    def __init__(self, input_channel=1280, hw=7, num_classes=1000, reduction='mean', activation="None"):
        super(MobileNetV2Head, self).__init__()
        if reduction:
            self.flatten = GlobalPooling(reduction)
        else:
            self.flatten = nn.Flatten()
            input_channel = input_channel * hw * hw
        self.dense = nn.Dense(input_channel, num_classes, weight_init='ones', has_bias=False)
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Softmax":
            self.activation = nn.Softmax()
        else:
            self.need_activation = False

    def construct(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        if self.need_activation:
            x = self.activation(x)
        return x
    
def train_head():
    train_dataset = create_dataset(config=config)
    eval_dataset = create_dataset(config=config)
    step_size = train_dataset.get_dataset_size()
    
    backbone = MobileNetV2Backbone()
    for param in backbone.get_parameters():
       param.requires_grad = False
    load_checkpoint(config.pretrained_ckpt, net=backbone)

    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
    network = mobilenet_v2(backbone, head)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    lrs = build_lr(config.epochs * step_size, lr_max=config.lr_max, warmup_steps=3, decay_type=config.decay_type)
    opt = nn.Momentum(head.trainable_params(), lrs, config.momentum, config.weight_decay)
    net = nn.WithLossCell(head, loss)
    train_step = nn.TrainOneStepCell(net, opt)
    train_step.set_train()
    # 早停

    # 创建 EarlyStopping 对象
    early_stopping = EarlyStopping(patience=5, mode="auto")
    # 初始化最佳损失为无穷大
    best_loss = float('inf')

    # 初始化没有改进的周期数
    no_improve_epochs = 0

    # train
    history = list()
    features_path = config.features_path
    idx_list = list(range(step_size))
    for epoch in range(config.epochs):
        random.shuffle(idx_list)
        epoch_start = time.time()
        losses = []
        for j in idx_list:
            feature = Tensor(np.load(os.path.join(features_path, f"feature_{j}.npy")))
            label = Tensor(np.load(os.path.join(features_path, f"label_{j}.npy")))
            losses.append(train_step(feature, label).asnumpy())
        epoch_seconds = (time.time() - epoch_start)
        epoch_loss = np.mean(np.array(losses))
        
         # 如果损失没有改进，增加没有改进的周期数
        if epoch_loss >= best_loss:
            no_improve_epochs += 1
        else:
            # 如果损失有改进，更新最佳损失，并重置没有改进的周期数
            best_loss = epoch_loss
            no_improve_epochs = 0

        # 如果没有改进的周期数达到 patience，停止训练
        if no_improve_epochs >= early_stopping.patience:
            print("Early stopping")
            break
        
        history.append(epoch_loss)
        print("epoch: {}, time cost: {}, avg loss: {}".format(epoch + 1, epoch_seconds, epoch_loss))
        if (epoch + 1) % config.save_ckpt_epochs == 0:
            save_checkpoint(network, os.path.join(config.save_ckpt_path, f"mobilenetv2-{epoch+1}.ckpt"))
    
    # evaluate
    print('validating the model...')
    eval_model = Model(network, loss, metrics={'acc', 'loss'})
    acc = eval_model.eval(eval_dataset, dataset_sink_mode=False)
    print(acc)
    
    return history

def create_predict_iterator(net, dataset_path, config):
    dataset = create_dataset(config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")

    data_iter = dataset.create_dict_iterator()
    for i, data in enumerate(data_iter):
        image = data["image"]
        label = data["label"]
        features = net(image)
        yield features.asnumpy(), label.asnumpy()


if __name__ ==  "__main__":
    os.environ['GLOG_v'] = '2' # Log Level = Error
    has_gpu = (os.system('command -v nvidia-smi') == 0)
    print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')

    # 垃圾分类数据集标签，以及用于标签映射的字典。
    index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
            '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14, 
            '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
            '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
    inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
                8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
                15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
                22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

    # 训练超参
    config = EasyDict({
        "num_classes": 26, # 分类数，即输出层的维度
        "reduction": 'mean', # mean, max, Head部分池化采用的方式
        "image_height": 224,
        "image_width": 224,
        "batch_size": 24, # 鉴于CPU容器性能，太大可能会导致训练卡住
        "eval_batch_size": 10,
        "epochs": 25, # 请尝试修改以提升精度
        "lr_max": 0.05, # 请尝试修改以提升精度
        "decay_type": 'exponential', # 请尝试修改以提升精度
        "momentum": 0.9, # 请尝试修改以提升精度
        "weight_decay": 0.005, # 请尝试修改以提升精度
        "dataset_path": "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
        "features_path": "./results/garbage_26x100_features", # 临时目录，保存冻结层Feature Map，可随时删除
        "class_index": index,
        "save_ckpt_epochs": 1,
        "save_ckpt_path": './results/ckpt_mobilenetv2',
        "pretrained_ckpt": './src_mindspore/mobilenetv2-200_1067_cpu_gpu.ckpt',
        "export_path": './results/mobilenetv2.mindir'
        
    })

    ds = create_dataset(config=config, training=False)
    data_iter = ds.create_dict_iterator(output_numpy=True)
    data = next(data_iter)
    images = data['image']
    labels = data['label']

    backbone = MobileNetV2Backbone()
    load_checkpoint(config.pretrained_ckpt, net=backbone)
    extract_features(backbone, config.dataset_path, config)


    if os.path.exists(config.save_ckpt_path):
        shutil.rmtree(config.save_ckpt_path)
    os.makedirs(config.save_ckpt_path)

    history = train_head()

    plt.plot(history, label='train_loss')
    plt.legend()
    plt.show()

    CKPT = f'mobilenetv2-{config.epochs}.ckpt'
    print("Chosen checkpoint is", CKPT)









