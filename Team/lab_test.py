import math
import numpy as np
import os
import cv2
import random
import shutil
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
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

from src_mindspore.dataset import create_dataset # 数据处理脚本
from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2 # 模型定义脚本

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
    "lr_max": 0.1, # 请尝试修改以提升精度
    "decay_type": 'square', # 请尝试修改以提升精度
    "momentum": 0.9, # 请尝试修改以提升精度
    "weight_decay": 0.001, # 请尝试修改以提升精度
    "dataset_path": "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "features_path": "./results/garbage_26x100_features", # 临时目录，保存冻结层Feature Map，可随时删除
    "class_index": index,
    "save_ckpt_epochs": 1,
    "save_ckpt_path": './results/ckpt_mobilenetv2',
    "pretrained_ckpt": './src_mindspore/mobilenetv2-200_1067_cpu_gpu.ckpt',
    "export_path": './results/mobilenetv2.mindir'
    
})


class GlobalPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:
        reduction: mean or max, which means AvgPooling or MaxpPooling.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self, reduction='mean'):
        super(GlobalPooling, self).__init__()
        if reduction == 'max':
            self.mean = ms.ops.ReduceMax(keep_dims=False)
        else:
            self.mean = ms.ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class MobileNetV2Head(nn.Cell):
    """
    MobileNetV2Head architecture.

    Args:
        input_channel (int): Number of channels of input.
        hw (int): Height and width of input, 7 for MobileNetV2Backbone with image(224, 224).
        num_classes (int): Number of classes. Default is 1000.
        reduction: mean or max, which means AvgPooling or MaxpPooling.
        activation: Activation function for output logits.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> MobileNetV2Head(num_classes=1000)
    """

    def __init__(self, input_channel=1280, hw=7, num_classes=1000, reduction='mean', activation="Sigmod", dropout_rate=0.5):
        super(MobileNetV2Head, self).__init__()
        if reduction:
            self.flatten = GlobalPooling(reduction)
        else:
            self.flatten = nn.Flatten()
            input_channel = input_channel * hw * hw
        
        # self.dense = nn.Dense(input_channel, num_classes, weight_init=nn.initializer.HeNormal(), has_bias=False)
        # self.dense = nn.Dense(input_channel, num_classes, weight_init=HeNormal(), has_bias=False)
        hidden_units_1 = 256
        hidden_units_2 = 512
        hidden_units_3 = 256
        self.dense1 = nn.Dense(input_channel, hidden_units_1, weight_init=HeNormal(), has_bias=False)
        self.dense2 = nn.Dense(hidden_units_1, num_classes, weight_init=HeNormal(), has_bias=False)
        # self.dense3 = nn.Dense(hidden_units_2, hidden_units_3, weight_init=HeNormal(), has_bias=False)
        # self.dense4 = nn.Dense(hidden_units_3, num_classes, weight_init=HeNormal(), has_bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        # self.bn = nn.BatchNorm2d(num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
            self.need_activation = True
        elif activation == "Softmax":
            self.activation = nn.Softmax()
            self.need_activation = True
        else:
            self.need_activation = False
    def construct(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.softmax(x)
        x = self.dense2(x)
        x = self.bn(x)
        x = self.dropout(x)
        if self.need_activation:
            x = self.activation(x)
        return x
    

def image_process(image):
    """Precess one image per time.
    
    Args:
        image: shape (H, W, C)
    """
    mean=[0.485*255, 0.456*255, 0.406*255]
    std=[0.229*255, 0.224*255, 0.225*255]
    image = (np.array(image) - mean) / std
    image = image.transpose((2,0,1))
    img_tensor = Tensor(np.array([image], np.float32))
    return img_tensor

def infer_one(network, image_path):
    image = Image.open(image_path).resize((config.image_height, config.image_width))
    logits = network(image_process(image))
    pred = np.argmax(logits.asnumpy(), axis=1)[0]
    print(image_path, inverted[pred])
    # 显示图像和标签
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(f'Label: {inverted[pred]}')
    plt.show()
    return pred

def infer(images):
    backbone = MobileNetV2Backbone()
    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
    network = mobilenet_v2(backbone, head)
    print('加载模型路径:',os.path.join(config.save_ckpt_path, CKPT))
    load_checkpoint(os.path.join(config.save_ckpt_path, CKPT), net=network)
    for img in images:
        infer_one(network, img)


CKPT = f'mobilenetv2-{config.epochs}.ckpt'
test_images = list()
folder = os.path.join(config.dataset_path, 'val/00_08') # Hats
for img in os.listdir(folder):
    test_images.append(os.path.join(folder, img))

infer(test_images)