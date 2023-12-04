# 导入相关库
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from minepy import MINE
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from IPython.display import display
from datetime import datetime as dt
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split



def processing_data(data_path):
   
    """
    数据处理
    :param data_path: 数据集路径
    :return: feature1,feature2,label:处理后的特征数据、标签数据
    """
    feature1,feature2,label = None, None, None
    # -------------------------- 实现数据处理部分代码 ----------------------------
 #导入医疗数据
    data_xls = pd.ExcelFile(data_path)
    data={}
    
    #查看数据名称与大小
    for name in data_xls.sheet_names:
            df = data_xls.parse(sheet_name=name,header=None)
            data[name] = df
    
    #获取 特征1 特征2 类标    
    feature1_raw = data['Feature1']
    feature2_raw = data['Feature2']
    label = data['label']


    # 初始化一个 scaler，并将它施加到特征上
    scaler = MinMaxScaler()
    feature1 = pd.DataFrame(scaler.fit_transform(feature1_raw))
    feature2 = pd.DataFrame(scaler.fit_transform(feature2_raw))

    # ------------------------------------------------------------------------
    
    return feature1,feature2,label


def feature_select(feature1, feature2, label): 
    """
    特征选择
    :param  feature1,feature2,label: 数据处理后的输入特征数据，标签数据
    :return: new_features,label:特征选择后的特征数据、标签数据
    """ 
    new_features= None
    # -------------------------- 实现特征选择部分代码 ----------------------------

    # ------------------------------------------------------------------------
    # 返回筛选后的数据
    return new_features,label

def data_split(features,labels):

    """
    数据切分
    :param  features,label: 特征选择后的输入特征数据、类标数据
    :return: X_train, X_val, X_test,y_train, y_val, y_test:数据切分后的训练数据、验证数据、测试数据
    """ 
    
    X_train, X_val, X_test,y_train, y_val, y_test=None, None,None, None, None, None
    # -------------------------- 实现数据切分部分代码 ----------------------------

    # ------------------------------------------------------------------------

    return X_train, X_val, X_test,y_train, y_val, y_test


def search_model(X_train, y_train,X_val,y_val, model_save_path):
    """
    创建、训练、优化和保存深度学习模型
    :param X_train, y_train: 训练集数据
    :param X_val,y_val: 验证集数据
    :param save_model_path: 保存模型的路径和名称
    :return:
    """
    # --------------------- 实现模型创建、训练、优化和保存等部分的代码 ---------------------

    # 保存模型（请写好保存模型的路径及名称）
    # -------------------------------------------------------------------------

    
def load_and_model_prediction(X_test,y_test,save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型优化过程中的参数选择，测试集数据的准确率、召回率、F-score 等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param X_test,y_test: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------

    # ---------------------------------------------------------------------------



def main():
    """
    监督学习模型训练流程, 包含数据处理、特征选择、训练优化模型、模型保存、评价模型等。  
    如果对训练出来的模型不满意, 你可以通过修改数据处理方法、特征选择方法、调整模型类型和参数等方法重新训练模型, 直至训练出你满意的模型。  
    如果你对自己训练出来的模型非常满意, 则可以进行测试提交! 
    :return:
    """
    data_path = ""  # 数据集路径
    
    save_model_path = ''  # 保存模型路径和名称

    # 获取数据 预处理
    feature1,feature2,label = processing_data(data_path)
    print(feature1)
   
    #特征选择
    new_features,label = feature_select(feature1, feature2, label)
   
    #数据划分
    X_train, X_val, X_test,y_train, y_val, y_test = data_split(new_features,label)
    
    # 创建、训练和保存模型
    search_model(X_train, y_train,X_val,y_val, save_model_path)

    # 评估模型
    load_and_model_prediction(X_test,y_test,save_model_path)


if __name__ == '__main__':
    main()