"""
数据处理工具模块

提供常用的数据加载和预处理功能。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


class DataPreprocessor:
    """数据预处理器"""
    
    @staticmethod
    def normalize(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        数据标准化
        
        参数:
            data: 输入数据
            method: 标准化方法 ('minmax' 或 'zscore')
            
        返回:
            标准化后的数据
        """
        if method == 'minmax':
            min_val = data.min()
            max_val = data.max()
            if max_val - min_val == 0:
                return data
            return (data - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean = data.mean()
            std = data.std()
            if std == 0:
                return data
            return (data - mean) / std
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    @staticmethod
    def train_test_split(X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练集和测试集
        
        参数:
            X: 特征数据
            y: 标签数据
            test_size: 测试集比例
            random_state: 随机种子
            
        返回:
            X_train, X_test, y_train, y_test
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        test_size_n = int(n_samples * test_size)
        
        test_indices = indices[:test_size_n]
        train_indices = indices[test_size_n:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    @staticmethod
    def one_hot_encode(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        """
        独热编码
        
        参数:
            labels: 标签数组
            num_classes: 类别数量
            
        返回:
            独热编码后的数组
        """
        if num_classes is None:
            num_classes = int(labels.max()) + 1
        
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1
        return one_hot


class CustomDataset(Dataset):
    """自定义PyTorch数据集"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        """
        初始化数据集
        
        参数:
            X: 特征数据
            y: 标签数据
            transform: 数据转换函数
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class DataLoader:
    """数据加载器包装类"""
    
    @staticmethod
    def create_dataloader(X: np.ndarray, y: np.ndarray, 
                         batch_size: int = 32, 
                         shuffle: bool = True,
                         transform=None) -> TorchDataLoader:
        """
        创建PyTorch数据加载器
        
        参数:
            X: 特征数据
            y: 标签数据
            batch_size: 批次大小
            shuffle: 是否打乱数据
            transform: 数据转换函数
            
        返回:
            PyTorch DataLoader对象
        """
        dataset = CustomDataset(X, y, transform)
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def load_csv(filepath: str, label_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        从CSV文件加载数据
        
        参数:
            filepath: CSV文件路径
            label_column: 标签列名称
            
        返回:
            特征DataFrame和标签Series (如果指定)
        """
        df = pd.read_csv(filepath)
        
        if label_column:
            labels = df[label_column]
            features = df.drop(columns=[label_column])
            return features, labels
        
        return df, None
