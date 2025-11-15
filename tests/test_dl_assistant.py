"""
深度学习助手单元测试
"""

import unittest
import numpy as np
import torch
import sys
import os

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_assistant import (
    DataPreprocessor,
    DataLoader,
    ModelBuilder,
    ModelHelper,
    Trainer,
    EarlyStopping
)


class TestDataPreprocessor(unittest.TestCase):
    """测试数据预处理器"""
    
    def test_normalize_minmax(self):
        """测试MinMax标准化"""
        data = np.array([1, 2, 3, 4, 5]).astype(float)
        normalized = DataPreprocessor.normalize(data, method='minmax')
        self.assertAlmostEqual(normalized.min(), 0.0)
        self.assertAlmostEqual(normalized.max(), 1.0)
    
    def test_normalize_zscore(self):
        """测试Z-score标准化"""
        data = np.array([1, 2, 3, 4, 5]).astype(float)
        normalized = DataPreprocessor.normalize(data, method='zscore')
        self.assertAlmostEqual(normalized.mean(), 0.0, places=10)
        self.assertAlmostEqual(normalized.std(), 1.0, places=10)
    
    def test_train_test_split(self):
        """测试数据集划分"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = DataPreprocessor.train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
    
    def test_one_hot_encode(self):
        """测试独热编码"""
        labels = np.array([0, 1, 2, 0, 1])
        one_hot = DataPreprocessor.one_hot_encode(labels, num_classes=3)
        
        self.assertEqual(one_hot.shape, (5, 3))
        self.assertEqual(one_hot.sum(), 5)  # 每行只有一个1


class TestModelBuilder(unittest.TestCase):
    """测试模型构建器"""
    
    def test_build_mlp(self):
        """测试MLP构建"""
        model = ModelBuilder.build_mlp(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=5,
            activation='relu'
        )
        
        self.assertIsInstance(model, torch.nn.Module)
        
        # 测试前向传播
        x = torch.randn(2, 10)
        output = model(x)
        self.assertEqual(output.shape, (2, 5))
    
    def test_build_cnn(self):
        """测试CNN构建"""
        model = ModelBuilder.build_cnn(
            input_channels=3,
            num_classes=10,
            conv_layers=[(32, 3, 1)]
        )
        
        self.assertIsInstance(model, torch.nn.Module)
        
        # 测试前向传播
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (2, 10))
    
    def test_build_rnn(self):
        """测试RNN构建"""
        model = ModelBuilder.build_rnn(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            output_size=5,
            rnn_type='lstm'
        )
        
        self.assertIsInstance(model, torch.nn.Module)
        
        # 测试前向传播
        x = torch.randn(2, 15, 10)  # batch_size, seq_len, input_size
        output = model(x)
        self.assertEqual(output.shape, (2, 5))


class TestModelHelper(unittest.TestCase):
    """测试模型辅助工具"""
    
    def test_count_parameters(self):
        """测试参数统计"""
        model = ModelBuilder.build_mlp(
            input_dim=10,
            hidden_dims=[20],
            output_dim=5
        )
        
        num_params = ModelHelper.count_parameters(model)
        # (10*20 + 20) + (20*5 + 5) = 220 + 105 = 325
        self.assertEqual(num_params, 325)


class TestEarlyStopping(unittest.TestCase):
    """测试早停机制"""
    
    def test_early_stopping_min(self):
        """测试最小值早停"""
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # 第一次调用
        self.assertFalse(early_stopping(1.0))
        # 改善
        self.assertFalse(early_stopping(0.8))
        # 没有改善 - counter=1
        self.assertFalse(early_stopping(0.9))
        # counter=2
        self.assertFalse(early_stopping(0.9))
        # counter=3, 触发早停
        self.assertTrue(early_stopping(0.9))
    
    def test_early_stopping_max(self):
        """测试最大值早停"""
        early_stopping = EarlyStopping(patience=2, mode='max')
        
        # 第一次调用
        self.assertFalse(early_stopping(0.5))
        # 改善
        self.assertFalse(early_stopping(0.7))
        # 没有改善 - counter=1
        self.assertFalse(early_stopping(0.6))
        # counter=2, 触发早停
        self.assertTrue(early_stopping(0.6))


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def test_create_dataloader(self):
        """测试创建数据加载器"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        dataloader = DataLoader.create_dataloader(X, y, batch_size=10)
        
        # 检查批次数量
        batches = list(dataloader)
        self.assertEqual(len(batches), 10)
        
        # 检查批次形状
        batch_x, batch_y = batches[0]
        self.assertEqual(batch_x.shape, (10, 10))
        self.assertEqual(batch_y.shape, (10,))


if __name__ == '__main__':
    unittest.main()
