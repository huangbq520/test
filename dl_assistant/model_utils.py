"""
模型工具模块

提供常用的深度学习模型构建和辅助功能。
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ModelHelper:
    """模型辅助工具类"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """
        统计模型参数数量
        
        参数:
            model: PyTorch模型
            
        返回:
            可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def save_model(model: nn.Module, filepath: str):
        """
        保存模型
        
        参数:
            model: PyTorch模型
            filepath: 保存路径
        """
        torch.save(model.state_dict(), filepath)
        print(f"模型已保存至: {filepath}")
    
    @staticmethod
    def load_model(model: nn.Module, filepath: str, device: str = 'cpu') -> nn.Module:
        """
        加载模型
        
        参数:
            model: PyTorch模型实例
            filepath: 模型文件路径
            device: 运行设备
            
        返回:
            加载参数后的模型
        """
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"模型已从 {filepath} 加载")
        return model
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        冻结模型层参数
        
        参数:
            model: PyTorch模型
            layer_names: 要冻结的层名称列表，None表示冻结所有层
        """
        if layer_names is None:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        
        print(f"已冻结层: {layer_names if layer_names else '所有层'}")
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        解冻模型层参数
        
        参数:
            model: PyTorch模型
            layer_names: 要解冻的层名称列表，None表示解冻所有层
        """
        if layer_names is None:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
        
        print(f"已解冻层: {layer_names if layer_names else '所有层'}")


class ModelBuilder:
    """模型构建器"""
    
    @staticmethod
    def build_mlp(input_dim: int, 
                  hidden_dims: List[int], 
                  output_dim: int,
                  activation: str = 'relu',
                  dropout: float = 0.0) -> nn.Module:
        """
        构建多层感知机(MLP)
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数 ('relu', 'tanh', 'sigmoid')
            dropout: Dropout比例
            
        返回:
            MLP模型
        """
        layers = []
        prev_dim = input_dim
        
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        
        if activation not in activation_map:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_map[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def build_cnn(input_channels: int,
                  num_classes: int,
                  conv_layers: List[Tuple[int, int, int]] = None) -> nn.Module:
        """
        构建简单的CNN模型
        
        参数:
            input_channels: 输入通道数
            num_classes: 分类数量
            conv_layers: 卷积层配置 [(out_channels, kernel_size, stride), ...]
            
        返回:
            CNN模型
        """
        if conv_layers is None:
            conv_layers = [(32, 3, 1), (64, 3, 1)]
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                layers = []
                in_ch = input_channels
                
                for out_ch, kernel, stride in conv_layers:
                    layers.extend([
                        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2)
                    ])
                    in_ch = out_ch
                
                self.features = nn.Sequential(*layers)
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(in_ch, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = self.adaptive_pool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return SimpleCNN()
    
    @staticmethod
    def build_rnn(input_size: int,
                  hidden_size: int,
                  num_layers: int,
                  output_size: int,
                  rnn_type: str = 'lstm',
                  bidirectional: bool = False) -> nn.Module:
        """
        构建RNN模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: RNN层数
            output_size: 输出维度
            rnn_type: RNN类型 ('rnn', 'lstm', 'gru')
            bidirectional: 是否双向
            
        返回:
            RNN模型
        """
        class SimpleRNN(nn.Module):
            def __init__(self):
                super(SimpleRNN, self).__init__()
                
                if rnn_type == 'lstm':
                    self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                                      batch_first=True, bidirectional=bidirectional)
                elif rnn_type == 'gru':
                    self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                                     batch_first=True, bidirectional=bidirectional)
                elif rnn_type == 'rnn':
                    self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                                     batch_first=True, bidirectional=bidirectional)
                else:
                    raise ValueError(f"不支持的RNN类型: {rnn_type}")
                
                fc_input_size = hidden_size * 2 if bidirectional else hidden_size
                self.fc = nn.Linear(fc_input_size, output_size)
            
            def forward(self, x):
                out, _ = self.rnn(x)
                out = self.fc(out[:, -1, :])
                return out
        
        return SimpleRNN()
