"""
中文深度学习代码助手

这是一个帮助简化深度学习开发流程的Python工具包。
提供了数据处理、模型构建、训练等常用功能。
"""

__version__ = "0.1.0"
__author__ = "深度学习助手团队"

from .data_utils import DataLoader, DataPreprocessor
from .model_utils import ModelBuilder, ModelHelper
from .training_utils import Trainer, EarlyStopping

__all__ = [
    'DataLoader',
    'DataPreprocessor', 
    'ModelBuilder',
    'ModelHelper',
    'Trainer',
    'EarlyStopping',
]
