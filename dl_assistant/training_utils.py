"""
训练工具模块

提供模型训练和评估的辅助功能。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, List
from tqdm import tqdm
import numpy as np


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        初始化早停
        
        参数:
            patience: 容忍轮数
            min_delta: 最小改善值
            mode: 'min' 表示越小越好, 'max' 表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"mode必须是'min'或'max', 当前为: {mode}")
    
    def __call__(self, current_score: float) -> bool:
        """
        检查是否需要早停
        
        参数:
            current_score: 当前分数
            
        返回:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cpu',
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        """
        初始化训练器
        
        参数:
            model: PyTorch模型
            device: 运行设备 ('cpu' 或 'cuda')
            criterion: 损失函数
            optimizer: 优化器
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters())
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        
        参数:
            dataloader: 训练数据加载器
            
        返回:
            包含损失和准确率的字典
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='训练中')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                '损失': f'{loss.item():.4f}',
                '准确率': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        参数:
            dataloader: 验证数据加载器
            
        返回:
            包含损失和准确率的字典
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='验证中')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    '损失': f'{loss.item():.4f}',
                    '准确率': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 10,
            early_stopping: Optional[EarlyStopping] = None,
            save_best_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping: 早停对象
            save_best_path: 保存最佳模型的路径
            
        返回:
            训练历史记录
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            print(f"训练损失: {train_metrics['loss']:.4f}, "
                  f"训练准确率: {train_metrics['accuracy']:.2f}%")
            
            # 验证
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])
                
                print(f"验证损失: {val_metrics['loss']:.4f}, "
                      f"验证准确率: {val_metrics['accuracy']:.2f}%")
                
                # 保存最佳模型
                if save_best_path and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save(self.model.state_dict(), save_best_path)
                    print(f"已保存最佳模型至: {save_best_path}")
                
                # 早停检查
                if early_stopping:
                    if early_stopping(val_metrics['loss']):
                        print(f"\n早停触发! 在第 {epoch + 1} 轮停止训练")
                        break
        
        return {
            'train_loss': self.train_losses,
            'train_acc': self.train_accuracies,
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies
        }
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            dataloader: 数据加载器
            
        返回:
            预测结果数组
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc='预测中'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
