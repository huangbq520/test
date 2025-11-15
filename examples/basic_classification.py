"""
基础分类任务示例

演示如何使用深度学习助手进行简单的分类任务
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dl_assistant import (
    DataPreprocessor, 
    DataLoader, 
    ModelBuilder, 
    ModelHelper,
    Trainer, 
    EarlyStopping
)


def main():
    """主函数"""
    print("=" * 60)
    print("基础分类任务示例")
    print("=" * 60)
    
    # 1. 生成示例数据
    print("\n步骤 1: 生成示例数据...")
    np.random.seed(42)
    
    # 创建3类数据，每类300个样本
    n_samples = 300
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples * n_classes, n_features)
    y = np.repeat(np.arange(n_classes), n_samples)
    
    # 为每一类添加特征偏移，使其更容易分类
    for i in range(n_classes):
        X[y == i] += i * 2
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 2. 数据预处理
    print("\n步骤 2: 数据预处理...")
    
    # 数据标准化
    X_normalized = DataPreprocessor.normalize(X, method='zscore')
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = DataPreprocessor.train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建数据加载器
    train_loader = DataLoader.create_dataloader(
        X_train, y_train, batch_size=32, shuffle=True
    )
    test_loader = DataLoader.create_dataloader(
        X_test, y_test, batch_size=32, shuffle=False
    )
    
    # 3. 构建模型
    print("\n步骤 3: 构建模型...")
    model = ModelBuilder.build_mlp(
        input_dim=n_features,
        hidden_dims=[64, 32, 16],
        output_dim=n_classes,
        activation='relu',
        dropout=0.3
    )
    
    print(f"模型架构:\n{model}")
    
    # 统计参数数量
    num_params = ModelHelper.count_parameters(model)
    print(f"\n模型参数数量: {num_params:,}")
    
    # 4. 设置训练器
    print("\n步骤 4: 设置训练器...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    trainer = Trainer(
        model=model,
        device=device,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001)
    )
    
    # 5. 训练模型
    print("\n步骤 5: 训练模型...")
    
    # 设置早停
    early_stopping = EarlyStopping(patience=10, mode='min')
    
    # 开始训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=100,
        early_stopping=early_stopping,
        save_best_path='models/best_classification_model.pth'
    )
    
    # 6. 评估模型
    print("\n步骤 6: 评估模型...")
    val_metrics = trainer.validate(test_loader)
    print(f"\n最终测试准确率: {val_metrics['accuracy']:.2f}%")
    print(f"最终测试损失: {val_metrics['loss']:.4f}")
    
    # 7. 进行预测
    print("\n步骤 7: 进行预测...")
    predictions = trainer.predict(test_loader)
    print(f"预测结果示例 (前10个): {predictions[:10]}")
    print(f"真实标签 (前10个): {y_test[:10]}")
    
    # 计算准确率
    accuracy = (predictions == y_test).mean() * 100
    print(f"\n预测准确率: {accuracy:.2f}%")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    # 创建模型保存目录
    import os
    os.makedirs('models', exist_ok=True)
    
    main()
