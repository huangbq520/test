"""
时间序列预测示例

演示如何使用深度学习助手进行时间序列预测任务
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


def generate_time_series(n_samples=1000, seq_length=50, n_features=1):
    """
    生成时间序列数据
    
    参数:
        n_samples: 样本数量
        seq_length: 序列长度
        n_features: 特征数量
        
    返回:
        时间序列数据和标签
    """
    # 生成正弦波加噪声的时间序列
    t = np.linspace(0, 100, n_samples * seq_length)
    data = np.sin(t) + 0.1 * np.random.randn(n_samples * seq_length)
    
    # 创建序列
    X = []
    y = []
    
    for i in range(n_samples):
        start_idx = i * (seq_length // 2)
        if start_idx + seq_length + 1 > len(data):
            break
        X.append(data[start_idx:start_idx + seq_length])
        y.append(data[start_idx + seq_length])
    
    X = np.array(X).reshape(-1, seq_length, n_features).astype(np.float32)
    y = np.array(y).astype(np.float32)
    
    # 将回归问题转换为分类问题（预测上升/下降/持平）
    y_class = np.zeros(len(y), dtype=np.int64)
    threshold = 0.1
    y_class[y > threshold] = 1  # 上升
    y_class[y < -threshold] = 2  # 下降
    # y_class == 0 表示持平
    
    return X, y_class


def main():
    """主函数"""
    print("=" * 60)
    print("时间序列预测示例")
    print("=" * 60)
    
    # 1. 生成时间序列数据
    print("\n步骤 1: 生成时间序列数据...")
    seq_length = 30
    n_features = 1
    
    X, y = generate_time_series(n_samples=800, seq_length=seq_length, n_features=n_features)
    print(f"序列数据形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print(f"类别分布 (0=持平, 1=上升, 2=下降): {np.bincount(y)}")
    
    # 2. 数据预处理
    print("\n步骤 2: 数据预处理...")
    
    # 标准化
    X_flat = X.reshape(len(X), -1)
    X_normalized = DataPreprocessor.normalize(X_flat, method='zscore')
    X_normalized = X_normalized.reshape(-1, seq_length, n_features)
    
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
    
    # 3. 构建RNN模型
    print("\n步骤 3: 构建LSTM模型...")
    model = ModelBuilder.build_rnn(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        output_size=3,  # 3个类别
        rnn_type='lstm',
        bidirectional=True
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
    early_stopping = EarlyStopping(patience=20, mode='min')
    
    # 开始训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=100,
        early_stopping=early_stopping,
        save_best_path='models/best_rnn_model.pth'
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
    
    # 统计各类别的预测情况
    print("\n各类别预测统计:")
    for i in range(3):
        class_name = ['持平', '上升', '下降'][i]
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == y_test[mask]).mean() * 100
            print(f"  {class_name}: {class_acc:.2f}% (共{mask.sum()}个样本)")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    # 创建模型保存目录
    import os
    os.makedirs('models', exist_ok=True)
    
    main()
