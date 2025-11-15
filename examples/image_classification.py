"""
图像分类示例

演示如何使用深度学习助手进行图像分类任务（使用MNIST风格的数据）
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


def generate_simple_images(n_samples=1000, img_size=28, n_classes=10):
    """
    生成简单的图像数据用于演示
    
    参数:
        n_samples: 样本数量
        img_size: 图像大小
        n_classes: 类别数量
        
    返回:
        图像数据和标签
    """
    # 生成随机图像数据 (模拟灰度图像)
    images = np.random.randn(n_samples, 1, img_size, img_size).astype(np.float32)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # 为每个类别添加特定的模式
    for i in range(n_classes):
        mask = labels == i
        # 在特定区域添加亮点
        x_pos = (i % 3) * 8 + 5
        y_pos = (i // 3) * 8 + 5
        images[mask, 0, y_pos:y_pos+3, x_pos:x_pos+3] += 2.0
    
    return images, labels


def main():
    """主函数"""
    print("=" * 60)
    print("图像分类任务示例")
    print("=" * 60)
    
    # 1. 生成图像数据
    print("\n步骤 1: 生成示例图像数据...")
    n_samples = 1000
    img_size = 28
    n_classes = 10
    
    images, labels = generate_simple_images(n_samples, img_size, n_classes)
    print(f"图像数据形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"类别分布: {np.bincount(labels)}")
    
    # 2. 数据预处理
    print("\n步骤 2: 数据预处理...")
    
    # 标准化图像数据
    images_flat = images.reshape(n_samples, -1)
    images_normalized = DataPreprocessor.normalize(images_flat, method='zscore')
    images_normalized = images_normalized.reshape(n_samples, 1, img_size, img_size)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = DataPreprocessor.train_test_split(
        images_normalized, labels, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建数据加载器
    train_loader = DataLoader.create_dataloader(
        X_train, y_train, batch_size=64, shuffle=True
    )
    test_loader = DataLoader.create_dataloader(
        X_test, y_test, batch_size=64, shuffle=False
    )
    
    # 3. 构建CNN模型
    print("\n步骤 3: 构建CNN模型...")
    model = ModelBuilder.build_cnn(
        input_channels=1,
        num_classes=n_classes,
        conv_layers=[(16, 3, 1), (32, 3, 1), (64, 3, 1)]
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
    early_stopping = EarlyStopping(patience=15, mode='min')
    
    # 开始训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=50,
        early_stopping=early_stopping,
        save_best_path='models/best_cnn_model.pth'
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
