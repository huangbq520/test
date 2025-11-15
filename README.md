# 中文深度学习代码助手 (Chinese Deep Learning Code Assistant)

一个简化深度学习开发流程的Python工具包，提供中文文档和易用的API接口。

## 🌟 特性

- 🔧 **数据处理**: 提供数据加载、预处理、标准化等常用功能
- 🧠 **模型构建**: 快速构建MLP、CNN、RNN等常见模型
- 📊 **训练工具**: 简化的训练流程，支持早停、模型保存等
- 🇨🇳 **中文支持**: 完整的中文文档和注释
- 💡 **易于使用**: 简洁的API设计，降低学习成本

## 📦 安装

### 基础安装

```bash
pip install -r requirements.txt
```

### 开发安装

```bash
pip install -e .
```

## 🚀 快速开始

### 1. 数据预处理

```python
from dl_assistant import DataPreprocessor, DataLoader
import numpy as np

# 创建示例数据
X = np.random.randn(1000, 10)
y = np.random.randint(0, 3, 1000)

# 数据标准化
X_normalized = DataPreprocessor.normalize(X, method='zscore')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = DataPreprocessor.train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

# 创建数据加载器
train_loader = DataLoader.create_dataloader(X_train, y_train, batch_size=32)
test_loader = DataLoader.create_dataloader(X_test, y_test, batch_size=32, shuffle=False)
```

### 2. 构建模型

```python
from dl_assistant import ModelBuilder

# 构建多层感知机
model = ModelBuilder.build_mlp(
    input_dim=10,
    hidden_dims=[64, 32],
    output_dim=3,
    activation='relu',
    dropout=0.2
)

# 构建CNN模型
cnn_model = ModelBuilder.build_cnn(
    input_channels=3,
    num_classes=10,
    conv_layers=[(32, 3, 1), (64, 3, 1)]
)

# 构建RNN模型
rnn_model = ModelBuilder.build_rnn(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    output_size=3,
    rnn_type='lstm',
    bidirectional=True
)
```

### 3. 训练模型

```python
from dl_assistant import Trainer, EarlyStopping
import torch.nn as nn
import torch.optim as optim

# 创建训练器
trainer = Trainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001)
)

# 设置早停
early_stopping = EarlyStopping(patience=5, mode='min')

# 训练模型
history = trainer.fit(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=50,
    early_stopping=early_stopping,
    save_best_path='best_model.pth'
)
```

### 4. 模型评估和预测

```python
from dl_assistant import ModelHelper

# 统计模型参数
num_params = ModelHelper.count_parameters(model)
print(f'模型参数数量: {num_params:,}')

# 验证模型
val_metrics = trainer.validate(test_loader)
print(f"测试准确率: {val_metrics['accuracy']:.2f}%")

# 进行预测
predictions = trainer.predict(test_loader)
```

## 📚 完整示例

查看 `examples/` 目录获取更多示例：

- `basic_classification.py` - 基础分类任务
- `image_classification.py` - 图像分类示例
- `time_series.py` - 时间序列预测

## 🔧 模块说明

### data_utils
数据处理工具模块，包含：
- `DataPreprocessor`: 数据预处理器
- `DataLoader`: 数据加载器
- `CustomDataset`: 自定义数据集

### model_utils
模型工具模块，包含：
- `ModelBuilder`: 模型构建器
- `ModelHelper`: 模型辅助工具

### training_utils
训练工具模块，包含：
- `Trainer`: 训练器
- `EarlyStopping`: 早停机制

## 🛠️ 依赖项

- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- tqdm >= 4.60.0

## 📝 许可证

MIT License

## 🤝 贡献

欢迎贡献代码和提出建议！

## 📧 联系方式

如有问题或建议，请提交Issue。