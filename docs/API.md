# API 文档

## 目录
- [数据处理模块 (data_utils)](#数据处理模块)
- [模型工具模块 (model_utils)](#模型工具模块)
- [训练工具模块 (training_utils)](#训练工具模块)

---

## 数据处理模块

### DataPreprocessor

数据预处理器类，提供常用的数据预处理方法。

#### normalize()

```python
@staticmethod
def normalize(data: np.ndarray, method: str = 'minmax') -> np.ndarray
```

**功能**: 数据标准化

**参数**:
- `data` (np.ndarray): 输入数据
- `method` (str): 标准化方法
  - `'minmax'`: 最小-最大标准化 (默认)
  - `'zscore'`: Z-score标准化

**返回**: 标准化后的数据

**示例**:
```python
X = np.array([1, 2, 3, 4, 5])
X_norm = DataPreprocessor.normalize(X, method='minmax')
```

#### train_test_split()

```python
@staticmethod
def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    random_state: Optional[int] = None)
```

**功能**: 划分训练集和测试集

**参数**:
- `X` (np.ndarray): 特征数据
- `y` (np.ndarray): 标签数据
- `test_size` (float): 测试集比例 (默认0.2)
- `random_state` (int, optional): 随机种子

**返回**: (X_train, X_test, y_train, y_test)

**示例**:
```python
X_train, X_test, y_train, y_test = DataPreprocessor.train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### one_hot_encode()

```python
@staticmethod
def one_hot_encode(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray
```

**功能**: 独热编码

**参数**:
- `labels` (np.ndarray): 标签数组
- `num_classes` (int, optional): 类别数量

**返回**: 独热编码后的数组

**示例**:
```python
labels = np.array([0, 1, 2, 0, 1])
one_hot = DataPreprocessor.one_hot_encode(labels, num_classes=3)
```

### DataLoader

数据加载器包装类。

#### create_dataloader()

```python
@staticmethod
def create_dataloader(X: np.ndarray, y: np.ndarray, 
                     batch_size: int = 32, 
                     shuffle: bool = True,
                     transform=None)
```

**功能**: 创建PyTorch数据加载器

**参数**:
- `X` (np.ndarray): 特征数据
- `y` (np.ndarray): 标签数据
- `batch_size` (int): 批次大小 (默认32)
- `shuffle` (bool): 是否打乱数据 (默认True)
- `transform`: 数据转换函数 (可选)

**返回**: PyTorch DataLoader对象

**示例**:
```python
train_loader = DataLoader.create_dataloader(
    X_train, y_train, batch_size=64, shuffle=True
)
```

---

## 模型工具模块

### ModelBuilder

模型构建器类，提供快速构建常见模型的方法。

#### build_mlp()

```python
@staticmethod
def build_mlp(input_dim: int, 
              hidden_dims: List[int], 
              output_dim: int,
              activation: str = 'relu',
              dropout: float = 0.0) -> nn.Module
```

**功能**: 构建多层感知机(MLP)

**参数**:
- `input_dim` (int): 输入维度
- `hidden_dims` (List[int]): 隐藏层维度列表
- `output_dim` (int): 输出维度
- `activation` (str): 激活函数 ('relu', 'tanh', 'sigmoid', 'leaky_relu')
- `dropout` (float): Dropout比例 (默认0.0)

**返回**: MLP模型

**示例**:
```python
model = ModelBuilder.build_mlp(
    input_dim=784,
    hidden_dims=[256, 128, 64],
    output_dim=10,
    activation='relu',
    dropout=0.3
)
```

#### build_cnn()

```python
@staticmethod
def build_cnn(input_channels: int,
              num_classes: int,
              conv_layers: List[Tuple[int, int, int]] = None) -> nn.Module
```

**功能**: 构建卷积神经网络(CNN)

**参数**:
- `input_channels` (int): 输入通道数
- `num_classes` (int): 分类数量
- `conv_layers` (List[Tuple]): 卷积层配置 [(out_channels, kernel_size, stride), ...]

**返回**: CNN模型

**示例**:
```python
model = ModelBuilder.build_cnn(
    input_channels=3,
    num_classes=10,
    conv_layers=[(32, 3, 1), (64, 3, 1), (128, 3, 1)]
)
```

#### build_rnn()

```python
@staticmethod
def build_rnn(input_size: int,
              hidden_size: int,
              num_layers: int,
              output_size: int,
              rnn_type: str = 'lstm',
              bidirectional: bool = False) -> nn.Module
```

**功能**: 构建循环神经网络(RNN)

**参数**:
- `input_size` (int): 输入特征维度
- `hidden_size` (int): 隐藏层大小
- `num_layers` (int): RNN层数
- `output_size` (int): 输出维度
- `rnn_type` (str): RNN类型 ('rnn', 'lstm', 'gru')
- `bidirectional` (bool): 是否双向 (默认False)

**返回**: RNN模型

**示例**:
```python
model = ModelBuilder.build_rnn(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    output_size=10,
    rnn_type='lstm',
    bidirectional=True
)
```

### ModelHelper

模型辅助工具类。

#### count_parameters()

```python
@staticmethod
def count_parameters(model: nn.Module) -> int
```

**功能**: 统计模型可训练参数数量

**参数**:
- `model` (nn.Module): PyTorch模型

**返回**: 参数总数

**示例**:
```python
num_params = ModelHelper.count_parameters(model)
print(f'模型参数: {num_params:,}')
```

#### save_model()

```python
@staticmethod
def save_model(model: nn.Module, filepath: str)
```

**功能**: 保存模型参数

**参数**:
- `model` (nn.Module): PyTorch模型
- `filepath` (str): 保存路径

**示例**:
```python
ModelHelper.save_model(model, 'model.pth')
```

#### load_model()

```python
@staticmethod
def load_model(model: nn.Module, filepath: str, device: str = 'cpu') -> nn.Module
```

**功能**: 加载模型参数

**参数**:
- `model` (nn.Module): PyTorch模型实例
- `filepath` (str): 模型文件路径
- `device` (str): 运行设备 (默认'cpu')

**返回**: 加载参数后的模型

**示例**:
```python
model = ModelHelper.load_model(model, 'model.pth', device='cuda')
```

---

## 训练工具模块

### Trainer

模型训练器类。

#### __init__()

```python
def __init__(self, 
             model: nn.Module,
             device: str = 'cpu',
             criterion: Optional[nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None)
```

**功能**: 初始化训练器

**参数**:
- `model` (nn.Module): PyTorch模型
- `device` (str): 运行设备 ('cpu' 或 'cuda')
- `criterion` (nn.Module, optional): 损失函数 (默认CrossEntropyLoss)
- `optimizer` (Optimizer, optional): 优化器 (默认Adam)

**示例**:
```python
trainer = Trainer(
    model=model,
    device='cuda',
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
)
```

#### fit()

```python
def fit(self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping: Optional[EarlyStopping] = None,
        save_best_path: Optional[str] = None) -> Dict[str, List[float]]
```

**功能**: 训练模型

**参数**:
- `train_loader` (DataLoader): 训练数据加载器
- `val_loader` (DataLoader, optional): 验证数据加载器
- `epochs` (int): 训练轮数 (默认10)
- `early_stopping` (EarlyStopping, optional): 早停对象
- `save_best_path` (str, optional): 保存最佳模型的路径

**返回**: 训练历史记录字典

**示例**:
```python
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    early_stopping=EarlyStopping(patience=10),
    save_best_path='best_model.pth'
)
```

#### validate()

```python
def validate(self, dataloader: DataLoader) -> Dict[str, float]
```

**功能**: 验证模型

**参数**:
- `dataloader` (DataLoader): 验证数据加载器

**返回**: 包含loss和accuracy的字典

**示例**:
```python
metrics = trainer.validate(test_loader)
print(f"准确率: {metrics['accuracy']:.2f}%")
```

#### predict()

```python
def predict(self, dataloader: DataLoader) -> np.ndarray
```

**功能**: 使用模型进行预测

**参数**:
- `dataloader` (DataLoader): 数据加载器

**返回**: 预测结果数组

**示例**:
```python
predictions = trainer.predict(test_loader)
```

### EarlyStopping

早停机制类。

#### __init__()

```python
def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min')
```

**功能**: 初始化早停

**参数**:
- `patience` (int): 容忍轮数 (默认7)
- `min_delta` (float): 最小改善值 (默认0.0)
- `mode` (str): 监控模式 ('min'越小越好, 'max'越大越好)

**示例**:
```python
early_stopping = EarlyStopping(patience=10, mode='min')
```

#### __call__()

```python
def __call__(self, current_score: float) -> bool
```

**功能**: 检查是否需要早停

**参数**:
- `current_score` (float): 当前分数

**返回**: 是否应该停止训练

**示例**:
```python
if early_stopping(val_loss):
    print("触发早停!")
    break
```
