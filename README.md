# 基于UNet的街景图像分割项目

本项目实现了基于UNet架构的街景图像分割方法，主要针对CamVid数据集进行预训练、跨域适应和小目标优化实验。

## 🚀 项目概述

本项目包含三个主要实验部分：
- **Part 1**: 预训练和评估（Sunny数据集）
- **Part 2**: 跨域适应实验（Cloudy数据集）  
- **Part 3**: 小目标分割优化

项目支持两种UNet模型：
- 标准UNet
- 改进的UNet（增强小目标分割能力）

## 📊 数据集

项目使用CamVid数据集，包含14个语义分割类别：
- Sky, Building, Pole, Road, LaneMarking, SideWalk, Pavement
- Tree, SignSymbol, Fence, Car_Bus, Pedestrian, Bicyclist, Unlabelled

数据集包含两种天气条件：
- **Sunny**: 晴天场景
- **Cloudy**: 多云场景

## 🛠️ 环境配置

### 系统要求
- Python 3.10+
- CUDA 11.8+ 
- Windows 10/11 或 Linux

### 方法一：使用Conda环境 (推荐)

1. **安装Anaconda或Miniconda**
   ```bash
   # 下载并安装Anaconda
   # https://www.anaconda.com/products/distribution
   ```

2. **创建并激活环境**
   ```bash
   # 使用environment.yml创建环境
   conda env create -f environment.yml
   
   # 激活环境
   conda activate unet
   ```

### 方法二：使用pip安装

1. **创建虚拟环境**
   ```bash
   python -m venv unet_env
   
   # Windows
   unet_env\Scripts\activate
   
   # Linux/Mac
   source unet_env/bin/activate
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 核心依赖包

- **深度学习框架**: PyTorch 2.4.1
- **图像处理**: PIL, OpenCV, scikit-image
- **数据科学**: NumPy, Matplotlib, tqdm
- **其他**: colorama, networkx

## 📁 项目结构

```
M/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包
├── environment.yml             # Conda环境配置
├── config.py                   # 项目配置文件
├── run.py                      # 主实验运行脚本
├── 技术报告.md                  # 详细技术报告
│
├── CamVid/                     # 数据集目录
│   ├── sunny/                  # 晴天数据
│   └── cloudy/                 # 多云数据
│
├── models/                     # 模型定义
│   ├── unet.py                 # 标准UNet
│   └── improved_unet.py        # 改进UNet
│
├── data/                       # 数据处理模块
│   ├── dataset.py              # 数据集类
│   ├── augmentation.py         # 数据增强
│   └── statistics/             # 数据统计
│
├── losses/                     # 损失函数
│   └── losses.py               # 各种损失函数实现
│
├── utils/                      # 工具函数
│   ├── metrics.py              # 评估指标
│   ├── training.py             # 训练工具
│   ├── object_metrics.py       # 目标级指标
│   └── visualization.py        # 可视化工具
│
├── outputs/                    # 实验结果
│   ├── part1_pretrain_evaluation/
│   ├── part2_domain_adaptation/
│   └── part3_small_object_optimization/
│
└── visualization/              # 预测结果可视化
    ├── unet_predictions/
    └── improved_unet_predictions/
```

## 🏃‍♂️ 运行指南

### 完整实验流程

使用主脚本运行所有三个部分的实验：

```bash
python run.py
```

### 分步骤运行

#### Part 1: 预训练和评估
```bash
python part1_pretrain_evaluation.py
```
- 在Sunny数据集上训练UNet模型
- 评估模型性能并生成可视化结果
- 保存最佳模型权重

#### Part 2: 跨域适应实验
```bash
python part2_domain_adaptation.py
```
- 使用预训练模型在Cloudy数据集上进行微调
- 比较不同损失函数和优化器的效果
- 生成跨域适应性能报告

#### Part 3: 小目标优化
```bash
python part3_small_object_optimization.py
```
- 针对小目标类别（Car_Bus, LaneMarking, Pedestrian）进行优化
- 比较标准UNet和改进UNet的性能
- 生成小目标分割改进分析

### 单独预测
```bash
python predict_and_save.py
```

### 配置参数

主要配置参数在 `config.py` 中设置：

```python
# 数据路径
DATA_ROOT_SUNNY = 'CamVid/sunny'
DATA_ROOT_CLOUDY = 'CamVid/cloudy'

# 图像尺寸
IMG_SIZE = [384, 384]

# 类别数量
NUM_CLASSES = 14

# 小目标类别
SMALL_OBJECT_CLASSES = ['Car_Bus', 'LaneMarking', 'Pedestrian', ...]
```

## 📈 实验结果

### Part 1: 预训练结果
- 模型在Sunny数据集上的基准性能
- 各类别的分割精度和IoU指标
- 最佳和最差预测样例展示

### Part 2: 跨域适应结果
- 不同损失函数（CE Loss, Dice Loss, Combined Loss）的比较
- 不同优化器（Adam, AdamW, SGD, ASGD）的性能对比
- 训练和验证指标的对比分析

### Part 3: 小目标优化结果
- 改进UNet在小目标分割上的提升效果
- Car_Bus, LaneMarking, Pedestrian类别的详细分析
- 基线模型与改进模型的性能对比

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小batch size
   # 在config.py中调整BATCH_SIZE参数
   ```

2. **数据集路径错误**
   ```bash
   # 确保CamVid数据集正确放置在项目根目录
   # 检查config.py中的路径配置
   ```

3. **依赖包版本冲突**
   ```bash
   # 建议使用conda环境隔离
   conda env create -f environment.yml
   ```

### 性能优化建议

- 使用GPU加速训练（需要CUDA支持）
- 根据显存大小调整batch_size
- 使用数据并行训练加速

## 📝 技术报告

详细的技术实现和实验分析请参考 `技术报告.md` 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目仅供学习和研究使用。

## 📞 联系方式

如有问题或建议，请通过Issue或邮件联系。

---

**最后更新**: 2025年6月27日
