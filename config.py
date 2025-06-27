"""
项目配置文件
"""
import os

# 数据路径配置
DATA_ROOT_SUNNY = 'CamVid/sunny'
DATA_ROOT_CLOUDY = 'CamVid/cloudy'

# 预训练模型路径
PRETRAINED_MODEL_PATH = 'outputs/part1_pretrain_evaluation/best_model.pth'
PRETRAINED_OUTPUT_PATH = 'outputs/part1_pretrain_evaluation'
# 图像尺寸
IMG_SIZE = [384, 384]

# 类别数量
NUM_CLASSES = 14

# 类别标签映射
CLASS_LABELS_DICT = {
    "Sky": 0, "Building": 1, "Pole": 2, "Road": 3, "LaneMarking": 4, 
    "SideWalk": 5, "Pavement": 6, "Tree": 7, "SignSymbol": 8, "Fence": 9, 
    "Car_Bus": 10, "Pedestrian": 11, "Bicyclist": 12, "Unlabelled": 13
}

# 小目标类别（用于Part 3）
SMALL_OBJECT_CLASSES = ['Car_Bus', 'LaneMarking', 'Pedestrian']
SMALL_OBJECT_IDS = [10, 4, 11]

# 训练超参数
TRAINING_CONFIG = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'device': 'cuda',
    'num_workers': 16
}

# 实验配置（用于Part 2）
EXPERIMENT_CONFIGS = [
    {'name': 'CE Loss + Adam', 'loss': 'ce', 'optimizer': 'adam'},
    {'name': 'Dice Loss + Adam', 'loss': 'dice', 'optimizer': 'adam'},
    {'name': 'Combined Loss + Adam', 'loss': 'combined', 'optimizer': 'adam'},
    {'name': 'Combined Loss + AdamW', 'loss': 'combined', 'optimizer': 'adamw'},
    {'name': 'Combined Loss + SGD', 'loss': 'combined', 'optimizer': 'sgd'},
    {'name': 'Combined Loss + ASGD', 'loss': 'combined', 'optimizer': 'asgd'}
]

# 输出目录
OUTPUT_DIRS = {
    'part1': 'outputs/part1_pretrain_evaluation',
    'part2': 'outputs/part2_domain_adaptation', 
    'part3': 'outputs/part3_small_object_optimization'
}

# 确保输出目录存在
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)
