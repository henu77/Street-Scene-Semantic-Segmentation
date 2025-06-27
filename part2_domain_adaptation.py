"""
第二部分：跨域适应实验
在cloudy数据集上测试预训练模型，并使用不同损失函数和优化器进行微调
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD, ASGD

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data.dataset import CamVidLoader
from data.augmentation import get_augmentation
from models.unet import UNet
from losses.losses import CrossEntropyLoss, DiceLoss, CombinedLoss
from utils.training import train_model, evaluate_model
from utils.visualization import plot_metrics_comparison


def get_loss_function(loss_type):
    """根据类型获取损失函数"""
    if loss_type == 'ce':
        return CrossEntropyLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'combined':
        return CombinedLoss()
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


def get_optimizer(model, optimizer_type, lr):
    """根据类型获取优化器"""
    if optimizer_type == 'adam':
        return Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'adamw':
        return AdamW(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == 'asgd':
        return ASGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")


def evaluate_pretrained_on_cloudy():
    """评估预训练模型在cloudy数据集上的性能"""
    print("评估预训练模型在cloudy数据集上的性能...")
    
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练模型
    model = UNet(3, NUM_CLASSES, width=32, bilinear=True)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 加载cloudy测试数据
    test_dataset = CamVidLoader(
        root=DATA_ROOT_CLOUDY,
        split='test',
        is_aug=False,
        img_size=IMG_SIZE,
        is_pytorch_transform=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # 评估
    _, _, _, test_accuracy, test_iou = evaluate_model(model, test_loader, device)
    
    print(f"预训练模型在cloudy数据集上的性能:")
    print(f"  全局准确率: {test_accuracy:.4f}")
    print(f"  平均IoU: {test_iou:.4f}")
    
    return test_accuracy, test_iou


def run_experiment(config, experiment_name):
    """运行单个实验"""
    print(f"\n开始实验: {experiment_name}")
    print("-" * 50)
    
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = UNet(3, NUM_CLASSES, width=32, bilinear=True).to(device)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True))
    
    # 获取损失函数和优化器
    criterion = get_loss_function(config['loss'])
    optimizer = get_optimizer(model, config['optimizer'], TRAINING_CONFIG['learning_rate'])
    
    # 数据增强
    aug_fun = get_augmentation(is_strong=False)
    
    # 加载数据
    train_dataset = CamVidLoader(
        root=DATA_ROOT_CLOUDY,
        split='train',
        is_aug=True,
        aug=aug_fun,
        img_size=IMG_SIZE,
        is_pytorch_transform=True
    )
    
    val_dataset = CamVidLoader(
        root=DATA_ROOT_CLOUDY,
        split='val',
        is_aug=False,
        img_size=IMG_SIZE,
        is_pytorch_transform=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        drop_last=False
    )
    
    # 创建保存目录
    save_dir = os.path.join(OUTPUT_DIRS['part2'], experiment_name.replace(' ', '_').replace('+', ''))
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练模型
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        device=device,
        save_path=save_dir
    )
    
    # 测试评估
    test_dataset = CamVidLoader(
        root=DATA_ROOT_CLOUDY,
        split='test',
        is_aug=False,
        img_size=IMG_SIZE,
        is_pytorch_transform=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    _, _, _, test_accuracy, test_iou = evaluate_model(model, test_loader, device)
    
    # 保存结果
    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(f"实验: {experiment_name}\n")
        f.write(f"损失函数: {config['loss']}\n")
        f.write(f"优化器: {config['optimizer']}\n")
        f.write(f"测试准确率: {test_accuracy:.4f}\n")
        f.write(f"测试IoU: {test_iou:.4f}\n")
    
    print(f"实验 {experiment_name} 完成!")
    print(f"  测试准确率: {test_accuracy:.4f}")
    print(f"  测试IoU: {test_iou:.4f}")
    
    return test_accuracy, test_iou


def main():
    """主函数"""
    print("=" * 60)
    print("第二部分：跨域适应实验")
    print("=" * 60)
    
    # 1. 首先评估预训练模型在cloudy数据集上的性能
    pretrain_acc, pretrain_iou = evaluate_pretrained_on_cloudy()
    
    # 2. 运行所有实验配置
    results = {}
    for config in EXPERIMENT_CONFIGS:
        experiment_name = config['name']
        test_acc, test_iou = run_experiment(config, experiment_name)
        results[experiment_name] = {
            'accuracy': test_acc,
            'iou': test_iou,
            'config': config
        }
    
    # 3. 生成对比图表
    print("\n生成指标对比图...")
    experiment_dirs = []
    experiment_names = []
    
    for config in EXPERIMENT_CONFIGS:
        dir_name = config['name'].replace(' ', '_').replace('+', '')
        exp_dir = os.path.join(OUTPUT_DIRS['part2'], dir_name)
        if os.path.exists(exp_dir):
            experiment_dirs.append(exp_dir)
            experiment_names.append(config['name'])
    
    if experiment_dirs:
        plot_metrics_comparison(
            experiment_dirs, experiment_names,
            save_dir=OUTPUT_DIRS['part2']
        )
    
    # 4. 生成总结报告
    print("\n生成总结报告...")
    report_path = os.path.join(OUTPUT_DIRS['part2'], 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("第二部分：跨域适应实验总结报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 预训练模型在cloudy数据集上的性能:\n")
        f.write(f"   全局准确率: {pretrain_acc:.4f}\n")
        f.write(f"   平均IoU: {pretrain_iou:.4f}\n\n")
        
        f.write("2. 不同配置的微调结果:\n")
        f.write(f"{'实验名称':<25} {'准确率':<10} {'IoU':<10} {'损失函数':<12} {'优化器':<10}\n")
        f.write("-" * 70 + "\n")
        
        # 按准确率排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for exp_name, result in sorted_results:
            f.write(f"{exp_name:<25} {result['accuracy']:<10.4f} {result['iou']:<10.4f} "
                   f"{result['config']['loss']:<12} {result['config']['optimizer']:<10}\n")
        
        f.write("\n3. 关键发现:\n")
        best_exp = sorted_results[0]
        f.write(f"   - 最佳配置: {best_exp[0]}\n")
        f.write(f"   - 最佳准确率: {best_exp[1]['accuracy']:.4f}\n")
        f.write(f"   - 最佳IoU: {best_exp[1]['iou']:.4f}\n")
        f.write(f"   - 相比预训练模型提升: 准确率 {best_exp[1]['accuracy'] - pretrain_acc:.4f}, "
               f"IoU {best_exp[1]['iou'] - pretrain_iou:.4f}\n")
    
    print(f"总结报告已保存到: {report_path}")
    print("第二部分完成！")


if __name__ == "__main__":
    main()
