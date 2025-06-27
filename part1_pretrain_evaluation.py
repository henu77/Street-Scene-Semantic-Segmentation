"""
完整的预训练和评估流程
包含：
1. 在Sunny情况下训练模型
2. 模型预训练及性能评估
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from models.unet import UNet
from data.dataset import CamVidLoader
from data.augmentation import get_augmentation
from losses.losses import CombinedLoss, CrossEntropyLoss, DiceLoss
from utils.training import train_model, evaluate_model
from utils.visualization import plot_images
from utils.metrics import global_accuracy_metric, global_iou_metric




def train_sunny_model():
    """训练sunny数据集上的模型"""
    print("=" * 60)
    print("第一步：在Sunny数据集上训练U-Net模型")
    print("=" * 60)

    # 设置设备
    device = torch.device(
        TRAINING_CONFIG["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = PRETRAINED_OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据增强
    train_transform = get_augmentation(is_strong=True)
    val_transform = get_augmentation(is_strong=False)
    
    # 加载数据集
    print("\n正在加载数据集...")
    train_dataset = CamVidLoader(
        root=DATA_ROOT_SUNNY,
        split="train",
        img_size=IMG_SIZE,
        is_aug=True,
        aug=train_transform,
    )

    val_dataset = CamVidLoader(
        root=DATA_ROOT_SUNNY,
        split="val",
        img_size=IMG_SIZE,
        is_aug=True,
        aug=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        drop_last=False,
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 创建模型
    print("\n创建U-Net模型...")
    model = UNet(3, NUM_CLASSES, width=32, bilinear=True)
    model = model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    # 损失函数和优化器
    criterion = CombinedLoss(alpha=0.5)  # 使用组合损失
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])

    # 使用training.py中的train_model函数进行训练
    print("\n开始训练...")
    print("=" * 60)
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=TRAINING_CONFIG["num_epochs"],
        device=device,
        save_path=output_dir
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    
    # 最终评估
    print("\n进行最终评估...")
    # 加载最佳模型进行评估
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        trained_model.load_state_dict(torch.load(best_model_path))
        print(f"加载最佳模型: {best_model_path}")
    
    _, _, _, test_accuracy, test_iou = evaluate_model(trained_model, val_loader, device)
    
    print(f"最终验证指标:")
    print(f"- 准确率: {test_accuracy:.4f}")
    print(f"- IoU: {test_iou:.4f}")
    
    return trained_model


def evaluate_pretrained_model():
    """评估预训练模型"""
    print("\n" + "=" * 60)
    print("第二步：模型预训练及性能评估")
    print("=" * 60)
    
    # 设置设备
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    print(f"加载预训练模型: {PRETRAINED_MODEL_PATH}")
    try:
        model = UNet(3, NUM_CLASSES, width=32, bilinear=True)
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 加载测试数据集
    print(f"加载数据集: {DATA_ROOT_SUNNY}")
    test_dataset = CamVidLoader(
        root=DATA_ROOT_SUNNY,
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
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 模型评估
    print("\n开始模型评估...")
    all_images, all_labels, all_preds, test_accuracy, test_iou = evaluate_model(
        model, test_loader, device
    )
    
    print(f"\n评估结果:")
    print(f"全局准确率: {test_accuracy:.4f}")
    print(f"平均IoU: {test_iou:.4f}")
    
    # 计算每个样本的准确率并排序
    print("\n计算每个样本的准确率...")
    sample_accuracies = []
    for i in range(len(all_labels)):
        accuracy = global_accuracy_metric(all_labels[i], all_preds[i])
        sample_accuracies.append((i, accuracy))
    
    # 按准确率排序
    sample_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("前5个最高准确率样本:")
    for i, (idx, acc) in enumerate(sample_accuracies[:5]):
        print(f"  样本 {idx}: {acc:.4f}")
    
    print("后5个最低准确率样本:")
    for i, (idx, acc) in enumerate(sample_accuracies[-5:]):
        print(f"  样本 {idx}: {acc:.4f}")
    
    # 创建评估结果输出目录
    eval_output_dir = OUTPUT_DIRS['part1']
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # 可视化前5个最好的结果
    top5_indices = [idx for idx, _ in sample_accuracies[:5]]
    top5_images = all_images[top5_indices]
    top5_labels = all_labels[top5_indices]
    top5_preds = all_preds[top5_indices]
    
    save_path = os.path.join(eval_output_dir, 'top5_results.png')
    plot_images(
        top5_images, top5_labels, top5_preds,
        num_images=5,
        title='Top 5 Images based on Global Accuracy',
        save_path=save_path
    )
    
    # 可视化最差的5个结果
    worst5_indices = [idx for idx, _ in sample_accuracies[-5:]]
    worst5_images = all_images[worst5_indices]
    worst5_labels = all_labels[worst5_indices]
    worst5_preds = all_preds[worst5_indices]
    
    save_path = os.path.join(eval_output_dir, 'worst5_results.png')
    plot_images(
        worst5_images, worst5_labels, worst5_preds,
        num_images=5,
        title='Worst 5 Images based on Global Accuracy',
        save_path=save_path
    )
    
    # 保存评估结果
    results_path = os.path.join(eval_output_dir, 'results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("模型预训练及性能评估结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"全局准确率: {test_accuracy:.4f}\n")
        f.write(f"平均IoU: {test_iou:.4f}\n")
        f.write(f"测试样本数量: {len(test_dataset)}\n")
        f.write("\n前5个最高准确率样本:\n")
        for i, (idx, acc) in enumerate(sample_accuracies[:5]):
            f.write(f"  样本 {idx}: {acc:.4f}\n")
        f.write("\n后5个最低准确率样本:\n")
        for i, (idx, acc) in enumerate(sample_accuracies[-5:]):
            f.write(f"  样本 {idx}: {acc:.4f}\n")
    
    print(f"\n结果已保存到: {results_path}")
    return model


def main():
    """主函数"""
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='预训练和评估流程')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='eval',
                       help='执行模式: train(仅训练), eval(仅评估), both(训练+评估)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='指定模型路径(用于评估模式)')
    
    args = parser.parse_args()
    
    # 将global声明移到函数开头
    global PRETRAINED_MODEL_PATH
    
    try:
        print("=" * 80)
        print("完整的预训练和评估流程")
        print("=" * 80)
        
        if args.mode in ['train', 'both']:
            print("执行训练模式...")
            trained_model = train_sunny_model()
        
        if args.mode in ['eval', 'both']:
            print("执行评估模式...")
            # 确定模型路径
            model_path = args.model_path if args.model_path else PRETRAINED_MODEL_PATH
            
            if os.path.exists(model_path):
                # 如果指定了自定义模型路径，临时修改全局变量
                if args.model_path:
                    original_path = PRETRAINED_MODEL_PATH
                    PRETRAINED_MODEL_PATH = args.model_path
                    try:
                        evaluate_pretrained_model()
                    finally:
                        PRETRAINED_MODEL_PATH = original_path
                else:
                    evaluate_pretrained_model()
            else:
                print(f"\n预训练模型文件不存在: {model_path}")
                if args.mode == 'eval':
                    print("错误：评估模式需要存在的模型文件")
                    return
                else:
                    print("跳过详细评估步骤")

        print("\n" + "=" * 80)
        print("所有任务完成!")
        print("=" * 80)

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()