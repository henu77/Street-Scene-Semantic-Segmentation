"""
可视化工具
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.ndimage import gaussian_filter1d
from data.dataset import from_label_to_rgb
# 计算全局准确率
from .metrics import global_accuracy_metric, global_iou_metric
from config import *

def plot_images(imgs, labels, preds, num_images=5, title='Top 5 Images based on Global Accuracy', save_path=None):
    """绘制图像及其对应的标签和预测结果"""
    # 定义均值，与dataset.py中保持一致
    mean = np.array([104.00699/255.0, 116.66877/255.0, 122.67892/255.0])
    
    # 设置图像 - 调整子图间距
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    fig.suptitle(title, fontsize=16, y=0.995)  # 调整主标题位置，避免重合

    for i in range(num_images):
        # 加载图像
        img = imgs[i].copy()  # 创建副本避免修改原始数据
        
        # 反归一化处理：将减去的均值加回来
        img_denorm = img.transpose(1, 2, 0)  # CHW -> HWC
        img_denorm = img_denorm + mean  # 加回均值
        img_denorm = np.clip(img_denorm, 0, 1)  # 裁剪到 [0,1] 范围
        
        # 获取标签和预测结果
        label = labels[i]
        pred = preds[i]

        # 绘制原始图像（反归一化后）
        axes[i, 0].imshow(img_denorm)
        axes[i, 0].axis('off')

        # 绘制标签
        label_rgb = from_label_to_rgb(label)
        axes[i, 1].imshow(label_rgb)
        axes[i, 1].axis('off')

        # 计算准确率和IoU
        global_accuracy = global_accuracy_metric(label, pred)
        global_iou = global_iou_metric(label, pred, num_classes=NUM_CLASSES)
        
        # 绘制预测结果
        pred_rgb = from_label_to_rgb(pred)
        axes[i, 2].imshow(pred_rgb)
        # 将指标信息显示在图像底部，使用xlabel
        axes[i, 2].axis('off')
        
        # 使用text在图像底部显示指标信息
        axes[i, 2].text(0.5, -0.05, f'Acc: {global_accuracy:.3f} | IoU: {global_iou:.3f}', 
                        transform=axes[i, 2].transAxes, ha='center', va='top',
                        fontsize=10)
        


        # 只为第一行设置列标题，并调整位置避免与主标题重合
        if i == 0:
            axes[i, 0].set_title('Image', fontsize=14)
            axes[i, 1].set_title('Label', fontsize=14)
            axes[i, 2].set_title('Predicted', fontsize=14)
    
    # 调整子图布局，为主标题留出空间
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def plot_training_curves(train_loss, val_loss, train_acc, val_acc, train_iou, val_iou, save_path=None, smooth=True, sigma=2.0):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # 应用平滑处理
    if smooth:
        train_loss_smooth = smooth_data(train_loss, sigma)
        val_loss_smooth = smooth_data(val_loss, sigma)
        train_acc_smooth = smooth_data(train_acc, sigma)
        val_acc_smooth = smooth_data(val_acc, sigma)
        train_iou_smooth = smooth_data(train_iou, sigma)
        val_iou_smooth = smooth_data(val_iou, sigma)
    else:
        train_loss_smooth = train_loss
        val_loss_smooth = val_loss
        train_acc_smooth = train_acc
        val_acc_smooth = val_acc
        train_iou_smooth = train_iou
        val_iou_smooth = val_iou
    
    plt.subplot(1, 3, 1)
    if smooth:
        # 绘制原始数据（低透明度）
        plt.plot(train_loss, label='_nolegend_', alpha=0.3, color='blue')
        plt.plot(val_loss, label='_nolegend_', alpha=0.3, color='orange')
        # 绘制平滑数据
        plt.plot(train_loss_smooth, label='Train Loss (Smoothed)', color='blue', linewidth=2)
        plt.plot(val_loss_smooth, label='Validation Loss (Smoothed)', color='orange', linewidth=2)
    else:
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    if smooth:
        plt.plot(train_acc, label='_nolegend_', alpha=0.3, color='blue')
        plt.plot(val_acc, label='_nolegend_', alpha=0.3, color='orange')
        plt.plot(train_acc_smooth, label='Train Global Accuracy (Smoothed)', color='blue', linewidth=2)
        plt.plot(val_acc_smooth, label='Validation Global Accuracy (Smoothed)', color='orange', linewidth=2)
    else:
        plt.plot(train_acc, label='Train Global Accuracy')
        plt.plot(val_acc, label='Validation Global Accuracy')
    plt.title('Global Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if smooth:
        plt.plot(train_iou, label='_nolegend_', alpha=0.3, color='blue')
        plt.plot(val_iou, label='_nolegend_', alpha=0.3, color='orange')
        plt.plot(train_iou_smooth, label='Train Mean IoU (Smoothed)', color='blue', linewidth=2)
        plt.plot(val_iou_smooth, label='Validation Mean IoU (Smoothed)', color='orange', linewidth=2)
    else:
        plt.plot(train_iou, label='Train Mean IoU')
        plt.plot(val_iou, label='Validation Mean IoU')
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()


def smooth_data(data, sigma=2.0):
    """应用高斯平滑到数据"""
    return gaussian_filter1d(data, sigma=sigma)


def plot_metrics_comparison(experiment_dirs, experiment_names, colors=None, save_dir='./'):
    """绘制多个实验的指标对比（支持csv格式）"""
    import pandas as pd
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 训练指标对比
    plt.figure(figsize=(18, 6))
    
    # 1. 训练损失
    plt.subplot(1, 3, 1)
    for i, exp_dir in enumerate(experiment_dirs):
        metrics_path = os.path.join(exp_dir, 'metrics.csv')
        try:
            metrics = pd.read_csv(metrics_path)
            raw_data = metrics['train_loss'].values
            plt.plot(raw_data, label='_nolegend_', color=colors[i], alpha=0.2)
            smoothed_data = smooth_data(raw_data)
            plt.plot(smoothed_data, label=experiment_names[i], color=colors[i], linewidth=2)
        except Exception as e:
            print(f"无法加载 {metrics_path}: {e}")

    plt.title('Training Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)

    # 2. 训练准确率
    plt.subplot(1, 3, 2)
    for i, exp_dir in enumerate(experiment_dirs):
        metrics_path = os.path.join(exp_dir, 'metrics.csv')
        try:
            metrics = pd.read_csv(metrics_path)
            raw_data = metrics['train_accuracy'].values
            plt.plot(raw_data, label='_nolegend_', color=colors[i], alpha=0.2)
            smoothed_data = smooth_data(raw_data)
            plt.plot(smoothed_data, label=experiment_names[i], color=colors[i], linewidth=2)
        except Exception as e:
            print(f"无法加载 {metrics_path}: {e}")

    plt.title('Training Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)

    # 3. 训练IoU
    plt.subplot(1, 3, 3)
    for i, exp_dir in enumerate(experiment_dirs):
        metrics_path = os.path.join(exp_dir, 'metrics.csv')
        try:
            metrics = pd.read_csv(metrics_path)
            raw_data = metrics['train_iou'].values
            plt.plot(raw_data, label='_nolegend_', color=colors[i], alpha=0.2)
            smoothed_data = smooth_data(raw_data)
            plt.plot(smoothed_data, label=experiment_names[i], color=colors[i], linewidth=2)
        except Exception as e:
            print(f"无法加载 {metrics_path}: {e}")

    plt.title('Training Mean IoU Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean IoU', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练指标对比图已保存到: {save_path}")
    plt.show()

    # 验证指标对比
    plt.figure(figsize=(18, 6))
    
    # 验证损失
    plt.subplot(1, 3, 1)
    for i, exp_dir in enumerate(experiment_dirs):
        metrics_path = os.path.join(exp_dir, 'metrics.csv')
        try:
            metrics = pd.read_csv(metrics_path)
            raw_data = metrics['val_loss'].values
            plt.plot(raw_data, label='_nolegend_', color=colors[i], alpha=0.2)
            smoothed_data = smooth_data(raw_data)
            plt.plot(smoothed_data, label=experiment_names[i], color=colors[i], linewidth=2)
        except Exception as e:
            print(f"无法加载 {metrics_path}: {e}")

    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)

    # 验证准确率
    plt.subplot(1, 3, 2)
    for i, exp_dir in enumerate(experiment_dirs):
        metrics_path = os.path.join(exp_dir, 'metrics.csv')
        try:
            metrics = pd.read_csv(metrics_path)
            raw_data = metrics['val_accuracy'].values
            plt.plot(raw_data, label='_nolegend_', color=colors[i], alpha=0.2)
            smoothed_data = smooth_data(raw_data)
            plt.plot(smoothed_data, label=experiment_names[i], color=colors[i], linewidth=2)
        except Exception as e:
            print(f"无法加载 {metrics_path}: {e}")

    plt.title('Validation Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)

    # 验证IoU
    plt.subplot(1, 3, 3)
    for i, exp_dir in enumerate(experiment_dirs):
        metrics_path = os.path.join(exp_dir, 'metrics.csv')
        try:
            metrics = pd.read_csv(metrics_path)
            raw_data = metrics['val_iou'].values
            plt.plot(raw_data, label='_nolegend_', color=colors[i], alpha=0.2)
            smoothed_data = smooth_data(raw_data)
            plt.plot(smoothed_data, label=experiment_names[i], color=colors[i], linewidth=2)
        except Exception as e:
            print(f"无法加载 {metrics_path}: {e}")

    plt.title('Validation Mean IoU Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean IoU', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'validation_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"验证指标对比图已保存到: {save_path}")
    plt.show()


def visualize_segmentation_comparison(gt, pred, indices, class_name, class_id, save_path=None):
    """可视化分割结果对比"""
    plt.figure(figsize=(20, 6*len(indices)))
    
    plt.suptitle(f'{class_name} Segmentation Comparison (Green=Correct, Red=Missed, Blue=False Positive)', 
                 fontsize=16, y=0.98)
    
    for i, idx in enumerate(indices):
        gt_mask_i = gt[idx] == class_id
        pred_mask_i = pred[idx] == class_id
        
        # 计算差异
        missed = gt_mask_i & ~pred_mask_i  # 假阴性
        false_pos = ~gt_mask_i & pred_mask_i  # 假阳性
        correct = gt_mask_i & pred_mask_i  # 真阳性
        
        # 可视化
        plt.subplot(len(indices), 3, i*3+1)
        plt.imshow(gt_mask_i, cmap='gray')
        plt.title(f'Image {idx} - Ground Truth')
        plt.axis('off')
        
        plt.subplot(len(indices), 3, i*3+2)
        plt.imshow(pred_mask_i, cmap='gray')
        plt.title(f'Image {idx} - Prediction')
        plt.axis('off')
        
        plt.subplot(len(indices), 3, i*3+3)
        overlay = np.zeros((gt_mask_i.shape[0], gt_mask_i.shape[1], 3))
        overlay[correct] = [0, 1, 0]  # 绿色：正确
        overlay[missed] = [1, 0, 0]   # 红色：遗漏
        overlay[false_pos] = [0, 0, 1]  # 蓝色：假阳性
        plt.imshow(overlay)
        plt.title(f'Image {idx} - Comparison')
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"分割对比图已保存到: {save_path}")
    
    plt.show()
