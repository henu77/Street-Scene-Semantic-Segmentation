"""
第三部分：小目标分割优化
评估小目标分割性能并实现改进策略
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from PIL import Image
import argparse

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data.dataset import CamVidLoader
from data.augmentation import get_augmentation
from models.improved_unet import UNetImprove
from losses.losses import CombinedLoss
from utils.training import train_model, evaluate_model
from utils.object_metrics import compute_object_level_stats_percentile
from utils.visualization import visualize_segmentation_comparison
from models.unet import UNet

def evaluate_baseline_small_objects():
    """评估基线模型的小目标分割性能"""
    print("评估基线模型的小目标分割性能...")
    
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练模型
    model = UNet(3, NUM_CLASSES, width=32, bilinear=True).to(device)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # 加载测试数据
    test_dataset = CamVidLoader(
        root=DATA_ROOT_SUNNY,
        split='test',
        is_aug=False,
        img_size=IMG_SIZE,
        is_pytorch_transform=True
    )
    
    # 获取所有预测结果
    gt_mask_list = []
    pred_mask_list = []
    img_name_list = []
    
    print("处理测试数据...")
    for i in range(len(test_dataset)):
        img, gt_mask, img_name = test_dataset[i]
        img = img.to(device).unsqueeze(0)
        
        with torch.no_grad():
            pred_mask = model(img)
        
        pred_mask = pred_mask.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
        gt_mask_list.append(gt_mask.numpy())
        pred_mask_list.append(pred_mask)
        img_name_list.append(img_name)
        
        if (i + 1) % 20 == 0:
            print(f"已处理 {i+1}/{len(test_dataset)} 张图像")
    
    gt_mask_list = np.array(gt_mask_list)
    pred_mask_list = np.array(pred_mask_list)
    
    # 评估小目标类别的性能
    baseline_results = {}
    print("\n基线模型小目标评估结果:")
    print("=" * 60)
    
    for class_name, class_id in zip(SMALL_OBJECT_CLASSES, SMALL_OBJECT_IDS):
        print(f"\n评估类别: {class_name}")
        stats = compute_object_level_stats_percentile(
            gt_mask_list, pred_mask_list, class_name,
            iou_threshold=0.25, size_percentile=50
        )
        baseline_results[class_name] = {
            'precision_small': stats[0],
            'recall_small': stats[1],
            'f1_small': stats[2],
            'precision_large': stats[3],
            'recall_large': stats[4],
            'f1_large': stats[5]
        }
    
    # 生成对比可视化
    print("\n生成分割对比可视化...")
    car_indices = [11, 22, 33, 44, 55]
    lane_indices = [66, 77, 88, 99, 110]
    ped_indices = [16, 26, 36, 46, 86]
    
    for class_name, class_id, indices in zip(
        ['Car_Bus', 'LaneMarking', 'Pedestrian'],
        [10, 4, 11],
        [car_indices, lane_indices, ped_indices]
    ):
        save_path = os.path.join(OUTPUT_DIRS['part3'], f'{class_name}_baseline_comparison.png')
        visualize_segmentation_comparison(
            gt_mask_list, pred_mask_list, indices, class_name, class_id, save_path
        )
    
    return baseline_results, gt_mask_list, pred_mask_list, img_name_list


def train_improved_model():
    """训练改进版U-Net模型"""
    print("\n训练改进版U-Net模型...")
    
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    # 创建改进版模型
    model = UNetImprove(n_channels=3, n_classes=NUM_CLASSES, bilinear=True, width=32).to(device)
    
    # 损失函数和优化器
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    # 数据增强
    aug_fun = get_augmentation(is_strong=True)
    
    # 数据加载器
    train_dataset = CamVidLoader(
        root=DATA_ROOT_SUNNY,
        split='train',
        is_aug=True,
        aug=aug_fun,
        img_size=IMG_SIZE,
        is_pytorch_transform=True
    )
    
    val_dataset = CamVidLoader(
        root=DATA_ROOT_SUNNY,
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
    
    # 训练模型
    save_path = os.path.join(OUTPUT_DIRS['part3'], 'improved_unet_model')
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        device=device,
        save_path=save_path
    )
    
    return model, save_path


def evaluate_improved_model(model_path):
    """评估改进版模型的性能"""
    print("\n评估改进版模型性能...")
    
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    # 加载改进版模型
    model = UNetImprove(n_channels=3, n_classes=NUM_CLASSES, bilinear=True, width=32).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth')))
    model.eval()
    
    # 加载测试数据
    test_dataset = CamVidLoader(
        root=DATA_ROOT_SUNNY,
        split='test',
        is_aug=False,
        img_size=IMG_SIZE,
        is_pytorch_transform=True
    )
    
    # 获取预测结果
    all_pred = []
    all_gt = []
    all_img_path = []
    
    print("处理测试数据...")
    for i in range(len(test_dataset)):
        img, gt_mask, img_name = test_dataset[i]
        img = img.to(device).unsqueeze(0)
        
        with torch.no_grad():
            pred_mask = model(img)
        
        pred_mask = pred_mask.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
        all_pred.append(pred_mask)
        all_gt.append(gt_mask.numpy())
        all_img_path.append(img_name)
        
        if (i + 1) % 20 == 0:
            print(f"已处理 {i+1}/{len(test_dataset)} 张图像")
    
    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)
    
    # 计算整体指标
    from utils.metrics import global_accuracy_metric, global_iou_metric
    global_accuracy = global_accuracy_metric(all_gt, all_pred)
    mean_iou = global_iou_metric(all_gt, all_pred, NUM_CLASSES)
    
    print(f"改进版模型整体性能:")
    print(f"  全局准确率: {global_accuracy:.4f}")
    print(f"  平均IoU: {mean_iou:.4f}")
    
    # 评估小目标性能
    improved_results = {}
    print("\n改进版模型小目标评估结果:")
    print("=" * 60)
    
    for class_name in SMALL_OBJECT_CLASSES:
        print(f"\n评估类别: {class_name}")
        stats = compute_object_level_stats_percentile(
            all_gt, all_pred, class_name,
            iou_threshold=0.25, size_percentile=50
        )
        improved_results[class_name] = {
            'precision_small': stats[0],
            'recall_small': stats[1],
            'f1_small': stats[2],
            'precision_large': stats[3],
            'recall_large': stats[4],
            'f1_large': stats[5]
        }
    
    return improved_results, all_gt, all_pred, all_img_path, global_accuracy, mean_iou


def generate_improvement_comparison(baseline_gt, baseline_pred, improved_gt, improved_pred, img_paths):
    """生成改进前后对比可视化"""
    print("\n生成改进前后对比可视化...")
    
    # 加载原始图像
    all_imgs = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        all_imgs.append(img)
    
    # 选择代表性图像
    car_indices = [11, 22, 33, 44, 55]
    lane_indices = [66, 77, 88, 99, 110]
    ped_indices = [16, 26, 36, 46, 86]
    
    def visualize_model_improvement(images, gt_masks, pred_masks_base, pred_masks_improved, 
                                  indices, class_name, class_id, save_path):
        """可视化模型改进效果"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(25, 7*len(indices)))
        plt.suptitle(f'{class_name} Segmentation Results Comparison (Green=Correct, Red=Missed, Blue=False Positive)', 
                     fontsize=16, y=0.98)
        
        for i, idx in enumerate(indices):
            # 获取当前类别的掩码
            gt_mask = gt_masks[idx] == class_id
            pred_base = pred_masks_base[idx] == class_id
            pred_improved = pred_masks_improved[idx] == class_id
            
            # 计算基线模型的差异
            base_missed = gt_mask & ~pred_base
            base_false_pos = ~gt_mask & pred_base
            base_correct = gt_mask & pred_base
            
            # 计算改进模型的差异
            improved_missed = gt_mask & ~pred_improved
            improved_false_pos = ~gt_mask & pred_improved
            improved_correct = gt_mask & pred_improved
            
            # 可视化
            # 原始图像
            plt.subplot(len(indices), 4, i*4+1)
            img = np.array(images[idx])
            plt.imshow(img)
            plt.title(f'Image {idx} - Original Image')
            plt.axis('off')
            
            # 真实掩码
            plt.subplot(len(indices), 4, i*4+2)
            plt.imshow(gt_mask, cmap='gray')
            plt.title(f'Image {idx} - Ground Truth Mask')
            plt.axis('off')
            
            # 基线U-Net预测结果
            plt.subplot(len(indices), 4, i*4+3)
            overlay_base = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
            overlay_base[base_correct] = [0, 1, 0]
            overlay_base[base_missed] = [1, 0, 0]
            overlay_base[base_false_pos] = [0, 0, 1]
            plt.imshow(overlay_base)
            plt.title(f'Image {idx} - Baseline U-Net')
            plt.axis('off')
            
            # 改进U-Net预测结果
            plt.subplot(len(indices), 4, i*4+4)
            overlay_improved = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
            overlay_improved[improved_correct] = [0, 1, 0]
            overlay_improved[improved_missed] = [1, 0, 0]
            overlay_improved[improved_false_pos] = [0, 0, 1]
            plt.imshow(overlay_improved)
            plt.title(f'Image {idx} - Improved U-Net')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"改进对比图已保存到: {save_path}")
        plt.close()
    
    # 生成对比图
    for class_name, class_id, indices in zip(
        ['Car_Bus', 'LaneMarking', 'Pedestrian'],
        [10, 4, 11],
        [car_indices, lane_indices, ped_indices]
    ):
        save_path = os.path.join(OUTPUT_DIRS['part3'], f'{class_name}_improvement_comparison.png')
        visualize_model_improvement(
            all_imgs, baseline_gt, baseline_pred, improved_pred,
            indices, class_name, class_id, save_path
        )


def generate_report(baseline_results, improved_results=None, global_acc=None, mean_iou=None):
    """生成分析报告"""
    print("\n生成详细分析报告...")
    report_path = os.path.join(OUTPUT_DIRS['part3'], 'improvement_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("第三部分：小目标分割优化分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 改进策略说明:\n")
        f.write("   - SE注意力机制：增强特征表示能力\n")
        f.write("   - CBAM注意力模块：同时关注通道和空间注意力\n")
        f.write("   - 膨胀卷积：替代最大池化，保持更多细节信息\n\n")
        
        if improved_results is not None and global_acc is not None and mean_iou is not None:
            f.write("2. 整体性能提升:\n")
            f.write(f"   改进版模型全局准确率: {global_acc:.4f}\n")
            f.write(f"   改进版模型平均IoU: {mean_iou:.4f}\n\n")
            
            f.write("3. 小目标性能对比:\n")
            f.write(f"{'类别':<12} {'指标':<12} {'基线':<10} {'改进':<10} {'提升':<10}\n")
            f.write("-" * 60 + "\n")
            
            for class_name in SMALL_OBJECT_CLASSES:
                baseline = baseline_results[class_name]
                improved = improved_results[class_name]
                
                f.write(f"{class_name:<12} {'精确率_小':<12} {baseline['precision_small']:<10.4f} "
                       f"{improved['precision_small']:<10.4f} "
                       f"{improved['precision_small'] - baseline['precision_small']:<10.4f}\n")
                
                f.write(f"{'':<12} {'召回率_小':<12} {baseline['recall_small']:<10.4f} "
                       f"{improved['recall_small']:<10.4f} "
                       f"{improved['recall_small'] - baseline['recall_small']:<10.4f}\n")
                
                f.write(f"{'':<12} {'F1分数_小':<12} {baseline['f1_small']:<10.4f} "
                       f"{improved['f1_small']:<10.4f} "
                       f"{improved['f1_small'] - baseline['f1_small']:<10.4f}\n")
                
                f.write("-" * 60 + "\n")
            
            f.write("\n4. 关键发现:\n")
            # 计算平均提升
            avg_precision_improvement = np.mean([
                improved_results[cls]['precision_small'] - baseline_results[cls]['precision_small']
                for cls in SMALL_OBJECT_CLASSES
            ])
            avg_recall_improvement = np.mean([
                improved_results[cls]['recall_small'] - baseline_results[cls]['recall_small']
                for cls in SMALL_OBJECT_CLASSES
            ])
            avg_f1_improvement = np.mean([
                improved_results[cls]['f1_small'] - baseline_results[cls]['f1_small']
                for cls in SMALL_OBJECT_CLASSES
            ])
            
            f.write(f"   - 小目标平均精确率提升: {avg_precision_improvement:.4f}\n")
            f.write(f"   - 小目标平均召回率提升: {avg_recall_improvement:.4f}\n")
            f.write(f"   - 小目标平均F1分数提升: {avg_f1_improvement:.4f}\n")
            
            # 找出表现最好的类别
            best_class = max(SMALL_OBJECT_CLASSES, 
                            key=lambda x: improved_results[x]['f1_small'] - baseline_results[x]['f1_small'])
            best_improvement = improved_results[best_class]['f1_small'] - baseline_results[best_class]['f1_small']
            
            f.write(f"   - 改进效果最显著的类别: {best_class} (F1提升: {best_improvement:.4f})\n")
            
            f.write("\n5. 技术贡献分析:\n")
            f.write("   - SE注意力机制有效提升了特征的表示能力\n")
            f.write("   - CBAM模块通过通道和空间注意力的结合，更好地关注小目标区域\n")
            f.write("   - 膨胀卷积替代最大池化减少了信息损失，保持了更多细节\n")
            f.write("   - 组合损失函数平衡了像素级和区域级的优化目标\n")
        else:
            f.write("2. 基线模型小目标评估结果:\n")
            f.write(f"{'类别':<12} {'精确率_小':<12} {'召回率_小':<12} {'F1分数_小':<12}\n")
            f.write("-" * 60 + "\n")
            
            for class_name in SMALL_OBJECT_CLASSES:
                baseline = baseline_results[class_name]
                f.write(f"{class_name:<12} {baseline['precision_small']:<12.4f} "
                       f"{baseline['recall_small']:<12.4f} {baseline['f1_small']:<12.4f}\n")
    
    print(f"分析报告已保存到: {report_path}")


def run_eval_only():
    """仅评估模式"""
    print("运行模式: 仅评估基线模型")
    baseline_results, baseline_gt, baseline_pred, img_paths = evaluate_baseline_small_objects()
    generate_report(baseline_results)


def run_train_and_eval():
    """训练和评估模式"""
    print("运行模式: 训练改进模型并评估")
    
    # 1. 评估基线模型
    baseline_results, baseline_gt, baseline_pred, img_paths = evaluate_baseline_small_objects()
    
    # 2. 训练改进版模型
    improved_model, model_save_path = train_improved_model()
    
    # 3. 评估改进版模型
    improved_results, improved_gt, improved_pred, _, global_acc, mean_iou = evaluate_improved_model(model_save_path)
    
    # 4. 生成改进前后对比
    generate_improvement_comparison(baseline_gt, baseline_pred, improved_gt, improved_pred, img_paths)
    
    # 5. 生成详细报告
    generate_report(baseline_results, improved_results, global_acc, mean_iou)


def run_eval_improved():
    """仅评估改进模型"""
    print("运行模式: 仅评估改进模型")
    
    # 检查改进模型是否存在
    model_path = os.path.join(OUTPUT_DIRS['part3'], 'improved_unet_model')
    if not os.path.exists(os.path.join(model_path, 'best_model.pth')):
        print("错误: 改进模型不存在，请先运行训练模式或完整模式")
        return
    
    # 1. 评估基线模型（用于对比）
    baseline_results, baseline_gt, baseline_pred, img_paths = evaluate_baseline_small_objects()
    
    # 2. 评估改进版模型
    improved_results, improved_gt, improved_pred, _, global_acc, mean_iou = evaluate_improved_model(model_path)
    
    # 3. 生成改进前后对比
    generate_improvement_comparison(baseline_gt, baseline_pred, improved_gt, improved_pred, img_paths)
    
    # 4. 生成详细报告
    generate_report(baseline_results, improved_results, global_acc, mean_iou)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='第三部分：小目标分割优化')
    parser.add_argument('--mode', type=str, choices=['eval', 'train', 'eval_improved', 'all'], 
                       default='eval_improved', help='运行模式')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("第三部分：小目标分割优化")
    print("=" * 60)
    
    if args.mode == 'eval':
        # 仅评估基线模型
        run_eval_only()
    elif args.mode == 'train':
        # 训练改进模型
        print("运行模式: 仅训练改进模型")
        improved_model, model_save_path = train_improved_model()
        print("训练完成！")
    elif args.mode == 'eval_improved':
        # 评估改进模型（包含对比）
        run_eval_improved()
    elif args.mode == 'all':
        # 完整流程：评估基线 + 训练改进 + 评估改进 + 对比
        run_train_and_eval()
    
    print("第三部分完成！")


if __name__ == "__main__":
    main()