"""
训练工具
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metrics import global_accuracy_metric, global_iou_metric
from .visualization import plot_training_curves


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=50, device='cuda', save_path=None):
    """训练模型"""
    # 检查保存路径是否存在，如果不存在则创建
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 记录训练指标
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_train_accuracy = []
    epoch_val_accuracy = []
    epoch_train_iou = []
    epoch_val_iou = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_batch_accuracy_metrics = []
        train_batch_iou_metrics = []
        
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Train"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            
            outputs = torch.softmax(outputs, dim=1)
            # 计算指标
            pred_labels = outputs.argmax(dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            
            batch_accuracy = global_accuracy_metric(true_labels, pred_labels)
            batch_iou = global_iou_metric(true_labels, pred_labels, outputs.shape[1])
            
            train_batch_accuracy_metrics.append(batch_accuracy)
            train_batch_iou_metrics.append(batch_iou)
        
        # 计算训练平均指标
        train_avg_accuracy = np.mean(train_batch_accuracy_metrics)
        train_avg_iou = np.mean(train_batch_iou_metrics)
        train_avg_loss = train_running_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_batch_accuracy_metrics = []
        val_batch_iou_metrics = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Val"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                
                outputs = torch.softmax(outputs, dim=1)
                # 计算指标
                pred_labels = outputs.argmax(dim=1).cpu().numpy()
                true_labels = labels.cpu().numpy()
                
                batch_accuracy = global_accuracy_metric(true_labels, pred_labels)
                batch_iou = global_iou_metric(true_labels, pred_labels, outputs.shape[1])
                
                val_batch_accuracy_metrics.append(batch_accuracy)
                val_batch_iou_metrics.append(batch_iou)
        
        # 计算验证平均指标
        val_avg_accuracy = np.mean(val_batch_accuracy_metrics)
        val_avg_iou = np.mean(val_batch_iou_metrics)
        val_avg_loss = val_running_loss / len(val_loader)
        
        # 记录指标
        epoch_train_loss.append(train_avg_loss)
        epoch_val_loss.append(val_avg_loss)
        epoch_train_accuracy.append(train_avg_accuracy)
        epoch_val_accuracy.append(val_avg_accuracy)
        epoch_train_iou.append(train_avg_iou)
        epoch_val_iou.append(val_avg_iou)
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_avg_loss:.4f}, Train Acc: {train_avg_accuracy:.4f}, Train IoU: {train_avg_iou:.4f}')
        print(f'Val Loss: {val_avg_loss:.4f}, Val Acc: {val_avg_accuracy:.4f}, Val IoU: {val_avg_iou:.4f}')
        print('-' * 60)
        
        # 保存最佳模型
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            if save_path:
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print(f'Best model saved at epoch {epoch+1}')
      # 保存训练指标
    if save_path:
        # 创建DataFrame保存metrics
        metrics_df = pd.DataFrame({
            'epoch': range(1, num_epochs + 1),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_accuracy': epoch_train_accuracy,
            'val_accuracy': epoch_val_accuracy,
            'train_iou': epoch_train_iou,
            'val_iou': epoch_val_iou
        })
        
        # 保存为CSV格式
        csv_path = os.path.join(save_path, 'metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f'训练指标已保存到: {csv_path}')
        
        
        # 绘制并保存训练曲线（包含平滑）
        plot_training_curves(
            epoch_train_loss, epoch_val_loss,
            epoch_train_accuracy, epoch_val_accuracy,
            epoch_train_iou, epoch_val_iou,
            save_path=os.path.join(save_path, 'metrics_plot.png')
        )
    
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())
            all_images.append(images.cpu().numpy())
    
    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    
    # 计算指标
    test_accuracy = global_accuracy_metric(all_labels, all_preds)
    test_iou = global_iou_metric(all_labels, all_preds, outputs.shape[1])
    
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Mean IoU: {test_iou:.4f}')
    
    return all_images, all_labels, all_preds, test_accuracy, test_iou
