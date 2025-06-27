"""
评估指标计算
"""
import numpy as np


def global_accuracy_metric(y_true, y_pred):
    """计算全局准确率"""
    # 展平数组以计算准确率
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算正确预测的数量
    correct_predictions = np.sum(y_true_flat == y_pred_flat)
    # 计算总像素数量
    total_pixels = y_true_flat.size
    # 计算准确率
    accuracy = correct_predictions / total_pixels
    return accuracy


def global_iou_metric(y_true, y_pred, num_classes):
    """计算全局IoU指标"""
    # 展平数组以计算IoU
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 初始化交集和并集计数
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    for i in range(num_classes):
        # 计算每个类别的交集和并集
        intersection[i] = np.sum((y_true_flat == i) & (y_pred_flat == i))
        union[i] = np.sum((y_true_flat == i) | (y_pred_flat == i))

    # 计算每个类别的IoU
    iou = intersection / (union + 1e-6)  # 添加小值以避免除零

    # 计算平均IoU
    mean_iou = np.mean(iou)
    
    return mean_iou


def compute_class_wise_iou(y_true, y_pred, num_classes):
    """计算每个类别的IoU"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    class_iou = {}
    for i in range(num_classes):
        intersection = np.sum((y_true_flat == i) & (y_pred_flat == i))
        union = np.sum((y_true_flat == i) | (y_pred_flat == i))
        
        if union == 0:
            class_iou[i] = 0.0
        else:
            class_iou[i] = intersection / union
    
    return class_iou

# 计算每个类别的准确率
def compute_class_wise_accuracy(y_true, y_pred, num_classes):
    """计算每个类别的准确率"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    class_accuracy = {}
    for i in range(num_classes):
        correct_predictions = np.sum((y_true_flat == i) & (y_pred_flat == i))
        total_pixels = np.sum(y_true_flat == i)
        
        if total_pixels == 0:
            class_accuracy[i] = 0.0
        else:
            class_accuracy[i] = correct_predictions / total_pixels
    
    return class_accuracy

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """计算混淆矩阵"""
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    for i in range(len(y_true_flat)):
        confusion_matrix[y_true_flat[i]][y_pred_flat[i]] += 1
    
    return confusion_matrix
