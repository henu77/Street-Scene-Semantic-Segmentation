"""
目标级别评估指标
"""
import numpy as np
from scipy.ndimage import label

class_labels_dict = {
    "Sky": 0,
    "Building": 1,
    "Pole": 2,
    "Road": 3,
    "LaneMarking": 4,
    "SideWalk": 5,
    "Pavement": 6,
    "Tree": 7,
    "SignSymbol": 8,
    "Fence": 9,
    "Car_Bus": 10,
    "Pedestrian": 11,
    "Bicyclist": 12,
    "Unlabelled": 13
}


def compute_object_level_stats_percentile(
    gt_masks: np.ndarray, 
    pred_masks: np.ndarray,
    class_label: str,
    iou_threshold: float = 0.25,
    size_percentile: float = 50.0
):
    """
    计算目标级别指标（精确率、召回率、F1），分别针对'小'和'大'目标，
    边界由指定百分位数的GT目标面积阈值确定。
    
    参数:
        gt_masks: 真实标签掩码
        pred_masks: 预测标签掩码
        class_label: 类别名称
        iou_threshold: IoU阈值
        size_percentile: 大小百分位数阈值
    
    返回:
        precision_small, recall_small, f1_small, precision_large, recall_large, f1_large
    """
    class_idx = class_labels_dict[class_label]

    # 第一步：收集所有GT目标面积
    all_gt_areas = []
    frame_data = []

    N = len(gt_masks)
    for i in range(N):
        gt_frame = gt_masks[i]
        pred_frame = pred_masks[i]

        # 获取感兴趣类别的二值掩码
        gt_binary = (gt_frame == class_idx).astype(np.uint8)
        pred_binary = (pred_frame == class_idx).astype(np.uint8)

        # 标记连通区域
        gt_labeled, gt_num_objs = label(gt_binary)
        pred_labeled, pred_num_objs = label(pred_binary)

        # 收集GT目标
        gt_objects = []
        for obj_id in range(1, gt_num_objs + 1):
            coords = np.where(gt_labeled == obj_id)
            area = len(coords[0])
            all_gt_areas.append(area)
            gt_objects.append(coords)

        # 收集预测目标
        pred_objects = []
        for obj_id in range(1, pred_num_objs + 1):
            coords = np.where(pred_labeled == obj_id)
            pred_objects.append(coords)

        # 保存供第二步使用
        frame_data.append({
            "gt_objects": gt_objects,
            "pred_objects": pred_objects
        })

    # 如果没有GT目标，返回零
    if len(all_gt_areas) == 0:
        return 0, 0, 0, 0, 0, 0

    # 确定动态大小阈值
    dynamic_threshold = np.percentile(all_gt_areas, size_percentile)

    # 第二步：基于IoU匹配和大小分类计数
    tp_small = fp_small = fn_small = 0
    tp_large = fp_large = fn_large = 0
    total_small_gt = total_large_gt = 0
    total_small_pred = total_large_pred = 0

    for i in range(N):
        gt_objects = frame_data[i]["gt_objects"]
        pred_objects = frame_data[i]["pred_objects"]

        # 标记GT目标
        labeled_gt_objs = []
        for coords in gt_objects:
            area = len(coords[0])
            size_cat = "small" if area < dynamic_threshold else "large"
            pixel_set = set(zip(coords[0], coords[1]))
            labeled_gt_objs.append((pixel_set, size_cat))

        # 标记预测目标
        labeled_pred_objs = []
        for coords in pred_objects:
            area = len(coords[0])
            size_cat = "small" if area < dynamic_threshold else "large"
            pixel_set = set(zip(coords[0], coords[1]))
            labeled_pred_objs.append((pixel_set, size_cat))

        matched_gt = set()
        matched_pred = set()

        # 匹配GT和预测目标
        for gt_idx, (gt_pixels, gt_size_cat) in enumerate(labeled_gt_objs):
            best_iou = 0.0
            best_pred_idx = None
            
            for pred_idx, (pred_pixels, pred_size_cat) in enumerate(labeled_pred_objs):
                intersection = len(gt_pixels.intersection(pred_pixels))
                union = len(gt_pixels.union(pred_pixels))
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            # 如果IoU超过阈值，认为匹配成功
            if best_iou >= iou_threshold and best_pred_idx is not None:
                if gt_size_cat == "small":
                    tp_small += 1
                else:
                    tp_large += 1
                matched_gt.add(gt_idx)
                matched_pred.add(best_pred_idx)

        # 计算假阴性（未匹配的GT目标）
        for gt_idx, (_, gt_size_cat) in enumerate(labeled_gt_objs):
            if gt_idx not in matched_gt:
                if gt_size_cat == "small":
                    fn_small += 1
                else:
                    fn_large += 1

        # 计算假阳性（未匹配的预测目标）
        for pred_idx, (_, pred_size_cat) in enumerate(labeled_pred_objs):
            if pred_idx not in matched_pred:
                if pred_size_cat == "small":
                    fp_small += 1
                else:
                    fp_large += 1

        # 统计总数
        for coords, size_cat in labeled_gt_objs:
            if size_cat == "small":
                total_small_gt += 1
            else:
                total_large_gt += 1
        
        for coords, size_cat in labeled_pred_objs:
            if size_cat == "small":
                total_small_pred += 1
            else:
                total_large_pred += 1

    # 计算指标
    precision_small = tp_small / (tp_small + fp_small + 1e-10)
    recall_small = tp_small / (tp_small + fn_small + 1e-10)
    f1_small = 2 * precision_small * recall_small / (precision_small + recall_small + 1e-10)

    precision_large = tp_large / (tp_large + fp_large + 1e-10)
    recall_large = tp_large / (tp_large + fn_large + 1e-10)
    f1_large = 2 * precision_large * recall_large / (precision_large + recall_large + 1e-10)

    # 打印总结表格
    print(f"\n{'Metric':<30}{'Small Objects':<20}{'Large Objects':<20}")
    print("="*70)
    print(f"{'Class Name':<30}{class_label:<20}")
    print(f"{'Size Threshold (pixels)':<30}{dynamic_threshold:<20.2f}")
    print("="*70)
    print(f"{'Total Ground Truths':<30}{total_small_gt:<20}{total_large_gt:<20}")
    print(f"{'Total Predicted':<30}{total_small_pred:<20}{total_large_pred:<20}")
    print(f"{'Precision':<30}{precision_small:<20.4f}{precision_large:<20.4f}")
    print(f"{'Recall':<30}{recall_small:<20.4f}{recall_large:<20.4f}")
    print(f"{'F1 Score':<30}{f1_small:<20.4f}{f1_large:<20.4f}")
    print(f"{'False Positives (FP)':<30}{fp_small:<20}{fp_large:<20}")
    print(f"{'False Negatives (FN)':<30}{fn_small:<20}{fn_large:<20}")
    print(f"{'True Positives (TP)':<30}{tp_small:<20}{tp_large:<20}")
    print("="*70)

    return precision_small, recall_small, f1_small, precision_large, recall_large, f1_large
