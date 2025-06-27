import os
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from skimage.io import imread
from data.dataset import from_rgb_to_label, class_labels_dict, label_colours
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv

# 设置字体路径
font_path = "SIMSUN.TTC"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


class DatasetAnalyzer:
    """数据集统计分析器"""

    def __init__(self, root_path):
        self.root_path = root_path
        self.class_names = list(class_labels_dict.keys())
        self.n_classes = len(self.class_names)
        self.results = {}

    def analyze_dataset(self, dataset_name):
        """分析单个数据集"""
        dataset_path = os.path.join(self.root_path, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"数据集路径不存在: {dataset_path}")
            return None

        print(f"\n正在分析数据集: {dataset_name}")

        # 初始化统计变量
        split_stats = {}
        total_pixel_counts = np.zeros(self.n_classes)
        total_pixels = 0
        class_appearance = defaultdict(set)  # 每个类别出现在哪些图像中
        class_areas = defaultdict(list)  # 每个类别在各图像中的面积

        # 分析每个split
        for split in ["train", "val", "test"]:
            split_path = os.path.join(dataset_path, split)
            split_label_path = os.path.join(dataset_path, f"{split}_labels")

            if not os.path.exists(split_path) or not os.path.exists(split_label_path):
                print(f"跳过不存在的split: {split}")
                split_stats[split] = {"count": 0}
                continue

            # 获取图像文件列表
            img_files = [
                f
                for f in os.listdir(split_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            split_pixel_counts = np.zeros(self.n_classes)
            split_pixels = 0
            split_class_appearance = defaultdict(int)

            print(f"  处理 {split} split: {len(img_files)} 张图像")

            for i, img_file in enumerate(img_files):
                if i % 50 == 0:
                    print(f"    处理进度: {i+1}/{len(img_files)}")

                # 构建标签文件路径
                base_name = os.path.splitext(img_file)[0]
                label_file = base_name + "_L.png"
                label_path = os.path.join(split_label_path, label_file)

                if not os.path.exists(label_path):
                    print(f"    警告: 标签文件不存在 {label_path}")
                    continue

                try:
                    # 读取并转换标签
                    lbl_rgb = imread(label_path)
                    lbl = from_rgb_to_label(lbl_rgb)

                    # 统计像素数
                    unique_labels, counts = np.unique(lbl, return_counts=True)
                    img_total_pixels = lbl.shape[0] * lbl.shape[1]

                    split_pixels += img_total_pixels
                    total_pixels += img_total_pixels

                    # 统计每个类别
                    for label_id, count in zip(unique_labels, counts):
                        if label_id < self.n_classes:
                            split_pixel_counts[label_id] += count
                            total_pixel_counts[label_id] += count

                            # 记录类别出现
                            class_appearance[label_id].add(f"{split}_{img_file}")
                            split_class_appearance[label_id] += 1

                            # 记录面积（像素数）
                            class_areas[label_id].append(count)

                except Exception as e:
                    print(f"    错误处理文件 {label_path}: {e}")
                    continue

            # 保存split统计信息
            split_stats[split] = {
                "count": len(img_files),
                "total_pixels": split_pixels,
                "pixel_counts": split_pixel_counts.tolist(),
                "class_appearance": dict(split_class_appearance),
            }

        # 计算总体统计
        total_images = sum(split_stats[s]["count"] for s in ["train", "val", "test"])

        # 计算像素占比
        pixel_ratios = (
            total_pixel_counts / total_pixels
            if total_pixels > 0
            else np.zeros(self.n_classes)
        )

        # 计算出现频率
        appearance_freq = {i: len(class_appearance[i]) for i in range(self.n_classes)}

        # 计算最大最小面积
        area_stats = {}
        for class_id in range(self.n_classes):
            if class_areas[class_id]:
                area_stats[class_id] = {
                    "max_area": max(class_areas[class_id]),
                    "min_area": min(class_areas[class_id]),
                    "mean_area": np.mean(class_areas[class_id]),
                    "total_instances": len(class_areas[class_id]),
                }
            else:
                area_stats[class_id] = {
                    "max_area": 0,
                    "min_area": 0,
                    "mean_area": 0,
                    "total_instances": 0,
                }

        # 整合结果
        result = {
            "dataset_name": dataset_name,
            "total_images": total_images,
            "total_pixels": total_pixels,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "split_distribution": {
                "train": split_stats.get("train", {}).get("count", 0),
                "val": split_stats.get("val", {}).get("count", 0),
                "test": split_stats.get("test", {}).get("count", 0),
            },
            "pixel_statistics": {
                "total_pixel_counts": total_pixel_counts.tolist(),
                "pixel_ratios": pixel_ratios.tolist(),
                "pixel_percentages": (pixel_ratios * 100).tolist(),
            },
            "appearance_frequency": appearance_freq,
            "area_statistics": area_stats,
            "split_details": split_stats,
        }

        return result

    def print_statistics(self, result):
        """打印统计结果"""
        if result is None:
            return

        print(f"\n{'='*60}")
        print(f"数据集统计报告: {result['dataset_name']}")
        print(f"{'='*60}")

        # 基本信息
        print(f"\n📊 基本信息:")
        print(f"总图像数量: {result['total_images']}")
        print(f"类别数量: {result['n_classes']}")
        print(f"总像素数: {result['total_pixels']:,}")

        # 数据集分布
        print(f"\n📁 数据集分布:")
        total = result["total_images"]
        splits = result["split_distribution"]
        for split_name, count in splits.items():
            ratio = count / total * 100 if total > 0 else 0
            print(f"{split_name:>6}: {count:>6} 张 ({ratio:>5.1f}%)")

        # 像素占比统计
        print(f"\n🎨 各类别像素占比:")
        pixel_stats = result["pixel_statistics"]
        print(f"{'类别':<15} {'像素数':<12} {'占比':<10} {'百分比':<10}")
        print("-" * 50)
        for i, class_name in enumerate(result["class_names"]):
            pixel_count = int(pixel_stats["total_pixel_counts"][i])
            ratio = pixel_stats["pixel_ratios"][i]
            percentage = pixel_stats["pixel_percentages"][i]
            print(
                f"{class_name:<15} {pixel_count:<12,} {ratio:<10.6f} {percentage:<10.2f}%"
            )

        # 出现频率
        print(f"\n📈 类别出现频率:")
        print(f"{'类别':<15} {'出现图像数':<12} {'出现率':<10}")
        print("-" * 40)
        for i, class_name in enumerate(result["class_names"]):
            freq = result["appearance_frequency"][i]
            freq_ratio = freq / total * 100 if total > 0 else 0
            print(f"{class_name:<15} {freq:<12} {freq_ratio:<10.2f}%")

        # 面积统计
        print(f"\n📏 类别面积统计:")
        print(
            f"{'类别':<15} {'实例数':<8} {'最小面积':<10} {'最大面积':<10} {'平均面积':<10}"
        )
        print("-" * 65)
        for i, class_name in enumerate(result["class_names"]):
            area_stat = result["area_statistics"][i]
            instances = area_stat["total_instances"]
            min_area = area_stat["min_area"]
            max_area = area_stat["max_area"]
            mean_area = area_stat["mean_area"]
            print(
                f"{class_name:<15} {instances:<8} {min_area:<10} {max_area:<10} {mean_area:<10.1f}"
            )
    def save_to_csv(self, result, output_dir):
            """将统计结果保存为CSV文件"""
            if result is None:
                print("没有结果可保存")
                return

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存像素统计数据
            pixel_stats_file = os.path.join(output_dir, f"{result['dataset_name']}_pixel_statistics.csv")
            with open(pixel_stats_file, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["类别", "像素数", "占比", "百分比"])
                for i, class_name in enumerate(result["class_names"]):
                    writer.writerow([
                        class_name,
                        int(result["pixel_statistics"]["total_pixel_counts"][i]),
                        result["pixel_statistics"]["pixel_ratios"][i],
                        result["pixel_statistics"]["pixel_percentages"][i],
                    ])

            # 保存类别面积统计数据
            area_stats_file = os.path.join(output_dir, f"{result['dataset_name']}_area_statistics.csv")
            with open(area_stats_file, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["类别", "实例数", "最小面积", "最大面积", "平均面积"])
                for i, class_name in enumerate(result["class_names"]):
                    area_stat = result["area_statistics"][i]
                    writer.writerow([
                        class_name,
                        area_stat["total_instances"],
                        area_stat["min_area"],
                        area_stat["max_area"],
                        area_stat["mean_area"],
                    ])

            print(f"统计结果已保存为CSV文件: {output_dir}")
    def create_visualization(self, results, output_dir):
        """创建可视化图表"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 从dataset中获取类别颜色，转换为matplotlib格式
        def get_class_colors():
            colors = []
            for color_list in label_colours:
                # 取每个类别的第一个颜色，并转换为0-1范围
                rgb = [c / 255.0 for c in color_list[0]]
                colors.append(rgb)
            return colors

        class_colors = get_class_colors()

        for dataset_name, result in results.items():
            if result is None:
                continue

            # 1. 数据集划分扇形图
            plt.figure(figsize=(8, 8))
            splits = list(result["split_distribution"].keys())
            counts = list(result["split_distribution"].values())

            # 过滤掉数量为0的split
            filtered_splits = []
            filtered_counts = []
            for split, count in zip(splits, counts):
                if count > 0:
                    filtered_splits.append(split)
                    filtered_counts.append(count)

            if filtered_counts:
                colors_split = ["#FF6B6B", "#4ECDC4", "#45B7D1"][: len(filtered_splits)]
                
                # 自定义显示百分比和数量
                def autopct_format(pct, all_vals):
                    absolute = int(round(pct / 100. * sum(all_vals)))
                    return f"{pct:.1f}%\n({absolute})"

                wedges, texts, autotexts = plt.pie(
                    filtered_counts,
                    labels=None,
                    autopct=lambda pct: autopct_format(pct, filtered_counts),  # 自定义显示内容
                    startangle=90,
                    colors=colors_split,
                )
                # 调整百分比字体大小
                for autotext in autotexts:
                    autotext.set_fontsize(15)  # 设置百分比字体大小为 15，可根据需要调整
                plt.title(f"{dataset_name} - 训练/验证/测试集分布", fontsize=15)
                plt.axis("equal")
                # 设置 图注
                plt.legend(filtered_splits, loc="upper right", fontsize=15)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"{dataset_name}_split_distribution.png"),
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            plt.close()

            # 2. 像素分布柱状图（使用类别颜色）
            plt.figure(figsize=(15, 8))
            percentages = result["pixel_statistics"]["pixel_percentages"]
            class_names = result["class_names"]

            # 创建柱状图
            x_pos = np.arange(len(class_names))
            bars = plt.bar(x_pos, percentages, color=class_colors[: len(class_names)])

            plt.title(f"{dataset_name} - 各类别像素占比分布", fontsize=15)
            plt.xlabel("类别", fontsize=15)
            plt.ylabel("像素占比 (%)", fontsize=15)
            plt.xticks(x_pos, class_names, rotation=45, ha="right", fontsize=12)
            plt.yticks(fontsize=12)

            # 添加数值标签
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                if pct > 0.1:  # 只为占比大于0.1%的类别添加标签
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{pct:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=15,
                    )

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{dataset_name}_pixel_distribution.png"),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
        plt.figure(figsize=(12, 6))
        n_cols = 5  # 每行显示的色块数量
        n_rows = (len(class_names) + n_cols - 1) // n_cols  # 计算行数
        cell_width = 2
        cell_height = 1

        fig, ax = plt.subplots(figsize=(n_cols * cell_width, n_rows * cell_height))
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.axis("off")

        for i, class_name in enumerate(class_names):
            row = i // n_cols
            col = i % n_cols
            color = class_colors[i]
            rect = plt.Rectangle(
                (col, n_rows - row - 1), 1, 1, color=color, transform=ax.transData
            )
            ax.add_patch(rect)
            ax.text(
                col + 0.5,
                n_rows - row - 0.5,
                class_name,
                color="white",
                ha="center",
                va="center",
                fontsize=15,
                weight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{dataset_name}_class_color_blocks.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        

        print(f"可视化图表已保存到: {output_dir}")


def main():
    """主函数"""
    # 数据集根路径
    root_path = "CamVid"  # 根据实际路径修改

    # 创建分析器
    analyzer = DatasetAnalyzer(root_path)

    # 分析两个数据集
    datasets = ["sunny", "cloudy"]
    all_results = {}

    for dataset in datasets:
        result = analyzer.analyze_dataset(dataset)
        all_results[dataset] = result
        analyzer.print_statistics(result)
        csv_output_dir = os.path.join(os.path.dirname(__file__), "data/statistics")
        analyzer.save_to_csv(result, csv_output_dir)

    # 创建可视化
    output_dir = os.path.join(os.path.dirname(__file__), "data/visualization")
    analyzer.create_visualization(all_results, output_dir)

    # 打印汇总对比
    print(f"\n{'='*80}")
    print("数据集对比汇总")
    print(f"{'='*80}")

    print(f"{'指标':<20} {'Sunny':<15} {'Cloudy':<15}")
    print("-" * 50)

    for dataset in datasets:
        if all_results[dataset] is not None:
            result = all_results[dataset]
            print(f"总图像数:{'':<12} {result['total_images']:<15}")
            print(f"训练集:{'':<14} {result['split_distribution']['train']:<15}")
            print(f"验证集:{'':<14} {result['split_distribution']['val']:<15}")
            print(f"测试集:{'':<14} {result['split_distribution']['test']:<15}")
            print(f"总像素数:{'':<12} {result['total_pixels']:<15,}")
            print("-" * 50)


if __name__ == "__main__":
    main()
