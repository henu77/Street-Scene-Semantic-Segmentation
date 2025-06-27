#!/usr/bin/env python3
# filepath: /home/malong/project/my_project/M/run_all_experiments.py
"""
主实验脚本：按顺序运行所有三个部分的实验
- Part 1: 预训练和评估 (Sunny数据集)
- Part 2: 跨域适应实验 (Cloudy数据集)
- Part 3: 小目标分割优化
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *

def print_separator(title):
    """打印分隔符"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n{'='*20} 步骤 {step_num}: {title} {'='*20}")

def check_prerequisites():
    """检查运行前提条件"""
    print("检查运行前提条件...")
    
    # 检查数据目录
    if not os.path.exists(DATA_ROOT_SUNNY):
        print(f"❌ Sunny数据集目录不存在: {DATA_ROOT_SUNNY}")
        return False
    
    if not os.path.exists(DATA_ROOT_CLOUDY):
        print(f"❌ Cloudy数据集目录不存在: {DATA_ROOT_CLOUDY}")
        return False
    
    # 检查输出目录，不存在则创建
    for output_key, output_dir in OUTPUT_DIRS.items():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ 创建输出目录: {output_dir}")
        else:
            print(f"✅ 输出目录存在: {output_dir}")
    
    # 检查必需的脚本文件
    required_scripts = [
        'part1_pretrain_evaluation.py',
        'part2_domain_adaptation.py', 
        'part3_small_object_optimization.py'
    ]
    
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"❌ 脚本文件不存在: {script}")
            return False
        else:
            print(f"✅ 脚本文件存在: {script}")
    
    print("✅ 所有前提条件检查通过!")
    return True

def run_part1(mode='both'):
    """运行第一部分：预训练和评估"""
    print_step(1, "预训练和评估 (Sunny数据集)")
    
    start_time = time.time()
    
    try:
        # 运行第一部分脚本
        cmd = [sys.executable, 'part1_pretrain_evaluation.py', '--mode', mode]
        print(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        if result.returncode == 0:
            print("✅ Part 1 执行成功!")
            print("标准输出:")
            print(result.stdout)
        else:
            print("❌ Part 1 执行失败!")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Part 1 执行超时!")
        return False
    except Exception as e:
        print(f"❌ Part 1 执行异常: {e}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"Part 1 执行时间: {elapsed_time/60:.2f} 分钟")
    
    # 检查是否生成了预训练模型
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"⚠️  警告: 预训练模型文件未找到: {PRETRAINED_MODEL_PATH}")
        # 检查输出目录中是否有模型文件
        pretrain_output_dir = PRETRAINED_OUTPUT_PATH
        if os.path.exists(os.path.join(pretrain_output_dir, 'best_model.pth')):
            print(f"✅ 在输出目录中找到模型文件: {pretrain_output_dir}/best_model.pth")
        else:
            print("❌ 未找到训练好的模型文件，后续步骤可能失败")
            return False
    else:
        print(f"✅ 预训练模型文件存在: {PRETRAINED_MODEL_PATH}")
    
    return True

def run_part2():
    """运行第二部分：跨域适应实验"""
    print_step(2, "跨域适应实验 (Cloudy数据集)")
    
    start_time = time.time()
    
    try:
        # 运行第二部分脚本
        cmd = [sys.executable, 'part2_domain_adaptation.py']
        print(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)  # 3小时超时
        
        if result.returncode == 0:
            print("✅ Part 2 执行成功!")
            print("标准输出:")
            print(result.stdout)
        else:
            print("❌ Part 2 执行失败!")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Part 2 执行超时!")
        return False
    except Exception as e:
        print(f"❌ Part 2 执行异常: {e}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"Part 2 执行时间: {elapsed_time/60:.2f} 分钟")
    
    # 检查输出结果
    part2_output = OUTPUT_DIRS['part2']
    if os.path.exists(os.path.join(part2_output, 'summary_report.txt')):
        print(f"✅ Part 2 结果报告生成成功: {part2_output}/summary_report.txt")
    else:
        print("⚠️  Part 2 结果报告未生成")
    
    return True

def run_part3():
    """运行第三部分：小目标分割优化"""
    print_step(3, "小目标分割优化")
    
    start_time = time.time()
    
    try:
        # 运行第三部分脚本
        cmd = [sys.executable, 'part3_small_object_optimization.py']
        print(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)  # 3小时超时
        
        if result.returncode == 0:
            print("✅ Part 3 执行成功!")
            print("标准输出:")
            print(result.stdout)
        else:
            print("❌ Part 3 执行失败!")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Part 3 执行超时!")
        return False
    except Exception as e:
        print(f"❌ Part 3 执行异常: {e}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"Part 3 执行时间: {elapsed_time/60:.2f} 分钟")
    
    # 检查输出结果
    part3_output = OUTPUT_DIRS['part3']
    if os.path.exists(os.path.join(part3_output, 'improvement_analysis_report.txt')):
        print(f"✅ Part 3 结果报告生成成功: {part3_output}/improvement_analysis_report.txt")
    else:
        print("⚠️  Part 3 结果报告未生成")
    
    return True

def generate_final_summary():
    """生成最终总结报告"""
    print_step("总结", "生成最终实验报告")
    
    summary_path = os.path.join(os.getcwd(), 'final_experiment_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("完整实验流程总结报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"实验执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Part 1 结果
        f.write("第一部分：预训练和评估 (Sunny数据集)\n")
        f.write("-" * 40 + "\n")
        part1_results = os.path.join(OUTPUT_DIRS['part1'], 'results.txt')
        if os.path.exists(part1_results):
            f.write("✅ 执行成功\n")
            with open(part1_results, 'r', encoding='utf-8') as p1:
                f.write(p1.read())
        else:
            f.write("❌ 结果文件未找到\n")
        f.write("\n\n")
        
        # Part 2 结果
        f.write("第二部分：跨域适应实验 (Cloudy数据集)\n")
        f.write("-" * 40 + "\n")
        part2_results = os.path.join(OUTPUT_DIRS['part2'], 'summary_report.txt')
        if os.path.exists(part2_results):
            f.write("✅ 执行成功\n")
            with open(part2_results, 'r', encoding='utf-8') as p2:
                f.write(p2.read())
        else:
            f.write("❌ 结果文件未找到\n")
        f.write("\n\n")
        
        # Part 3 结果
        f.write("第三部分：小目标分割优化\n")
        f.write("-" * 40 + "\n")
        part3_results = os.path.join(OUTPUT_DIRS['part3'], 'improvement_analysis_report.txt')
        if os.path.exists(part3_results):
            f.write("✅ 执行成功\n")
            with open(part3_results, 'r', encoding='utf-8') as p3:
                f.write(p3.read())
        else:
            f.write("❌ 结果文件未找到\n")
        f.write("\n\n")
        
        # 输出目录信息
        f.write("输出文件位置:\n")
        f.write("-" * 20 + "\n")
        for part, output_dir in OUTPUT_DIRS.items():
            f.write(f"{part}: {output_dir}\n")
    
    print(f"✅ 最终总结报告已生成: {summary_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行完整的语义分割实验流程')
    parser.add_argument('--skip-part1', action='store_true', help='跳过Part 1 (假设模型已存在)')
    parser.add_argument('--skip-part2', action='store_true', help='跳过Part 2')
    parser.add_argument('--skip-part3', action='store_true', help='跳过Part 3')
    parser.add_argument('--part1-mode', choices=['train', 'eval', 'both'], default='both',
                       help='Part 1 的执行模式')
    parser.add_argument('--continue-on-error', action='store_true', 
                       help='在某个部分失败时继续执行后续部分')
    
    args = parser.parse_args()
    
    # 记录总开始时间
    total_start_time = time.time()
    
    print_separator("完整语义分割实验流程")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查前提条件
    if not check_prerequisites():
        print("❌ 前提条件检查失败，退出程序")
        return 1
    
    success_count = 0
    total_parts = 3
    
    try:
        # 运行Part 1
        if not args.skip_part1:
            if run_part1(args.part1_mode):
                success_count += 1
                print("✅ Part 1 完成!")
            else:
                print("❌ Part 1 失败!")
                if not args.continue_on_error:
                    print("退出程序")
                    return 1
        else:
            print("⏭️  跳过 Part 1")
            # 检查预训练模型是否存在
            if not os.path.exists(PRETRAINED_MODEL_PATH):
                print(f"❌ 跳过Part 1但预训练模型不存在: {PRETRAINED_MODEL_PATH}")
                return 1
        
        # 运行Part 2
        if not args.skip_part2:
            if run_part2():
                success_count += 1
                print("✅ Part 2 完成!")
            else:
                print("❌ Part 2 失败!")
                if not args.continue_on_error:
                    print("退出程序")
                    return 1
        else:
            print("⏭️  跳过 Part 2")
        
        # 运行Part 3
        if not args.skip_part3:
            if run_part3():
                success_count += 1
                print("✅ Part 3 完成!")
            else:
                print("❌ Part 3 失败!")
                if not args.continue_on_error:
                    print("退出程序")
                    return 1
        else:
            print("⏭️  跳过 Part 3")
        
        # 生成最终总结
        generate_final_summary()
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n❌ 执行过程中发生异常: {e}")
        return 1
    
    # 计算总执行时间
    total_elapsed_time = time.time() - total_start_time
    
    print_separator("实验流程完成")
    print(f"总执行时间: {total_elapsed_time/3600:.2f} 小时")
    print(f"成功完成: {success_count} / {total_parts} 个部分")
    
    if success_count == total_parts:
        print("🎉 所有实验部分都成功完成!")
        return 0
    else:
        print(f"⚠️  有 {total_parts - success_count} 个部分未成功完成")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)