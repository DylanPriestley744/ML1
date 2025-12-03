"""
数据分析和可视化
分析训练集的基本情况：类别分布、图片尺寸等
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 数据路径
TRAIN_DIR = r"d:\ML1\neu-plant-seedling-classification-2025\dataset-for-task1\dataset-for-task1\train"
TEST_DIR = r"d:\ML1\neu-plant-seedling-classification-2025\dataset-for-task1\dataset-for-task1\test"

def analyze_dataset():
    """分析数据集基本信息"""
    print("="*50)
    print("数据集分析")
    print("="*50)
    
    # 统计类别和数量
    categories = os.listdir(TRAIN_DIR)
    print(f"\n类别数量: {len(categories)}")
    print("\n各类别样本数量:")
    
    total_images = 0
    image_sizes = []
    
    for category in categories:
        category_path = os.path.join(TRAIN_DIR, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            total_images += count
            print(f"  {category}: {count} 张")
            
            # 分析部分图片尺寸
            for i, img_name in enumerate(images[:10]):  # 每类抽样10张
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    image_sizes.append(img.shape[:2])  # (height, width)
    
    print(f"\n总训练样本数: {total_images}")
    
    # 测试集数量
    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"测试集样本数: {len(test_images)}")
    
    # 图片尺寸分析
    if image_sizes:
        heights, widths = zip(*image_sizes)
        print(f"\n图片尺寸统计 (采样{len(image_sizes)}张):")
        print(f"  高度范围: {min(heights)} - {max(heights)}")
        print(f"  宽度范围: {min(widths)} - {max(widths)}")
        print(f"  平均尺寸: {int(np.mean(heights))} x {int(np.mean(widths))}")
    
    # 可视化类别分布
    plt.figure(figsize=(10, 6))
    category_counts = {}
    for category in categories:
        category_path = os.path.join(TRAIN_DIR, category)
        if os.path.isdir(category_path):
            count = len([f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            category_counts[category] = count
    
    plt.bar(category_counts.keys(), category_counts.values(), color='skyblue')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title('训练集类别分布', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("\n类别分布图已保存为 data_distribution.png")
    plt.show()
    
    # 展示每类的样本示例
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for idx, category in enumerate(sorted(categories)):
        category_path = os.path.join(TRAIN_DIR, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                img_path = os.path.join(category_path, images[0])
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(category, fontsize=10)
                axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    print("样本示例图已保存为 sample_images.png")
    plt.show()

if __name__ == "__main__":
    analyze_dataset()
