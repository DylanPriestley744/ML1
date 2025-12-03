"""
特征提取模块
实现多种传统机器学习特征提取方法：
1. HOG (Histogram of Oriented Gradients) - 方向梯度直方图
2. LBP (Local Binary Patterns) - 局部二值模式（自实现，无需 scikit-image）
3. Color Histogram - 颜色直方图
4. GLCM (Gray Level Co-occurrence Matrix) - 灰度共生矩阵（自实现，无需 scikit-image）
"""

import cv2
import numpy as np

class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self, image_size=(128, 128)):
        """
        初始化特征提取器
        
        Args:
            image_size: 统一的图像尺寸 (height, width)
        """
        self.image_size = image_size
        
        # HOG参数
        self.hog = cv2.HOGDescriptor(
            _winSize=(128, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        # LBP参数（自实现使用 radius=1, points=8 的经典设置）
        self.lbp_radius = 1
        self.lbp_points = 8
        
    def preprocess_image(self, img):
        """
        图像预处理：调整大小和归一化
        
        Args:
            img: BGR格式的输入图像
            
        Returns:
            预处理后的图像
        """
        # 调整大小
        img_resized = cv2.resize(img, self.image_size)
        return img_resized
    
    def extract_hog_features(self, img):
        """
        提取HOG特征
        
        Args:
            img: 预处理后的BGR图像
            
        Returns:
            HOG特征向量
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 提取HOG特征
        hog_features = self.hog.compute(gray)
        
        return hog_features.flatten()
    
    def extract_lbp_features(self, img):
        """
        提取LBP特征
        
        Args:
            img: 预处理后的BGR图像
            
        Returns:
            LBP直方图特征
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 自实现 LBP（P=8, R=1，邻域为整数坐标的固定八邻域）
        h, w = gray.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        center = gray[1:-1, 1:-1]
        neighbors = [
            gray[0:-2, 1:-1],  # 上
            gray[0:-2, 2:  ],  # 右上
            gray[1:-1, 2:  ],  # 右
            gray[2:  , 2:  ],  # 右下
            gray[2:  , 1:-1],  # 下
            gray[2:  , 0:-2],  # 左下
            gray[1:-1, 0:-2],  # 左
            gray[0:-2, 0:-2],  # 左上
        ]
        for i, nb in enumerate(neighbors):
            lbp |= ((nb >= center).astype(np.uint8) << i)
        # 计算直方图（256 bins），并归一化
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=False)
        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist
    
    def extract_color_histogram(self, img):
        """
        提取颜色直方图特征（HSV空间）
        
        Args:
            img: 预处理后的BGR图像
            
        Returns:
            颜色直方图特征
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 计算每个通道的直方图
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        return np.array(hist_features)
    
    def extract_glcm_features(self, img):
        """
        提取GLCM纹理特征
        
        Args:
            img: 预处理后的BGR图像
            
        Returns:
            GLCM特征向量
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 量化到 8 级（0..7），降低计算量
        q = (gray.astype(np.uint16) * 8) // 256
        q = q.astype(np.uint8)

        levels = 8
        # 方向偏移（distance=1）：0°, 45°, 90°, 135°
        offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]  # (dy, dx)

        def glcm_for_offset(arr, dy, dx):
            h, w = arr.shape
            if dy >= 0:
                a = arr[0:h-dy, :]
                b = arr[dy:h,   :]
            else:
                a = arr[-dy:h, :]
                b = arr[0:h+dy, :]
            if dx >= 0:
                a = a[:, 0:w-dx]
                b = b[:, dx:w]
            else:
                a = a[:, -dx:w]
                b = b[:, 0:w+dx]
            # 计算共生矩阵
            cm = np.zeros((levels, levels), dtype=np.float64)
            # 将 (a,b) 像素对映射到线性索引累加
            idx = a.reshape(-1) * levels + b.reshape(-1)
            counts = np.bincount(idx, minlength=levels*levels)
            cm = counts.reshape(levels, levels).astype(np.float64)
            # 对称化
            cm = cm + cm.T
            # 归一化
            s = cm.sum()
            if s > 0:
                cm /= s
            return cm

        def glcm_props(cm):
            # 计算纹理属性
            i = np.arange(levels).reshape(-1, 1)
            j = np.arange(levels).reshape(1, -1)
            diff = i - j
            contrast = (diff**2 * cm).sum()
            dissimilarity = (np.abs(diff) * cm).sum()
            homogeneity = (cm / (1.0 + diff**2)).sum()
            energy = np.sqrt((cm**2).sum())  # 常见定义为 sqrt(sum p^2)
            # 相关性
            pi = cm.sum(axis=1)
            pj = cm.sum(axis=0)
            mui = (np.arange(levels) * pi).sum()
            muj = (np.arange(levels) * pj).sum()
            si = np.sqrt(((np.arange(levels) - mui)**2 * pi).sum())
            sj = np.sqrt(((np.arange(levels) - muj)**2 * pj).sum())
            if si > 1e-12 and sj > 1e-12:
                # sum_{i,j} ((i-mu_i)(j-mu_j) p_ij) / (si*sj)
                corr = (((i - mui) * (j - muj) * cm).sum()) / (si * sj)
            else:
                corr = 1.0
            return np.array([contrast, dissimilarity, homogeneity, energy, corr], dtype=np.float64)

        feats = []
        for dy, dx in offsets:
            cm = glcm_for_offset(q, dy, dx)
            feats.append(glcm_props(cm))
        return np.concatenate(feats, axis=0)
    
    def extract_color_moments(self, img):
        """
        提取颜色矩特征 (均值、标准差、偏度)
        
        Args:
            img: 预处理后的BGR图像
            
        Returns:
            颜色矩特征
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        features = []
        for i in range(3):
            channel = hsv[:, :, i]
            # 均值
            mean = np.mean(channel)
            # 标准差
            std = np.std(channel)
            # 偏度
            skewness = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
            
            features.extend([mean, std, skewness])
        
        return np.array(features)
    
    def extract_all_features(self, img):
        """
        提取所有特征并拼接
        
        Args:
            img: 原始BGR图像
            
        Returns:
            组合特征向量
        """
        # 预处理
        img_processed = self.preprocess_image(img)
        
        # 提取各种特征
        hog_feat = self.extract_hog_features(img_processed)
        lbp_feat = self.extract_lbp_features(img_processed)
        color_hist_feat = self.extract_color_histogram(img_processed)
        glcm_feat = self.extract_glcm_features(img_processed)
        color_moment_feat = self.extract_color_moments(img_processed)
        
        # 拼接所有特征
        all_features = np.concatenate([
            hog_feat,
            lbp_feat,
            color_hist_feat,
            glcm_feat,
            color_moment_feat
        ])
        
        return all_features


def test_feature_extraction():
    """测试特征提取"""
    import os
    
    # 读取一张示例图片
    train_dir = r"d:\ML1\neu-plant-seedling-classification-2025\dataset-for-task1\dataset-for-task1\train"
    categories = os.listdir(train_dir)
    
    sample_category = categories[0]
    sample_path = os.path.join(train_dir, sample_category)
    sample_images = [f for f in os.listdir(sample_path) if f.endswith('.png')]
    
    if sample_images:
        img_path = os.path.join(sample_path, sample_images[0])
        img = cv2.imread(img_path)
        
        print("测试特征提取...")
        extractor = FeatureExtractor()
        
        print(f"原始图像尺寸: {img.shape}")
        
        # 测试各个特征
        hog_feat = extractor.extract_hog_features(extractor.preprocess_image(img))
        print(f"HOG特征维度: {hog_feat.shape}")
        
        lbp_feat = extractor.extract_lbp_features(extractor.preprocess_image(img))
        print(f"LBP特征维度: {lbp_feat.shape}")
        
        color_feat = extractor.extract_color_histogram(extractor.preprocess_image(img))
        print(f"颜色直方图特征维度: {color_feat.shape}")
        
        glcm_feat = extractor.extract_glcm_features(extractor.preprocess_image(img))
        print(f"GLCM特征维度: {glcm_feat.shape}")
        
        color_moment_feat = extractor.extract_color_moments(extractor.preprocess_image(img))
        print(f"颜色矩特征维度: {color_moment_feat.shape}")
        
        all_feat = extractor.extract_all_features(img)
        print(f"\n组合特征总维度: {all_feat.shape}")


if __name__ == "__main__":
    test_feature_extraction()
