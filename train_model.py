"""
模型训练和评估
使用交叉验证训练多种机器学习模型：SVM, Random Forest, XGBoost
"""

import os
import cv2
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import FeatureExtractor

# 路径配置
TRAIN_DIR = r"d:\ML1\neu-plant-seedling-classification-2025\dataset-for-task1\dataset-for-task1\train"
TEST_DIR = r"d:\ML1\neu-plant-seedling-classification-2025\dataset-for-task1\dataset-for-task1\test"
MODEL_DIR = "models"

# 创建模型保存目录
os.makedirs(MODEL_DIR, exist_ok=True)


class PlantClassifier:
    """植物分类器"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def load_training_data(self):
        """加载训练数据并提取特征"""
        print("加载训练数据...")
        
        X = []  # 特征
        y = []  # 标签
        
        categories = sorted(os.listdir(TRAIN_DIR))
        
        for category in tqdm(categories, desc="处理类别"):
            category_path = os.path.join(TRAIN_DIR, category)
            if not os.path.isdir(category_path):
                continue
                
            images = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in tqdm(images, desc=f"  {category}", leave=False):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 提取特征
                    features = self.feature_extractor.extract_all_features(img)
                    X.append(features)
                    y.append(category)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n数据加载完成!")
        print(f"特征矩阵形状: {X.shape}")
        print(f"标签数量: {len(y)}")
        print(f"类别: {np.unique(y)}")
        
        return X, y
    
    def train_with_cross_validation(self, X, y, model_type='svm', n_splits=5):
        """
        使用交叉验证训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            model_type: 模型类型 ('svm', 'rf', 'xgb')
            n_splits: 交叉验证折数
        """
        print(f"\n{'='*60}")
        print(f"训练模型: {model_type.upper()}")
        print(f"{'='*60}")
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建模型
        if model_type == 'svm':
            model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=200, max_depth=20, 
                                          min_samples_split=5, random_state=42, n_jobs=-1)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(n_estimators=200, max_depth=6, 
                                     learning_rate=0.1, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=skf, scoring='accuracy', n_jobs=-1)
        
        print(f"\n交叉验证结果 ({n_splits}-fold):")
        print(f"  各折准确率: {cv_scores}")
        print(f"  平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # 在全部训练数据上训练最终模型
        print(f"\n在全部训练数据上训练最终模型...")
        model.fit(X_scaled, y_encoded)
        
        # 训练集准确率
        train_pred = model.predict(X_scaled)
        train_acc = accuracy_score(y_encoded, train_pred)
        print(f"训练集准确率: {train_acc:.4f}")
        
        # 保存模型
        self.model = model
        self.save_model(model_type)
        
        # 打印分类报告
        print(f"\n分类报告:")
        print(classification_report(y_encoded, train_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # 混淆矩阵
        self.plot_confusion_matrix(y_encoded, train_pred, model_type)
        
        return cv_scores.mean()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'混淆矩阵 - {model_name.upper()}')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存为 confusion_matrix_{model_name}.png")
        plt.close()
    
    def save_model(self, model_type):
        """保存模型和相关对象"""
        model_path = os.path.join(MODEL_DIR, f'{model_type}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{model_type}_scaler.pkl')
        encoder_path = os.path.join(MODEL_DIR, f'{model_type}_encoder.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        print(f"\n模型已保存到: {MODEL_DIR}")
    
    def load_model(self, model_type):
        """加载模型和相关对象"""
        model_path = os.path.join(MODEL_DIR, f'{model_type}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{model_type}_scaler.pkl')
        encoder_path = os.path.join(MODEL_DIR, f'{model_type}_encoder.pkl')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        print(f"模型已从 {MODEL_DIR} 加载")
    
    def predict_test_set(self, output_file='submission-for-task1.csv'):
        """预测测试集并生成提交文件"""
        print("\n预测测试集...")
        
        test_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        predictions = []
        
        for img_name in tqdm(test_images, desc="预测"):
            img_path = os.path.join(TEST_DIR, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                # 提取特征
                features = self.feature_extractor.extract_all_features(img)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # 预测
                pred_encoded = self.model.predict(features_scaled)[0]
                pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
                
                predictions.append({
                    'image': img_name,
                    'label': pred_label
                })
        
        # 保存预测结果
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)
        print(f"\n预测结果已保存到: {output_file}")
        print(f"预测了 {len(predictions)} 张图片")
        
        # 显示预测类别分布
        print("\n预测类别分布:")
        print(df['label'].value_counts())
        
        return df


def main():
    """主训练流程"""
    print("="*60)
    print("植物分类模型训练")
    print("="*60)
    
    # 创建分类器
    classifier = PlantClassifier()
    
    # 加载训练数据
    X, y = classifier.load_training_data()
    
    # 训练多个模型并比较
    results = {}
    
    # 1. SVM
    print("\n" + "="*60)
    print("训练 SVM 模型")
    print("="*60)
    svm_score = classifier.train_with_cross_validation(X, y, model_type='svm', n_splits=5)
    results['SVM'] = svm_score
    
    # 2. Random Forest
    print("\n" + "="*60)
    print("训练 Random Forest 模型")
    print("="*60)
    classifier_rf = PlantClassifier()
    rf_score = classifier_rf.train_with_cross_validation(X, y, model_type='rf', n_splits=5)
    results['Random Forest'] = rf_score
    
    # 3. XGBoost
    print("\n" + "="*60)
    print("训练 XGBoost 模型")
    print("="*60)
    classifier_xgb = PlantClassifier()
    xgb_score = classifier_xgb.train_with_cross_validation(X, y, model_type='xgb', n_splits=5)
    results['XGBoost'] = xgb_score
    
    # 比较结果
    print("\n" + "="*60)
    print("模型性能比较")
    print("="*60)
    for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:20s}: {score:.4f}")
    
    # 选择最佳模型进行预测
    best_model = max(results, key=results.get)
    print(f"\n最佳模型: {best_model} (准确率: {results[best_model]:.4f})")
    
    # 使用最佳模型预测测试集
    if best_model == 'SVM':
        best_classifier = classifier
        model_type = 'svm'
    elif best_model == 'Random Forest':
        best_classifier = classifier_rf
        model_type = 'rf'
    else:
        best_classifier = classifier_xgb
        model_type = 'xgb'
    
    # 重新加载最佳模型（确保使用正确的模型）
    best_classifier.load_model(model_type)
    
    # 预测测试集
    best_classifier.predict_test_set()
    
    print("\n训练和预测完成!")


if __name__ == "__main__":
    main()
