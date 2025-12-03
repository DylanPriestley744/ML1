"""
XGBoost 超参调优 + PCA 降维（不使用深度学习）
- 使用 RandomizedSearchCV 做随机搜索
- 使用 StratifiedKFold 分层交叉验证
- 保存最佳模型，并用其预测 test 集，生成 submission-for-task1_tuned.csv
- 同时生成 ID,Category 表头的 submission-for-task1_tuned_formatted.csv

用法（PowerShell）：
    .\.venv\Scripts\Activate
    python tune_xgb.py --n_iter 40 --pca 200 --scoring f1_macro --cv 5
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import joblib
import cv2
import time

from train_model import PlantClassifier, TEST_DIR, MODEL_DIR


def build_search_pipeline(pca_dim: int, random_state: int = 42, use_pca: bool = True):
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=pca_dim, random_state=random_state)))
    steps.append(('clf', xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        tree_method='hist',
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=-1,
        random_state=random_state,
    )))
    return Pipeline(steps)


def get_param_distributions():
    return {
        'clf__n_estimators': [100, 200, 400, 800],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__max_depth': [3, 4, 6, 8, 10],
        'clf__subsample': [0.6, 0.7, 0.8, 1.0],
        'clf__colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'clf__min_child_weight': [1, 3, 5, 10],
        'clf__gamma': [0, 0.1, 0.2, 0.5],
        'clf__reg_alpha': [0, 0.01, 0.1, 1.0],
        'clf__reg_lambda': [0.5, 1.0, 5.0],
    }


def load_or_extract_data(use_cache: bool, random_state: int = 42):
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    feat_path = os.path.join(cache_dir, 'features.npy')
    label_path = os.path.join(cache_dir, 'labels.npy')
    classes_path = os.path.join(cache_dir, 'classes.npy')

    pc = PlantClassifier()

    if use_cache and os.path.exists(feat_path) and os.path.exists(label_path) and os.path.exists(classes_path):
        print('[CACHE] 读取已缓存特征...')
        X = np.load(feat_path)
        y = np.load(label_path)
        pc.label_encoder.classes_ = np.load(classes_path)
    else:
        print('[CACHE] 未命中或关闭缓存，开始特征提取...')
        X, y = pc.load_training_data()
        # 拟合编码器再保存类别顺序
        pc.label_encoder.fit(y)
        if use_cache:
            np.save(feat_path, X)
            np.save(label_path, y)
            np.save(classes_path, pc.label_encoder.classes_)
            print(f'[CACHE] 已缓存到 {cache_dir}/')
    y_enc = pc.label_encoder.transform(y)
    return pc, X, y, y_enc


def run_search(n_iter: int, pca_dim: int, scoring: str, cv_splits: int, repeats: int, use_cache: bool, random_state: int = 42, use_pca: bool = True):
    # 1) 加载或提取特征
    start_time = time.time()
    pc, X, y, y_enc = load_or_extract_data(use_cache=use_cache, random_state=random_state)

    # 2) 构建流水线 + CV 定义
    pipeline = build_search_pipeline(pca_dim, random_state, use_pca=use_pca)
    param_dist = get_param_distributions()
    if repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=repeats, random_state=random_state)
        print(f'[CV] 使用 RepeatedStratifiedKFold: {cv_splits} 折 x {repeats} 次')
    else:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        print(f'[CV] 使用 StratifiedKFold: {cv_splits} 折')

    # 3) 搜索器
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )

    print(f'[SEARCH] 开始随机搜索 n_iter={n_iter}, scoring={scoring}, pca_dim={pca_dim}')
    search.fit(X, y_enc)

    # 4) 结果输出
    print("\n========== 搜索完成 ==========")
    print("Best score (CV):", search.best_score_)
    print("Best params:\n", search.best_params_)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, 'xgb_pca_tuned.joblib')
    joblib.dump({'pipeline': search.best_estimator_, 'label_encoder': pc.label_encoder}, best_model_path)
    print(f"[SAVE] 最佳模型保存: {best_model_path}")

    # 5) 保存搜索日志
    cv_results = pd.DataFrame(search.cv_results_)
    log_path = os.path.join(MODEL_DIR, 'xgb_search_log.csv')
    cv_results.to_csv(log_path, index=False)
    print(f"[LOG] 搜索日志保存: {log_path}")

    # 6) 简单频率统计（显示前 10 行参数与分数）
    top = cv_results.sort_values(by='mean_test_score', ascending=False).head(10)
    print('\n[TOP-10] 参数组合 (mean_test_score 排序):')
    cols_show = [c for c in cv_results.columns if c.startswith('param_')] + ['mean_test_score', 'std_test_score']
    print(top[cols_show])

    elapsed = time.time() - start_time
    print(f"[TIME] 总耗时: {elapsed:.1f} 秒")

    return search.best_estimator_, pc.label_encoder


def predict_test_and_save(best_estimator, label_encoder, output_csv: str, output_csv_formatted: str):
    # 读取测试集图片文件名
    test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"测试集图片数: {len(test_images)}")

    # 使用与 PlantClassifier 相同的特征提取器
    from feature_extraction import FeatureExtractor
    extractor = FeatureExtractor()

    X_test = []
    names = []

    for img_name in test_images:
        img_path = os.path.join(TEST_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        feats = extractor.extract_all_features(img)
        X_test.append(feats)
        names.append(img_name)

    X_test = np.array(X_test)

    # 直接用 pipeline 预测（会自动做 scaler+pca+clf）
    y_pred_enc = best_estimator.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    # 保存 image,label 格式
    pd.DataFrame({'image': names, 'label': y_pred}).to_csv(output_csv, index=False)
    print(f"保存: {output_csv}")

    # 保存 ID,Category 格式
    pd.DataFrame({'ID': names, 'Category': y_pred}).to_csv(output_csv_formatted, index=False)
    print(f"保存: {output_csv_formatted}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=40, help='随机搜索次数（越大越稳，越耗时）')
    parser.add_argument('--pca', type=int, default=200, help='PCA降维目标维度')
    parser.add_argument('--scoring', type=str, default='accuracy', help='评分指标，如 accuracy 或 f1_macro')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--repeats', type=int, default=1, help='重复交叉验证次数 (1 表示不重复)')
    parser.add_argument('--no_cache', action='store_true', help='禁用特征缓存')
    parser.add_argument('--no_pca', action='store_true', help='跳过PCA，仅使用StandardScaler+XGB')
    args = parser.parse_args()

    best_estimator, label_encoder = run_search(
        n_iter=args.n_iter,
        pca_dim=args.pca,
        scoring=args.scoring,
        cv_splits=args.cv,
        repeats=args.repeats,
        use_cache=(not args.no_cache),
        use_pca=(not args.no_pca),
    )

    predict_test_and_save(
        best_estimator,
        label_encoder,
        output_csv='submission-for-task1_tuned.csv',
        output_csv_formatted='submission-for-task1_tuned_formatted.csv'
    )
