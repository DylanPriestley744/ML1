"""
主运行脚本 - 植物分类项目
包含数据分析、模型训练和预测的完整流程
"""

import os
import sys

def main():
    """主流程"""
    print("="*70)
    print(" "*20 + "植物分类项目")
    print("="*70)
    
    print("\n请选择要执行的操作:")
    print("1. 数据分析 - 查看数据集统计信息和可视化")
    print("2. 特征提取测试 - 测试特征提取功能")
    print("3. 模型训练 - 训练SVM、Random Forest、XGBoost模型")
    print("4. 测试集预测 - 使用已训练模型预测测试集")
    print("5. 完整流程 - 依次执行所有步骤")
    print("0. 退出")
    
    choice = input("\n请输入选项 (0-5): ").strip()
    
    if choice == '1':
        print("\n执行数据分析...")
        from data_analysis import analyze_dataset
        analyze_dataset()
        
    elif choice == '2':
        print("\n执行特征提取测试...")
        from feature_extraction import test_feature_extraction
        test_feature_extraction()
        
    elif choice == '3':
        print("\n执行模型训练...")
        from train_model import main as train_main
        train_main()
        
    elif choice == '4':
        print("\n执行测试集预测...")
        print("请选择要使用的模型:")
        print("1. SVM")
        print("2. Random Forest")
        print("3. XGBoost")
        
        model_choice = input("请输入选项 (1-3): ").strip()
        model_map = {'1': 'svm', '2': 'rf', '3': 'xgb'}
        
        if model_choice in model_map:
            from train_model import PlantClassifier
            model_type = model_map[model_choice]
            
            classifier = PlantClassifier()
            
            # 检查模型是否存在
            model_path = os.path.join("models", f'{model_type}_model.pkl')
            if not os.path.exists(model_path):
                print(f"\n错误: 模型文件不存在 ({model_path})")
                print("请先执行选项3进行模型训练!")
                return
            
            classifier.load_model(model_type)
            classifier.predict_test_set()
        else:
            print("无效的选项!")
            
    elif choice == '5':
        print("\n执行完整流程...")
        
        # 1. 数据分析
        print("\n" + "="*70)
        print("步骤 1/3: 数据分析")
        print("="*70)
        from data_analysis import analyze_dataset
        analyze_dataset()
        
        # 2. 特征提取测试
        print("\n" + "="*70)
        print("步骤 2/3: 特征提取测试")
        print("="*70)
        from feature_extraction import test_feature_extraction
        test_feature_extraction()
        
        # 3. 模型训练和预测
        print("\n" + "="*70)
        print("步骤 3/3: 模型训练和预测")
        print("="*70)
        from train_model import main as train_main
        train_main()
        
        print("\n" + "="*70)
        print("完整流程执行完毕!")
        print("="*70)
        
    elif choice == '0':
        print("\n退出程序。")
        sys.exit(0)
        
    else:
        print("\n无效的选项，请重新运行程序。")


if __name__ == "__main__":
    main()
