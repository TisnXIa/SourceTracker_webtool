import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, roc_auc_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# 创建输出文件夹
os.makedirs('Output', exist_ok=True)

# 设置随机种子确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

class MicrobialSourceTracker:
    def __init__(self):
        self.asv_table = None
        self.map_df = None
        self.X = None
        self.y = None
        self.models = {}
        self.scaler = None
        
    def load_data(self, asv_file, map_file):
        """加载ASV丰度表和映射文件"""
        # 加载ASV表
        self.asv_table = pd.read_csv(asv_file, sep='\t', skiprows=1, index_col=0)
        # 转置：样本作为行，ASV作为列
        self.asv_table = self.asv_table.T
        
        # 加载映射文件（处理可能的BOM或特殊字符）
        # 先读取原始内容，手动处理BOM和不可见字符
        with open(map_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 移除BOM字符（如果存在）
        if lines and lines[0].startswith('\ufeff'):
            lines[0] = lines[0][1:]
        
        # 解析数据
        header = lines[0].strip().split('\t')
        data = []
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                data.append(parts)
        
        self.map_df = pd.DataFrame(data, columns=header)
        
        # 检查并修复列名
        if '#SampleID' not in self.map_df.columns:
            # 尝试手动修复列名（可能有不可见字符）
            self.map_df.columns = ['#SampleID', 'Env', 'SourceSink']
        
        # 清理样本ID中的不可见字符
        self.map_df['#SampleID'] = self.map_df['#SampleID'].str.replace(r'[^\w\-]', '', regex=True)
        
        self.map_df.set_index('#SampleID', inplace=True)
        
        # 确保样本顺序一致
        asv_samples = set(self.asv_table.index)
        map_samples = set(self.map_df.index)
        
        # 调试：打印样本匹配情况
        print(f"\nASV表中的样本: {sorted(asv_samples)}")
        print(f"Map文件中的样本: {sorted(map_samples)}")
        
        common_samples = list(asv_samples & map_samples)
        self.asv_table = self.asv_table.loc[common_samples]
        self.map_df = self.map_df.loc[common_samples]
        
        print(f"Loaded {self.asv_table.shape[0]} samples with {self.asv_table.shape[1]} ASVs")
        print(f"Sample distribution:\n{self.map_df['SourceSink'].value_counts()}")
        print(f"Environment distribution:\n{self.map_df['Env'].value_counts()}")
        
    def preprocess_data(self, method='standard'):
        """数据预处理"""
        # 特征矩阵
        self.X = self.asv_table.values
        
        # 标签：source=0, sink=1
        self.y = (self.map_df['SourceSink'] == 'sink').astype(int).values
        
        # 数据标准化
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 划分训练集和测试集（保持源/汇比例）
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def exploratory_analysis(self):
        """探索性数据分析"""
        print("\n=== 探索性数据分析 ===")
        
        # 样本丰度分布
        sample_sums = self.asv_table.sum(axis=1)
        print(f"\n样本测序深度统计:")
        print(f"  均值: {sample_sums.mean():.1f}")
        print(f"  中位数: {sample_sums.median():.1f}")
        print(f"  最小值: {sample_sums.min():.1f}")
        print(f"  最大值: {sample_sums.max():.1f}")
        
        # ASV稀疏性分析
        zero_ratio = (self.asv_table == 0).mean().mean()
        print(f"\nASV矩阵稀疏度: {zero_ratio*100:.1f}%")
        
        # PCA可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['SourceSink'] = self.map_df['SourceSink'].values
        pca_df['Env'] = self.map_df['Env'].values
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='SourceSink', 
                       palette=['blue', 'red'], s=100)
        plt.title('PCA - Source vs Sink')
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Env', 
                       palette=['green', 'orange'], s=100)
        plt.title('PCA - Environment')
        plt.tight_layout()
        plt.savefig('Output/pca_visualization.png', dpi=300)
        print("\nPCA可视化已保存到 Output/pca_visualization.png")
        
        # t-SNE可视化
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        X_tsne = tsne.fit_transform(self.X_scaled)
        tsne_df = pd.DataFrame(X_tsne, columns=['tSNE1', 'tSNE2'])
        tsne_df['SourceSink'] = self.map_df['SourceSink'].values
        tsne_df['Env'] = self.map_df['Env'].values
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=tsne_df, x='tSNE1', y='tSNE2', hue='SourceSink', 
                       palette=['blue', 'red'], s=100)
        plt.title('t-SNE - Source vs Sink')
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=tsne_df, x='tSNE1', y='tSNE2', hue='Env', 
                       palette=['green', 'orange'], s=100)
        plt.title('t-SNE - Environment')
        plt.tight_layout()
        plt.savefig('Output/tsne_visualization.png', dpi=300)
        print("t-SNE可视化已保存到 Output/tsne_visualization.png")
        
    def train_traditional_models(self):
        """训练传统机器学习模型"""
        print("\n=== 训练传统机器学习模型 ===")
        
        # 根据样本数量确定交叉验证折数
        min_class_count = min(np.sum(self.y_train == 0), np.sum(self.y_train == 1))
        cv_folds = min(3, min_class_count)  # 使用3折或更少（取决于最小类别样本数）
        print(f"\n样本数量较少，使用 {cv_folds} 折交叉验证")
        
        # 定义模型
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # 超参数网格（简化以适应小样本）
        param_grids = {
            'Logistic Regression': {'C': [0.1, 1, 10]},
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, None]},
            'Gradient Boosting': {'n_estimators': [50], 'learning_rate': [0.1]},
            'SVM': {'C': [1, 10], 'gamma': ['scale']}
        }
        
        results = []
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            # 网格搜索
            grid = GridSearchCV(model, param_grids[name], cv=cv_folds, n_jobs=-1)
            grid.fit(self.X_train, self.y_train)
            
            # 最佳模型
            best_model = grid.best_estimator_
            self.models[name] = best_model
            
            # 预测
            y_pred = best_model.predict(self.X_test)
            y_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            # 评估
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            
            print(f"  最佳参数: {grid.best_params_}")
            print(f"  交叉验证准确率: {grid.best_score_:.4f}")
            print(f"  测试集准确率: {accuracy:.4f}")
            print(f"  AUC-ROC: {roc_auc:.4f}")
            
            results.append({
                'Model': name,
                'BestParams': grid.best_params_,
                'CVAccuracy': grid.best_score_,
                'TestAccuracy': accuracy,
                'AUC': roc_auc
            })
            
            # 特征重要性（针对树模型）
            if hasattr(best_model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'ASV': self.asv_table.columns,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                print(f"  Top 10重要ASVs:\n{importances}")
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv('Output/traditional_model_results.csv', index=False)
        print("\n传统模型结果已保存到 Output/traditional_model_results.csv")
        
        return results_df
        
    def train_neural_network(self):
        """训练深度学习模型"""
        print("\n=== 训练深度学习模型 ===")
        
        # 构建神经网络
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 早停和模型保存
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        checkpoint = ModelCheckpoint('Output/best_nn_model.h5', monitor='val_loss', save_best_only=True)
        
        # 训练
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=8,
            callbacks=[early_stopping, checkpoint],
            verbose=0
        )
        
        # 评估
        loss, accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        y_proba = model.predict(self.X_test)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        print(f"测试集准确率: {accuracy:.4f}")
        print(f"AUC-ROC: {roc_auc:.4f}")
        
        # 保存模型
        model.save('Output/neural_network_model.h5')
        print("神经网络模型已保存到 Output/neural_network_model.h5")
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Output/nn_training_curve.png', dpi=300)
        print("训练曲线已保存到 Output/nn_training_curve.png")
        
        self.models['Neural Network'] = model
        
        return {'Model': 'Neural Network', 'TestAccuracy': accuracy, 'AUC': roc_auc}
    
    def perform_source_tracking(self):
        """执行溯源分析 - 预测sink样本的来源比例"""
        print("\n=== 执行溯源分析 ===")
        
        # 获取源样本和汇样本
        source_samples = self.map_df[self.map_df['SourceSink'] == 'source'].index
        sink_samples = self.map_df[self.map_df['SourceSink'] == 'sink'].index
        
        source_data = self.asv_table.loc[source_samples]
        sink_data = self.asv_table.loc[sink_samples]
        
        # 获取源样本的环境类型（动态获取，不硬编码）
        source_envs = self.map_df[self.map_df['SourceSink'] == 'source']['Env'].unique()
        print(f"\n源样本环境类型: {source_envs}")
        
        # 计算每个汇样本中各来源环境的贡献比例
        # 使用简单的丰度相似性方法
        source_by_env = {}
        for env in source_envs:
            env_samples = self.map_df[(self.map_df['SourceSink'] == 'source') & 
                                     (self.map_df['Env'] == env)].index
            if len(env_samples) > 0:
                source_by_env[env] = source_data.loc[env_samples].mean(axis=0)
                print(f"  环境 {env}: {len(env_samples)} 个源样本")
        
        # 如果只有单一环境来源，添加一个"Unknown"类别用于剩余比例
        if len(source_by_env) == 1:
            source_by_env['Unknown'] = pd.Series(0, index=self.asv_table.columns)
        
        # 预测每个sink样本的来源贡献
        contributions = []
        for sink in sink_samples:
            sink_profile = sink_data.loc[sink]
            total = 0
            contrib = {'SampleID': sink}
            
            for env, source_profile in source_by_env.items():
                if env == 'Unknown':
                    # Unknown类别使用一个很小的固定值
                    contrib[env] = 0.01
                    total += 0.01
                else:
                    # 计算余弦相似度作为贡献度
                    similarity = np.dot(sink_profile, source_profile) / (
                        np.linalg.norm(sink_profile) * np.linalg.norm(source_profile)
                    )
                    contrib[env] = similarity
                    total += similarity
            
            # 归一化
            if total > 0:
                for env in source_by_env.keys():
                    contrib[env] /= total
            
            contributions.append(contrib)
        
        contrib_df = pd.DataFrame(contributions)
        contrib_df.set_index('SampleID', inplace=True)
        
        # 保存溯源结果
        contrib_df.to_csv('Output/source_tracking_results.csv')
        print("溯源分析结果已保存到 Output/source_tracking_results.csv")
        print("\n溯源分析结果:")
        print(contrib_df)
        
        # 可视化溯源结果
        plt.figure(figsize=(10, 6))
        contrib_df.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Source Tracking Results - Environment Contributions')
        plt.xlabel('Sink Samples')
        plt.ylabel('Contribution Proportion')
        plt.legend(title='Source Environment')
        plt.tight_layout()
        plt.savefig('Output/source_tracking_barplot.png', dpi=300)
        print("\n溯源结果可视化已保存到 Output/source_tracking_barplot.png")
        
        return contrib_df
    
    def run_complete_analysis(self, asv_file, map_file):
        """运行完整的分析流程"""
        print("="*60)
        print("微生物溯源分析流程启动")
        print("="*60)
        
        # 步骤1: 加载数据
        self.load_data(asv_file, map_file)
        
        # 步骤2: 预处理
        self.preprocess_data()
        
        # 步骤3: 探索性分析
        self.exploratory_analysis()
        
        # 步骤4: 训练传统ML模型
        self.train_traditional_models()
        
        # 步骤5: 训练神经网络
        self.train_neural_network()
        
        # 步骤6: 执行溯源分析
        self.perform_source_tracking()
        
        print("\n" + "="*60)
        print("微生物溯源分析流程完成")
        print("="*60)

if __name__ == "__main__":
    tracker = MicrobialSourceTracker()
    tracker.run_complete_analysis(
        asv_file='ASV_table.txt',
        map_file='map.txt'
    )