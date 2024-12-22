import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# 加载预处理后的数据
data_path = 'processed_audio_data.csv'
data_df = pd.read_csv(data_path)

# 编码 Language 列为数值
le = LabelEncoder()
data_df['Language'] = le.fit_transform(data_df['Language'])
print(le.classes_)  # 查看 LabelEncoder 的分类顺序

# 假设 data_df 包含音频特征和目标变量 'StoryType'
# 分离特征和目标
X = data_df.drop(columns=['StoryType'])  # 特征
y = data_df['StoryType']                # 目标

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')  # 保存标准化器
print("标准化器已保存")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 应用 SMOTE 增强训练集
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# print(f"增强后的训练集大小: {X_train_resampled.shape}, {y_train_resampled.shape}")

#Logistic Regression

# 定义 Logistic Regression 参数网格
logistic_param_grid = {
    'C': [0.1, 1, 10, 100],  # 正则化强度
    'solver': ['lbfgs', 'liblinear']  # 优化算法
}

# 使用 GridSearchCV 进行优化
logistic_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), logistic_param_grid, cv=5, scoring='accuracy')
logistic_grid.fit(X_train, y_train)

# 最佳 Logistic Regression 模型
best_logistic_model = logistic_grid.best_estimator_

# 预测和评估
y_pred_logistic = best_logistic_model.predict(X_test)
print("Optimized Logistic Regression Results:")
print(f"Best Parameters: {logistic_grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic)}")
print(classification_report(y_test, y_pred_logistic))

# Random Forest

# 定义 Random Forest 参数网格
rf_param_grid = {
    'n_estimators': [50, 100, 150],  # 决策树数量
    'max_depth': [None, 10, 20],     # 最大深度
    'min_samples_split': [2, 5, 10], # 节点分裂所需的最小样本数
    'min_samples_leaf': [1, 2, 4]    # 叶子节点的最小样本数
}

# 使用 GridSearchCV 进行优化
# 临时禁用并行化
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1  # 禁用并行化，-1是启用
)
rf_grid.fit(X_train, y_train)

# print(type(X_train))  # 应该是 <class 'numpy.ndarray'>
# print(type(y_train))  # 应该是 <class 'numpy.ndarray' 或 'pandas.Series'>

# 最佳 Random Forest 模型
best_rf_model = rf_grid.best_estimator_

# 预测和评估
y_pred_rf = best_rf_model.predict(X_test)
print("Optimized Random Forest Results:")
print(f"Best Parameters: {rf_grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# SVM

# 定义 SVM 参数网格
svm_param_grid = {
    'C': [0.1, 1, 10, 100],          # 正则化参数
    'kernel': ['linear', 'rbf'],     # 核函数类型
    'gamma': ['scale', 'auto']       # 核函数系数
}

# 使用 GridSearchCV 进行优化
svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train, y_train)

# 最佳 SVM 模型
best_svm_model = svm_grid.best_estimator_

# 预测和评估
y_pred_svm = best_svm_model.predict(X_test)
print("Optimized SVM Results:")
print(f"Best Parameters: {svm_grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(classification_report(y_test, y_pred_svm))

# 模型比较
print("Model Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logistic)}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")

# 保存最佳 SVM 模型
joblib.dump(best_svm_model, 'svm_model.pkl')
print("SVM 模型已保存为 svm_model.pkl")

