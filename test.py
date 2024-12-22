import joblib
import librosa
import numpy as np
import pandas as pd

# 加载 SVM 模型和标准化器
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')  # 假设保存了标准化器

# 定义特征提取函数
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
        'rms': np.mean(librosa.feature.rms(y=y)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
    }
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(1, 14):
        features[f'mfcc_{i}'] = np.mean(mfccs[i-1])
    return features

# 提取新音频文件的特征
file_path = 'Mystory_test/00034.wav'  # 替换为实际文件路径
new_features = extract_features(file_path)

# 手动添加 Language
new_features['Language'] = 1  # 0 表示中文  1 表示英文
# 如果数据比较多，也可以从文件名或路径推断 Language

# 转为 DataFrame 并标准化
# 确保特征顺序一致
new_features_df = pd.DataFrame([new_features])
new_features_df = new_features_df[['zcr', 'rms', 'spectral_centroid', 'spectral_bandwidth'] + [f'mfcc_{i}' for i in range(1, 14)]+ ['Language']]
new_features_scaled = scaler.transform(new_features_df)

# 预测故事类型
predicted_class = svm_model.predict(new_features_scaled)
print(f"预测的故事类型是: {predicted_class[0]}")# 0 表示假的，1 表示真的
