import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 读取属性文件
csv_path = 'CBU0521DD_stories_attributes.csv'  # 替换为实际路径
attributes = pd.read_csv(csv_path)

# 查看数据
print(attributes.head())

def extract_features(file_path):
    """
    提取音频文件的特征
    :param file_path: 音频文件路径
    :return: 包含多个特征的字典
    """
    try:
        # 加载音频文件
        y, sr = librosa.load(file_path, sr=None)
        print(f"Loaded file: {file_path}, y: {len(y)}, sr: {sr}")

        # 提取特征
        features = {
            'zcr': np.mean(librosa.feature.zero_crossing_rate(y=y)),  # 零交叉率
            'rms': np.mean(librosa.feature.rms(y=y)),                 # 均方根能量
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),  # 频谱质心
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),# 频谱带宽
        }

        # 提取MFCC（取前13个系数的均值）
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(1, 14):
            features[f'mfcc_{i}'] = np.mean(mfccs[i - 1])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

audio_dir = 'CBU0521DD_stories'  # 替换为实际音频文件路径

# 创建用于存储特征的列表
data = []

for idx, row in attributes.iterrows():
    file_name = row['filename']  # 假设CSV中有音频文件名列
    file_path = os.path.join(audio_dir, file_name)
    # print(f"Processing file: {file_path}")
    
    # 提取音频特征
    features = extract_features(file_path)
    # print(f"Extracted features for {file_path}: {features}")
    if features:
        features['Language'] = row['Language']      # 添加语言属性
        features['StoryType'] = row['Story_type']    # 添加故事类型（目标变量）
        data.append(features)
    # else:
    #    print(f"Failed to extract features for {file_path}")

# 转为DataFrame
data_df = pd.DataFrame(data)

# 查看提取结果
print(data_df.head())

# 编码目标变量
label_encoder = LabelEncoder()
data_df['StoryType'] = label_encoder.fit_transform(data_df['StoryType'])

# 查看编码后的结果
print(data_df['StoryType'].value_counts())

# 分离特征和目标
X = data_df.drop(columns=['StoryType'])  # 特征
y = data_df['StoryType']                # 目标

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查训练集和测试集维度
print(f"训练集维度: {X_train.shape}, 测试集维度: {X_test.shape}")

#保存处理后的数据
data_df.to_csv('processed_audio_data.csv', index=False)
