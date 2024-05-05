import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# 加载声音文件并提取特征
def extract_features(dataset):
    # 使用Librosa库加载声音文件
    sound, sample_rate = librosa.load('dataset/Data Train/cat/cat1.wav', sr=None)

    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=13)

    return mfccs


# 创建数据集
def create_dataset(file_paths, labels):
    features = []
    for file_path in file_paths:
        features.append(extract_features(file_path))
    return np.array(features), np.array(labels)


# 构建深度学习模型
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 加载数据集和标签
file_paths = ['dataset/Data Train/cat/cat1.wav', 'dataset/Data Train/cat/cat2.wav', 'dataset/Data Train/dog/dog1.wav', 'dataset/Data Train/dog/dog2.wav', 'dataset/Data Train/dog/dog-bark3.wav', 'dataset/Data Train/dog/dog_bark2.wav', 'dataset/Data Train/duck/duck1.wav', 'dataset/Data Train/duck/duck2.wav', 'dataset/Data Train/horse/horse1.wav', 'dataset/Data Train/horse/horse2.wav']  # 声音文件路径
labels = ['cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'duck', 'duck', 'horse', 'horse']  # 对应的标签

# 创建训练和测试数据集
X, y = create_dataset(file_paths, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
input_shape = X_train.shape[1:]
num_classes = len(np.unique(y))
model = build_model(input_shape, num_classes)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# cat, sr5 = librosa.load('cat2.wav')
# dog, sr6 = librosa.load('dog2.wav')
# dog, sr6 = librosa.load('dog3.wav')
# dog, sr7 = librosa.load('dog4.wav')
# duck, sr7 = librosa.load('duck2.wav')
# horse, sr8 = librosa.load('horse2.wav')
# horse, sr10 = librosa.load('horse2.wav')

# cat_mfcc = librosa.feature.mfcc(y=cat, sr=sr5)
# dog_mfcc = librosa.feature.mfcc(y=dog, sr=sr6)
# dog_mfcc = librosa.feature.mfcc(y=dog, sr=sr6)
# dog_mfcc = librosa.feature.mfcc(y=dog, sr=sr7)
# duck_mfcc = librosa.feature.mfcc(y=duck, sr=sr7)
# horse_mfcc = librosa.feature.mfcc(y=horse, sr=sr8)
# horse_mfcc = librosa.feature.mfcc(y=horse, sr=sr10)

# # 生成波形图
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(y, sr=22050)
# plt.show()