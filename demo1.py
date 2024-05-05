import librosa
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import IPython.display as ipd
import librosa.util

import matplotlib.pyplot as plt
import librosa.display

import warnings
warnings.filterwarnings("ignore")

# 加载数据集
def load_data():
    cat, sr1 = librosa.load('cat1.wav')
    dog, sr2 = librosa.load('dog1.wav')
    duck, sr3 = librosa.load('duck1.wav')
    horse, sr4 = librosa.load('horse1.wav')

    # 提取MFCC特征，这里也就是不同人声音音色提取
    cat_mfcc = librosa.feature.mfcc(y=cat, sr=sr1)
    dog_mfcc = librosa.feature.mfcc(y=dog, sr=sr2)
    duck_mfcc = librosa.feature.mfcc(y=duck, sr=sr3)
    horse_mfcc = librosa.feature.mfcc(y=horse, sr=sr4)

    # 将不同人声音色的MFCC特征合并成一个数据集
    X = np.concatenate((cat_mfcc.T, dog_mfcc.T, duck_mfcc.T, horse_mfcc.T), axis=0)
    # 修改标签向量
    # "cat" -> "cat", "dog" -> "dog", "duck" -> "duck", "horse" -> "horse"
    y = np.concatenate((np.full(len(cat_mfcc.T), "cat"), np.full(len(dog_mfcc.T), "dog"),np.full(len(duck_mfcc.T), "duck"), np.full(len(horse_mfcc.T), "horse")))
    return X, y

# 加载数据集
X, y = load_data()

# 训练模型
def train(X, y):
    # 将数据集分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用逻辑回归算法进行多类别分类
    model = LogisticRegression(multi_class='ovr')
    # 训练模型
    model.fit(X_train, y_train)

    return model

# 训练模型
model = train(X, y)



# 测试模型
def predict(model, file_path='cat2.wav'):
    # 加载音频文件并提取MFCC特征
    y, sr = librosa.load('cat2.wav')
    # 将音频数据标准化为浮点数格式
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # 进行多类别分类预测
    label = model.predict(mfcc.T)
    proba = model.predict_proba(mfcc.T)

    # 获取概率最大的类别标签
    max_prob_idx = np.argmax(proba[0])
    max_prob_label = label[max_prob_idx]
    return max_prob_label
result = predict(model, 'cat2.wav')
# 生成波形图
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=22050)
plt.show()
print('音色为：', result)

