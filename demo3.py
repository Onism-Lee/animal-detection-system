import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import warnings
import joblib


class VoiceClassifier:
    def __init__(self, audio_files, model_file='voice_classifier_model.pkl'):
        self.audio_files = audio_files
        self.model_file = model_file
        self.model = None

    def extract_features(self, audio_file):
        y, sr = librosa.load(audio_file)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        return np.vstack((mfcc, chroma, spectral_centroid, spectral_flux)).T

    def load_data(self):
        X = []
        y = []
        for audio_file, label in self.audio_files:
            mfcc = self.extract_features(audio_file)
            X.extend(mfcc)
            y.extend([label] * len(mfcc))
        return np.array(X), np.array(y)

    def train(self):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        self.model = model
        joblib.dump(model, self.model_file)

    def predict(self, audio_file):
        if self.model is None:
            self.model = joblib.load(self.model_file)
        mfcc = self.extract_features(audio_file)
        label = self.model.predict(mfcc)
        proba = self.model.predict_proba(mfcc)
        max_prob_idx = np.argmax(proba[0])
        max_prob_label = label[max_prob_idx]
        return max_prob_label

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    audio_files = [
        ('cat1.wav', 'Cat'),
        # ('cat2.wav', 'cat'),

        ('dog1.wav', 'Dog'),
        # ('dog-bark.wav', 'dog'),

        ('duck1.wav', 'Duck'),
        # ('duck2.wav', 'duck'),

        ('horse1.wav', 'Horse'),
        # ('horse2.wav', 'horse')

        # ('lion-roar1.wav', 'lion'),
        # ('lion-roar2.wav', 'lion')
    ]

    classifier = VoiceClassifier(audio_files)
    classifier.train()

    # 测试模型
    test_audio_file = 'cat2.wav'
    result = classifier.predict(test_audio_file)
    print(result)
