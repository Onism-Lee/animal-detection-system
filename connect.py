import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import warnings
import joblib
import pymysql

class VoiceClassifier:
    def __init__(self, audio_files, model_file='voice_classifier_model.pkl'):
        self.audio_files = audio_files
        self.model_file = model_file
        self.model = None

    # ... (VoiceClassifier 类的其余部分，包括 extract_features、load_data、train 和 predict 函数)

# 定义数据库连接函数
def connect_to_db(host, port, db, user, password):
    try:
        conn = pymysql.connect(host=host, port=port, db=db, user=user, password=password)
        return conn
    except pymysql.Error as e:
        print(f"Database connection error: {e}")
        return None

# 更新停车位状态的函数
def update_parking_status(conn, id, is_available):
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute('UPDATE `parking_status_parkinglot` SET `is_available` = %s WHERE `id` = %s',
                               (is_available, id))
            conn.commit()
        except pymysql.Error as e:
            print(f"SQL execution error: {e}")

# 处理声音分类结果并更新数据库
def process_output(output, conn):
    c = []
    d = []
    for j in output:
        c.append(j)
    for x in range(len(c)):
        if c[x] == 'D':
            break
        elif c[x] == '1':
            if c[x - 2] == '1':
                continue
            else:
                d.append(c[x + 2])
        else:
            continue
    for y in range(0, 10):
        if str(y) in d:
            update_parking_status(conn, int(y), '1')
        else:
            update_parking_status(conn, int(y), '0')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # 连接到数据库
    host = localhost
    port = 3306
    db = 'ani'
    user = 'root'
    password = '111111'
    conn = connect_to_db(host, port, db, user, password)

    if conn:
        audio_files = [
            ('cat1.wav', 'Cat'),
            # ('cat2.wav', 'cat'),

            ('dog1.wav', 'Dog'),
            # ('dog-bark.wav', 'dog'),

            ('duck1.wav', 'Duck'),
            # ('duck2.wav', 'duck'),

            ('horse1.wav', 'Horse')
            # ('horse2.wav', 'horse')

            # ('lion-roar1.wav', 'lion'),
            # ('lion-roar2.wav', 'lion')
        ]

        classifier = VoiceClassifier(audio_files)
        classifier.train()

        test_audio_file = 'dog3.wav'
        result = classifier.predict(test_audio_file)
        print(result)

        # 更新数据库中的停车位状态
        process_output(result, conn)

        conn.close()
