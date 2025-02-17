import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 各画家の名前（ラベル）
artists = ['Vincent_van_Gogh',"Leonardo_da_Vinci"]
data_dir = '/content/drive/MyDrive/archive/images'  # Google Drive上のデータディレクトリ

# 画像データとラベルを取得
images = []
labels = []

for label, artist in enumerate(artists):
    artist_dir = os.path.join(data_dir, artist)
    for img_name in os.listdir(artist_dir):
        img_path = os.path.join(artist_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = cv2.resize(img, (50, 50))
            images.append(img)
            labels.append(label)

# NumPy配列に変換
X = np.array(images)
y = np.array(labels)

# データをシャッフルして分割
X, y = X / 255.0, np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ラベルをone-hotエンコーディング
y_train = to_categorical(y_train, num_classes=len(artists))
y_test = to_categorical(y_test, num_classes=len(artists))

# モデルの構築
input_tensor = Input(shape=(50, 50, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# 転移学習のトップモデル
model_top = Sequential()
model_top.add(Flatten(input_shape=vgg16.output_shape[1:]))
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(128, activation='relu'))
#model_top.add(Dropout(0.5))
model_top.add(Dense(64, activation='relu'))
#model_top.add(Dropout(0.5))
model_top.add(Dense(len(artists), activation='softmax'))

model = Model(inputs=vgg16.input, outputs=model_top(vgg16.output))

# VGG16の一部の層を固定
for layer in vgg16.layers:
    layer.trainable = False

# モデルのコンパイル
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)
# データ増強用のImageDataGeneratorを設定
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# データ増強用ジェネレーターを定義
train_generator = datagen.flow(X_train, y_train, batch_size=32)
# モデルの訓練
history = model.fit(
    train_generator,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)
# モデルの保存
save_dir = '/content/drive/MyDrive/saved_model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model.save(os.path.join(save_dir, 'artist_classifier.h5'))
print("モデルを保存しました。")

# モデルの評価
eval_result = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {eval_result[0]:.4f}")
print(f"Test Accuracy: {eval_result[1]:.4f}")

# 画像を分類する関数
def predict_artist(img_path):
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = cv2.resize(img, (50, 50))
    img = img.astype('float') / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = np.argmax(pred, axis=1)[0]
    return artists[label]

# テスト用
sample_image_path = os.path.join(data_dir,  '/content/drive/MyDrive/archive/images/Vincent_van_Gogh/Vincent_van_Gogh_281.jpg')  # 適宜パスを変更
predicted_artist = predict_artist(sample_image_path)
print(f"Predicted Artist: {predicted_artist}")