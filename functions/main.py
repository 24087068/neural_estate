import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# IMPORTS:
# Local Import:
def load_local():
    csv_train = "../data/train.csv"
    img_train = "../data/train/"
    df_train_meta = pd.read_csv(csv_train)
    csv_test = "../data/test.csv"
    img_test = "../data/test/"
    df_test_meta = pd.read_csv(csv_test)
    return df_train_meta, img_train, df_test_meta, img_test
# Alt Colab import:
def load_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    csv_train = "/content/drive/MyDrive/neural_estate_data_colab/train.csv"
    img_train = "/content/drive/MyDrive/neural_estate_data_colab/train/"
    df_train_meta = pd.read_csv(csv_train)
    csv_test = "/content/drive/MyDrive/neural_estate_data_colab/test.csv"
    img_test = "/content/drive/MyDrive/neural_estate_data_colab/test/"
    df_test_meta = pd.read_csv(csv_test)
    return df_train_meta, img_train, df_test_meta, img_test

# EDA:
# Images:
def img_show(df, img_path):
    plt.figure(figsize=(20, 20))
    for i, (idx, row) in enumerate(df.sample(9).iterrows()):
        plt.subplot(3, 3, i + 1)
        img = cv2.imread(img_path + str(int(row['House ID'])) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(row['Price'])
        plt.axis('off')
    plt.show()
# Correlation:
def corr_show(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

# MLP:
def prep_mlp_data(df_train, df_test):
    features = ['Area', 'Bedrooms', 'Bathrooms', 'Latitude', 'Longitude']
    x_train = df_train[features].values.astype('float32')

    # GenAi: Target Scaling And Huber Loss For Hitting MALE Target.
    y_train = np.log1p(df_train['Price'].values.astype('float32'))
    x_test = df_test[features].values.astype('float32')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, scaler

def build_mlp():
    model = keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    # GenAi: Target Scaling And Huber Loss For Hitting MALE Target.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='huber',
                  metrics=['mape'])
    return model

# CNN:
def load_images(df, img_path, img_size=(128, 128)):
    imgs = []
    for house_id in df['House ID']:
        img = cv2.imread(img_path + str(int(house_id)) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        imgs.append(img)
    return np.array(imgs, dtype='float32') / 255.0

def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.1),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='huber', metrics=['mape'])
    return model


