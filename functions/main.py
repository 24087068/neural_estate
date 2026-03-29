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


# MULTIMODAL:
def prep_multimodal_data(df_train, df_test, img_train_path, img_test_path, img_size=224):
    """Load and prepare image + tabular data for the multimodal model."""
    features = ['Area', 'Bedrooms', 'Bathrooms', 'Latitude', 'Longitude']

    # Tabular features
    x_tab_train = df_train[features].values.astype('float32')
    x_tab_test = df_test[features].values.astype('float32')
    scaler = StandardScaler()
    x_tab_train = scaler.fit_transform(x_tab_train)
    x_tab_test = scaler.transform(x_tab_test)

    # Images (resized to img_size x img_size for EfficientNetB0)
    x_img_train = load_images(df_train, img_train_path, img_size=(img_size, img_size))
    x_img_test = load_images(df_test, img_test_path, img_size=(img_size, img_size))

    # Log-transform target (GenAi: Target Scaling And Huber Loss For Hitting MALE Target.)
    y_train = np.log1p(df_train['Price'].values.astype('float32'))

    return x_img_train, x_tab_train, y_train, x_img_test, x_tab_test, scaler


def build_multimodal(img_size=224):
    """
    Build a multimodal model with two parallel branches:
      - Image branch: EfficientNetB0 (frozen) + dense head
      - Tabular branch: fully-connected layers on the 5 metadata features
    The branches are merged via Concatenate, followed by shared dense layers.
    Returns (compiled model, base_model) so the caller can fine-tune the base.
    """
    from tensorflow.keras.applications import EfficientNetB0

    # --- Image branch ---
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False  # frozen in phase 1

    img_input = layers.Input(shape=(img_size, img_size, 3), name='image_input')
    # Data augmentation inside the model (active only during training)
    x_img = layers.RandomFlip('horizontal')(img_input)
    x_img = layers.RandomBrightness(0.1)(x_img)
    x_img = layers.RandomContrast(0.1)(x_img)
    x_img = base_model(x_img, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dense(256, activation='relu')(x_img)
    x_img = layers.Dropout(0.4)(x_img)  # L2-achtige regularisatie via Dropout

    # --- Tabular branch ---
    tab_input = layers.Input(shape=(5,), name='tabular_input')
    x_tab = layers.Dense(64, activation='relu')(tab_input)
    x_tab = layers.BatchNormalization()(x_tab)
    x_tab = layers.Dense(32, activation='relu')(x_tab)

    # --- Merge via Concatenate ---
    merged = layers.Concatenate()([x_img, x_tab])

    # --- Shared head ---
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, name='output')(x)

    model = keras.Model(inputs=[img_input, tab_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='huber',
        metrics=['mape']
    )
    return model, base_model


