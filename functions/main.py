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


# Transfer learning
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def prepare_data_tl(df_train_meta, img_train, df_test_meta, img_test):
    """Maak image paths en houd alleen bestaande afbeeldingen over."""
    train_df_tl = df_train_meta.copy()
    test_df_tl = df_test_meta.copy()

    train_df_tl["image_path"] = train_df_tl["House ID"].apply(
        lambda x: os.path.join(img_train, f"{int(x)}.jpg")
    )
    test_df_tl["image_path"] = test_df_tl["House ID"].apply(
        lambda x: os.path.join(img_test, f"{int(x)}.jpg")
    )

    train_df_tl = train_df_tl[
        train_df_tl["image_path"].apply(os.path.exists)
    ].reset_index(drop=True)

    test_df_tl = test_df_tl[
        test_df_tl["image_path"].apply(os.path.exists)
    ].reset_index(drop=True)

    return train_df_tl, test_df_tl


def split_data_tl(train_df_tl, test_size=0.2):
    """Maak train/validation split."""
    return train_test_split(train_df_tl, test_size=test_size, random_state=42)


def load_train_image_tl(path, price):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    img = tf.cast(img, tf.float32)

    return img, tf.cast(price, tf.float32)


def load_test_image_tl(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    img = tf.cast(img, tf.float32) 

    return img


def make_train_val_datasets_tl(train_df_tl, val_df_tl):
    """Maak tf.data datasets voor train en validatie."""
    train_ds_tl = tf.data.Dataset.from_tensor_slices(
        (train_df_tl["image_path"].values, train_df_tl["Price"].values)
    )
    val_ds_tl = tf.data.Dataset.from_tensor_slices(
        (val_df_tl["image_path"].values, val_df_tl["Price"].values)
    )

    train_ds_tl = (
        train_ds_tl
        .map(load_train_image_tl, num_parallel_calls=AUTOTUNE)
        .shuffle(200)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_ds_tl = (
        val_ds_tl
        .map(load_train_image_tl, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds_tl, val_ds_tl


def make_test_dataset_tl(test_df_tl):
    """Maak tf.data dataset voor test."""
    test_ds_tl = tf.data.Dataset.from_tensor_slices(
        test_df_tl["image_path"].values
    )

    test_ds_tl = (
        test_ds_tl
        .map(load_test_image_tl, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return test_ds_tl


def build_model_tl():
    """Bouw EfficientNetB0 transfer learning model voor regressie."""
    base_model_tl = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model_tl.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)

    x = base_model_tl(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1)(x)

    model_tl = models.Model(inputs, outputs)

    model_tl.compile(
        optimizer="adam",
        loss="huber",
        metrics=["mae"]
    )

    return model_tl, base_model_tl


def train_head_tl(model_tl, train_ds_tl, val_ds_tl, epochs=6):
    """Train alleen de nieuwe output-head."""
    history_tl = model_tl.fit(
        train_ds_tl,
        validation_data=val_ds_tl,
        epochs=epochs,
        callbacks=[
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3)
    ]
    )
    return history_tl


def fine_tune_model_tl(
    model_tl,
    base_model_tl,
    train_ds_tl,
    val_ds_tl,
    epochs=15,
    unfreeze_last=100
):
    """Fine-tune de bovenste lagen van EfficientNet."""
    base_model_tl.trainable = True

    for layer in base_model_tl.layers[:-unfreeze_last]:
        layer.trainable = False

    model_tl.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="huber",
        metrics=["mae"]
    )

    history_tl = model_tl.fit(
        train_ds_tl,
        validation_data=val_ds_tl,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
    )
    return history_tl


def plot_history_tl(history_tl, title):
    """Plot loss en MAE."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_tl.history["loss"], label="train")
    plt.plot(history_tl.history["val_loss"], label="val")
    plt.title(f"{title} - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_tl.history["mae"], label="train")
    plt.plot(history_tl.history["val_mae"], label="val")
    plt.title(f"{title} - MAE")
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_test_tl(
    model_tl,
    test_ds_tl,
    test_df_tl,
    file_name="submission_transfer.csv"
):
    """Maak voorspellingen voor de testset en sla submission op."""
    preds_tl = model_tl.predict(test_ds_tl).flatten()
    preds_tl = np.expm1(preds_tl)
    preds_tl = np.maximum(preds_tl, 0)

    submission_tl = test_df_tl[["House ID"]].copy()
    submission_tl["Price"] = preds_tl
    submission_tl.to_csv(file_name, index=False)

    return submission_tl