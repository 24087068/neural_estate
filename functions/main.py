import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

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
    csv_train = "/content/drive/MyDrive/data/train.csv"
    img_train = "/content/drive/MyDrive/data/train/"
    df_train_meta = pd.read_csv(csv_train)
    csv_test = "/content/drive/MyDrive/data/test.csv"
    img_test = "/content/drive/MyDrive/data/test/"
    df_test_meta = pd.read_csv(csv_test)
    return df_train_meta, img_train, df_test_meta, img_test

# EDA:
# Images showcase:
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