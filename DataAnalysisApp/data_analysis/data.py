import os
import pandas as pd
import random
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from DataAnalysisApp import my_forms


class PrepareDataset():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_dataset(self):
        # ボストンの住宅価格
        if self.dataset_name == 'boston':
            dataset = datasets.load_boston()
        # アイリス（アヤメ）の種類
        elif self.dataset_name == 'iris':
            dataset = datasets.load_iris()
        # 糖尿病の進行状況
        elif self.dataset_name == 'diabetes':
            dataset = datasets.load_diabetes()
        # 手書き文字（数字）
        elif self.dataset_name == 'digits':
            dataset = datasets.load_digits()
        # 生理学的（physiological）測定結果と運動（exercise）測定結果
        elif self.dataset_name == 'linnerud':
            dataset = datasets.load_linnerud()
        # ワインの種類
        elif self.dataset_name == 'wine':
            dataset = datasets.load_wine()
        # がんの診断結果
        elif self.dataset_name == 'breast_cancer':
            dataset = datasets.load_breast_cancer()
        # 同一人物の様々な状態の顔画像（40人 x 10枚）
        elif self.dataset_name == 'olivetti_faces':
            dataset = datasets.fetch_olivetti_faces()
        # トピック別のニュース記事
        elif self.dataset_name == '20newsgroups':
            dataset = datasets.fetch_20newsgroups()
        # fetch_20newsgroups()の特徴抽出済みバージョン
        elif self.dataset_name == '20newsgroups_vectorized':
            dataset = datasets.fetch_20newsgroups_vectorized()
        # 有名人の顔写真
        elif self.dataset_name == 'lfw_people':
            dataset = datasets.fetch_lfw_people()
        # 有名人の顔写真
        elif self.dataset_name == 'lfw_pairs':
            dataset = datasets.fetch_lfw_pairs()
        # 森林の木の種類
        elif self.dataset_name == 'covtype':
            dataset = datasets.fetch_covtype()
        # カテゴリ別のニュース（ベクトル化済み）
        elif self.dataset_name == 'rcv1':
            dataset = datasets.fetch_rcv1()
        # ネットワークの侵入検知
        elif self.dataset_name == 'kddcup99':
            dataset = datasets.fetch_kddcup99()
        # カリフォルニアの住宅価格
        elif self.dataset_name == 'california_housing':
            dataset = datasets.fetch_california_housing()
        return dataset

    def create_dataframe(self):
        dataset = self.load_dataset()
        toy_dataset_names_keys = [key[0] for key in my_forms.toy_dataset_choice]
        if self.dataset_name in toy_dataset_names_keys:
            df_dataset = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        else:
            df_dataset = pd.DataFrame(dataset.data)
        return df_dataset

    def correlation(self, save_jpg_base_path):
        df_dataset = self.create_dataframe()
        correlation_matrix = df_dataset.corr().round(2)
        plt.figure(figsize=(15,15))
        plt.title('Feature Correlation')
        sns.heatmap(data=correlation_matrix, annot=True)
        corr_heat_jpg_path = os.path.join(save_jpg_base_path, f'corr_heat_{str(time.time())}.jpg')
        plt.savefig(corr_heat_jpg_path)
        return corr_heat_jpg_path

    def data_train_test_split(self, test_size=0.2, random_state=0):
        dataset = self.load_dataset()
        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def data_train_val_test_split(self, test_size=0.2, val_size=0.3, random_state=0):
        dataset = self.load_dataset()
        x_train_val, x_test, y_train_val, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size, random_state=random_state)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size, random_state=random_state)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def normalize(self, split='train_test'):
        scaler = StandardScaler()
        if split=='train_test':
            x_train, x_test, _, _ = self.data_train_test_split()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.fit_transform(x_test)
            return x_train_scaled, x_test_scaled
        elif split=='train_val_test':
            x_train, x_val, x_test, _, _, _ = self.data_train_val_test_split()
            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.fit_transform(x_val)
            x_test_scaled = scaler.fit_transform(x_test)
            return x_train_scaled, x_val_scaled, x_test_scaled
        elif split==False:
            df_dataset = self.create_dataframe()
            data_scaled = scaler.fit_transform(df_dataset)
            return data_scaled
        else:
            raise Exception('split value Error!')


    def get_label(self):
        dataset = self.load_dataset()
        return list(map(str, dataset.target_names))


    def draw_img(self, save_jpg_base_path, h=64, w=64):
        df_dataset = self.create_dataframe()
        fig, ax = plt.subplots(3,3)
        plt.suptitle('Part of the image dataset')
        for i in range(3):
            for j in range(3):
                ax[i][j].imshow(df_dataset.loc[random.randrange(len(df_dataset)),:].values.reshape(h,w))
        img_data_jpg_path = os.path.join(save_jpg_base_path, f'img_data_{str(time.time())}.jpg')
        plt.savefig(img_data_jpg_path)
        return img_data_jpg_path