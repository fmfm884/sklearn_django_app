import os
import time
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score


class ClfEval:
    def __init__(self, true, pred):
        self.true = true
        self.pred = pred

    def confusion_matrix(self, target_names, save_jpg_base_path):
        df_cmx = pd.DataFrame(confusion_matrix(self.true, self.pred), index=target_names, columns=target_names)
        plt.figure()
        sns.heatmap(df_cmx, square=True, cbar=True, annot=True, cmap='Blues')
        confusion_jpg_path = os.path.join(save_jpg_base_path, f'confusion_{str(time.time())}.jpg')
        plt.savefig(confusion_jpg_path)
        return confusion_jpg_path

    def pr_curve(self, save_jpg_base_path):
        precision, recall, _ = precision_recall_curve(self.true, self.pred)
        plt.figure()
        plt.plot(precision, recall)
        plt.xlabel('Precision')
        plt.ylabel('recall')
        pr_curve_jpg_path = os.path.join(save_jpg_base_path, f'pr_{str(time.time())}.jpg')
        plt.savefig(pr_curve_jpg_path)
        return pr_curve_jpg_path

    def clf_report(self, target_names):
        return pd.DataFrame(classification_report(self.true, self.pred, target_names=target_names, output_dict=True))


class RegEval:
    def __init__(self, true, pred):
        self.true = []
        if len(true.shape) > 1:
            for obj_var in range(true.shape[1]):
                self.true.append(true[:, obj_var])
        else:
            self.true.append(true)
        self.pred = pred

    def mae(self):
        return [mean_absolute_error(true, pred) for true, pred in zip(self.true, self.pred)]

    def mse(self):
        return [mean_squared_error(true, pred) for true, pred in zip(self.true, self.pred)]

    def rmse(self):
        return [np.sqrt(mse) for mse in self.mse()]

    def r2(self):
        return [r2_score(true, pred) for true, pred in zip(self.true, self.pred)]

    def reg_report(self, obj_var):
        columns = ['mae', 'mse', 'rmse', 'r2']
        values = [self.mae(), self.mse(), self.rmse(), self.r2()]
        report = dict(zip(columns, values))
        return pd.DataFrame(report, columns=columns, index=obj_var)


class ClstEval:
    def plot_data(df_data, save_jpg_base_path, labels):
        #colors = list(mcolors.TABLEAU_COLORS)
        #colors = list(mcolors.CSS4_COLORS)
        # 20色
        colors = []
        cmap_names = ['tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r']
        for cmap_name in cmap_names:
            cm = plt.cm.get_cmap(cmap_name)
            for color in cm.colors:
                colors.append(color)
        columns = list(df_data.columns)
        plt.figure()
        for idx_label, label in enumerate(labels):
            plt.scatter(df_data.iloc[:,0][df_data[columns[-1]]==idx_label], df_data.iloc[:,1][df_data[columns[-1]]==idx_label], s=10, color=colors[idx_label], label=label)
        plt.legend()
        data_jpg_path = os.path.join(save_jpg_base_path, f'cluster_{str(time.time())}.jpg')
        plt.savefig(data_jpg_path)
        return data_jpg_path
    
    def multi_plot_data(df_data, save_jpg_base_path, labels):
        #colors = list(mcolors.TABLEAU_COLORS) # 10色
        #colors = list(mcolors.CSS4_COLORS) # 148色
        # 20色
        colors = []
        cmap_names = ['tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r']
        for cmap_name in cmap_names:
            cm = plt.cm.get_cmap(cmap_name)
            for color in cm.colors:
                colors.append(color)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Prediction')
        for idx in list(df_data['pred'].unique()):
            plt.scatter(df_data.iloc[:,0][df_data['pred']==idx], df_data.iloc[:,1][df_data['pred']==idx], s=10, color=colors[idx])
        plt.subplot(1,2,2)
        plt.title('True')
        for idx_label, label in enumerate(labels):
            plt.scatter(df_data.iloc[:,0][df_data['true']==idx_label], df_data.iloc[:,1][df_data['true']==idx_label], s=10, color=colors[idx_label], label=label)
        plt.legend().get_frame().set_alpha(0.4)
        data_jpg_path = os.path.join(save_jpg_base_path, f'cluster_{str(time.time())}.jpg')
        plt.savefig(data_jpg_path)
        return data_jpg_path