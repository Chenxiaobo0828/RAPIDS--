## -------------------- ROC曲线 -----------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


def plot_ROC(y_true, y_pred, name, j, Dir): #真实数据、预测数据、保存名、保存位置
    plt.close()
    plt.figure(figsize=(5.7, 5.3))
    cutoff = pd.DataFrame(); List = pd.DataFrame(); J=0
    color = ['#2A8F2A', '#0F88FE', '#F43906', '#AE1F1E', "#3926B8", "#498FCC", "#FEAF63", "#FE6363", '#103F91', '#8E78BB']

    # ----------- 计算 AUC ----------------
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = round(auc(fpr, tpr), 3)
    if j == 0:
        cutoff.loc[0, j] = threshold[np.argmax(tpr - fpr)]
        print('  cutoff = ' + "%.3f" % cutoff.loc[0, j])

    # ----------- 画 ROC 曲线 ----------------
    plt.plot(fpr, tpr, color=color[j - 1], lw=1.5,
             label='  AUC = ' + "%.3f" % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    plt.subplots_adjust(left=0.105, right=0.965, top=0.965, bottom=0.1)
    plt.legend(loc="lower right", fontsize=9)

    y_pred = np.where(y_pred >= cutoff.loc[0, j], 1, 0)
    TN = metrics.confusion_matrix(y, y_pred)[0, 0]
    FP = metrics.confusion_matrix(y, y_pred)[0, 1]
    FN = metrics.confusion_matrix(y, y_pred)[1, 0]
    TP = metrics.confusion_matrix(y, y_pred)[1, 1]

    J = 1
    List.loc[J, 'Model'] = '神经网络'
    List.loc[J, 'AUC (95% CI)'] = "%.3f" % roc_auc
    List.loc[J, 'Accuracy'] = "%.3f" % ((TP + TN) / (TP + TN + FP + FN))
    List.loc[J, 'Sensitivity'] = "%.3f" % (TP / (TP + FN))
    List.loc[J, 'Specificity'] = "%.3f" % (TN / (FP + TN))
    List.loc[J, 'PPV'] = "%.3f" % (TP / (TP + FP))
    List.loc[J, 'NPV'] = "%.3f" % (TN / (TN + FN))
    List.loc[J, 'F1_score'] = "%.3f" % (2 / ((TP + FP) / TP + (TP + FN) / TP))
    # Matthews correlation coefficient https://mp.weixin.qq.com/s/Ucw-6sBCDQlOWHkBY-ihHA
    List.loc[J, 'MCC'] = "%.3f" % ((TP * TN - FP * FN) / pow((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 0.5))
    List.loc[J, 'Cutoff'] = "%.3f" % (cutoff.loc[0, j])
    
    plt.savefig(Dir + name + r'\ROC_curve.tiff', dpi=300)
    List.to_csv(Dir + name + r'\模型性能表.csv', encoding='gbk', index=False)
