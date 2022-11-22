import itertools
import os
import re
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class modelUtil:
    # 从一个文件夹里读取有指定前缀和后缀的文件
    # 用于当模型中有多个的时候能够一次性读取并保存为csv
    def readModel(self,path,head,tail):
        path_name = os.listdir(path)
        need_files_path = []
        for i in range(0, len(path_name)):
            file_path = path + str('/') + str(path_name[i])
            pat = '(' + head + '.*)' + tail
            data_name = re.findall(pat, file_path)
            if data_name != []:
                # print(data_name)
                need_files_path.append(path+'/'+data_name[0]+tail)
            else:
                pass
        need_files_path.sort()
        return need_files_path

    def calcAvg(self,path ,head):
        files_path = self.readModel(path,head,'load.csv')
        f1es = []
        prees = []
        acces = []
        reces = []
        print('path-----type------mean')
        for file_path in files_path:
            data = pd.read_csv(file_path)
            acces = list(data['acc'])
            acces_mean = sum(acces) / len(acces)
            print(file_path , "----acc------" ,acces_mean)
            prees = list(data['prec'])
            prees_mean = sum(prees) / len(prees)
            print(file_path , "----prec------" , prees_mean)
            reces = list(data['rec'])
            reces_mean = sum(reces) / len(reces)
            print(file_path, "----reces------" , reces_mean)
            f1es = list(data['f1'])
            f1es_mean = sum(f1es) / len(f1es)
            print(file_path , "----f1------" , f1es_mean)
            print("----next file------")

    def calcProcessAvg(self, path, head):
        files_path = self.readModel(path, head, 'processload.csv')
        f1es = []
        prees = []
        acces = []
        reces = []
        print('path-----type------mean')
        for file_path in files_path:
            data = pd.read_csv(file_path)
            acces = list(data['acc'])
            acces_mean = sum(acces) / len(acces)
            print(file_path, "----acc------", acces_mean)
            processacc = list(data['processacc'])
            processacc_mean = sum(processacc) / len(processacc)
            print(file_path, "----processacc------", processacc_mean)
            prees = list(data['prec'])
            prees_mean = sum(prees) / len(prees)
            print(file_path, "----prec------", prees_mean)
            reces = list(data['rec'])
            reces_mean = sum(reces) / len(reces)
            print(file_path, "----reces------", reces_mean)
            f1es = list(data['f1'])
            f1es_mean = sum(f1es) / len(f1es)
            print(file_path, "----f1------", f1es_mean)
            print("----next file------")

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    unique = unique_labels(y_true, y_pred)
    classes = classes[unique]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)
    # plt.figure(figsize =(400,400))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            # ax.text(j, i,'')
    fig.tight_layout()

    plt.show()
    return ax


# 以下代码为绘制混淆矩阵的代码
def plot_confusion_matrix2(y_true, y_pred, classes,
                          normalize=False,title="混淆矩阵",
                          cmap=plt.cm.Blues, save_flg=False):
    # classes = [str(i) for i in range(7)]
    # classes = ['<5%', '5%', '7%', '9%', '11%', '13%', '15%']
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    unique = unique_labels(y_true, y_pred)
    classes = classes[unique]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=7.5, fontfamily="SimSun")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    plt.rcParams['font.sans-serif'] = ['TimesNewRoman']
    plt.rcParams['axes.unicode_minus'] = False
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # plt.ylabel('True label', fontsize=15, fontfamily="SimSun")
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    if save_flg:
        plt.savefig("./confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    # modelUtil().calcAvg('./log/','2013closed_resnet18_att_fold')
    # modelUtil().calcAvg('./log/', '2013closed_resnet_att_fold')
    # modelUtil().calcProcessAvg('./log/', '2013open_resnet_fold')
    modelUtil().calcAvg('./log/', '2013open_resnet_att_fold')