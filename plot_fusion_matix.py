from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
import seaborn as sns
# 开始画图
# y_test为真实label，y_pred为预测label，classes为类别名称，是个ndarray数组，内容为string类型的标签
from DrawConfusion import DrawConfusionMatrix
from xtUtil import plot_confusion_matrix, modelUtil, plot_confusion_matrix2

# files_path = ['./log/2013open_resnet_fold2_val_checkpoint.pth.tarload.csv',
#               './log/2013open_resnet_fold1_val_checkpoint.pth.tarload.csv',
#               './log/2013open_resnet_fold0_val_model_best.pth.tarload.csv'
#               ]
# files_path = ['./log/2013open_resnet_att_fold0_val_16_model_best.pth.tarload.csv',
#               './log/2013open_resnet_att_fold1_val_21_checkpoint.pth.tarload.csv',
#               './log/2013open_resnet_att_fold2_val_50_checkpoint.pth.tarload.csv'
#               ]
files_path = ['./log/2013open_resnet_att_fold0_val_16_model_best.pth.tarprocessload.csv',
              './log/2013open_resnet_att_fold1_val_21_checkpoint.pth.tarprocessload.csv',
              # './log/2013open_resnet_att_fold2_val_50_checkpoint.pth.tarprocessload.csv'
              ]
# files_path = ['./log/2013open_resnet_fold0_val_model_best.pth.tarload.csv',
#               './log/2013open_resnet_fold1_val_model_best.pth.tarload.csv',
#               './log/2013open_resnet_fold2_val_checkpoint.pth.tarload.csv'
#               ]
# files_path = ['./log/2013closed_resnet_att_fold0_loss_checkpoint.pth.tarload.csv',
#               './log/2013closed_resnet_att_fold1_val_checkpoint.pth.tarload.csv',
#               './log/2013closed_resnet_att_fold2_val_model_best.pth.tarload.csv'
#               ]
# files_path = ['./log/2012wcom_resnet_fold0_val_checkpoint.pth.tarload.csv',
#               './log/2012wcom_resnet_fold1_val_checkpoint.pth.tarload.csv',
#               './log/2012wcom_resnet_fold2_val_model_best.pth.tarload.csv'
#               ]
# files_path = ['./log/receipt_resnet_att_fold0_val_checkpoint.pth.tarload.csv',
#               './log/receipt_resnet_att_fold1_val_model_best.pth.tarload.csv',
#               './log/receipt_resnet_att_fold2_val_checkpoint.pth.tarload.csv'
#               ]

class_names = np.array([i for i in range(26)])  # 按你的实际需要修改名称
# class_names = np.array(['0',"1", "2","3",'4','5'])  # 按你的实际需要修改名称
# class_names = np.array([1, 2, 3])  # 按你的实际需要修改名称
truthes = []
predictiones = []
acces = []
truthes = []
predictiones = []
for file_path in files_path:
    # truthes = []
    # predictiones = []
    data = pd.read_csv(file_path)
    # data['truth'] = data['truth'].astype('object')
    truth = data['target'].tolist()
    for t in truth:
        t = literal_eval(t)
        truthes.append(t)

    prediction = data['prediction'].tolist()
    for p in prediction:
        p = literal_eval(p)
        predictiones.append(p)
    from itertools import chain
    target = list(chain.from_iterable(truthes))
    preds_a = list(chain.from_iterable(predictiones))
    acc = accuracy_score(preds_a, target)
    acces.append(acc)
    print("acc:",acc)

print("avg acc : ",sum(acces)/len(acces))
from itertools import chain
target = list(chain.from_iterable(truthes))
preds_a = list(chain.from_iterable(predictiones))
acc = accuracy_score(preds_a, target)
print("acc:",acc)
plot_confusion_matrix(target, preds_a, classes=class_names, normalize=True,title=' ')
# plot_confusion_matrix2(target, preds_a, classes=class_names, normalize=True,title=' ')


# from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(target, preds_a)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             )
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()

# labels_name = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# drawconfusionmatrix = DrawConfusionMatrix(labels_name=class_names)  # 实例化
# prediction_array = np.asarray(preds_a).astype('int64')
# truth_array = np.array(target).astype('int64')
# drawconfusionmatrix.update(prediction_array,
#                            truth_array)  # 将新批次的predict和label更新（保存）
# drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵
#
# confusion_mat = drawconfusionmatrix.getMatrix()  # 你也可以使用该函数获取混淆矩阵(ndarray)
# print(confusion_mat)