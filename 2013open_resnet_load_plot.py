import datetime
from time import perf_counter, time

from sklearn import metrics

seed = 123
import pandas as pd
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import getResnet as Resnet
from resnet_nam import AverageMeter, save_checkpoint
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from receiptData import getData, get2013openData

# from deeplearning.myExperiment.resnet50 import ResNet50

criterion = None

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def validate(val_loader, model, criterion, file_path):

    losses = AverageMeter()
    preces = AverageMeter()
    acces = AverageMeter()
    reces = AverageMeter()
    f1es = AverageMeter()

    df = pd.DataFrame(columns=['time', 'Epoch', 'Loss', 'acc','prec','rec','f1','truth','prediction'])  # 列名
    df.to_csv("./log/"+file_path +"load.csv", index=False,mode='w')  # 路径可以根据需要更改

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):

        with torch.no_grad():
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
        preds_a = np.argmax(output.detach().cpu().numpy(), axis=1)
        target = np.argmax(target, axis=1)
        # measure accuracy and record loss

        acc = accuracy_score(preds_a, target)
        prec, rec, f1, _ = metrics.precision_recall_fscore_support(
            target, preds_a, average="weighted")
        print('\n',metrics.classification_report(
           target, preds_a))
        # prec = precision_score(preds_a, target)
        # rec = recall_score(preds_a, target)
        # f1 = f1_score(preds_a, target)
        losses.update(loss.item(), input.size(0))
        preces.update(prec, input.size(0))
        acces.update(acc, input.size(0))
        reces.update(rec, input.size(0))
        f1es.update(f1, input.size(0))

        # 开始画图
        # y_test为真实label，y_pred为预测label，classes为类别名称，是个ndarray数组，内容为string类型的标签
        # class_names = np.array(["1", "2","3"])  # 按你的实际需要修改名称
        # plot_confusion_matrix(target, preds_a, classes=class_names, normalize=False)

        # 将训练结果放入csv中
        time = "%s" % datetime.datetime.now()

        list = [time, i, loss, acc, prec, rec, f1, target.tolist(),preds_a.tolist()]
        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list])
        data.to_csv("./log/"+file_path +'load.csv', mode='a', header=False, index=False)

        if i % 50 == 0:
            print('validate_Epoch: [{0}][{1}/{2}]\t'
                  'Loss@{loss.val:.4f} ({loss.avg:.4f})\t'
                  'prec@{prec.val:.3f}({prec.avg:.4f})\t'
                  'acc@{rec.val:.3f}({acc.avg:.4f})\t'
                  'rec@{rec.val:.3f}({rec.avg:.4f})\t'
                  'f1@{f1.val:.3f}({f1.avg:.4f})\t'
                .format(
                999, i, len(val_loader), loss=losses, prec=preces,acc = acces,rec = reces,f1 = f1es))

    print(' * acces@1 {acces.avg:.3f}'
          .format(acces=acces))

    return preces


def fit_and_score(path):

    print()

    model = Resnet.ResNet18().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    # define loss function (criterion) and optimizer
    global criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # reload_states = torch.load("./fold/all_states_.pth")
    # reload_states = torch.load("./fold/final_all_states.pth")

    reload_states = torch.load(path)
    # print(reload_states["optimizer"])
    model.load_state_dict(reload_states["state_dict"], strict=False)
    optimizer.load_state_dict(reload_states["optimizer"])
    return model,optimizer


if __name__ == '__main__':


    print("上传标识符：",9)
    space = {
        # 'kernel_size': hp.choice('kernel_size', [3]),
        # 'kernel_size': hp.choice('kernel_size', [3, 7, 9]),
        'batch_size': hp.choice('batch_size', [5, 6, 7]),
             # 'batch_size': hp.choice('batch_size', [5]),
        'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
        'n_classes': 26}


    Y_1, Y_2, Y_3, _, _, _ = get2013openData()

    outfile = open("2013open_resent18_load" + '.log', 'w')
    namedataset = "receipt"
    # lr, num_epochs, batch_size = 0.05, 100 , 50
    batch_size = 50
    # n_iter = 20
    n_iter = 20
    from xtUtil import modelUtil, plot_confusion_matrix

    for f in range(3):
        files_path = modelUtil().readModel('./checkpoints', '2013open_resnet_fold'+str(f), 'pth.tar')
        for num,file_path in enumerate(files_path):
            print("Fold n.", f)
            if f == 0:
                y_a_train = np.concatenate((Y_1, Y_2))
                y_a_test = Y_3
            elif f == 1:
                y_a_train = np.concatenate((Y_2, Y_3))
                y_a_test = Y_1
            elif f == 2:
                y_a_train = np.concatenate((Y_1, Y_3))
                y_a_test = Y_2

            # X_a_train = np.load("./bpi_challenge_2013_open_problems/" + "bpi_challenge_2013_open_problems_train_fold_" + str(f) + ".npy")
            # X_a_train_new = np.asarray(X_a_train)
            # X_a_train_new = X_a_train_new / 255.0
            # X_a_train_reshape = np.transpose(X_a_train_new, (0, 3, 1, 2))

            X_a_test = np.load("./bpi_challenge_2013_open_problems/" + "bpi_challenge_2013_open_problems_test_fold_" + str(f) + ".npy")
            X_a_test_new = np.asarray(X_a_test)
            X_a_test_new = X_a_test_new / 255.0
            X_a_test_reshape = np.transpose(X_a_test_new, (0, 3, 1, 2))

            # input_data, target_data = torch.FloatTensor(X_a_train_reshape), torch.FloatTensor(y_a_train)
            # # https://zhuanlan.zhihu.com/p/371516520
            # train_dataset = Data.TensorDataset(input_data, target_data,)

            test_input_data, test_target_data = torch.FloatTensor(X_a_test_reshape), torch.FloatTensor(y_a_test)
            test_dataset = Data.TensorDataset(test_input_data, test_target_data)
            test_loader = Data.DataLoader(test_dataset, batch_size, True)

            # model selection
            print('Starting model selection...')


            best_model,_ = fit_and_score(file_path)

            # evaluate
            print('Evaluating final model...')

            # 对文件名字进行处理./log/./checkpoints/2013open_resnet_attention_fold0_1_checkpoint.pth.tarload.csv
            file_name = file_path.split('/')[2]

            accuracy = validate(test_loader, best_model, criterion, file_name)
            outfile.write("\nmax accuracy----" + str(accuracy.max))
            outfile.write("\navg accuracy----" + str(accuracy.avg))
            outfile.flush()
    outfile.close()