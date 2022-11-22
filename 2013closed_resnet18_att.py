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
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from tqdm import tqdm
from receiptData import get2013closedData

# from deeplearning.myExperiment.resnet50 import ResNet50
modelnum = 0
criterion = None

def train(train_loader, model, criterion, optimizer, epochs):

    losses = AverageMeter()
    preces = AverageMeter()

    # switch to train mode
    model.train()
    for epoch in range(epochs):
        for i, (input, target) in enumerate(train_loader):

            input_var = input.cuda()
            target_var =target.cuda()

            optimizer.zero_grad()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            preds_a = np.argmax(output.detach().cpu().numpy(), axis=1)
            target = np.argmax(target, axis=1)
            prec = accuracy_score(preds_a, target)
            losses.update(loss.item(), input.size(0))
            preces.update(prec, input.size(0))
            # compute gradient and do SGD step

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@{Prec.val:.3f}({Prec.avg:.4f})\t'
                     .format(
                    epoch, i, len(train_loader), loss=losses, Prec=preces))
    return losses


def validate(val_loader, model, criterion, f):

    losses = AverageMeter()
    preces = AverageMeter()
    acces = AverageMeter()
    reces = AverageMeter()
    f1es = AverageMeter()
    if f == 0 :
        df = pd.DataFrame(columns=['time', 'Epoch', 'Loss', 'acc','prec','rec','f1'])  # 列名
        df.to_csv("./log/2013closed_resnet18_att"+".csv", index=False,mode='w')  # 路径可以根据需要更改

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
        # prec = precision_score(preds_a, target)
        # rec = recall_score(preds_a, target)
        # f1 = f1_score(preds_a, target)
        losses.update(loss.item(), input.size(0))
        preces.update(prec, input.size(0))
        acces.update(acc, input.size(0))
        reces.update(rec, input.size(0))
        f1es.update(f1, input.size(0))

        # 将训练结果放入csv中
        time = "%s" % datetime.datetime.now()

        list = [time, i, loss, acc,prec,rec,f1]
        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list])
        data.to_csv('./log/2013closed_resnet18_att.csv', mode='a', header=False, index=False)

        if i % 50 == 0:
            print('validate_Epoch: [{0}][{1}/{2}]\t'
                  'Loss@{loss.val:.4f} ({loss.avg:.4f})\t'
                  'prec@{prec.val:.3f}({prec.avg:.4f})\t'
                  'acc@{rec.val:.3f}({acc.avg:.4f})\t'
                  'rec@{rec.val:.3f}({rec.avg:.4f})\t'
                  'f1@{f1.val:.3f}({f1.avg:.4f})\t'
                .format(

                f, i, len(val_loader), loss=losses, prec=preces,acc = acces,rec = reces,f1 = f1es))

    print(' * Prec@1 {Prec.avg:.3f}'
          .format(Prec=preces))

    return acces

def fit_and_score(params):

    print(params)
    start_time = perf_counter()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr,
    #                                 momentum=0.9,
    #                                 weight_decay=5e-4)
    model = Resnet.ResNet18().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    # define loss function (criterion) and optimizer
    global criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[60, 120, 160], gamma=0.2)

    train_loader = Data.DataLoader(train_dataset, 2**params['batch_size'], True)

    scores = train(train_loader, model, criterion, optimizer, epochs=200)
    score =scores.min

    print('train loss:',score)


    end_time = perf_counter()
    global best_score, best_model, best_time, best_numparameters, best_optimizer

    if best_score > score:
        best_score = score
        best_model = model
        best_optimizer = optimizer
        best_numparameters = sum(p.numel() for p in best_model.parameters())
        best_time = end_time - start_time
        save_checkpoint({
            'state_dict': best_model.state_dict(),
            'best_loss': score,
            'best_prec1': score,
            'optimizer': best_optimizer.state_dict(),
        }, True, "2013closed_resnet_attention_fold"+str(params['f']))

    global modelnum
    modelnum = modelnum + 1
    save_checkpoint({
        'state_dict': best_model.state_dict(),
        'best_loss': score,
        'best_prec1': score,
        'optimizer': best_optimizer.state_dict(),
    }, False, "2013closed_resnet_attention_fold"+str(params['f'])+'_'+str(modelnum))
    return {'loss': score, 'status': STATUS_OK, 'n_epochs': scores.count, 'n_params': best_numparameters,
            'time': end_time - start_time}

if __name__ == '__main__':

    print("上传标识符：",9)



    # Y_1,Y_2,Y_3,_,_,_ = getData()
    Y_1, Y_2, Y_3, _, _, _ = get2013closedData()

    outfile = open("2013closed_resent18_att" + '.log', 'w')
    namedataset = "2013closed"
    # lr, num_epochs, batch_size = 0.05, 100 , 50
    batch_size = 50
    # n_iter = 20
    n_iter = 1
    for f in range(3):

        space = {
        # 'kernel_size': hp.choice('kernel_size', [3]),
        # 'kernel_size': hp.choice('kernel_size', [3, 7, 9]),
        'batch_size': hp.choice('batch_size', [5]),
        'learning_rate': 0.0005,
        'f':f}

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

        X_a_train = np.load("./bpi_challenge_2013_closed_problems/" + "bpi_challenge_2013_closed_problems_train_fold_" + str(f) + ".npy")
        X_a_train_new = np.asarray(X_a_train)
        X_a_train_new = X_a_train_new / 255.0
        X_a_train_reshape = np.transpose(X_a_train_new, (0, 3, 1, 2))

        X_a_test = np.load("./bpi_challenge_2013_closed_problems/" + "bpi_challenge_2013_closed_problems_test_fold_" + str(f) + ".npy")
        X_a_test_new = np.asarray(X_a_test)
        X_a_test_new = X_a_test_new / 255.0
        X_a_test_reshape = np.transpose(X_a_test_new, (0, 3, 1, 2))

        input_data, target_data = torch.FloatTensor(X_a_train_reshape), torch.FloatTensor(y_a_train)
        # https://zhuanlan.zhihu.com/p/371516520
        train_dataset = Data.TensorDataset(input_data, target_data,)

        test_input_data, test_target_data = torch.FloatTensor(X_a_test_reshape), torch.FloatTensor(y_a_test)
        test_dataset = Data.TensorDataset(test_input_data, test_target_data)
        test_loader = Data.DataLoader(test_dataset, batch_size, True)

        # model selection
        print('Starting model selection...')

        best_score = np.inf
        best_model = None
        best_time = 0
        best_numparameters = 0

        trials = Trials()
        best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials,
                    rstate=np.random.default_rng(seed + f))
        best_params = hyperopt.space_eval(space, best)

        outfile.write("\nResnet18 attention Hyperopt trials")
        outfile.write("\ntid,loss,learning_rate,batch_size,time,n_epochs,n_params,perf_time")
        for trial in trials.trials:
            outfile.write("\n%d,%f,%f,%s,%d,%d,%d,%f" % (trial['tid'],
                                                                trial['result']['loss'],
                                                                # trial['misc']['vals']['learning_rate'],
                                                                0.0005,
                                                                trial['misc']['vals']['batch_size'][0] + 7,
                                                                (trial['refresh_time'] - trial['book_time']).total_seconds(),
                                                                trial['result']['n_epochs'],
                                                                trial['result']['n_params'],
                                                                trial['result']['time'],
                                                                ))
        outfile.write("\n\nBest parameters:")
        print(best_params, file=outfile)
        outfile.write("\nModel parameters: %d" % best_numparameters)
        outfile.write('\nBest Time taken: %f' % best_time)

        # evaluate
        print('Evaluating final model...')

        accuracy = validate(test_loader, best_model, criterion , f)

        outfile.write("max accuracy----\n" + str(accuracy.max))
        outfile.write("avg accuracy----\n" + str(accuracy.avg))

        outfile.flush()

    outfile.close()


