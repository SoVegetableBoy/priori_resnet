import datetime
from time import perf_counter, time

import pm4py
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
from receiptData import getData, get2013openData, get2013Open_Train_Test_process_Loader
import warnings
warnings.filterwarnings("ignore")
# from deeplearning.myExperiment.resnet50 import ResNet50

criterion = None


def validate(val_loader, model, criterion, file_path):

    losses = AverageMeter()
    preces = AverageMeter()
    acces = AverageMeter()
    reces = AverageMeter()
    f1es = AverageMeter()

    df1 = pd.DataFrame(columns=['time', 'Epoch', 'Loss', 'acc','prec','rec','f1'])  # 列名
    df2 = pd.DataFrame(columns=['time', 'Epoch', 'acc', 'processacc', 'prec', 'rec', 'f1','target','prediction'])  # 列名
    df1.to_csv("./log/"+file_path +"load.csv", index=False,mode='w')  # 路径可以根据需要更改
    df2.to_csv("./log/" + file_path + "processload.csv", index=False, mode='w')  # 路径可以根据需要更改
    # switch to evaluate mode
    model.eval()

    for i, (input, target,traces) in enumerate(val_loader):

        with torch.no_grad():
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
        preds_a = np.argmax(output.detach().cpu().numpy(), axis=1) + 1
        target = np.argmax(target, axis=1) + 1
        # measure accuracy and record loss
        acc = accuracy_score(preds_a, target)

        prec, rec, f1, _ = metrics.precision_recall_fscore_support(
            target, preds_a, average="weighted")
        # print('\n',metrics.classification_report(
        #    target, preds_a))
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
        data.to_csv("./log/"+file_path + 'load.csv', mode='a', header=False, index=False)

        Top1, processAcc, best_pred = TopAccuracy(output, target, traces, topk=(1, 3))
        prec2, rec2, f12, _ = metrics.precision_recall_fscore_support(
            target, best_pred, average="weighted")
        # 这里修改一下，将loss改成未加入先验知识的精确度
        processlist = [time, i, Top1.item(), processAcc.item(), prec2, rec2, f12,target.tolist(),best_pred.tolist()]
        data = pd.DataFrame([processlist])
        data.to_csv("./log/" + file_path + 'processload.csv', mode='a', header=False, index=False)

        print("ACC:", acc,"------Top1:", Top1, "----processAcc:", processAcc)

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



def TopAccuracy(output, target,traces, topk=(1,)):
    net, initial_marking, final_marking = pm4py.read_pnml('./fold/2013open_petri_heuristics_miner_num.pnml')
    # 创建一个字典，将每一个fitness放入
    # 创建一个数组，放入一个批次的字典
    # from pm4py.visualization.petri_net import visualizer as pn_visualizer
    # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    # pn_visualizer.view(gviz)
    batch_fitness = []
    batch_best = []

    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() + 1
    # 这里和原来的代码有点不符合，注释修改
    # y_a_test = np.argmax(target.cpu().numpy(), axis=1)
    # y_a_test_view = torch.Tensor(y_a_test).cuda().view(1, -1)
    pred_source = pred.cpu()

    y_a_test_view_expand = target.expand_as(pred)
    correct = pred_source.eq(y_a_test_view_expand)
    res = []
    for k in topk:
        correct_contiguous = correct[:k].contiguous()
        correct_view = correct_contiguous.view(-1)
        correct_k = correct_view.float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    # 将pred中取出五个进行筛选，计算fitness
    # 前缀迹如何得到呢，原来的代码是将数字映射到2**24 -1 上，然后24个二进制数，不够的补零，再拆分成RGB
    for i,trace in enumerate(traces):
        top_five = [x[i] for x in pred_source]
        fitness_dic = {}
        for j,p in enumerate(top_five):
            trace = trace[(trace>0)]
            list_sum = np.hstack((trace,p))

            # 做一个rework判断，如果有循环就不进行操作，因为模型对循环支持不太好
            # s = Solution()
            # repeat = s.findRepeatNumber(list_sum.tolist())
            # if repeat is None:
            CaseID = np.ones(len(list_sum), dtype=np.int32)
            ActivityID = np.array(list_sum).astype('int32')
            Timestamp = np.ones(len(list_sum), dtype=np.int32)
            df = pd.DataFrame({'CaseID': CaseID, 'ActivityID': ActivityID, 'Timestamp': Timestamp})
            event_log = pm4py.format_dataframe(df, case_id='CaseID', activity_key='ActivityID',
                                               timestamp_key='Timestamp')
            token_based_replay = pm4py.conformance_diagnostics_token_based_replay(event_log, net, initial_marking,
                                                                                  final_marking)
            fitness = token_based_replay[0].get("trace_fitness")

            # 将每一个值放入字典中
            # fitness_dic.update({ p : fitness})
            # 修改一下，如果百分之一百确认，可以放入集合中，如果不是就是0
            # if fitness >= 0.80 and j == 0:
            if res[0] < 60:
                print(list_sum)
            if len(list_sum) > 1:
                if j == 0:
                    # if fitness >= 0.9:
                    fitness_dic.update({p: fitness})
                    # elif fitness < 0.9:
                    #     print("laji < 0.9 :", list_sum,"---fitness:", fitness,"---source:",pred_source[i])
                    #     fitness_dic.update({p: fitness})
                else:
                    fitness_dic.update({p: fitness})
            else:
                fitness_dic.update({p: fitness})
            # else:
            #     fitness_dic.update({p: 0})

            # else:
            #     fitness_dic.update({p: 1})
            #     if j != 0:
            #         fitness_dic.update({p: 0})

        fitness_dic_order = sorted(fitness_dic.items(), key=lambda x: x[1], reverse=True)
        batch_fitness.append(fitness_dic_order)
        # batch_best.append(fitness_dic_order)

    best_pred = [i[0][0] for i in batch_fitness]
    #这还必须是Tensor因为list没有eq这个方法
    best_pred = torch.tensor(best_pred)
    # 注释掉下面计算准确度的代码，直接放到validate中嘿嘿，
    # 算了直接修改，不动以前的代码吧

    correct = best_pred.eq(y_a_test_view_expand)
    correct_contiguous = correct[:1].contiguous()
    correct_view = correct_contiguous.view(-1)
    correct_k = correct_view.float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))

    return res[0], res[2], best_pred


def fit_and_score(path):

    print()

    model = Resnet.ResNet18().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00017)
    # define loss function (criterion) and optimizer
    global criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # reload_states = torch.load("./fold/all_states_.pth")
    # reload_states = torch.load("./fold/final_all_states.pth")


    reload_states = torch.load(path)

    model.load_state_dict(reload_states["state_dict"], strict=False)
    optimizer.load_state_dict(reload_states["optimizer"])
    return model,optimizer


if __name__ == '__main__':


    print("上传标识符：",9)

    # Y_1, Y_2, Y_3,Y_source_1,Y_source_2,Y_source_3 = get2013openData()

    outfile = open("2013open_resent18_att_load" + '.log', 'w')
    namedataset = "2013open"
    # lr, num_epochs, batch_size = 0.05, 100 , 50
    batch_size = 50
    # n_iter = 20
    n_iter = 20
    from xtUtil import modelUtil
    for f in range(3):
        files_path = modelUtil().readModel('./checkpoints', '2013open_resnet_att_fold'+str(f), 'pth.tar')
        # files_path = ['./checkpoints/2013open_resnet_att_fold0_val_16_model_best.pth.tar']
        for num,file_path in enumerate(files_path):
            print("Fold n.", f)
            _, test_loader = get2013Open_Train_Test_process_Loader(batch_size=batch_size,f = f)

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


