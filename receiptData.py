import os
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

run_dir = os.path.dirname(os.path.abspath(__file__))

def getOpenTrace():
    # namedataset = "Bpic2013_closed_bertFeature
    # df = pd.read_csv('./deeplearning/fold/' + namedataset + 'myfeature.csv', header=None)
    # fold1, fold2, fold3 = get_size_fold(namedataset)
    fold1, fold2, fold3 = 595, 559, 378
    df = pd.read_csv('./fold/BPIC2013open_bertFeature.csv', header=None)
    sorce = df[0].values.tolist()
    source_int = []
    source_valid_len = []
    for i in sorce:
        j = i.split(" ")
        j = list(map(int,j))
        source_int.append(j)

    # 实现二
    def apply_to_zeros(lst, dtype=np.int64):
        inner_max_len = max(map(len, lst))
        result = np.zeros([len(lst), inner_max_len], dtype)
        for i, row in enumerate(lst):
            for j, val in enumerate(row):
                result[i][j] = val
        return result

    source_int = apply_to_zeros(source_int)
    # df_fold = pd.read_csv('./deeplearning/fold/' + namedataset + '.txt', header=None)  # ,encoding='windows-1252')
    # df_fold.columns = ["CaseID", "Activity", "Resource", "Timestamp"]
    # cont_trace = df_fold['CaseID'].value_counts(dropna=False)
    # max_trace = max(cont_trace)
    # max_trace = 25
    # trace = df.iloc[:, :max_trace]
    # trace_1 = trace[:fold1]
    # trace_2 = trace[fold1:(fold1 + fold2)]
    # trace_3 = trace[(fold1 + fold2):]
    trace_1 = source_int[:fold1]
    trace_2 = source_int[fold1:(fold1 + fold2)]
    trace_3 = source_int[(fold1 + fold2):]

    # trace_5 = torch.tensor(trace_1)
    # trace_4 = trace_5[(trace_5>0)]
    # trace_6 = np.hstack((trace_4,[100]))
    return trace_1,trace_2,trace_3

# # 找回本来的映射
# def token_based_log(log):
#     net, initial_marking, final_marking = pm4py.read_pnml('petri.pnml')
#     gviz = pn_visualizer.apply(net, initial_marking, final_marking)
#     pn_visualizer.view(gviz)
#     fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking,
#                                              variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
#     return fitness

def get_size_fold(namedataset):
    fold1 = pd.read_csv(run_dir + '/fold/' + namedataset + '_premiereFold0' + '.txt', header=None)
    fold2 = pd.read_csv(run_dir + '/fold/' + namedataset + '_premiereFold1' + '.txt', header=None)
    fold3 = pd.read_csv(run_dir + '/fold/' + namedataset + '_premiereFold2' + '.txt', header=None)

    fold1.columns = ["CaseID", "Activity", "Resource", "Timestamp"]
    fold2.columns = ["CaseID", "Activity", "Resource", "Timestamp"]
    fold3.columns = ["CaseID", "Activity", "Resource", "Timestamp"]

    n_caseid1 = fold1['CaseID'].nunique()
    n_caseid2 = fold2['CaseID'].nunique()
    n_caseid3 = fold3['CaseID'].nunique()

    len_fold1 = len(fold1)
    len_fold2 = len(fold2)
    len_fold3 = len(fold3)

    nos1 = len_fold1 - n_caseid1
    nos2 = len_fold2 - n_caseid2
    nos3 = len_fold3 - n_caseid3

    return nos1, nos2, nos3

def get_image_size(num_col):
    import math
    matx = round(math.sqrt(num_col))
    if num_col>(matx*matx):
        matx = matx + 1
        padding = (matx*matx) - num_col
    else:
        padding = (matx*matx) - num_col
    return matx, padding
def get2013open_Data_change_Onehot():
    df = pd.read_csv(run_dir + '/fold/bpi_challenge_2013_open_problemsfeature.csv', header=None)
    num_col = df.iloc[:, :-1] # remove target column
    num_col = len(df. columns)
    # fold1, fold2, fold3 = get_size_fold("receipt")
    fold1, fold2, fold3 = 595, 559, 378
    target = df[df.columns[-1]]
    df_labels = np.unique(list(target))
    maxlable = int(max(df_labels))

    # img_size, pad = get_image_size(num_col)
    #
    # label_encoder = preprocessing.LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(df_labels)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #
    # onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    # onehot_encoder.fit(integer_encoded)
    # onehot_encoded = onehot_encoder.transform(integer_encoded)
    #
    # train_integer_encoded = label_encoder.transform(target).reshape(-1, 1)
    # train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)

    # 修改编码，不使用系统自带的onehot，将上面代码注释了
    results = np.zeros(shape=(len(target), maxlable))
    for i, sample in enumerate(target):
        index = int(sample)
        results[i, index-1] = 1  # i : 该词所在的句子  j ：该词位于句子中的索引  index：该词在词典中的索引+1

    # 暂时性的让代码注释，回归原来的 编码方式
    # y_one_hot = np.asarray(results)
    y_one_hot = np.asarray(results)

    # n_classes = len(df_labels)
    # 经过检查，argmax是从0开始的，那么在本程序中就是整体偏移2个单位，加上2，就是原来的标签
    # y_a_test = np.argmax(y_one_hot, axis=1)

    Y_1 = y_one_hot[:fold1]
    Y_2 = y_one_hot[fold1:(fold1+fold2)]
    Y_3 = y_one_hot[(fold1+fold2):]



    return Y_1,Y_2,Y_3
def getData():
    df = pd.read_csv(run_dir + '/fold/receiptmyfeature.csv', header=None)
    num_col = df.iloc[:, :-1] # remove target column
    num_col = len(df. columns)
    fold1, fold2, fold3 = get_size_fold("receipt")
    target = df[df.columns[-1]]
    target_list = list(map(int ,target.tolist()))
    df_labels = np.unique(list(target))

    img_size, pad = get_image_size(num_col)

    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df_labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    train_integer_encoded = label_encoder.transform(target).reshape(-1, 1)
    train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
    y_one_hot = np.asarray(train_onehot_encoded)
    n_classes = len(df_labels)

    Y_1 = y_one_hot[:fold1]
    Y_2 = y_one_hot[fold1:(fold1+fold2)]
    Y_3 = y_one_hot[(fold1+fold2):]

    Y_source_1 = target_list[:fold1]
    Y_source_2 = target_list[fold1:(fold1 + fold2)]
    Y_source_3 = target_list[(fold1 + fold2):]

    return Y_1,Y_2,Y_3,Y_source_1,Y_source_2,Y_source_3
def get2013openData():
    df = pd.read_csv(run_dir + '/fold/bpi_challenge_2013_open_problemsfeature.csv', header=None)
    num_col = df.iloc[:, :-1] # remove target column
    num_col = len(df. columns)
    # fold1, fold2, fold3 = get_size_fold("receipt")
    # fold1, fold2, fold3 =24000, 24000, 24414
    fold1, fold2, fold3 = 595, 559, 378

    target = df[df.columns[-1]]
    target_list = list(map(int ,target.tolist()))
    df_labels = np.unique(list(target))

    img_size, pad = get_image_size(num_col)

    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df_labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    train_integer_encoded = label_encoder.transform(target).reshape(-1, 1)
    train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
    y_one_hot = np.asarray(train_onehot_encoded)
    n_classes = len(df_labels)

    Y_1 = y_one_hot[:fold1]
    Y_2 = y_one_hot[fold1:(fold1+fold2)]
    Y_3 = y_one_hot[(fold1+fold2):]

    Y_source_1 = target_list[:fold1]
    Y_source_2 = target_list[fold1:(fold1 + fold2)]
    Y_source_3 = target_list[(fold1 + fold2):]

    return Y_1,Y_2,Y_3,Y_source_1,Y_source_2,Y_source_3
def get2013closedData():
    df = pd.read_csv(run_dir + '/fold/bpi_challenge_2013_closed_problemsfeature.csv', header=None)
    num_col = df.iloc[:, :-1] # remove target column
    num_col = len(df. columns)
    # fold1, fold2, fold3 = get_size_fold("receipt")
    # fold1, fold2, fold3 =24000, 24000, 24414
    # fold1, fold2, fold3 = 595, 559, 378
    fold1, fold2, fold3 = 1800,1800,1573

    target = df[df.columns[-1]]
    target_list = list(map(int ,target.tolist()))
    df_labels = np.unique(list(target))

    img_size, pad = get_image_size(num_col)

    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df_labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    train_integer_encoded = label_encoder.transform(target).reshape(-1, 1)
    train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
    y_one_hot = np.asarray(train_onehot_encoded)
    n_classes = len(df_labels)

    Y_1 = y_one_hot[:fold1]
    Y_2 = y_one_hot[fold1:(fold1+fold2)]
    Y_3 = y_one_hot[(fold1+fold2):]

    Y_source_1 = target_list[:fold1]
    Y_source_2 = target_list[fold1:(fold1 + fold2)]
    Y_source_3 = target_list[(fold1 + fold2):]

    return Y_1,Y_2,Y_3,Y_source_1,Y_source_2,Y_source_3
def getProcess():

    df = pd.read_csv(run_dir + '/fold/receiptmyfeature.csv', header=None)
    num_col = df.iloc[:, :-1] # remove target column
    num_col = len(df. columns)
    fold1, fold2, fold3 = get_size_fold("receipt")
    target = df[df.columns[-1]]
    df_labels = np.unique(list(target))

    img_size, pad = get_image_size(num_col)

    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df_labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    train_integer_encoded = label_encoder.transform(target).reshape(-1, 1)
    train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
    y_one_hot = np.asarray(train_onehot_encoded)
    n_classes = len(df_labels)

    Y_1 = y_one_hot[:fold1]
    Y_2 = y_one_hot[fold1:(fold1+fold2)]
    Y_3 = y_one_hot[(fold1+fold2):]
    return Y_1,Y_2,Y_3
def getTrain_Test_Loader(batch_size=5,f = 0):
    Y_1,Y_2,Y_3 = getData()
    if f == 0:
        y_a_train = np.concatenate((Y_1, Y_2))
        y_a_test = Y_3
    elif f == 1:
        y_a_train = np.concatenate((Y_2, Y_3))
        y_a_test = Y_1
    elif f == 2:
        y_a_train = np.concatenate((Y_1, Y_3))
        y_a_test = Y_2

    X_a_train = np.load("./image/receipt/" + "receipt_train_fold_addActivity_" + str(f) + ".npy")
    X_a_train_new = np.asarray(X_a_train)
    X_a_train_reshape = np.transpose(X_a_train_new, (0, 3, 1, 2))
    X_a_test = np.load("./image/receipt/" + "receipt_test_fold_addActivity_" + str(f) + ".npy")
    X_a_test_new = np.asarray(X_a_test)
    X_a_test_reshape = np.transpose(X_a_test_new, (0, 3, 1, 2))
    # y_a_train = np.concatenate((Y_1, Y_2))
    # y_a_test = Y_3

    input_data, target_data = torch.FloatTensor(X_a_train_reshape), torch.FloatTensor(y_a_train)
    # https://zhuanlan.zhihu.com/p/371516520
    dataset = Data.TensorDataset(input_data, target_data)
    train_loader = Data.DataLoader(dataset, batch_size, True)

    test_input_data, test_target_data = torch.FloatTensor(X_a_test_reshape), torch.FloatTensor(y_a_test)
    test_dataset = Data.TensorDataset(test_input_data, test_target_data)
    test_loader = Data.DataLoader(test_dataset, batch_size, True)
    return train_loader,test_loader
def get2013Open_Train_Test_process_Loader(batch_size=5,f = 0):
    Y_1,Y_2,Y_3 = get2013open_Data_change_Onehot()
    trace_1,trace_2,trace_3 = getOpenTrace()
    if f == 0:
        y_a_train = np.concatenate((Y_1, Y_2))
        y_a_test = Y_3
        trace_test = trace_3
        # Y_test_source = Y_source_3
    elif f == 1:
        y_a_train = np.concatenate((Y_2, Y_3))
        y_a_test = Y_1
        trace_test = trace_1
        # Y_test_source = Y_source_1
    elif f == 2:
        y_a_train = np.concatenate((Y_1, Y_3))
        y_a_test = Y_2
        trace_test = trace_2
        # Y_test_source = Y_source_2

    X_a_train = np.load("./bpi_challenge_2013_open_problems/"
                        + "bpi_challenge_2013_open_problems_train_fold_" + str(f) + ".npy")
    X_a_train_new = np.asarray(X_a_train)
    X_a_train_reshape = np.transpose(X_a_train_new, (0, 3, 1, 2))
    X_a_test = np.load("./bpi_challenge_2013_open_problems/" +
                       "bpi_challenge_2013_open_problems_test_fold_" + str(f) + ".npy")
    X_a_test_new = np.asarray(X_a_test)
    X_a_test_new = X_a_test_new / 255.0
    X_a_test_reshape = np.transpose(X_a_test_new, (0, 3, 1, 2))
    # y_a_train = np.concatenate((Y_1, Y_2))
    # y_a_test = Y_3

    input_data, target_data = torch.FloatTensor(X_a_train_reshape), torch.FloatTensor(y_a_train)
    # https://zhuanlan.zhihu.com/p/371516520
    dataset = Data.TensorDataset(input_data, target_data)
    train_loader = Data.DataLoader(dataset, batch_size, True)

    trace_test = np.array(trace_test)
    # Y_test_source = np.array(Y_test_source)

    test_input_data, test_target_data ,test_trace_data = torch.FloatTensor(X_a_test_reshape), torch.FloatTensor(y_a_test),torch.Tensor(trace_test)
    test_dataset = Data.TensorDataset(test_input_data, test_target_data,test_trace_data)
    test_loader = Data.DataLoader(test_dataset, batch_size, True)
    return train_loader,test_loader
def get2013open_Train_Test_process_Loader(batch_size=5,f = 0):
    Y_1,Y_2,Y_3 = getData_change_Onehot()
    trace_1,trace_2,trace_3 = getTrace()
    if f == 0:
        y_a_train = np.concatenate((Y_1, Y_2))
        y_a_test = Y_3
        trace_test = trace_3
        # Y_test_source = Y_source_3
    elif f == 1:
        y_a_train = np.concatenate((Y_2, Y_3))
        y_a_test = Y_1
        trace_test = trace_1
        # Y_test_source = Y_source_1
    elif f == 2:
        y_a_train = np.concatenate((Y_1, Y_3))
        y_a_test = Y_2
        trace_test = trace_2
        # Y_test_source = Y_source_2

    X_a_train = np.load("./image/receipt/" + "receipt_train_fold_addActivitynosacalr_" + str(f) + ".npy")
    X_a_train_new = np.asarray(X_a_train)
    X_a_train_reshape = np.transpose(X_a_train_new, (0, 3, 1, 2))
    X_a_test = np.load("./image/receipt/" + "receipt_test_fold_addActivitynosacalr_" + str(f) + ".npy")
    X_a_test_new = np.asarray(X_a_test)
    X_a_test_reshape = np.transpose(X_a_test_new, (0, 3, 1, 2))
    # y_a_train = np.concatenate((Y_1, Y_2))
    # y_a_test = Y_3

    input_data, target_data = torch.FloatTensor(X_a_train_reshape), torch.FloatTensor(y_a_train)
    # https://zhuanlan.zhihu.com/p/371516520
    dataset = Data.TensorDataset(input_data, target_data)
    train_loader = Data.DataLoader(dataset, batch_size, True)

    trace_test = np.array(trace_test)
    # Y_test_source = np.array(Y_test_source)

    test_input_data, test_target_data ,test_trace_data = torch.FloatTensor(X_a_test_reshape), torch.FloatTensor(y_a_test),torch.Tensor(trace_test)
    test_dataset = Data.TensorDataset(test_input_data, test_target_data,test_trace_data)
    test_loader = Data.DataLoader(test_dataset, batch_size, True)
    return train_loader,test_loader
if __name__ == '__main__':
    getTrain_Test_process_Loader(batch_size=5, f=0)
