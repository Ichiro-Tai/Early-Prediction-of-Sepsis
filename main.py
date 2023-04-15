import os
import pprint
from torch.optim import Adam
from torch.utils.data import DataLoader

from preprocessing.Preprocessor import preprocess_data
from models.lstm import LSTM
from models.loss import Loss
from models.train_eval import train_eval
from dataloader.dataloader import DataSet
from tools import parse, python_utils

path_vital = "data/vital.csv"
path_similar = "data/similar.json"
path_grp_index = "data/group_index_dict.json"
generated_data_save_path = "generated_data"
path_master = "data/master.csv"
path_label = "data/label.csv"

args = parse.args
args.hard_mining = 0
args.gpu = 1
args.use_trend = max(args.use_trend, args.use_value)
args.use_value = max(args.use_trend, args.use_value)
args.rnn_size = args.embed_size
args.hidden_size = args.embed_size
args.split_nn = args.split_num + args.split_nor * 3
args.vocab_size = args.split_nn * 145 + 1

# All the parameters
p_dict = dict()
p_dict['args'] = args


if __name__ == "__main__":
    print('---Pre-processing raw data---')
    preprocess_data(path_vital, path_similar, path_grp_index, generated_data_save_path, path_master, path_label)    

    print('---Reading data---')
    gen_data_path = 'generated_data'
    patient_time_record_dict = python_utils.myreadjson(os.path.join(gen_data_path, 'patient_time_record_dict.json'))
    patient_master_dict = python_utils.myreadjson(os.path.join(gen_data_path, 'patient_master_dict.json'))
    patient_label_dict = python_utils.myreadjson(os.path.join(gen_data_path, 'patient_label_dict.json'))

    patients = list(patient_time_record_dict.keys())
    patients = list(patient_label_dict.keys())
    n = int(0.8 * len(patients))
    patient_train = patients[:n]
    patient_valid = patients[n:]

    print('---Setting data loaders---')
    # TODO
    train_dataset = DataSet(
                patient_train, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='train')
    train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
    val_dataset = DataSet(
                patient_valid, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='val')
    val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)

    p_dict['train_loader'] = train_loader
    p_dict['val_loader'] = val_loader

    print('---Initializing models---')
    net = LSTM(args)

    print('---Initializing loss---')
    my_train_metric = []
    p_dict['my_train_metric'] = my_train_metric

    if args.gpu:
        net = net.cuda()
        p_dict['loss'] = Loss().cuda()
    else:
        p_dict['loss'] = Loss()

    parameters = []
    for p in net.parameters():
        parameters.append(p)
    optimizer = Adam(parameters, lr=args.lr)
    p_dict['optimizer'] = optimizer
    p_dict['model'] = net
    start_epoch = 0

    p_dict['epoch'] = 0
    p_dict['best_metric'] = [0, 0]

    if args.phase == 'train':
        best_f1score = 0
        for epoch in range(p_dict['epoch'] + 1, args.epochs):
            p_dict['epoch'] = epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            train_eval(p_dict, 'train')
            train_eval(p_dict, 'val')

    with open('./generated_models/my_train_metric.txt', 'w') as ff:
        pprint.pprint(my_train_metric, ff)

    with open('./generated_models/my_train_acc.txt', 'w') as ff:
        my_train_acc = []
        for metric_dict in my_train_metric:
            tp = metric_dict['tp']
            tn = metric_dict['tn']
            fp = metric_dict['fp']
            fn = metric_dict['fn']
            accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
            my_train_acc.append(accuracy)
        pprint.pprint(my_train_acc)
        pprint.pprint(my_train_acc, ff)
