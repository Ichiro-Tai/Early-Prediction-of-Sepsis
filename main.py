from preprocessing.Preprocessor import preprocess_data
from models.lstm import LSTM
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
    # TODO

    print('---Setting data loaders---')
    # TODO

    print('---Initializing models---')
    net = LSTM(args)
