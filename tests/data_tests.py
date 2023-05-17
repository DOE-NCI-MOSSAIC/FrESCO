
import argparse
import json
import os
import sys

import torch
import numpy as np
import pandas as pd

from collections import Counter
sys.path.append("../")
from validate import validate_params
from data_loaders import data_utils
from training import training



def subset_check_X(dw, data, split):


    size = len(dw.inference_data["X"][split]) / data.shape[0]
    print(f"Original shape: {data.shape[0]} " +
          f"New shape: {len(dw.inference_data['X'][split])} " +
          f"Subset proportion: {size}")

    matches = []
    for i, idx in enumerate(dw.inference_data['X'][split].index):
        iidx = np.nonzero(data.index == idx)[0]
        x1 = data.iat[iidx[0]]
        x2 = dw.inference_data['X'][split].iat[i]
        ret = np.allclose(x1, x2)
        matches.append(ret)

    return [1 if m else 0 for m in matches]


def verify_subset(params, dw):
    """Test subetting original dataframe does not lose X infomation.

    """
    tasks = dw.model_args['data_kwargs']['tasks']
    splits = ['train', 'test', 'val']
    fold = 0
    data_path = dw.model_args['data_kwargs']['data_path']
    label2idx = []

    with open(os.path.join(data_path, 'id2labels_fold' + str(fold) + '.json'),
              'r', encoding='utf-8') as f:
            tmp = json.load(f)

    id2label = {task: {int(k): str(v) for k, v in labels.items()}
                      for task, labels in tmp.items()}

    for task in dw.dict_maps['id2label'].keys():
        label2idx.append({v: k for k, v in id2label[task].items()})

    df = pd.read_csv(os.path.join(data_path, 'data_fold' + str(fold) + '.csv'),
                     dtype=str, engine='c')

    for split in splits:
        data = df[df['split'] == split]['X'].apply(lambda x: np.array(json.loads(x), dtype=np.int32))
        val = subset_check_X(dw, data, split)
        if sum(val) != len(dw.inference_data["X"][split]):
            print("Error, X arrays do not match")
        else:
            print(f"Split: {split} X-arrays are correct")



def check_dataloader(dw):
    """Verify X, y, and idxs in dataloader match those on disk.

        Args:
            dw - DataHandler class


        Note: loads fold 0 by default

    """
    fold = 0
    data_path = dw.model_args['data_kwargs']['data_path']

    df = pd.read_csv(os.path.join(data_path, 'data_fold' + str(fold) + '.csv'),
                     dtype=str, engine='c')
    df["X"] = df["X"].apply(lambda x: np.asarray(json.loads(x),dtype=np.int32))

    splits =  {'train': 0, 'test': 1, 'val': 2}



    vocab_size = dw.inference_data['word_embedding'].shape[0]
    unk_tok = vocab_size - 1

    for split in splits.keys():
        check_pathreports(dw, df, split)

def check_pathreports(dw, df, split):

    df = df[df['split'] == split]
    res = {"X": [], "y": []}

    tasks = list(dw.dict_maps['id2label'].keys())
    label2id = []
    for task in dw.dict_maps['id2label'].keys():
        label2id.append({v: k for k, v in dw.dict_maps['id2label'][task].items()})

    data = data_utils.PathReports(dw.inference_data['X'][split],
                                  dw.inference_data['y'][split],
                                  tasks=dw.model_args['data_kwargs']['tasks'],
                                  label_encoders=dw.dict_maps['id2label'],
                                  max_len=dw.model_args['train_kwargs']['doc_max_len'])

    for i in range(len(data)):
        X = data[i]["X"]
        X = X[X != 0]  # 0 is <pad>
        # have to remove 0 from both
        # X = X[X != vocab_size-1]

        idx = data[i]['index']
        iidx = np.nonzero(df.index == idx)[0]

        X1 = df['X'].iat[iidx[0]]
        X1 = X1[X1 != 0]
        res["X"].append(np.allclose(X, X1))

        loader_y = [data[i][f'y_{task}'] for task in tasks]
        disk_y = [label2id[t][df[task].iat[iidx[0]]] for t, task in enumerate(tasks)]
        res['y'].append(np.allclose(loader_y, disk_y))

    for k, v in res.items():
        correct = np.all(v)
        if not correct:
            print(f"Error, split: {split} PathReports class {k} values are not correct")
        else:
            print(f"split: {split} PathReports class {k} are correct")


def check_loaded_data(params, dw):
    """Check loaded and processed y-values match those on disk.

        Sanity check to ensure the output, the Ys, match those generated from the
        data_generation pipeline.

    """
    tasks = dw.model_args['data_kwargs']['tasks']
    splits = ['train', 'test', 'val']
    fold = 0
    data_path = dw.model_args['data_kwargs']['data_path']

    label2idx = []

    with open(os.path.join(data_path, 'id2labels_fold' + str(fold) + '.json'),
              'r', encoding='utf-8') as f:
            tmp = json.load(f)

    id2label = {task: {int(k): str(v) for k, v in labels.items()}
                      for task, labels in tmp.items()}

    for task in dw.dict_maps['id2label'].keys():
        label2idx.append({v: k for k, v in id2label[task].items()})

    df = pd.read_csv(os.path.join(data_path, 'data_fold' + str(fold) + '.csv'),
                     dtype=str, engine='c')

    res = []
    for t, task in enumerate(tasks):
        for split in splits:
            processed = dw.inference_data["y"][split][task]
            loaded = df[df['split'] == split][task].map(label2idx[t])
            for i, idx in enumerate(processed.index):
                idx = np.nonzero(loaded.index == idx)[0]
                res.append(np.allclose(processed.iat[i], loaded.iloc[idx].values))
            correct = np.all(res)
            res = []
            print(f"Task: {task} split: {split} correct: {correct}")


def check_saved_data(params, dw):
    fold = 0
    data_path = dw.model_args['data_kwargs']['data_path']
    save_path = dw.model_args['save_name']

    # original data
    pre = pd.read_csv(os.path.join(data_path, 'data_fold' + str(fold) + '.csv'),
                      dtype=str, engine='c')

    post = pd.read_csv('predictions/' + save_path + "_preds.csv", dtype=str, engine='c')
    post.set_index('Unnamed: 0', inplace=True)

    index = np.array(post.index.values, dtype=np.int32)
    res = []

    for i, idx in enumerate(index):
        postdoc = post['recordDocumentId'].iat[i]
        iidx = pre.index.values[pre.index.values == idx][0]
        res.append(postdoc == pre['recordDocumentId'].iat[iidx])

    if not np.all(res):
        ctr = Counter(res)
        print(ctr)
    else:
        print("All processed data is correct.")


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', '-mp', type=str, default='',
                        help="""this is the location of the model
                                that will used to make predictions""")
    parser.add_argument('--data_path', '-dp', type=str, default='',
                        help="""where the data will load from. The default is
                                the path saved in the model""")
    parser.add_argument('--model_args', '-args', type=str, default='',
                        help="""file specifying the model or clc args; default is in
                                the model_suite directory""")
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    args = parser.parse_args()

    # 1. validate model/data args
    print("Validating kwargs in model_args.yml file")
    data_source = 'pre-generated'

    valid_params = validate_params.ValidateParams(args, data_source=data_source)

    valid_params.check_data_train_args()

    if valid_params.model_args['model_type'] == 'mthisan':
        valid_params.hisan_arg_check()
    elif valid_params.model_args['model_type'] == 'mtcnn':
        valid_params.mtcnn_arg_check()

    if valid_params.model_args['abstain_kwargs']['abstain_flag']:
        valid_params.check_abstain_args()

    valid_params.check_data_files()

    # 2. load data
    print("Loading data ")
    dw = data_utils.DataHandler(data_source, valid_params.model_args)
    dw.load_folds()

    print("\nVerifying subset is correct")
    verify_subset(valid_params, dw)

    # maps labels to ints, eg, C50 -> <some int>
    dw.convert_y()
    print("\nVerifying loaded data matches that on disk")
    check_loaded_data(valid_params, dw)

    # check torch dataLoders
    print("\nVerifying PathReports class is correct")
    check_dataloader(dw)


if __name__ == "__main__":
    main()
