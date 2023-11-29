
import argparse
import json
import os
import sys

# import torch
import numpy as np
import pandas as pd

from collections import Counter

from validate import validate_params
from data_loaders import data_utils
from training import training


class LabelDict(dict):
    """Create dict subclass for correct behaviour when mapping task unks."""
    def __init__(self, data, missing):
        super().__init__(data)
        self.data = data
        self.missing = missing

    def __getitem__(self, key):
        if key not in self.data:
            return self.__missing__(key)
        return self.data[key]

    def __missing__(self, key):
        """Assigns all missing keys to unknown in the label2id mapping."""
        return self.data[self.missing]


def subset_check_X(dw, data, split_idx):


    size = len(dw.inference_data["X"][split_idx]) / data.shape[0]
    print(f"Original shape: {data.shape[0]} " +
          f"New shape: {len(dw.inference_data['X'][split_idx])} " +
          f"Subset proportion: {size}")

    matches = []
    for i, idx in enumerate(dw.inference_data['X'][split_idx].index):
        iidx = np.nonzero(data.index == idx)[0]
        x1 = data.iat[iidx[0]]
        x2 = dw.inference_data['X'][split_idx].iat[i]
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

    for s, split in enumerate(splits):
        data = df[df['split'] == split]['X'].apply(lambda x: np.array(json.loads(x), dtype=np.int32))
        val = subset_check_X(dw, data, s)
        if sum(val) != len(dw.inference_data["X"][s]):
            print("Error, X arrays do not match")
        else:
            print(f"Split: {split} X-arrays are correct")



def verfiy_pathreports_class(dw):
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

    splits =  ['train', 'test', 'val']
    tasks = list(dw.dict_maps['id2label'].keys())
    label2id = {}

    for task in dw.dict_maps['id2label'].keys():
        label2id[task] = {v: k for k, v in dw.dict_maps['id2label'][task].items()}

    for split in splits:
        df_split = df[df['split'] == split]
        check_pathreports(dw, df_split, split, label2id)

def check_pathreports(dw, df, split, label2id):

    data = data_utils.PathReports(dw.inference_data['X'][split],
                                  dw.inference_data['y'][split],
                                  tasks=dw.model_args['data_kwargs']['tasks'],
                                  label_encoders=dw.dict_maps['id2label'],
                                  max_len=dw.model_args['train_kwargs']['doc_max_len'])


    res = {"X": [], "y": []}
    tasks = list(dw.dict_maps['id2label'].keys())
    
    for i in range(len(data)):
        X = data[i]["X"]
        X = X[X != 0]  # 0 is <pad>
        # have to remove 0 from both

        idx = data[i]['index']
        iidx = np.nonzero(df.index == idx)[0]

        X1 = df['X'].iat[iidx[0]]
        X1 = X1[X1 != 0]
        res["X"].append(np.allclose(X, X1))

        loader_y = [data[i][f'y_{task}'].item() for task in tasks]
        disk_y = [label2id[task][df[task].iat[iidx[0]]] if df[task].iat[iidx[0]] in label2id[task].keys() 
                  else int(label2id[task][dw.model_args['task_unks'][task]]) for task in tasks]
        
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

    with open(os.path.join(data_path, 'id2labels_fold' + str(fold) + '.json'),
              'r', encoding='utf-8') as f:
            tmp = json.load(f)

    id2label = {task: {int(k): str(v) for k, v in labels.items()}
                      for task, labels in tmp.items()}
    label2idx = {}
    for task in tasks:
        label2idx[task] = {v: k for k, v in id2label[task].items()}
    
    df = pd.read_csv(os.path.join(data_path, 'data_fold' + str(fold) + '.csv'),
                     dtype=str, engine='c')

    res = []
    for task in tasks:
        label_dict = LabelDict(label2idx[task], dw.model_args['task_unks'][task])
        for split in splits:
            processed = dw.inference_data["y"][split][task]
            loaded = df[df['split'] == split][task].map(label_dict)
            for i, idx in enumerate(processed.index):
                idx = np.nonzero(loaded.index == idx)[0]
                res.append(np.allclose(processed.iat[i], loaded.iloc[idx].values))
                if not np.allclose(processed.iat[i], loaded.iloc[idx].values):
                    print(processed.iat[i], loaded.iloc[idx].values)
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

    # 1. validate model/data args
    print("Validating kwargs in model_args.yml file")
    cache_class = [False]
    data_source = 'pre-generated'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", "-m", type=str, default='ie',
                        help="""which type of model to create. Must be either
                                IE (information extraction) or clc (case-level context).""")
    parser.add_argument('--model_path', '-mp', type=str, default='',
                       help="""this is the location of the model
                               that will used to make predictions""")
    parser.add_argument('--data_path', '-dp', type=str, default='',
                        help="""where the data will load from. The default is
                                the path saved in the model""")
    parser.add_argument('--model_args', '-args', type=str, default='',
                        help="""file specifying the model or clc args; default is in
                                the model_suite directory""")

    args = parser.parse_args()

    valid_params = validate_params.ValidateParams(cache_class, args,  data_source=data_source)

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
    dw = data_utils.DataHandler(data_source, valid_params.model_args, cache_class)
    dw.load_folds()

    print("\nVerifying subset is correct")
    # verify_subset(valid_params, dw)

    # maps labels to ints, eg, C50 -> <some int>
    dw.convert_y()
    print("\nVerifying loaded data matches that on disk")
    check_loaded_data(valid_params, dw)

    # check torch dataLoders
    print("\nVerifying PathReports class is correct")
    verfiy_pathreports_class(dw)
    
    print("Checking postprocessing data")
    # check_saved_data(valid_params, dw)

if __name__ == "__main__":
    main()
