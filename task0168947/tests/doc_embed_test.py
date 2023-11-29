import os
import json
import sys
import argparse
import random
import pickle
import torch
import numpy as np
import pandas as pd
import polars as pl

from validate import exceptions
from validate import validate_params
from data_loaders import data_utils
from models import mthisan, mtcnn, mthisan_v1

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from warnings import warn



def load_model_v1(model_path, model_type, embeds, num_classes, device):

    if torch.cuda.is_available():
        model_dict = torch.load(model_path)
    else:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_dict = {k.replace('module.',''):v for k,v in model_dict.items()}
    model_args = model_dict['metadata_package']['Model_Suite']['mod_args']

    # Load Model
    if model_type == 'mtcnn':
        model = mtcnn.MTCNN(embeds, num_classes, **model_args["MTCNN_kwargs"])
    elif model_type == 'mthisan':
        model = mthisan_v1.MTHiSAN(embeds, num_classes, **model_args["MTHiSAN_kwargs"])
    model.to(device)

    model_dict = {k:v for k,v in model_dict.items() if k!='metadata_package'}
    model.load_state_dict(model_dict)
    model.eval()

    print('model_loaded')
    return model

def load_model_dict(model_path):
    """Load pretrained model from disk.

        Args:
            model_path: str, from command line args, points to saved model
            valid_params: ValidateParams class, with model_args dict
            data_path: str or None, using data from the trained model, or different one

        We check if the supplied path is valid and if the packages match needed
            to run the pretrained model.

    """
    if os.path.exists(model_path):
        if torch.cuda.is_available():
            model_dict = torch.load(model_path)
        else:
            model_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model_dict = {k.replace('module.',''):v for k,v in model_dict.items()}
    else:
        raise exceptions.ParamError("Provided model path does not exist")

    return model_dict


def load_model_v2(model_dict, device, dw):

    # model_args = model_dict['metadata_package']['Model_Suite']
    model_args = model_dict['metadata_package']['mod_args']

    if model_args['model_type'] == 'mthisan':
        model = mthisan.MTHiSAN(dw.inference_data['word_embedding'],
                                dw.num_classes,
                                **model_args['MTHiSAN_kwargs'])

    elif model_args['model_type'] == 'mtcnn':
        model = mtcnn.MTCNN(dw.inference_data['word_embedding'],
                            dw.num_classes,
                            **model_args['MTCNN_kwargs'])

    model.to(device)

    model_dict = {k: v for k,v in model_dict.items() if k!='metadata_package'}
    model_dict = {k.replace('module.',''): v for k,v in model_dict.items()}
    model.load_state_dict(model_dict)

    print('model loaded')

    return model


def generate_doc_embeds_v1(model, data_loader, device):

    model.eval()
    embeds = []
    with torch.no_grad():
        for b, batch in enumerate(data_loader):
            X = batch['X'].to(device)
            _,embed = model(X, return_embeds=True)
            embeds.append(embed.detach().cpu().numpy())
            sys.stdout.write("predicting sample %i of %i           \r"\
                             % ((b + 1) * data_loader.batch_size, len(data_loader.dataset)))
            sys.stdout.flush()
    print()
    embeds = np.vstack(embeds)
    return embeds


def create_doc_embeddings_v2(model, model_type, data_loader, tasks, device):
    """Generate document embeddings from trained model."""
    model.eval()
    if model_type == 'mtcnn':
        embed_dim = 900
    else:
        embed_dim = 400

    bs = data_loader.batch_size
    embeds = torch.empty((len(data_loader.dataset), embed_dim), device=device)
    ys = {task: torch.empty(len(data_loader.dataset), dtype=torch.int, device=device)
          for task in tasks}
    idxs = torch.empty((len(data_loader.dataset), ), dtype=torch.int, device=device)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            X = batch["X"].to(device, non_blocking=True)
            _, embed = model(X, return_embeds=True)
            if embed.shape[0] == bs:
                embeds[i*bs:(i+1)*bs] = embed
                idxs[i*bs:(i+1)*bs] = batch['index']
                for task in tasks:
                    if task != 'Ntask':
                        ys[task][i*bs:(i+1)*bs] = batch[f'y_{task}']
            else:
                embeds[-embed.shape[0]:] = embed
                idxs[-embed.shape[0]:] = batch['index']
                for task in tasks:
                    if task != "Ntask":
                        ys[task][-embed.shape[0]:] = batch[f'y_{task}']
    ys_np = {task: vals.cpu().numpy() for task, vals in ys.items()}
    outputs = {"X": embeds.cpu().numpy(), 'y': ys_np, 'index': idxs.cpu().numpy()}
    return outputs


class GroupedCases_v1(Dataset):
    def __init__(self,
                 doc_embeds,
                 df_Y,
                 tasks,
                 metadata,
                 label_encoders,
                 exclude_single=False,
                 shuffle_case_order=False,
                 split_by_tumorid=False,
                ):

        self.embed_size = len(doc_embeds[0])
        self.tasks = tasks
        self.shuffle_case_order = shuffle_case_order
        self.label_encoders = {}
        for task in tasks:
            le = {v:int(k) for k,v in label_encoders[task].items()}
            self.label_encoders[task] = le
        self.grouped_X = []
        self.grouped_ys = {task:[] for task in tasks}
        self.new_idx = []
        if split_by_tumorid:
            metadata['uid'] = metadata['registryId'] + metadata['patientId'].astype(str) + metadata['tumorId'].astype(str)
        else:
            metadata['uid'] = metadata['registryId'] + metadata['patientId'].astype(str)
        groups = metadata.reset_index().groupby('uid')
        self.max_seq_len = groups.agg('count').max().tolist()[0]
        try:
            df_Y_ = df_Y.reset_index()
        except:
            print('Warning: skipping df_y.reset_index()')

        for i,(name,group) in enumerate(groups):

            if exclude_single and len(group.index) == 1:
                continue

            g = group.sort_values(by='recordDocumentId')
            group = []
            indices = []
            labels = {task:[] for task in tasks}
            for idx in g.index:
                group.append(doc_embeds[idx])
                indices.append(idx)
                for task in tasks:
                    label = df_Y_[task][idx]
                    # labels[task].append(self.label_encoders[task][label])
                    labels[task].append(label)
            self.grouped_X.append(np.vstack(group))
            self.new_idx.append(indices)
            for task in tasks:
                self.grouped_ys[task].append(labels[task])

    def __len__(self):
        return len(self.grouped_X)

    def __getitem__(self,idx):

        seq = self.grouped_X[idx]
        ys = []
        for t,task in enumerate(self.tasks):
            y = self.grouped_ys[task][idx]
            ys.append(y)
        if self.shuffle_case_order:
            ys = np.array(ys).T
            shuffled = list(zip(seq,ys))
            random.shuffle(shuffled)
            seq,ys = zip(*shuffled)
            seq = np.array(seq)
            ys = np.array(ys).T

        array = np.zeros((self.max_seq_len,self.embed_size))
        l = len(seq)
        array[:l,:] = seq
        sample = {'X':torch.tensor(array,dtype=torch.float)}
        sample['len'] = torch.tensor(l,dtype=torch.long)

        for t,task in enumerate(self.tasks):
            y = ys[t]
            array = np.zeros(self.max_seq_len)
            array[:l] = y
            sample['y_%s' % task] = torch.tensor(array,dtype=torch.long)

        indices = self.new_idx[idx]
        array = np.zeros(self.max_seq_len)
        array[:l] = indices
        sample['new_idx'] = torch.tensor(array,dtype=torch.long)
        return sample


class GroupedCases_v2(Dataset):
    """Create grouped cases for torch DataLoaders.

        args:
            doc_embeds - document embeddings from trained model as torch.tensor
            Y - dict of integer Y values, keys are the splits
            idxs - indices of  doc embds and Ys associated with original data
            tasks - list of tasks
            metadata - dict of model metadata
            device - torch.device, either cuda or cpu
            exclude_single - are we omitting sinlge cases, default is True
            shuffle_case_order - shuffle cases, default is True
            split_by_tumor_id - split the cases by tumorId, default is True


    """
    def __init__(self,
                 doc_embeds,
                 Y,
                 idxs,
                 tasks,
                 metadata,
                 device,
                 exclude_single=True,
                 shuffle_case_order=True,
                 split_by_tumor_id=True):
        """Class for grouping cases for clc.

        """
        self.embed_size = doc_embeds.shape[1]
        self.tasks = tasks
        self.shuffle_case_order = shuffle_case_order
        self.label_encoders = {}  # label_encoders
        self.grouped_X = []
        self.grouped_y = {task: [] for task in self.tasks}
        self.new_idx = []
        self.device = device

        if split_by_tumor_id:
            metadata['uid'] = metadata['registryId'] + metadata['patientId'].astype(str) +\
                metadata['tumorId'].astype(str)
        else:
            metadata['uid'] = metadata['registryId'] + metadata['patientId'].astype(str)
       
        uids = metadata['uid'].tolist()
        metadata_idxs = metadata.index.tolist()
        uid_pl = pl.DataFrame({'index': metadata_idxs, 'uid': uids}).with_columns(pl.col("index").cast(pl.Int32))
       
        self.max_seq_len = uid_pl.groupby('uid').count().max().select("count").item()
        
        self.X_array = torch.zeros((self.max_seq_len, self.embed_size),
                                   device=device, dtype=torch.float32)
        self.y_array = torch.zeros((self.max_seq_len),
                                   device=device, dtype=torch.long)
        self.idx_array = torch.zeros((self.max_seq_len),
                                     device=device, dtype=torch.long)

        # num docs x 400
        X_pl = pl.Series('doc_embeds', doc_embeds) 
        # dict of numpy arrays, each 1 x num docs, keys are tasks
        y_pl = pl.from_dict(Y)
        # numpy array, idxs.shape[0] = num docs
        idx_pl = pl.Series('index', idxs)
        df_pl = pl.DataFrame([idx_pl, X_pl]).hstack(y_pl)
        
        pl_cols = [task for task in self.tasks]
        pl_cols.append("index")
        pl_cols.append("doc_embeds")

        groups_pl = (uid_pl.join(df_pl, on='index', how='inner')
                           .groupby("uid").agg([pl.col(col) for col in pl_cols]))
        del pl_cols[-2:]
        self.grouped_X = groups_pl.select("doc_embeds").to_series().to_list()
        
        self.new_idx = groups_pl.select("index").to_series() # .to_numpy()
        
        self.grouped_y = groups_pl.select(pl_cols).to_dict(as_series=False)

#        for uid, group in groups_pl:
#            print(group)
#            if exclude_single and len(group.index) == 1:
#                continue
#            _group = []
#            _idxs = []
#            labels = {task: [] for task in self.tasks}
#
#            for idx in group['index']:
#                print(idx)
#                data_idx = np.where(data['index']==idx)[0]
#                group.append(doc_embeds[data_idx])
#                _idxs.append(idx)
#                for task in self.tasks:
#                    labels[task].append(data[task][data_idx])
#
#            self.grouped_X.append(np.vstack(group))
#            self.new_idx.append(_idxs)
#            for task in self.tasks:
#                self.grouped_y[task].append(labels[task])

    def __len__(self):
        return len(self.grouped_X)

    def __getitem__(self, idx):
        seq = self.grouped_X[idx]
        ys = {}

        for task in self.tasks:
            ys[task] = np.array(self.grouped_y[task][idx]).flatten()

        if self.shuffle_case_order:
            ys = np.array(ys).T
            shuffled = list(zip(seq, ys))
            random.shuffle(shuffled)
            seq, ys = zip(*shuffled)
            seq = np.array(seq)
            ys = np.array(ys).T

        self.X_array[:] = 0.0
        self.y_array[:] = 0
        self.idx_array[:] = 0

        _len = len(seq)  # .shape[0]
        self.X_array[:_len, :] = torch.tensor(seq, device=self.device)  # seq.clone()
        sample = {"X": self.X_array}
        sample['len'] = torch.tensor(_len, dtype=torch.int, device=self.device)

        for task in self.tasks:
            self.y_array[:_len] = torch.tensor(ys[task], device=self.device)
            sample[f"y_{task}"] = self.y_array.clone()

        self.idx_array[:_len] = torch.tensor(self.new_idx[idx])
        sample['index'] = self.idx_array.clone()

        return sample

def make_grouped_cases_v2(dw, embeds_data, clc_args, device):
    """Created GroupedCases class for torch DataLoaders."""

    print("\nCreating GroupedCases loaders")
    datasets = {split: GroupedCases_v2(embeds_data[split]['X'],
                                    embeds_data[split]['y'],
                                    embeds_data[split]['index'],
                                    dw.model_args['data_kwargs']['tasks'],
                                    dw.metadata['metadata'][split],
                                    device,
                                    exclude_single=clc_args['data_kwargs']['exclude_single'],
                                    shuffle_case_order=clc_args['data_kwargs']['shuffle_case_order'],
                                    split_by_tumor_id=clc_args['data_kwargs']['split_by_tumorid']
                                    ) for split in dw.splits}

    # grouped_cases = {split: DataLoader(datasets[split],
    #                                         clc_args['train_kwargs']['batch_per_gpu'],
    #                                         shuffle=False) for split in dw.splits}
    return datasets


def make_grouped_cases_v1(dw, embeds_data, clc_args, device):
    """Created GroupedCases class for torch DataLoaders."""

    print("\nCreating GroupedCases loaders")

    datasets = {split: GroupedCases_v1(embeds_data[split],
                                    dw.inference_data['y'][split],
                                    dw.model_args['data_kwargs']['tasks'],
                                    dw.metadata['metadata'][split],
                                    dw.dict_maps['id2label'],
                                    exclude_single=clc_args['data_kwargs']['exclude_single'],
                                    shuffle_case_order=clc_args['data_kwargs']['shuffle_case_order'],
                                    split_by_tumorid=clc_args['data_kwargs']['split_by_tumorid']
                                    ) for i, split in enumerate(dw.splits)}

    grouped_cases = {split: DataLoader(datasets[split],
                                            clc_args['train_kwargs']['batch_per_gpu'],
                                            shuffle=False) for split in dw.splits}
    return datasets, grouped_cases


def main():
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
    clc_flag = True

    # 1. validate model/data args
    print("Validating kwargs in clc_args.yml file")
    cache_class = [False]
    data_source = 'pre-generated'

    clc_params = validate_params.ValidateClcParams(cache_class, args, data_source=data_source)

    clc_params.check_data_train_args()

    if clc_params.model_args['abstain_kwargs']['abstain_flag']:
        clc_params.check_abstain_args()

    if clc_params.model_args['data_kwargs']['reproducible']:
        seed = clc_params.model_args['data_kwargs']['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = load_model_dict(clc_params.model_args['data_kwargs']['model_path'])

    clc_params.model_args['data_kwargs']['data_path'] = model_dict['metadata_package']['mod_args']['data_kwargs']['data_path']
    clc_params.check_data_files(clc_params.model_args['data_kwargs']['data_path'])

    args.model_args = model_dict['metadata_package']['mod_args']
    print("Validating kwargs from pretrained model ")
    submodel_args = validate_params.ValidateParams(cache_class, args,
                                                   data_source=data_source,
                                                   model_args=args.model_args)

    submodel_args.check_data_train_args(from_pretrained=True)

    if clc_params.model_args['data_kwargs']['tasks'] != \
        submodel_args.model_args['data_kwargs']['tasks']:
        raise exceptions.ParamError("Tasks must be the same for both original and clc models")

    if clc_params.model_args['abstain_kwargs']['abstain_flag'] != \
        submodel_args.model_args['abstain_kwargs']['abstain_flag']:
        raise exceptions.ParamError("DAC must be true for both original and clc models")

    if clc_params.model_args['abstain_kwargs']['ntask_flag'] != \
        submodel_args.model_args['abstain_kwargs']['ntask_flag']:
        raise exceptions.ParamError("N-task must be true for both original and clc models")

    if submodel_args.model_args['model_type'] == 'mthisan':
        submodel_args.hisan_arg_check()
    elif submodel_args.model_args['model_type'] == 'mtcnn':
        warn("MTCNN will be deprecated; use HiSAN for all future models.", DeprecationWarning)
        submodel_args.mtcnn_arg_check()

    if submodel_args.model_args['abstain_kwargs']['abstain_flag']:
        submodel_args.check_abstain_args()

    submodel_args.check_data_files()

    # 2. load data
    print("Loading data and creating DataLoaders")

    dw = data_utils.DataHandler(data_source, submodel_args.model_args, cache_class)
    dw.load_folds(fold=0, subset_frac=1.0)

    # set add_noise = 0, AS 6/2
    data_loaders = dw.make_torch_dataloaders(0.0,  # submodel_args.model_args['data_kwargs']['add_noise'],
                                             reproducible=clc_params.model_args['data_kwargs']['reproducible'],
                                             shuffle_data=False,
                                             seed=seed)

    print('setting up v2')
    model = load_model_v2(model_dict, device, dw)
    print("Creating doc embeddings")

    outputs = {}

    for split in dw.splits:
        outputs[split] = create_doc_embeddings_v2(model,
                                               submodel_args.model_args['model_type'],
                                               data_loaders[split],
                                               dw.tasks,
                                               device)

    gc_v2 = make_grouped_cases_v2(dw, outputs, clc_params.model_args, device)

    print("Setting up v1")
    v1_model_path = '/home/spannausa/Gitlab/modularization_pipeline/Model_Suite/savedmodels/LAKY_model.h5'
    model_v1 = load_model_v1(v1_model_path, submodel_args.model_args['model_type'],
                            dw.inference_data['word_embedding'], dw.num_classes.values(), device)

    outputs_v1 = {split: generate_doc_embeds_v1(model_v1, data_loaders[split], device) for split in dw.splits}

    gc_v1 = make_grouped_cases_v1(dw, outputs_v1, clc_params.model_args, device)

    with open("GroupedCases_v1.pkl", "wb") as f:
        pickle.dump(gc_v1, f)

    with open("GroupedCases_v2.pkl", "wb") as f:
        pickle.dump(gc_v2, f)

    with open("embeds_v1.pkl", "wb") as f:
        pickle.dump(outputs_v1, f)

    with open("embeds_v2.pkl", "wb") as f:
        pickle.dump(outputs, f)

if __name__ == "__main__":
    main()
    
