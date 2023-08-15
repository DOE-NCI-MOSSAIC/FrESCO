"""

Module for loading pre-generated data.

"""
import json
import math
import os
import pickle
import random

import torch

import numpy as np
import pandas as pd
import polars as pl


from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from fresco.validate import exceptions


def word2int(tok, vocab):
    """Map words to tokens for random embeddings.

        If a word doesn't exist in the training/val split, it is mapped
        to 'unk', the unknown token.

    """
    unk = len(vocab)
    return [x if x is not None else unk for x in tok]

class LabelDict(dict):
    """
    Create dict subclass for correct behaviour when mapping task unks.
    """
    def __init__(self, data, missing):
        super().__init__(data)
        self.data = data
        self.missing = missing

    def __getitem__(self, key):
        if key not in self.data:
            return self.__missing__(key)
        return self.data[key]

    def __missing__(self, key):
        """
        Assigns all missing keys to unknown in the label2id mapping.
        """
        return self.data[self.missing]


class DataHandler():
    """
    Class for loading data.

    Attributes:
        data_source (str): Defines how data was generated. Needs to be 'pre-generated', 'pipeline', or 'official' (not implemented).
        model_args (dict): Keywords necessary to load data, build, and train a model_args.

    Note: The present implementation is only for 1 fold. We presently cannot generate more than one fold of data.
    """

    def __init__(self, data_source: str, model_args: dict, clc_flag: bool = False):
        self.data_source = data_source
        self.model_args = model_args

        print(f"Loading data from {self.model_args['data_kwargs']['data_path']}")

        self.inference_data = {}

        self.dict_maps = {}

        self.metadata = {'metadata': None, 'packages': None}

        self.splits = []
        self.num_classes = []
        self.train_size = 0
        self.val_size = 0
        self.test_size = 0

        self.tasks = self.model_args['data_kwargs']['tasks']
        self.weights = None
        if clc_flag:
            self.clc_flag = True
            self.grouped_cases = {}

    def load_folds(self, fold: int = 0, subset_frac: float = None):
        """
        Load data for each fold in the dataset.

        Parameters:
            fold (int): Integer number of the fold to load.
            subset_frac (float): Float value for proportion to load.

        Pre-condition: __init__ called and model_args is not None.

        Post-condition: Class attributes populated.

        Note: Case level context model will load fold 0 by default. See run_clc.py line 225.
        """

        data_loader = self.load_from_saved

        if fold is None:
            fold = self.model_args['data_kwargs']['fold_number']

        if self.model_args['train_kwargs']['class_weights'] is not None:
            self.load_weights(fold)

        if subset_frac is None:
            subset_frac = self.model_args['data_kwargs']['subset_proportion']
        else:
            subset_frac = subset_frac

        loaded = data_loader(fold, subset_frac)

        loaded['packages'] = {"mod_args": self.model_args,
                              "py_packs": {"torch": str(torch.__version__),
                                           "numpy": str(np.__version__),
                                           "pandas": str(pd.__version__)},
                              "fold": fold
                             }

        self.inference_data['X'] = loaded['X']
        self.inference_data['y'] = loaded['Y']
        if loaded['we'] is None:
            self.get_vocab()
        else:
            self.inference_data['word_embedding'] = loaded['we']
            self.dict_maps['id2word'] = loaded['id2word']

        self.dict_maps['id2label'] = loaded['id2label']

        self.metadata['metadata'] = loaded['metadata']

        self.metadata['packages'] = loaded['packages']

        self.num_classes = [len(self.dict_maps['id2label'][t].keys()) for
                            t in self.tasks]

    def load_from_saved(self, fold: int, subset_frac: float = None) -> dict:
        """
        Load data files.

        Arguments:
            fold (int): Fold number. Should always be 0 for now.
            subset_frac (float): Proportion of data to load.

        Post-condition:
            Modifies self.splits in-place.
        """

        loaded_data = {'metadata': {}, "X": {}, "Y": {}}
        data_path = self.model_args['data_kwargs']['data_path']

        with open(os.path.join(data_path, 'id2labels_fold' + str(fold) + '.json'),
                  'r', encoding='utf-8') as f:
            tmp = json.load(f)

        loaded_data['id2label'] = {task: {int(k): str(v) for k, v in labels.items()}
                                   for task, labels in tmp.items() if task in self.tasks}

        df = pd.read_csv(os.path.join(data_path, 'data_fold' + str(fold) + '.csv'),
                         dtype=str, engine='c', memory_map=True)

        self.splits = list(set(df['split'].values))
        if len(self.splits) == 3:
            self.splits = ['train', 'test', 'val']
        elif len(self.splits) == 2:
            if 'val' in self.splits:
                self.splits = sorted(self.splits)
            else:
                self.splits = ['train', 'test']

        for split in self.splits:
            loaded_data['X'][split] = df[df['split'] == split]['X'].apply(lambda x: np.array(json.loads(x),
                                                                                             dtype=np.int32))
            loaded_data['metadata'][split] = df[df['split'] == split][[v for v in df.columns
                                                             if v not in ['X', *loaded_data['id2label'].keys()]]]
            data_tasks = set([v for v in df.columns if v in loaded_data['id2label'].keys()])
            if len(data_tasks) > 0:
                loaded_data['Y'][split] = df[df['split'] == split][sorted(loaded_data['id2label'].keys())]

        if subset_frac < 1:
            rng = np.random.default_rng(self.model_args['train_kwargs']['random_seed'])
            for split in self.splits:
                data_size = len(loaded_data["X"][split])
                idxs = rng.choice(data_size, size=math.ceil(data_size*subset_frac),
                                  replace=False)
                loaded_data["X"][split] = loaded_data["X"][split].loc[loaded_data["X"][split].index[idxs]]
                loaded_data["Y"][split] = loaded_data["Y"][split].loc[loaded_data["Y"][split].index[idxs]]
                loaded_data['metadata'][split] = loaded_data['metadata'][split].loc[loaded_data['metadata'][split].index[idxs]]

        if os.path.exists(os.path.join(data_path, 'word_embeds_fold' + str(fold) + '.npy')):
            loaded_data['we'] = np.load(os.path.join(data_path, 'word_embeds_fold' + str(fold) + '.npy'))
        else:
            loaded_data['we'] = None

        if os.path.exists(os.path.join(data_path, 'id2word_fold' + str(fold) + '.json')):
            with open(os.path.join(data_path, 'id2word_fold' + str(fold) + '.json'),
                      'r', encoding='utf-8') as f:
                tmp = json.load(f)
            loaded_data['id2word'] = {int(k): str(v) for k, v in tmp.items()}
        else:
            loaded_data['id2word'] = None

        return loaded_data

    def get_vocab(self):
        """
        Get the vocab and word embeddings from tokenized data.
        """

        embedding_dim = 300

        X = self.inference_data['X']['train']

        if 'val' in self.inference_data['X'].keys():
            X = pd.concat([X, self.inference_data['X']['val']])

        # find all unique tokens
        s1 = set(X.iloc[0])
        for x in X:
            s1 = set(x).union(s1)

        vocab_len = np.max(list(s1))
        unk = vocab_len + 1 
        self.inference_data['X']['test'].apply(lambda d: word2int(d, s1))
        rng = np.random.default_rng(self.model_args['train_kwargs']['random_seed'])
        unk_embed = rng.normal(size=(1, embedding_dim), scale=0.1)
        random_embeds = rng.standard_normal(size=(vocab_len, embedding_dim), dtype=np.float32) * 0.1
        self.inference_data['word_embedding'] = np.concatenate((np.zeros(shape=(1, embedding_dim)),
                                                                random_embeds,
                                                                unk_embed),
                                                                axis=0)

    def convert_y(self):
        """
        Add task unknown labels to Y and map values to integers for inference.

        Post-condition:
            The data frame with the output, the ys, is modified in place by this function.
            It maps the string values to ints for inference, i.e., C50 -> 48 for the site task.

        Note: If loading data separately from creating torch dataloaders,
            this function should be called if you want ints and not strings.
        """


        missing_tasks = [v for v in self.inference_data['y']['train'].columns if
                         v not in self.model_args['task_unks'].keys()]

        if missing_tasks != []:
            raise exceptions.ParamError(f'the tasks {",".join(missing_tasks)} are missing from' +
                                         'task_unks in the the model_args.yml file')
        known_labels = {}
        label2id = {}

        for task in self.dict_maps['id2label'].keys():
            known_labels[task] = [str(v) for v in self.dict_maps['id2label'][task].values()]
            if self.model_args['task_unks'][task] not in known_labels[task]:
                raise exceptions.ParamError(f'for task {task} the task_unks ' +
                                            f'{self.model_args["task_unks"][task]} ' +
                                            'is not in the mapped labels')

        for task in self.dict_maps['id2label'].keys():
            label2id[task] = {v: k for k, v in self.dict_maps['id2label'][task].items()}
            label_dict = LabelDict(label2id[task], self.model_args['task_unks'][task])
            for split in self.splits:
                self.inference_data['y'][split][task] = self.inference_data['y'][split][task].map(label_dict)

    def load_weights(self, fold):
        """
        Loads class weights from pickle file or dict in model_args file.

        Args:
            fold (int): Data fold to be loaded

        """

        if (self.model_args['train_kwargs']['class_weights'] is not None) and self.model_args['abstain_kwargs']['abstain_flag']:
            raise exceptions.ParamError("Class weights cannot be used with dac or ntask")

        data_path = self.model_args['data_kwargs']['data_path']

        with open(os.path.join(data_path, 'id2labels_fold' + str(fold) + '.json'),
                  'r', encoding='utf-8') as f:
            tmp = json.load(f)
        id2label = {task: {int(k): str(v) for k, v in labels.items()}
                           for task, labels in tmp.items()}

        if isinstance(self.model_args['train_kwargs']['class_weights'], dict):
            self.weights = self.model_args['train_kwargs']['class_weights']

        elif isinstance(self.model_args['train_kwargs']['class_weights'], str):
            path = self.model_args['train_kwargs']['class_weights']
            if os.path.exists(path):
                with open(path, "rb") as f_in:
                    self.weights = pickle.load(f_in)
            else:
                raise exceptions.ParamError("Class weights path does not exist.")
        elif self.model_args['train_kwargs']['class_weights'] is None:
            self.weights = None
        else:
            raise exceptions.ParamError("Class weights must be dict, point to relevant .pickle file, or be None.")

        if self.weights is not None:
            # check that all tasks have class weight
            if len(id2label.keys()) > 1:
                raise exceptions.ParamError("Class weights for multi-task is not yet implemented.")
            for task in self.tasks:
                if task not in self.weights.keys():
                    raise exceptions.ParamError(f'Class weights for task {task} not specified ' +
                                                f'For unweighted classes specify as "{task}: ' +
                                                f'None" in {self.model_args["train_kwargs"]["class_weights"]}')

            for task in self.weights:
                # if there is only a single task, weights may be none
                if len(self.weights) == 1:
                    if not (isinstance(self.weights[task], list) or self.weights[task] is None):
                        raise exceptions.ParamError('Weights for single task must be a list or None.')

                # if there are multiple tasks, weights should be lists.
                elif len(self.weights) > 1:
                    if self.weights[task] is None:
                        raise exceptions.ParamError('Weights for multiple task must be lists.')

                # check that all class weight lists provided are the right length
                if isinstance(self.weights[task], list):
                    n_weights = len(self.weights[task])
                    if n_weights != len(id2label[task].values()):
                        raise exceptions.ParamError("Number of weights must be equal to the " +
                                                    "number of classes in each task. Task " +
                                                    f"{task} should have {len(id2label[task].values())} values")

    def make_torch_dataloaders(self, switch_rate: float,
                               reproducible: bool = False,
                               shuffle_data: bool = False,
                               seed: int = None) -> dict:
        """
        Create torch DataLoader classes for training module.

        Returns a dictionary of PyTorch DataLoaders (train, val, test) for the training module.

        Args:
            switch_rate (float): Proportion of words in each document to randomly flip.
            reproducible (bool): Set all random number seeds.
            shuffle_data (bool): Shuffle ordering of data within data loader.
            seed (int): Random number seed.
        """
        if reproducible:
            gen = torch.Generator()
            gen.manual_seed(seed)
            worker = self.seed_worker
        else:
            worker = None
            gen = None
        
        vocab_size = self.inference_data['word_embedding'].shape[0]
        unk_tok = vocab_size - 1
        
        pin_mem = bool(torch.cuda.is_available())

        if switch_rate == 0.0:
            _transform = None
        else:
            _transform = AddNoise(unk_tok,
                                  self.model_args['train_kwargs']['doc_max_len'],
                                  vocab_size,
                                  switch_rate,
                                  seed
                                  )

        loaders = {}


        # num multiprocessing workers for DataLoaders
        n_wkrs = 4
        print(f"Num workers: {n_wkrs}, reproducible: {self.model_args['data_kwargs']['reproducible']}")

        # maps labels to ints, eg, C50 -> <some int>
        self.convert_y()

        train_data = PathReports(self.inference_data['X']['train'], self.inference_data['y']['train'],
                                 tasks=self.model_args['data_kwargs']['tasks'],
                                 label_encoders=self.dict_maps['id2label'],
                                 max_len=self.model_args['train_kwargs']['doc_max_len'],
                                 transform=_transform
                                 )
        loaders['train'] = DataLoader(train_data,
                                      batch_size=self.model_args['train_kwargs']['batch_per_gpu'],
                                      shuffle=shuffle_data, pin_memory=pin_mem, num_workers=n_wkrs,
                                      worker_init_fn=worker, generator=gen)
        self.train_size = len(train_data)

        if 'val' in self.splits:
            val_data = PathReports(self.inference_data['X']['val'], self.inference_data['y']['val'],
                                   tasks=self.model_args['data_kwargs']['tasks'],
                                   label_encoders=self.dict_maps['id2label'],
                                   max_len=self.model_args['train_kwargs']['doc_max_len'],
                                   transform=None)

            loaders['val'] = DataLoader(val_data,
                                        batch_size=self.model_args['train_kwargs']['batch_per_gpu'],
                                        shuffle=shuffle_data, pin_memory=pin_mem, num_workers=n_wkrs,
                                        worker_init_fn=worker, generator=gen)
            self.val_size = len(val_data)
        else:
            self.val_size = 0

        if 'test' in self.splits:
            test_data = PathReports(self.inference_data['X']['test'], self.inference_data['y']['test'],
                                    tasks=self.model_args['data_kwargs']['tasks'],
                                    label_encoders=self.dict_maps['id2label'],
                                    max_len=self.model_args['train_kwargs']['doc_max_len'],
                                    transform=None)

            loaders['test'] = DataLoader(test_data,
                                         batch_size=self.model_args['train_kwargs']['batch_per_gpu'],
                                         shuffle=shuffle_data, pin_memory=pin_mem, num_workers=n_wkrs,
                                         worker_init_fn=worker, generator=gen)
            self.test_size = len(test_data)
        else:
            self.test_size = 0
        print(f"Training on {self.train_size} validate on {self.val_size}")

        return loaders

    def make_grouped_cases(self, doc_embeds, clc_args, device, reproducible:bool = True, seed: int = None):
        """Created GroupedCases class for torch DataLoaders."""

        if reproducible:
            gen = torch.Generator()
            gen.manual_seed(seed)
            worker = self.seed_worker
        else:
            worker = None
            gen = None
        
        datasets = {split: GroupedCases(doc_embeds[split]['X'],
                                        doc_embeds[split]['y'],
                                        doc_embeds[split]['index'],
                                        self.model_args['data_kwargs']['tasks'],
                                        self.metadata['metadata'][split],
                                        device,
                                        exclude_single=clc_args['data_kwargs']['exclude_single'],
                                        shuffle_case_order=clc_args['data_kwargs']['shuffle_case_order'],
                                        ) for split in self.splits}

        self.grouped_cases = {split: DataLoader(datasets[split],
                                                batch_size=clc_args['train_kwargs']['batch_per_gpu'],
                                                shuffle=False,
                                                worker_init_fn=worker,
                                                generator=gen) for split in self.splits}

    def inference_loader(self, reproducible: bool = True,
                         seed: int = None, batch_size: int = 128) -> dict:
        """
        Create torch DataLoader class for inference from a trained model.

        Returns a dictionary of PyTorch DataLoaders (test) for inference.

        Args:
            reproducible (bool): Set all random number seeds.
            seed (int): Random number generator seed.
            batch_size (int): Batch size for inference.
        """
        if reproducible:
            gen = torch.Generator()
            gen.manual_seed(seed)
            worker = self.seed_worker
        else:
            worker = None
            gen = None

        if torch.cuda.is_available():
            pin_mem = True
        else:
            pin_mem = False

        loaders = {}

        # num multiprocessing workers for DataLoaders
        n_wkrs = 4
        print(f"Num workers: {n_wkrs}, reproducible: {self.model_args['data_kwargs']['reproducible']}")

        # Y is None for predictions without ground truth.
        inference_data = PathReports(self.inference_data['X']['test'], None,
                                     tasks=self.model_args['data_kwargs']['tasks'],
                                     label_encoders=None,
                                     max_len=self.model_args['train_kwargs']['doc_max_len'],
                                     transform=None
                                    )
        loaders['test'] = DataLoader(inference_data,
                                      batch_size=self.model_args['train_kwargs']['batch_per_gpu'],
                                      shuffle=True, pin_memory=pin_mem, num_workers=n_wkrs,
                                      worker_init_fn=worker, generator=gen)

        return loaders

    @staticmethod
    def seed_worker(worker_id):
        """
        Set random seed for everything.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class AddNoise():
    """
    Optional transform object for the PathReports dataset.

    This transform adds a random amount of padding at the front of the document using `unk_token` to reduce HiSAN overfitting. It also randomly replaces words with randomly selected other words to reduce overfitting.

    Args:
        unk_token (int): Integer mapping for unknown tokens.
        max_pad_len (int): Maximum amount of padding at the front of the document.
        vocab_size (int): Size of the vocabulary matrix or the maximum integer value to use when randomly replacing word tokens.
        switch_rate (float, default: 0.1): Percentage of words to randomly replace with random tokens.
    """

    def __init__(self, unk_token, max_pad_len, vocab_size, switch_rate, seed=None):
        self.unk_token = unk_token
        self.max_pad_len = max_pad_len
        self.vocab_size = vocab_size
        self.switch_rate = switch_rate
        self.rng = np.random.default_rng(seed)

    def __call__(self, doc):
        pad_amt = self.rng.integers(0, self.max_pad_len)
        doc = [int(self.unk_token) for i in range(pad_amt)] + list(doc)
        r_idx = self.rng.choice(np.arange(len(doc)), size=int(len(doc)*self.switch_rate), replace=False)
        r_voc = self.rng.integers(1, self.vocab_size, r_idx.shape[0])
        doc = np.array(doc)
        doc[r_idx] = r_voc
        return doc


class PathReports(Dataset):
    def __init__(self, X, Y, tasks, label_encoders, max_len=3000, transform=None, multilabel=False):
        """
        Torch DataLoader class for cancer path reports from generate_data.py.

        Args:
            X (pandas.DataFrame): DataFrame of tokenized path report data. Entries are numpy arrays generated from generate_data.py.
            Y (pd.DataFrame): DataFrame of ground truth values.
            tasks (list[str]): List of tasks to generate labels for.
            label_encoders (dict): Dictionary of task-to-label encoders to convert raw labels into integers.
            max_len (int, default: 3000): Maximum length for a document. Should match the value in data_args.json. Longer documents will be cut, and shorter documents will be zero-padded.
            transform (object, default: None): Optional transform to apply to document tensors.
            pred_only(bool): For making predictions on unlabeled data.

        Outputs per batch:
            dict[str, torch.Tensor]: A sample dictionary with the following keys and values:
                - 'X': torch.Tensor (int) [max_len]: Document converted to integer word-mappings, zero-padded to max_len.
                - 'y_%s % task': torch.Tensor (int) [] or torch.Tensor (int) [num_classes]: Integer label for a given task if label encoders are used. One-hot vectors for a given task if label binarizers are used.
                - 'index': int: DataFrame index to match up with metadata stored in the original DataFrame.
        """

        self.X = X
        self.ys = {}
        self.ys_onehot = {}
        self.label_encoder = label_encoders
        self.num_classes = {}
        self.tasks = tasks
        self.transform = transform
        self.max_len = max_len
        self.multilabel = multilabel
        self.pred_only = False

        if Y is None:  # for predictions w/o ground truth
            self.pred_only = True
        else:
            for task in tasks:
                y = np.asarray(Y[task].values, dtype=np.int16)
                le = {v: int(k) for k, v in self.label_encoder[task].items()}
                # ignore abstention class if it exists
                if f'abs_{task}' in le:
                    del le[f'abs_{task}']
                self.num_classes[task] = len(le)
                self.ys[task] = y

                if self.multilabel:
                    y_onehot = np.zeros((len(y), len(le)), dtype=np.int16)
                    y_onehot[np.arange(len(y)), y] = 1
                    self.ys_onehot[task] = y_onehot

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        doc = self.X.iat[idx]
        if self.transform:
            doc = self.transform(doc)
        array = np.zeros(self.max_len, dtype=np.int32)
        doc = doc[:self.max_len]
        array[:doc.shape[0]] = doc
        sample = {'X': torch.tensor(array, dtype=torch.long),
                  'index': self.X.index[idx]}  # indexing allows us to keep track of the metadata associated with each X
        if not self.pred_only:
            for _, task in enumerate(self.tasks):
                if self.multilabel:
                    y = self.ys_onehot[task][idx]
                    sample[f'y_{task}'] = torch.tensor(y, dtype=torch.float)
                else:
                    y = self.ys[task][idx]
                    sample[f'y_{task}'] = torch.tensor(y, dtype=torch.long)
        return sample

class GroupedCases(Dataset):
    """Create grouped cases for torch DataLoaders.

        args:
            doc_embeds - document embeddings from trained model as np.ndarray
            Y - dict of integer Y values, keys are the splits
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
                 shuffle_case_order=True):
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

        groups = metadata['group'].tolist()
        metadata_idxs = metadata.index.tolist()

        groups_pl = pl.DataFrame({'index': metadata_idxs, 'group': groups}).with_columns(pl.col("index").cast(pl.Int32))

        self.max_seq_len = groups_pl.groupby('group').count().max().select("count").item()

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

        groups_pl = (groups_pl.join(df_pl, on='index', how='inner')
                           .groupby(by="group", maintain_order=True).agg([pl.col(col) for col in pl_cols]))
        del pl_cols[-2:]
        
        grouped_X = groups_pl.select("doc_embeds").to_series().to_list()
        self.grouped_X = []
        self.lens = []
        for X in grouped_X: 
            blank = torch.zeros((self.max_seq_len, self.embed_size), dtype=torch.float32)
            blank[:len(X),:] = torch.Tensor(X)
            self.grouped_X.append( blank )
            self.lens.append(len(X))
        self.new_idx = groups_pl.select("index").to_series()
        self.grouped_y = groups_pl.select(pl_cols).to_dict(as_series=False)

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
        
        sample = {"X": self.grouped_X[idx]}
        _len = self.lens[idx]
        sample['len'] = _len   # , device=self.device)

        y_array = torch.zeros((self.max_seq_len,), dtype=torch.long)
        for task in self.tasks:
            y_array[:_len] = torch.from_numpy(ys[task])
            sample[f"y_{task}"] = y_array.clone()

        idx_array = torch.zeros((self.max_seq_len,), dtype=torch.int16)
        idx_array[:_len] = torch.tensor(self.new_idx[idx])
        sample['index'] = idx_array.clone()

        return sample
