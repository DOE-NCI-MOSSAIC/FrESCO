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


class DataHandler:
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

        self.metadata = {"metadata": None, "packages": None}

        self.splits = []
        self.num_classes = {}
        self.train_size = 0
        self.val_size = 0
        self.test_size = 0

        self.tasks = self.model_args["data_kwargs"]["tasks"]
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
            fold = self.model_args["data_kwargs"]["fold_number"]

        if self.model_args["train_kwargs"]["class_weights"] is not None:
            self.load_weights(fold)

        if subset_frac is None:
            subset_frac = self.model_args["data_kwargs"]["subset_proportion"]
        else:
            subset_frac = subset_frac

        loaded = data_loader(fold, subset_frac)

        loaded["packages"] = {
            "mod_args": self.model_args,
            "py_packs": {
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
                "pandas": str(pd.__version__),
            },
            "fold": fold,
        }

        self.inference_data["X"] = loaded["X"]
        self.inference_data["y"] = loaded["Y"]
        if loaded["we"] is None:
            self.get_vocab()
        else:
            self.inference_data["word_embedding"] = loaded["we"]
            self.dict_maps["id2word"] = loaded["id2word"]

        self.dict_maps["id2label"] = loaded["id2label"]

        self.metadata["metadata"] = loaded["metadata"]

        self.metadata["packages"] = loaded["packages"]

        self.num_classes = {t: len(self.dict_maps["id2label"][t].keys()) for t in self.tasks}

    def load_from_saved(self, fold: int, subset_frac: float = None) -> dict:
        """
        Load data files.

        Arguments:
            fold (int): Fold number. Should always be 0 for now.
            subset_frac (float): Proportion of data to load.

        Post-condition:
            Modifies self.splits in-place.
        """

        loaded_data = {"metadata": {}, "X": {}, "Y": {}}
        data_path = self.model_args["data_kwargs"]["data_path"]

        with open(
            os.path.join(data_path, "id2labels_fold" + str(fold) + ".json"), "r", encoding="utf-8"
        ) as f:
            tmp = json.load(f)

        loaded_data["id2label"] = {
            task: {int(k): str(v) for k, v in labels.items()}
            for task, labels in tmp.items()
            if task in self.tasks
        }

        df = pd.read_csv(
            os.path.join(data_path, "data_fold" + str(fold) + ".csv"),
            dtype=str,
            engine="c",
            memory_map=True,
        )

        # check if data has 'split' column, if not, it is added on line 320
        # if not, this is added for inference loaders
        try:
            self.splits = list(set(df["split"].values))
        except KeyError:
            self.splits = []

        if len(self.splits) == 3:
            self.splits = ["train", "test", "val"]
        elif len(self.splits) == 2:
            if "val" in self.splits:
                self.splits = sorted(self.splits)
            else:
                self.splits = ["train", "test"]
        elif len(self.splits) == 0:
            self.splits = ["test"]
            df.insert(df.shape[1], column="split", value="test")

        for split in self.splits:
            loaded_data["X"][split] = df[df["split"] == split]["X"].apply(
                lambda x: np.array(json.loads(x), dtype=np.int32)
            )
            loaded_data["Y"][split] = df[df["split"] == split][sorted(loaded_data["id2label"].keys())]
            loaded_data["metadata"][split] = df[df["split"] == split][
                [v for v in df.columns if v not in ["X", *loaded_data["id2label"].keys()]]
            ]
            # data_tasks = set([v for v in df.columns if v in loaded_data['id2label'].keys()])
            # if len(data_tasks) > 0:
            #     loaded_data['Y'][split] = df[df['split'] == split][sorted(loaded_data['id2label'].keys())]

        if subset_frac < 1:
            rng = np.random.default_rng(self.model_args["train_kwargs"]["random_seed"])
            for split in self.splits:
                data_size = len(loaded_data["X"][split])
                idxs = rng.choice(data_size, size=math.ceil(data_size * subset_frac), replace=False)
                loaded_data["X"][split] = loaded_data["X"][split].loc[loaded_data["X"][split].index[idxs]]
                loaded_data["Y"][split] = loaded_data["Y"][split].loc[loaded_data["Y"][split].index[idxs]]
                loaded_data["metadata"][split] = loaded_data["metadata"][split].loc[
                    loaded_data["metadata"][split].index[idxs]
                ]

        if os.path.exists(os.path.join(data_path, "word_embeds_fold" + str(fold) + ".npy")):
            loaded_data["we"] = np.load(os.path.join(data_path, "word_embeds_fold" + str(fold) + ".npy"))
        else:
            loaded_data["we"] = None

        if os.path.exists(os.path.join(data_path, "id2word_fold" + str(fold) + ".json")):
            with open(
                os.path.join(data_path, "id2word_fold" + str(fold) + ".json"), "r", encoding="utf-8"
            ) as f:
                tmp = json.load(f)
            loaded_data["id2word"] = {int(k): str(v) for k, v in tmp.items()}
        else:
            loaded_data["id2word"] = None

        return loaded_data

    def get_vocab(self):
        """
        Get the vocab and word embeddings from tokenized data.
        """

        embedding_dim = 300

        X = self.inference_data["X"]["train"]

        if "val" in self.inference_data["X"].keys():
            X = pd.concat([X, self.inference_data["X"]["val"]])

        # find all unique tokens
        s1 = set(X.iloc[0])
        for x in X:
            s1 = set(x).union(s1)

        vocab_len = np.max(list(s1))
        unk = vocab_len + 1
        self.inference_data["X"]["test"].apply(lambda d: word2int(d, s1))
        rng = np.random.default_rng(self.model_args["train_kwargs"]["random_seed"])
        unk_embed = rng.normal(size=(1, embedding_dim), scale=0.1)
        random_embeds = rng.standard_normal(size=(vocab_len, embedding_dim), dtype=np.float32) * 0.1
        self.inference_data["word_embedding"] = np.concatenate(
            (np.zeros(shape=(1, embedding_dim)), random_embeds, unk_embed), axis=0
        )

    def convert_y(self):
        """
        Add task unknown labels to Y and map values to integers for inference.

        Post-condition:
            The data frame with the output, the ys, is modified in place by this function.
            It maps the string values to ints for inference, i.e., C50 -> 48 for the site task.

        Note: If loading data separately from creating torch dataloaders,
            this function should be called if you want ints and not strings.
        """

        missing_tasks = [
            v
            for v in self.inference_data["y"]["train"].columns
            if v not in self.model_args["task_unks"].keys()
        ]

        if missing_tasks != []:
            raise exceptions.ParamError(
                f'the tasks {",".join(missing_tasks)} are missing from'
                + "task_unks in the the model_args.yml file"
            )
        known_labels = {}
        label2id = {}

        for task in self.dict_maps["id2label"].keys():
            known_labels[task] = [str(v) for v in self.dict_maps["id2label"][task].values()]
            if self.model_args["task_unks"][task] not in known_labels[task]:
                raise exceptions.ParamError(
                    f"for task {task} the task_unks "
                    + f'{self.model_args["task_unks"][task]} '
                    + "is not in the mapped labels"
                )

        for task in self.dict_maps["id2label"].keys():
            label2id[task] = {v: k for k, v in self.dict_maps["id2label"][task].items()}
            label_dict = LabelDict(label2id[task], self.model_args["task_unks"][task])
            for split in self.splits:
                self.inference_data["y"][split][task] = self.inference_data["y"][split][task].map(
                    label_dict
                )

    def load_weights(self, fold):
        """
        Loads class weights from pickle file or dict in model_args file.

        Args:
            fold (int): Data fold to be loaded

        """

        if (self.model_args["train_kwargs"]["class_weights"] is not None) and self.model_args[
            "abstain_kwargs"
        ]["abstain_flag"]:
            raise exceptions.ParamError("Class weights cannot be used with dac or ntask")

        data_path = self.model_args["data_kwargs"]["data_path"]

        with open(
            os.path.join(data_path, "id2labels_fold" + str(fold) + ".json"), "r", encoding="utf-8"
        ) as f:
            tmp = json.load(f)
        id2label = {task: {int(k): str(v) for k, v in labels.items()} for task, labels in tmp.items()}

        if isinstance(self.model_args["train_kwargs"]["class_weights"], dict):
            self.weights = self.model_args["train_kwargs"]["class_weights"]

        elif isinstance(self.model_args["train_kwargs"]["class_weights"], str):
            path = self.model_args["train_kwargs"]["class_weights"]
            if os.path.exists(path):
                with open(path, "rb") as f_in:
                    self.weights = pickle.load(f_in)
            else:
                raise exceptions.ParamError("Class weights path does not exist.")
        elif self.model_args["train_kwargs"]["class_weights"] is None:
            self.weights = None
        else:
            raise exceptions.ParamError(
                "Class weights must be dict, point to relevant .pickle file, or be None."
            )

        if self.weights is not None:
            # check that all tasks have class weight
            if len(id2label.keys()) > 1:
                raise exceptions.ParamError("Class weights for multi-task is not yet implemented.")
            for task in self.tasks:
                if task not in self.weights.keys():
                    raise exceptions.ParamError(
                        f"Class weights for task {task} not specified "
                        + f'For unweighted classes specify as "{task}: '
                        + f'None" in {self.model_args["train_kwargs"]["class_weights"]}'
                    )

            for task in self.weights:
                # if there is only a single task, weights may be none
                if len(self.weights) == 1:
                    if not (isinstance(self.weights[task], list) or self.weights[task] is None):
                        raise exceptions.ParamError("Weights for single task must be a list or None.")

                # if there are multiple tasks, weights should be lists.
                elif len(self.weights) > 1:
                    if self.weights[task] is None:
                        raise exceptions.ParamError("Weights for multiple task must be lists.")

                # check that all class weight lists provided are the right length
                if isinstance(self.weights[task], list):
                    n_weights = len(self.weights[task])
                    if n_weights != len(id2label[task].values()):
                        raise exceptions.ParamError(
                            "Number of weights must be equal to the "
                            + "number of classes in each task. Task "
                            + f"{task} should have {len(id2label[task].values())} values"
                        )

    def make_torch_dataloaders(
        self,
        doc_embeds=None,
        switch_rate: float = 0.0,
        reproducible: bool = False,
        shuffle_data: bool = True,
        seed: int = None,
        clc_flag: bool = False,
        clc_args: dict = None,
        inference: bool = False,
    ) -> dict:
        """Create torch DataLoader classes for training module.

            Returns dict of pytorch DataLoaders (train, val) for training/inference module.

        Params:
            doc_embeds (numpy.ndarray): document embeddings from base model
            switch_rate (float): proporiton of words in each doc to randomly flip
            reproducible (bool): set all random number generator seeds
            shuffle_data (bool): shuffle data in torch dataloaders
            seed (int): seed foor random number generators
            clc_flag (bool): running a clc model or not
            clc_args (dict): dictionary of kwds for clc model
            inference (bool): running in inference or training mode

        """
        if reproducible:
            gen = torch.Generator()
            gen.manual_seed(seed)
            worker = self.seed_worker
        else:
            worker = None
            gen = None

        if not clc_flag:
            doc_embeds = None
            clc_args = None

        vocab_size = self.inference_data["word_embedding"].shape[0]
        unk_tok = vocab_size - 1

        _transform = None
        if switch_rate > 0.0:
            _transform = AddNoise(
                unk_tok, self.model_args["train_kwargs"]["doc_max_len"], vocab_size, switch_rate, seed
            )

        # num multiprocessing workers for DataLoaders, 4 * num_gpus
        num_workers = 4
        print(f"Num workers: {num_workers}, reproducible: {self.model_args['data_kwargs']['reproducible']}")

        pin_mem = bool(torch.cuda.is_available())

        self.convert_y()

        if inference:
            print("\nSetting up inference loaders\n")
            loaders = self._inference_loader(
                doc_embeds,
                shuffle_data=shuffle_data,
                batch_size=self.model_args["train_kwargs"]["batch_per_gpu"],
                clc_flag=clc_flag,
                clc_args=clc_args,
                worker=worker,
                random_num_gen=gen,
                num_workers=num_workers,
                pin_mem=pin_mem,
            )
        else:
            loaders = self._training_loader(
                transform=_transform,
                shuffle_data=shuffle_data,
                pin_mem=pin_mem,
                num_workers=num_workers,
                worker=worker,
                random_num_gen=gen,
            )
        return loaders

    def _training_loader(
        self,
        transform=None,
        shuffle_data: bool = False,
        pin_mem: bool = False,
        num_workers: int = 0,
        worker=None,
        random_num_gen=None,
    ) -> dict:
        """Create dataloaders for training step.

             Returns dict of pytorch DataLoaders (train, val) for training module.

        Params:
            transform (class): class which randomly flips tokens in training set to
                prevent overfitting
            shuffle_data (bool): shuffle data in torch dataloaders
            pin_mem (bool): pins gpu memory if running with gpu enabled
            num_workers (int): number of multiprocessing workers for DataLoader
            reproducible (bool): set all random number generator seeds
            worker (function): creates random number generator independently for each process
            random_num_gen (function): random number generator for each process

        """

        loaders = {}

        train_data = PathReports(
            self.inference_data["X"]["train"],
            self.inference_data["y"]["train"],
            tasks=self.model_args["data_kwargs"]["tasks"],
            label_encoders=self.dict_maps["id2label"],
            max_len=self.model_args["train_kwargs"]["doc_max_len"],
            transform=transform,
        )
        loaders["train"] = DataLoader(
            train_data,
            batch_size=self.model_args["train_kwargs"]["batch_per_gpu"],
            shuffle=shuffle_data,
            pin_memory=pin_mem,
            num_workers=num_workers,
            worker_init_fn=worker,
            generator=rng,
        )
        self.train_size = len(train_data)

        if "val" in self.splits:
            val_data = PathReports(
                self.inference_data["X"]["val"],
                self.inference_data["y"]["val"],
                tasks=self.model_args["data_kwargs"]["tasks"],
                label_encoders=self.dict_maps["id2label"],
                max_len=self.model_args["train_kwargs"]["doc_max_len"],
                transform=None,
            )
            loaders["val"] = DataLoader(
                val_data,
                batch_size=self.model_args["train_kwargs"]["batch_per_gpu"],
                shuffle=shuffle_data,
                pin_memory=pin_mem,
                num_workers=num_workers,
                worker_init_fn=worker,
                generator=rng,
            )
            self.val_size = len(val_data)
        else:
            loaders["val"] = None
            self.val_size = 0

        print(f"Training on {self.train_size} validate on {self.val_size}")

        return loaders

    def _inference_loader(
        self,
        doc_embeds=None,
        shuffle_data: bool = True,
        batch_size: int = 128,
        clc_flag: bool = False,
        clc_args: dict = None,
        worker=None,
        random_num_gen=None,
        num_workers: int = 0,
        pin_mem: bool = False,
    ) -> dict:
        """Create torch DataLoader classes for training module.

            Returns dict of pytorch DataLoaders for inference step.

        Params:
            doc_embeds (numpy.ndarray): document embeddings from base model
            shuffle_data (bool): shuffle data in torch dataloaders
            batch_size (int): batch size for inference
            clc_flag (bool): are we running a clc model?
            clc_args (dict): dict of clc_args
            worker (function): creates random number generator independently for each process
            random_num_gen (function): random number generator for each process
            num_workers (int): number of multiprocessing workers for DataLoader
            pin_mem (bool): pins gpu memory if running with gpu enabled

        """
        if clc_flag and clc_args is None:
            raise exceptions.ParamError(
                """Clc flag cannot be true without clc_args.
                                        Pass valid clc_args for inference."""
            )

        loaders = {}

        self.convert_y()

        if clc_flag:
            X = {}
            idxs = {}
            y = {}

            for split in self.splits:
                y[split] = {task: doc_embeds[split]["y"][task] for task in self.tasks}
                X[split] = doc_embeds[split]["X"]
                idxs[split] = doc_embeds[split]["index"]

                data = GroupedCases(
                    np.array(X[split]),
                    y[split],
                    np.array(idxs[split]),
                    self.model_args["data_kwargs"]["tasks"],
                    self.metadata["metadata"][split],
                    exclude_single=clc_args["data_kwargs"]["exclude_single"],
                    shuffle_case_order=clc_args["data_kwargs"]["shuffle_case_order"],
                    split_by_tumor_id=clc_args["data_kwargs"]["split_by_tumorid"],
                )

                loaders[split] = DataLoader(
                    data,
                    batch_size=batch_size,
                    shuffle=False,
                    worker_init_fn=worker,
                    generator=random_num_gen,
                )
        else:
            for split in self.splits:
                data = PathReports(
                    self.inference_data["X"][split],
                    self.inference_data["y"][split],
                    tasks=self.model_args["data_kwargs"]["tasks"],
                    label_encoders=self.dict_maps["id2label"],
                    max_len=self.model_args["train_kwargs"]["doc_max_len"],
                    transform=None,
                )
                loaders[split] = DataLoader(
                    data,
                    batch_size=self.model_args["train_kwargs"]["batch_per_gpu"],
                    shuffle=shuffle_data,
                    pin_memory=pin_mem,
                    num_workers=num_workers,
                    worker_init_fn=worker,
                    generator=random_num_gen,
                )

        return loaders

    def make_grouped_cases(self, doc_embeds, clc_args, reproducible=True, seed: int = None):
        """Created GroupedCases class for torch DataLoaders.

        Params:
            doc_embeds (numpy.ndarray): document embeddings from base model
            clc_args (dict): dict of clc_args
            reproducible (bool): set all random number generator seeds
            seed (int): seed for random number generator

        """

        if reproducible:
            gen = torch.Generator()
            gen.manual_seed(seed)
            worker = self.seed_worker
        else:
            worker = None
            gen = None

        datasets = {
            split: GroupedCases(
                doc_embeds[split]["X"],
                doc_embeds[split]["y"],
                doc_embeds[split]["index"],
                self.model_args["data_kwargs"]["tasks"],
                self.metadata["metadata"][split],
                exclude_single=clc_args["data_kwargs"]["exclude_single"],
                shuffle_case_order=clc_args["data_kwargs"]["shuffle_case_order"],
                split_by_tumor_id=clc_args["data_kwargs"]["split_by_tumorid"],
            )
            for split in self.splits
        }

        self.grouped_cases = {
            split: DataLoader(
                datasets[split],
                batch_size=clc_args["train_kwargs"]["batch_per_gpu"],
                shuffle=False,
                worker_init_fn=worker,
                generator=gen,
            )
            for split in self.splits
        }

    @staticmethod
    def seed_worker(worker_id):
        """Set random seed for everything."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class AddNoise:
    """
    optional transform object for PathReports dataset
      - adds random amount of padding at front of document using unk_tok to
        reduce hisan overfitting
      - randomly replaces words with randomly selected other words to reduce
        overfitting

    parameters:
        unk_token: int
            integer mapping for unknown tokens
        max_pad_len: int
            maximum amount of padding at front of document
        vocab_size: int
            size of vocabulary matrix or
            maximum integer value to use when randomly replacing word tokens
        switch_rate: float (default: 0.1)
            percentage of words to randomly replace with random tokens
        seed - int: seed for random number generator
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
        r_idx = self.rng.choice(np.arange(len(doc)), size=int(len(doc) * self.switch_rate), replace=False)
        r_voc = self.rng.integers(1, self.vocab_size, r_idx.shape[0])
        doc = np.array(doc)
        doc[r_idx] = r_voc
        return doc


class PathReports(Dataset):
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

    def __init__(self, X, Y, tasks, label_encoders, max_len=3000, transform=None, multilabel=False):
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
                if f"abs_{task}" in le:
                    del le[f"abs_{task}"]
                self.num_classes[task] = len(le)
                self.ys[task] = y

                if self.multilabel:
                    y_onehot = np.zeros((len(y), len(le)), dtype=np.int16)
                    y_onehot[np.arange(len(y)), y] = 1
                    self.ys_onehot[task] = y_onehot

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        """Get PathReports item.

        Returns dict with keys:
            X (torch.tensor): tensor of ints
            y_task (int): class labels
            idxs (int): index in original data to match up with metadata

        """
        doc = self.X.iat[idx]
        if self.transform is not None:
            doc = self.transform(doc)
        array = np.zeros(self.max_len, dtype=np.int32)
        doc = doc[: self.max_len]
        array[: doc.shape[0]] = doc
        sample = {
            "X": torch.tensor(array, dtype=torch.long),
            "index": self.X.index[idx],
        }  # indexing allows us to keep track of the metadata associated with each X
        if not self.pred_only:
            for _, task in enumerate(self.tasks):
                if self.multilabel:
                    y = self.ys_onehot[task][idx]
                    sample[f"y_{task}"] = torch.tensor(y, dtype=torch.float)
                else:
                    y = self.ys[task][idx]
                    sample[f"y_{task}"] = torch.tensor(y, dtype=torch.long)
        return sample


class GroupedCases(Dataset):
    """Create grouped cases for torch DataLoaders.

    args:
        doc_embeds (torch.tensor): document embeddings from trained model as np.ndarray
        Y (dict): dict of integer Y values, keys are the splits
        tasks (list): list of tasks
        metadata (dict): dict of model metadata
        exclude_single (bool): are we omitting sinlge cases, default is True
        shuffle_case_order (bool):shuffle cases, default is True
        split_by_tumor_id (bool): split the cases by tumorId, default is True


    """

    def __init__(self, doc_embeds, Y, idxs, tasks, metadata, exclude_single=True, shuffle_case_order=True):
        """Class for grouping cases for clc."""
        self.embed_size = doc_embeds.shape[1]
        self.tasks = tasks
        self.shuffle_case_order = shuffle_case_order
        self.label_encoders = {}  # label_encoders
        self.grouped_X = []
        self.grouped_y = {task: [] for task in self.tasks}
        self.new_idx = []

        if split_by_tumor_id:
            metadata["uid"] = (
                metadata["registryId"] + metadata["patientId"].astype(str) + metadata["tumorId"].astype(str)
            )
        else:
            metadata["uid"] = metadata["registryId"] + metadata["patientId"].astype(str)
        uids = metadata["uid"].tolist()
        metadata_idxs = metadata.index.tolist()

        uid_pl = pl.DataFrame({"index": metadata_idxs, "uid": uids}).with_columns(
            pl.col("index").cast(pl.Int32)
        )

        self.max_seq_len = uid_pl.groupby("uid").count().max().select("count").item()

        # num docs x 400 (hisan) or 900 (cnn)
        X_pl = pl.Series("doc_embeds", doc_embeds)
        # dict of numpy arrays, each 1 x num docs, keys are tasks
        y_pl = pl.from_dict(Y)
        # numpy array, idxs.shape[0] = num docs
        idx_pl = pl.Series("index", idxs)
        df_pl = pl.DataFrame([idx_pl, X_pl]).hstack(y_pl)

        pl_cols = list(self.tasks)
        pl_cols.append("index")
        pl_cols.append("doc_embeds")

        groups_pl = (
            uid_pl.join(df_pl, on="index", how="inner")
            .groupby(by="uid", maintain_order=True)
            .agg([pl.col(col) for col in pl_cols])
        )
        del pl_cols[-2:]

        grouped_X = groups_pl.select("doc_embeds").to_series().to_list()
        # Xs are doc embeddings from base model
        self.grouped_X = []
        self.lens = []

        for X in grouped_X:
            # preallocate max group len, torch doesn't like ragged tensors
            # the padding is masked out during training/inference
            blank = torch.zeros((self.max_seq_len, self.embed_size), dtype=torch.float32)
            blank[: len(X), :] = torch.tensor(X)
            self.grouped_X.append(blank)
            self.lens.append(len(X))

        # indices in original data to match up with metadata
        self.new_idx = groups_pl.select("index").to_series()
        # grouped ys from data
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
        sample["len"] = _len
        # pad to max sequence length, then mask to length of
        # an individual sequence
        y_array = torch.zeros(self.max_seq_len, dtype=torch.long)
        for task in self.tasks:
            y_array[:_len] = torch.from_numpy(ys[task])
            sample[f"y_{task}"] = y_array.clone()

        idx_array = torch.zeros(self.max_seq_len, dtype=torch.int)
        idx_array[:_len] = torch.tensor(self.new_idx[idx])
        sample["index"] = idx_array

        return sample
