"""

Module for loading pre-generated Bardi or mod_pipeline data.

"""
import json
import math
import os
import pickle
import random
from re import match
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from fresco.validate import exceptions


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


class DataHandler:
    """Class for loading data.

    Attributes:
        data_source - str: defines how data was generated, needs to be
            'pre-generated', 'pipeline', or 'official' (not implemented)
        model_args - dict: keywords necessary to load data, build, and train a model_args
        cache_class - list: the CachedClass will be passed through all
            modules and keep track of the pipeline arguements
        clc_flag - bool: flag for running clc or standard model

    Note: the present implementation is only for 1 fold. We presently cannot
    generate more than one fold of data.

    """

    def __init__(self, data_source: str, model_args: dict, cache_class: list, clc_flag: bool = False):
        self.data_source = data_source
        self.model_args = model_args
        self.cache_class = cache_class

        try:
            self.data_pipeline = self.model_args["data_kwargs"]["data_pipeline"]
        except KeyError:
            self.data_pipeline = "mod_pipeline"

        print(f"Loading data from {self.model_args['data_kwargs']['data_path']}")

        self.inference_data = {}

        self.dict_maps = {}

        self.metadata = {"metadata": None, "packages": None, "query": None, "schema": None, "commits": None}

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

    def load_folds(self, fold: int = 0, subset_frac: Optional[float] = None):
        """Load data for each fold in the dataset.

        Parameters:
            fold - int: which data fold to load
            subset_frac - float: what proportion of the data to load

        Pre-condition: self.__init__called and model_args is not None
        Post-condition: class attributes populated

        Case level context model will load fold 0 by default, see run_clc.py line 225.

        """

        if self.data_source == "pre-generated":
            data_loader = self.load_from_saved
        else:
            data_loader = self.load_from_cache

        if fold is None:
            fold = self.model_args["data_kwargs"]["fold_number"]

        if self.model_args["train_kwargs"]["class_weights"] is not None:
            self.load_weights(fold)

        if subset_frac is None:
            _subset_frac = self.model_args["data_kwargs"]["subset_proportion"]
        else:
            _subset_frac = 1.0
        loaded = data_loader(fold, _subset_frac)

        # need to check model_args tasks agree with id2label tasks and y_task

        try:
            model_metadata = loaded["pipeline_metadata"]["model_metadata"]
        except KeyError:
            model_metadata = None

        loaded["packages"] = {
            "mod_args": self.model_args,
            "py_packs": {
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
                "pandas": str(pd.__version__),
            },
            "model_metadata": model_metadata,
            "fold": fold,
        }

        self.inference_data["X"] = loaded["X"]
        self.inference_data["y"] = loaded["Y"]
        self.metadata["metadata"] = loaded["metadata"]
        self.inference_data["word_embedding"] = loaded["we"]
        self.dict_maps["id2label"] = loaded["id2label"]
        self.dict_maps["id2word"] = loaded["id2word"]

        self.metadata["pipeline_metadata"] = loaded["pipeline_metadata"]
        self.metadata["packages"] = loaded["packages"]

        self.num_classes = {t: len(self.dict_maps["id2label"][t].keys()) for t in self.tasks}

    def load_from_saved(self, fold: int, subset_frac: float = 1.0) -> dict:
        """Load data files.

        Arguments:
            fold - int: fold number, should always be 0 for now
            subset_frac - float: what proportion of the data to load

        Post-condition:
            Modifies self.splits in-place
        """

        data_path = self.model_args["data_kwargs"]["data_path"]

        # Creating a skeleton for loaded data
        loaded_data = {"metadata": {}, "X": {}, "Y": {}, "pipeline_metadata": {}}

        # Dictionaries are passed by reference so no need to set variables
        # The load methods will modify the dictionary in place
        if self.data_pipeline == "mod_pipeline":
            self.load_mod_pipeline_data(loaded_data, data_path, fold, subset_frac)
        elif self.data_pipeline == "bardi":
            self.load_bardi_data(loaded_data, data_path, subset_frac)

        return loaded_data

    def load_bardi_data(self, loaded_data: dict, data_path: str, subset_frac: float) -> None:
        """Load data that has been preprocessed with a Bardi pipeline

        Arguments:
            loaded_data: dictionary reference
                reference to the loaded_data dictionary
            data_path: str
                path to directory containing files for data, label mapping,
                vocab, embeddings, and metadata
            subset_frac: float
                proportion of data to load. used for debugging.
        """
        # Load data into a Pandas DataFrame
        data_files = self.model_args["data_kwargs"]["data_files"]["data"]
        data_file_paths = [os.path.join(data_path, f) for f in data_files]
        data = pd.concat(
            pd.read_parquet(path=data_file_path, engine="pyarrow") for data_file_path in data_file_paths
        )

        # Set the splits found in the data
        # Confirm they are a subset of the split values that FrESCO can handle
        self.splits = list(set(data["split"].values))
        if not set(self.splits).issubset(set(["train", "test", "val"])):
            raise ValueError(
                "The data provided split values that FrESCO is "
                "not designed to handle. Currently, it can only "
                'accept split values in ["train", "test", "val"]'
            )

        # Load and sort the distinct label names from the label mapping file
        id_to_label_file_name = self.model_args["data_kwargs"]["data_files"]["id_to_label_mapping"]
        id_to_label_file_path = os.path.join(data_path, id_to_label_file_name)
        with open(id_to_label_file_path, "r", encoding="utf-8") as f:
            id_to_label_mappings = json.load(f)
        data_tasks = sorted(set(id_to_label_mappings.keys()))

        # Load the id to label mappings to the loaded_data dictionary
        # JSON format requires that keys are strings. The layout of the label mapping
        # has ids as keys, but the ids need to be integers when used. So,
        # each task's individual mapping needs to be gone through and have the
        # ids cast back to an integer
        loaded_data["id2label"] = {
            data_task: {int(label_id): str(label) for label_id, label in id_to_label_mapping.items()}
            for data_task, id_to_label_mapping in id_to_label_mappings.items()
        }

        # Handle subset_proportion
        # If subset_proportion is less than 1.0 then a random subset of the overall data
        # will be selected
        if subset_frac < 1.0:
            random_seed = self.model_args["train_kwargs"]["random_seed"]
            rng = np.random.default_rng(random_seed)  # numpy random number generator
            subset_splits = []
            for split in self.splits:
                split_data = (data[data["split"] == split]).reset_index()
                data_size = len(split_data)  # number of rows in data
                idxs = rng.choice(
                    data_size, size=math.ceil(data_size * subset_frac), replace=False
                )  # randomly generate a subset of indices
                subset_split_data = (split_data.loc[idxs]).set_index(
                    "index"
                )  # Shrink data to subset if applicable
                subset_splits.append(subset_split_data)
            data = pd.concat(subset_splits)

        # Define which columns are 'metadata' columns (not encoded text or label cols)
        metadata_cols = [col for col in data.columns if col not in ["X", *data_tasks]]

        # Assign data to metadata, X, and Y keys of loaded_data dict organized by split
        for split in self.splits:
            # encoded text column
            loaded_data["X"][split] = data[data["split"] == split]["X"]
            # label columns
            loaded_data["Y"][split] = data[data["split"] == split][data_tasks]
            # other cols
            loaded_data["metadata"][split] = data[data["split"] == split][metadata_cols]

        # Load word embeddings
        embeddings_file_name = self.model_args["data_kwargs"]["data_files"]["embeddings"]
        embeddings_file_path = os.path.join(data_path, embeddings_file_name)
        loaded_data["we"] = np.load(embeddings_file_path)

        # Load vocabulary to the loaded_data dictionary
        # JSON format requires that keys are strings. The layout of the vocab
        # has encodings as keys, but the encodings need to be integers when used. So,
        # the vocab needs to be gone through and have the encodings cast back to an integer
        vocab_file_name = self.model_args["data_kwargs"]["data_files"]["vocab"]
        vocab_file_path = os.path.join(data_path, vocab_file_name)
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        loaded_data["id2word"] = {int(token_id): str(token) for token_id, token in vocab.items()}

        # Load pipeline metadata for data and model provenance
        # Metadata files are specified in the data_kwargs section of model_args as a list
        # of files. Files of type txt and json are opened and the contents are added to the
        # loaded_data dictionary under a parent key 'pipeline_metadata' and child key
        # <filename>. For any other file types, the path to the file is added.
        metadata_file_names = self.model_args["data_kwargs"]["data_files"]["metadata"]
        metadata_files = {
            metadata_file_name: os.path.join(data_path, metadata_file_name)
            for metadata_file_name in metadata_file_names
        }
        for metadata_file_name in metadata_files.keys():
            metadata_file_path = metadata_files[metadata_file_name]
            if match(r".*\.json", metadata_file_name):
                with open(metadata_file_path, "r", encoding="utf-8") as f:
                    loaded_data["pipeline_metadata"][metadata_file_name] = json.load(f)
            elif match(r".*\.txt", metadata_file_name):
                with open(metadata_file_path, "r", encoding="utf-8") as f:
                    loaded_data["pipeline_metadata"][metadata_file_name] = f.read()
            else:
                loaded_data["pipeline_metadata"][metadata_file_name] = metadata_file_path

    # def load_mod_pipeline_data(
    #     self, loaded_data: dict, data_path: str, fold: int, subset_frac: float
    # ) -> None:
        with open(
            os.path.join(data_path, "id2labels_fold" + str(fold) + ".json"), "r", encoding="utf-8"
        ) as f:
            tmp = json.load(f)

        loaded_data["id2label"] = {
            task: {int(k): str(v) for k, v in labels.items()} for task, labels in tmp.items()
        }

        df = pd.read_csv(
            os.path.join(data_path, "data_fold" + str(fold) + ".csv"),
            dtype=str,
            engine="c",
            memory_map=True,
        )

        # check if data has 'split' column, if not, it is added on line 320.
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

        loaded_data["we"] = np.load(os.path.join(data_path, "word_embeds_fold" + str(fold) + ".npy"))

        with open(
            os.path.join(data_path, "id2word_fold" + str(fold) + ".json"), "r", encoding="utf-8"
        ) as f:
            tmp = json.load(f)
        loaded_data["id2word"] = {int(k): str(v) for k, v in tmp.items()}

        with open(os.path.join(data_path, "metadata.json"), "r", encoding="utf-8") as f:
            loaded_data["pipeline_metadata"]["model_metadata"] = json.load(f)

        with open(os.path.join(data_path, "schema.json"), "r", encoding="utf-8") as f:
            loaded_data["pipeline_metadata"]["schema"] = json.load(f)

        with open(os.path.join(data_path, "query.txt"), "r", encoding="utf-8") as f:
            loaded_data["pipeline_metadata"]["query"] = f.read()

    def convert_y(self, inference: bool = False):
        """Add task unknown labels to Y and map values to integers for inference.

        Args:
            inference (bool): running in inference mode

        Post-condition:
            The data frame with the output, the ys, is modified in place by this function.
            It maps the string values to ints for inference, ie C50 -> 48 for the site task.

        Note: If loading data separate from creating torch dataloaders,
            this function should be called if you want ints and not strings.

        """
        missing_tasks = []

        if not inference:
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
        if (self.model_args["train_kwargs"]["class_weights"] is not None) and self.model_args[
            "abstain_kwargs"
        ]["abstain_flag"]:
            raise exceptions.ParamError("Class weights cannot be used with dac or ntask")

        data_path = self.model_args["data_kwargs"]["data_path"]

        if self.data_pipeline == "mod_pipeline":
            label_map = "id2labels_fold" + str(fold) + ".json"
        else:  # bardi data
            label_map = self.model_args["data_kwargs"]["data_files"]["id_to_label_mapping"]
        with open(os.path.join(data_path, label_map), "r", encoding="utf-8") as f:
            tmp = json.load(f)
        id2label = {task: {int(k): str(v) for k, v in labels.items()} for task, labels in tmp.items()}

        if isinstance(self.model_args["train_kwargs"]["class_weights"], dict):
            self.weights = self.model_args["train_kwargs"]["class_weights"]

        elif isinstance(self.model_args["train_kwargs"]["class_weights"], str):
            path = self.model_args["train_kwargs"]["class_weights"]
            print(f"Loading class weights from {path}")
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
            doc_embeds - numpy.ndarray: document embeddings from base model
            switch_rate - float: proporiton of words in each doc to ranndomly flip
            reproducible - bool: set all random number generator seeds
            shuffle_data - bool: shuffle data in torch dataloaders
            seed - int: seed foor random number generators
            clc_flag - bool: running a clc model or not
            clc_args - dict: dictionary of kwds for clc model
            inference - bool: running in inference or training mode

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

        _transform = None
        if switch_rate > 0.0:
            _transform = AddNoise(
                unk_tok, self.model_args["train_kwargs"]["doc_max_len"], vocab_size, switch_rate, seed
            )

        # num multiprocessing workers for DataLoaders, 4 * num_gpus
        num_workers = 4
        print(f"Num workers: {num_workers}, reproducible: {self.model_args['data_kwargs']['reproducible']}")

        pin_mem = bool(torch.cuda.is_available())

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
            transform - class: class which randomly flips tokens in training set to
                prevent overfitting
            shuffle_data - bool: shuffle data in torch dataloaders
            pin_mem - bool: pins gpu memory if running with gpu enabled
            num_workers - int: number of multiprocessing workers for DataLoader
            reproducible - bool: set all random number generator seeds
            worker - function: creates random number generator independently for each process
            random_num_gen - function: random number generator for each process 
        
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
            generator=random_num_gen,
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
                generator=random_num_gen,
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
            doc_embeds - numpy.ndarray: document embeddings from base model
            shuffle_data - bool: shuffle data in torch dataloaders
            batch_size - int: batch size for inference
            clc_flag - bool: are we running a clc model?
            clc_args - dict: dict of clc_args
            worker - function: creates random number generator independently for each process
            random_num_gen - function: random number generator for each process 
            num_workers - int: number of multiprocessing workers for DataLoader
            pin_mem - bool: pins gpu memory if running with gpu enabled

        """
        if clc_flag and clc_args is None:
            raise exceptions.ParamError(
                """Clc flag cannot be true without clc_args.
                                        Pass valid clc_args for inference."""
            )

        loaders = {}

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
                    data, batch_size=batch_size, shuffle=False, worker_init_fn=worker, generator=random_num_gen
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
                    generator=random_num_gen
                )

        return loaders

    def make_grouped_cases(self, doc_embeds, clc_args, reproducible=True, seed: int = None):
        """Created GroupedCases class for torch DataLoaders.
        
            Params:
                doc_embeds - numpy.ndarray: document embeddings from base model
                clc_args - dict: dict of clc_args
                reproducible - bool: set all random number generator seeds
                seed - int: seed for random number generator
        
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
    Torch dataloader class for cancer path reports from generate_data.py

    parameters:
      - X: pandas DataFrame of tokenized path report data, entries are
            numpy array
      - Y: pd.DataFrame
            dataframe ground truth values
      - tasks: list[string]
        list of tasks to generate labels for
      - label_encoders:
        dict (task:label encoders) to convert raw labels into integers
      - max_len: int (default: 3000)
        maximum length for document, should match value in model_args.yml
        longer documents will be cut, shorter documents will be 0-padded
      - transform: object (default: None)
        optional transform to apply to document tensors
        multilabel - bool: running a multilabel model or not?

    outputs per batch:
      - dict[str:torch.tensor]
        sample dictionary with following keys/vals:
          - 'X': torch.tensor (int) [max_len]
            document converted to integer word-mappings, 0-padded to max_len
          - 'y_%s % task': torch.tensor (int) [] or
                           torch.tensor (int) [num_classes]
            integer label for a given task if label encoders are used
            one hot vectors for a given task if label binarizers are used
            -'index': int of DataFrame index to match up with metadata stored in the original DataFrame
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
                X - tensor of ints
                y_task - class labels
                idxs - index in original data to match up with metadata
        
        """
        doc = self.X.iat[idx]
        if self.transform:
            doc = self.transform(doc)
        array = np.zeros(self.max_len, dtype=np.int32)
        doc = doc[: self.max_len]
        array[: doc.shape[0]] = doc
        sample = {"X": torch.tensor(array, dtype=torch.long), "index": self.X.index[idx]}
        # indexing allows us to keep track of the metadata associated with each doc
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
        doc_embeds - document embeddings from trained model as np.ndarray
        Y - dict of integer Y values, keys are the splits
        index - int of DataFrame index to match up with metadata stored in the original DataFrame
        tasks - list of tasks
        metadata - dict of model metadata
        exclude_single - are we omitting sinlge cases, default is True
        shuffle_case_order - shuffle cases, default is True
        split_by_tumor_id - split the cases by tumorId, default is True

    NOTE: arg exclude_single is presently unused. 10/23 - AS


    """

    def __init__(
        self,
        doc_embeds,
        Y,
        idxs,
        tasks,
        metadata,
        exclude_single=True,
        shuffle_case_order=True,
        split_by_tumor_id=True,
    ):
        """Class for grouping cases for clc."""
        self.embed_size = doc_embeds.shape[1]
        self.tasks = tasks
        self.shuffle_case_order = shuffle_case_order
        self.label_encoders = {}
        self.grouped_X = []
        self.grouped_y = {task: [] for task in self.tasks}
        self.new_idx = []

        registry = "_meta_registry"
        patient = "patient_id_number"
        tumor = "tumor_record_number"
       
        try:
            metadata["_meta_registry"]
        except KeyError:
            registry = "registryId"
            patient = "patientId"
            tumor = "tumorId"

        if split_by_tumor_id:
            metadata["uid"] = (
                metadata[registry] + metadata[patient].astype(str) + metadata[tumor].astype(str)
            )
        else:
            metadata["uid"] = metadata[registry] + metadata[patient].astype(str)
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
