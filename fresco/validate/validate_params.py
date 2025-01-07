"""

Module for sanity checking parameters for data and model building.

"""

import datetime
import json
import math
import os
from typing import Set

import yaml

from fresco.validate import exceptions


class ValidateParams:
    """Class to validate model-specific paramaters for MOSSAIC models.

    Args:
        cached_class (list): the CachedClass will be passed through all
            modules and keep track of the pipeline arguements
        cli_args: argparse list of command line args
        data_source (str): indicates where the data will come from,
        should be one of
            - pre-generated: data_args.yml will indicate the source.
            - pipeline:      the pipeline is used; uses cache_class
            - official:      the pipeline is used to generate "official" data; uses cache_class

    Pre-condition: cached_class is set in the calling function.

    Post-condition: model_args dict loaded and sanity checked.
    """

    def __init__(
        self,
        # cached_class: list,
        cli_args,
        data_source: str = "pre-generated",
        model_args: dict = None,
    ):
        if data_source in ["pre-generated", "pipeline", "official"]:
            self.data_source = data_source
            if data_source == "official":
                raise exceptions.ParamError(
                    "A run type of 'official' is not yet implemented."
                )
        else:
            raise exceptions.ParamError(
                "if run_type is not 'pre-generated', "
                + "use_pipeline.py must be run first."
            )

        if model_args is None:
            if len(cli_args.model_args) > 0:
                mod_args_file = cli_args.model_args
            else:
                mod_args_file = "model_args.yml"

            if os.path.isfile(mod_args_file):
                with open(mod_args_file, "r", encoding="utf-8") as f_in:
                    self.model_args = yaml.safe_load(f_in)
            else:
                raise exceptions.ParamError(
                    "within the Model_Suite the "
                    + "model_args.yml file is needed to set "
                    + "the model arguments"
                )
        else:
            self.model_args = model_args

        if (
            self.model_args["abstain_kwargs"]["ntask_flag"]
            and not self.model_args["abstain_kwargs"]["abstain_flag"]
        ):
            raise exceptions.ParamError("Ntask cannot be enabled without Abstention")

        if self.model_args["model_type"] not in ["mtcnn", "mthisan"]:
            raise exceptions.ParamError(
                "model type was not found "
                + "to be 'mtcnn' or 'mthisan'. "
                + "Currently these are the only expected options."
            )

        if self.model_args["save_name"] == "":
            self.save_name = f'model_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        else:
            self.save_name = self.model_args["save_name"]

        self.save_name = os.path.join("savedmodels", self.save_name)

        if not os.path.exists(os.path.dirname(self.save_name)):
            print(
                f"savepath {os.path.dirname(self.save_name)} does not exist, creating it"
            )
            os.makedirs(os.path.dirname(self.save_name))

        fold = self.model_args["data_kwargs"]["fold_number"]

        if len(cli_args.data_path) > 0:
            self.model_args["data_kwargs"]["data_path"] = cli_args.data_path

        if not os.path.exists(self.model_args["data_kwargs"]["data_path"]):
            raise exceptions.ParamError("User provided data path does not exist.")

        if isinstance(fold, int):
            self.model_args["data_kwargs"]["fold_number"] = fold
        else:
            raise exceptions.ParamError(
                "Model building does not presently support > 1 fold."
            )

        if (
            self.model_args["data_kwargs"]["subset_proportion"] > 1
            or self.model_args["data_kwargs"]["subset_proportion"] <= 0
        ):
            raise exceptions.ParamError(
                "subset proportion must be float value between 0 and 1."
            )

        # if cached_class[0]:
        #     self.model_args["train_kwargs"]["doc_max_len"] = cached_class[1].mod_args(
        #         "sequence_length"
        #     )
        #     self.save_name = os.path.join("Model_Suite", self.save_name)

        if not isinstance(self.model_args["data_kwargs"]["batch_per_gpu"], int):
            raise exceptions.ParamError("Batch size must be an int value.")
        if (
            self.model_args["data_kwargs"]["batch_per_gpu"] < 0
            or self.model_args["data_kwargs"]["batch_per_gpu"] > 2048
        ):
            raise exceptions.ParamError(
                "Batch size must be an int value between 1 and 2048."
            )

        if not isinstance(self.model_args["train_kwargs"]["mixed_precision"], bool):
            raise exceptions.ParamError("Mixed precision must be boolean.")

    def hisan_arg_check(self):
        """
        Check and modify hisan specific args.

        Parameters: none

        Pre-condtition: self.model_args is not None

        Post-condition:
            self.model_args['MTHiSAN_kwargs']['max_lines'] modified to be
            the ceiling of the  doc_max_len / max_words_per_line.

            self.model_args['train_kwargs']['doc_max_len'] modified to be
            max_words_per_line * max_lines
        """

        self.model_args["MTHiSAN_kwargs"]["max_lines"] = math.ceil(
            self.model_args["train_kwargs"]["doc_max_len"]
            / self.model_args["MTHiSAN_kwargs"]["max_words_per_line"]
        )

        self.model_args["train_kwargs"]["doc_max_len"] = (
            self.model_args["MTHiSAN_kwargs"]["max_words_per_line"]
            * self.model_args["MTHiSAN_kwargs"]["max_lines"]
        )

        if self.model_args["train_kwargs"]["class_weights"] is not None:
            self.check_weights()

    def mtcnn_arg_check(self):
        """Check the number of filters matchesthe number of windows."""
        if len(self.model_args["MTCNN_kwargs"]["num_filters"]) != len(
            self.model_args["MTCNN_kwargs"]["window_sizes"]
        ):
            raise exceptions.ParamError(
                "Number of filters must match the number of window_sizes."
            )

        if self.model_args["train_kwargs"]["class_weights"] is not None:
            self.check_weights()

    def check_data_train_args(self, from_pretrained: bool = False):
        """
        Verify arguements are appropriate for the chosen model options.

        Parameters: from_pretrained as bool, chekcing model args from a pretrained model,
            pretrained model args are different, some are copied from data_kwargs to train_kwargs

        Pre-condtition: self.model_args is not None
        Post- condition: self.model_args['train_kwargs']['doc_max_len'] is
            updated from the data_kwargs and
            'max_lines' is added to the hisan kw_args

        """
        schema = {
            "data_kwargs": [
                "doc_max_len",
                "tasks",
                "fold_number",
                "data_files",
                "data_path",
                # "mutual_info_filter",
                # "mutual_info_threshold",
                "subset_proportion",
                "add_noise",
                "add_noise_flag",
                # "multilabel",
                "random_seed",
                "batch_per_gpu",
                "reproducible",
            ],
            "MTHiSAN_kwargs": [
                "max_words_per_line",
                "att_heads",
                "att_dim_per_head",
                "att_dropout",
                "bag_of_embeddings",
                "embeddings_scale",
            ],
            "MTCNN_kwargs": [
                "window_sizes",
                "num_filters",
                "dropout",
                "bag_of_embeddings",
                "embeddings_scale",
            ],
            "train_kwargs": [
                "keywords",
                "max_epochs",
                "patience",
                "class_weights",
                "mixed_precision",
                "save_probs",
                "learning_rate",
                "beta_1",
                "beta_2"
            ],
        }

        if from_pretrained:
            schema["train_kwargs"] = [
                "batch_per_gpu",
                "class_weights",
                "doc_max_len",
                "keywords",
                "max_epochs",
                # "multilabel",
                "patience",
                "random_seed",
                "reproducible",
                "mixed_precision",
                "save_probs",
            ]
            if self.model_args["model_type"] == "mthisan":
                schema["MTHiSAN_kwargs"] = [
                    "max_words_per_line",
                    "max_lines",
                    "att_heads",
                    "att_dim_per_head",
                    "att_dropout",
                    "bag_of_embeddings",
                    "embeddings_scale",
                ]
            elif self.model_args["model_type"] == "mtcnn":
                schema["MTCNN_kwargs"] = [
                    "bag_of_embeddings",
                    "dropout",
                    "embeddings_scale",
                    "num_filters",
                    "window_sizes",
                ]

        model_kwds = [
            "MTCNN_kwargs",
            "MTHiSAN_kwargs",
            "Transformers_kwargs",
            "abstain_kwargs",
            "data_kwargs",
            "model_type",
            "save_name",
            "task_unks",
            "train_kwargs",
        ]

        if sorted(self.model_args.keys()) != model_kwds:
            print("\nReceived: ", sorted(self.model_args.keys()))
            print("Expected: ", model_kwds)
            raise exceptions.ParamError("model_arg keys do not match the schema")

        for kwrd, vals in schema.items():
            if kwrd == "abstain_kwargs":
                continue  # these are checked in a separate function
            if not set(vals).issubset(set(self.model_args[kwrd])):
                print("\nReceived: ", sorted(self.model_args[kwrd]))
                print("Expected: ", sorted(vals))
                raise exceptions.ParamError(
                    (f"model args {kwrd} does not have " + "the expected variables")
                )

        # copy data kwargs to train kwds
        copy_kwds = [
            "doc_max_len",
            "batch_per_gpu",
            "random_seed",
            # "multilabel",
            "reproducible",
        ]
        for word in copy_kwds:
            self.model_args["train_kwargs"].update(
                [(word, self.model_args["data_kwargs"][word])]
            )

    def check_abstain_args(self):
        """Verify keywords needed for abstention to work are present and valid."""
        abstain_kwargs = [
            "abstain_flag",
            "alphas",
            "max_abs",
            "min_acc",
            "abs_gain",
            "acc_gain",
            "alpha_scale",
            "tune_mode",
            "stop_limit",
            "stop_metric",
            "ntask_flag",
            "ntask_tasks",
            "ntask_alpha",
            "ntask_alpha_scale",
            "ntask_max_abs",
            "ntask_min_acc",
        ]

        if (
            self.model_args["train_kwargs"]["class_weights"] is not None
            and self.model_args["abstain_kwargs"]["abstain_flag"]
        ):
            raise exceptions.ParamError(
                "Class weighting is not presently implemented for the DAC."
            )

        if sorted(abstain_kwargs) != sorted(self.model_args["abstain_kwargs"]):
            print("\nReceived: ", sorted(self.model_args["abstain_kwargs"]))
            print("Expected: ", sorted(abstain_kwargs))
            raise exceptions.ParamError(
                ("model args abstain_kwargs does not have " + "the expected variables")
            )

        if set(self.model_args["abstain_kwargs"]["alphas"].keys()).isdisjoint(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Alpha tasks are not a subset of the data tasks."
            )

        if len(self.model_args["abstain_kwargs"]["alphas"]) > len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of abstain alphas is greater than number of tasks."
            )

        if len(self.model_args["abstain_kwargs"]["max_abs"]) != len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of max abstain rates is different than number of tasks."
            )

        if len(self.model_args["abstain_kwargs"]["min_acc"]) != len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of min acc rates is different than number of tasks."
            )

        if len(self.model_args["abstain_kwargs"]["alpha_scale"]) != len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of alpha scales is different than number of tasks."
            )

        if self.model_args["abstain_kwargs"]["ntask_flag"]:
            if set(self.model_args["abstain_kwargs"]["ntask_tasks"]).isdisjoint(
                self.model_args["data_kwargs"]["tasks"]
            ):
                raise exceptions.ParamError(
                    "Ntask tasks are not a subset of the data tasks."
                )

    def check_keyword_args(self):
        """Validate keyword args."""
        tasks = ["histology", "laterality", "site", "subsite", "behavior"]
        if not set(self.model_args["data_kwargs"]["tasks"]).issubset(tasks):
            raise exceptions.ParamError(
                "Keywords are only available for: "
                + "histology, laterality, site, subsite, behavior"
            )

    def check_weights(self):
        """Validate class weights path exists."""
        if isinstance(self.model_args["train_kwargs"]["class_weights"], str):
            path = self.model_args["train_kwargs"]["class_weights"]
            if not os.path.exists(path):
                raise exceptions.ParamError(
                    "Invalid path; please provide a valid path for " + "class weights"
                )

    def validate_tasks(self, data_tasks: Set[str]) -> None:
        """Confirm that tasks supplied in the id_to_label mapping aligns with tasks defined
        in the model_args

        Arguments:
            data_tasks: Set[str]
                the distinct set of tasks in the id_to_label mapping file

        Raises:
            Parameter Error if data tasks and model tasks do not align
        """
        model_tasks = set(self.model_args["data_kwargs"]["tasks"])
        if not model_tasks.issubset(data_tasks):
            raise exceptions.ParamError(
                (
                    "the tasks specified in the model_args file must "
                    "match the tasks supplied in the id_to_labels file."
                )
            )

    def validate_bardi_data_requirements(self, data_path: str) -> None:
        """Confirm that files specified in model_args exist

        Check the path of each file specified in model_args ensuring they exist.
        Open the id_to_label file and make sure the keys match the tasks specified
        in model_args

        Arguments:
            data_path: str
                path to a directory containing data files

        Returns:

        """

        # Confirm that data files were specified in model_args
        try:
            data_files = self.model_args["data_kwargs"]["data_files"]["data"]
        except KeyError:
            raise exceptions.ParamError("No data files were specified in model_args.")

        # Confirm that data files specified in model_args exist
        for file_name in data_files:
            file_path = os.path.join(data_path, file_name)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f"The file {file_path} specified in model_args could not be found."
                )

        # Load the set of tasks included in the label mapping and validate them
        # against tasks in model_args
        id_to_label_file_name = self.model_args["data_kwargs"]["data_files"][
            "id_to_label_mapping"
        ]
        id_to_label_file_path = os.path.join(data_path, id_to_label_file_name)
        if not os.path.isfile(id_to_label_file_path):
            raise FileNotFoundError(
                f"The file {id_to_label_file_path} specified in model_args could not be found."
            )
        with open(id_to_label_file_path, "r", encoding="utf-8") as f:
            id_to_label_mapping = json.load(f)
        data_tasks = set(id_to_label_mapping.keys())
        self.validate_tasks(data_tasks)

        # Confirm that a vocab exists
        vocab_file_name = self.model_args["data_kwargs"]["data_files"]["vocab"]
        vocab_file_path = os.path.join(data_path, vocab_file_name)
        if not os.path.isfile(vocab_file_path):
            raise FileNotFoundError(
                f"The file {vocab_file_path} specified in model_args could not be found."
            )

        # Confirm that the embeddings file exists
        embeddings_file_name = self.model_args["data_kwargs"]["data_files"][
            "embeddings"
        ]
        embeddings_file_path = os.path.join(data_path, vocab_file_name)
        if not os.path.isfile(embeddings_file_path):
            raise FileNotFoundError(
                f"The file {embeddings_file_path} specified in model_args could not be found."
            )

    def validate_mod_pipeline_data_requirements(self, data_path: str) -> None:
        """docstring"""

        fold = self.model_args["data_kwargs"]["fold_number"]
        data_files = [
            "data_fold.csv",
            "word_embeds_fold.npy",
            "id2labels_fold.json",
            "id2word_fold.json",
            "metadata.json",
            "schema.json",
            "query.txt",
        ]

        # Confirm that the required files are present in the supplied data path
        for file_name in data_files:
            file_path = os.path.join(
                data_path, file_name.replace("fold", f"fold{fold}")
            )
            if not os.path.isfile(file_path):
                raise exceptions.ParamError(f" the file {file_path} does not exist.")

        # Load the set of labels included in the id2labels file and validate them
        # against tasks in model_args
        id2label_path = os.path.join(data_path, f"id2labels_fold{str(fold)}.json")
        with open(id2label_path, "r", encoding="utf-8") as f:
            id2label_contents = json.load(f)
        data_tasks = set(id2label_contents.keys())
        self.validate_tasks(data_tasks)

    def check_data_files(self, data_path=None):
        """Verify the necessary data files exist and confirm that the contents
        of the id_to_labels file matches the model_args defined tasks

            Args:
                data_path: str, from argparser, optional path to dataset
                            setting data_path will override the path set in model_args.yml
        """

        # confirm the provided data_path exists
        if data_path:
            # user defined check
            if not os.path.exists(os.path.dirname(data_path)):
                raise exceptions.ParamError(
                    f"user defined data_path {data_path} does not exist, exiting"
                )
        else:
            # model_args defined check
            data_path = self.model_args["data_kwargs"]["data_path"]
            if not os.path.exists(os.path.dirname(data_path)):
                raise exceptions.ParamError(
                    f"model_args defined data_path {data_path} does not exist, exiting"
                )

        # Gather appropriate set of validation info for the pipeline utilized
        try:
            pipeline = self.model_args["data_kwargs"]["data_pipeline"]
        except KeyError:
            # defaulting to mod_pipeline validations to keep from breaking existing
            # workflows with existing mod_pipeline data
            pipeline = "mod_pipeline"
            print(
                "data_pipeline was not specified in model_args. "
                "FrESCO is defaulting to mod_pipeline specs. "
                "If you meant to specify GAuDI, please add "
                'data_pipeline: "gaudi" to data_kwargs.'
            )
        if pipeline == "mod_pipeline":
            self.validate_mod_pipeline_data_requirements(data_path)
        elif pipeline == "bardi":
            self.validate_bardi_data_requirements(data_path)
        else:
            raise exceptions.ParamError((f"data pipeline {pipeline} is not supported."))


class ValidateClcParams:
    """Class to validate model-specific paramaters for MOSSAIC models.

    Args:
        cached_class (list): the CachedClass will be passed through all
            modules and keep track of the pipeline arguements
        cli_args: argparse list of command line args
        data_source (str): indicates where the data will come from,
        should be one of
            - pre-generated: data_args.yml will indicate the source.
            - pipeline:      the pipeline is used; uses cache_class
            - official:      the pipeline is used to generate "official" data; uses cache_class

    Pre-condition: cached_class is set in the calling function.

    Post-condition: model_args dict loaded and sanity checked.
    """

    def __init__(
        self,
        # cached_class: list,
        cli_args,
        data_source: str = "pre-generated",
        model_args: dict = None,
    ):
        if data_source in ["pre-generated", "pipeline", "official"]:  #  or cached_class[0]:
            self.data_source = data_source
            if data_source == "official":
                raise exceptions.ParamError(
                    "A run type of 'official' is not yet implemented."
                )
        else:
            raise exceptions.ParamError(
                "if run_type is not 'pre-generated', "
                + "use_pipeline.py must be run first."
            )
        if model_args is None:
            if len(cli_args.model_args) > 0:
                mod_args_file = cli_args.model_args
            else:
                mod_args_file = "clc_args.yml"
            # if cached_class[0]:
            #     mod_args_file = os.path.join("Model_Suite", mod_args_file)

            if os.path.isfile(mod_args_file):
                with open(mod_args_file, "r", encoding="utf-8") as f_in:
                    self.model_args = yaml.safe_load(f_in)
            else:
                raise exceptions.ParamError(
                    "within the Model_Suite the "
                    + "clc_args.yml file is needed to set "
                    + "the model arguments"
                )
        else:
            self.model_args = model_args

        if (
            self.model_args["abstain_kwargs"]["ntask_flag"]
            and not self.model_args["abstain_kwargs"]["abstain_flag"]
        ):
            raise exceptions.ParamError("Ntask cannot be enables without Abstention")

        if self.model_args["save_name"] == "":
            self.save_name = f'model_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        else:
            self.save_name = self.model_args["save_name"]

        self.save_name = os.path.join("savedmodels", self.save_name)

        if not os.path.exists(os.path.dirname(self.save_name)):
            print(
                f"savepath {os.path.dirname(self.save_name)} does not exist, creating it"
            )
            os.makedirs(os.path.dirname(self.save_name))

        # if cached_class[0]:
        #     self.model_args["train_kwargs"]["doc_max_len"] = cached_class[1].mod_args(
        #         "sequence_length"
        #     )
        #     self.save_name = os.path.join("Model_Suite", self.save_name)

        if (
            self.model_args["data_kwargs"]["subset_proportion"] > 1
            or self.model_args["data_kwargs"]["subset_proportion"] <= 0
        ):
            raise exceptions.ParamError(
                "subset proportion must be float value between 0 and 1."
            )

        if not isinstance(self.model_args["train_kwargs"]["batch_per_gpu"], int):
            raise exceptions.ParamError("Batch size must be an int value.")

        if (
            self.model_args["train_kwargs"]["batch_per_gpu"] < 0
            or self.model_args["train_kwargs"]["batch_per_gpu"] > 2048
        ):
            raise exceptions.ParamError(
                "Batch size must be an int value between 1 and 2048."
            )

    def clc_arg_check(self):
        """
        Check and modify hisan specific args.

        Parameters: none

        Pre-condtition: self.model_args is not None

        Post-condition:
            self.model_args['MTHiSAN_kwargs']['max_lines'] modified to be
            the ceiling of the  doc_max_len / max_words_per_line.

            self.model_args['train_kwargs']['doc_max_len'] modified to be
            max_words_per_line * max_lines
        """

        if (
            self.model_args["model_kwargs"]["att_dropout"] > 1
            or self.model_args["model_kwargs"]["att_dropout"] < 0
        ):
            raise exceptions.ParamError("Attn dropout must be between 0 and 1")

        if not isinstance(self.model_args["train_kwargs"]["att_heads"], int):
            raise exceptions.ParamError("Attn heads mut be an int between 1 and 16")

        if (
            self.model_args["train_kwargs"]["att_heads"] > 16
            or self.model_args["train_kwargs"]["att_heads"] < 1
        ):
            raise exceptions.ParamError("Attn heads mut be an int between 1 and 16")

        if not isinstance(self.model_args["train_kwargs"]["att_dim_per_head"], int):
            raise exceptions.ParamError(
                "Attn dim per head mut be an int between 1 and 16"
            )

        if (
            self.model_args["train_kwargs"]["att_dim_per_head"] > 100
            or self.model_args["train_kwargs"]["att_dim_per_head"] < 1
        ):
            raise exceptions.ParamError(
                "Attn dim per head mut be an int between 1 and 100"
            )

    def check_data_train_args(self, from_pretrained: bool = False):
        """
        Verify arguements are appropriate for the chosen model options.

        Parameters: none

        Pre-condtition: self.model_args is not None
        Post-condition: self.model_args['train_kwargs']['doc_max_len'] is
            updated from the data_kwargs
        """
        schema = {
            "data_kwargs": [
                "tasks",
                "exclude_single",
                "shuffle_case_order",
                "split_by_tumorid",
                "model_path",
                "subset_proportion",
                "random_seed",
                "reproducible",
            ],
            "model_kwargs": [
                "att_dim_per_head",
                "att_heads",
                "att_dropout",
                "forward_mask",
            ],
            "train_kwargs": [
                "batch_per_gpu",
                "max_epochs",
                "patience",
                "class_weights",
                "mixed_precision",
                "save_probs",
            ],
        }

        model_kwds = [
            "model_kwargs",
            "abstain_kwargs",
            "data_kwargs",
            "save_name",
            "train_kwargs",
        ]
        if from_pretrained:
            schema["data_kwargs"].append("data_path")
            schema["train_kwargs"].extend(["reproducible", "random_seed"])

        if sorted(self.model_args.keys()) != sorted(model_kwds):
            print("\nReceived: ", sorted(self.model_args.keys()))
            print("Expected: ", model_kwds)
            raise exceptions.ParamError("model_arg keys do not match the schema")

        for kwrd, vals in schema.items():
            if kwrd == "abstain_kwargs":
                continue  # these are checked in a separate function
            if sorted(self.model_args[kwrd]) != sorted(vals):
                print("\nReceived: ", sorted(self.model_args[kwrd]))
                print("Expected: ", sorted(vals))
                raise exceptions.ParamError(
                    (f"model args {kwrd} does not have " + "the expected variables")
                )

        # copy data kwargs to train kwds
        copy_kwds = ["random_seed", "reproducible"]
        for word in copy_kwds:
            self.model_args["train_kwargs"].update(
                [(word, self.model_args["data_kwargs"][word])]
            )

    def check_abstain_args(self):
        """Verify keywords needed for abstention to work are present and valid."""
        abstain_kwargs = [
            "abstain_flag",
            "alphas",
            "max_abs",
            "min_acc",
            "abs_gain",
            "acc_gain",
            "alpha_scale",
            "tune_mode",
            "stop_limit",
            "stop_metric",
            "ntask_flag",
            "ntask_tasks",
            "ntask_alpha",
            "ntask_alpha_scale",
            "ntask_max_abs",
            "ntask_min_acc",
        ]

        if sorted(abstain_kwargs) != sorted(self.model_args["abstain_kwargs"]):
            print("\nReceived: ", sorted(self.model_args["abstain_kwargs"]))
            print("Expected: ", sorted(abstain_kwargs))
            raise exceptions.ParamError(
                ("model args abstain_kwargs does not have " + "the expected variables")
            )

        if set(self.model_args["abstain_kwargs"]["alphas"].keys()).isdisjoint(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Alpha tasks are not a subset of the data tasks."
            )

        if len(self.model_args["abstain_kwargs"]["alphas"]) > len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of abstain alphas is greater than number of tasks."
            )

        if len(self.model_args["abstain_kwargs"]["max_abs"]) != len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of max abstain rates is different than number of tasks."
            )

        if len(self.model_args["abstain_kwargs"]["min_acc"]) != len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of min acc rates is different than number of tasks."
            )

        if len(self.model_args["abstain_kwargs"]["alpha_scale"]) != len(
            self.model_args["data_kwargs"]["tasks"]
        ):
            raise exceptions.ParamError(
                "Number of alpha scales is different than number of tasks."
            )

        if (
            self.model_args["abstain_kwargs"]["ntask_flag"]
            and not self.model_args["abstain_kwargs"]["abstain_flag"]
        ):
            raise exceptions.ParamError("Ntask cannot be enables without Abstention")

        if self.model_args["abstain_kwargs"]["ntask_flag"]:
            if set(self.model_args["abstain_kwargs"]["ntask_tasks"]).isdisjoint(
                self.model_args["data_kwargs"]["tasks"]
            ):
                raise exceptions.ParamError(
                    "Ntask tasks are not a subset of the data tasks."
                )
