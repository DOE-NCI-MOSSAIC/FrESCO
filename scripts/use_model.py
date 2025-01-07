"""
    Top-level script for loading pre-trained model and making predictions
    on new data.

    typical call is: $python use_model.py -mp /path/to/model -dp /path/to/data/`
"""
import argparse
import os
import random

import torch
import yaml

import numpy as np

from fresco.validate import exceptions
from fresco.validate import validate_params
from fresco.data_loaders import data_utils
from fresco.abstention import abstention
from fresco.models import mthisan, mtcnn
from fresco.predict import predictions


def load_model_dict(model_path, data_path):
    """Load pretrained model from disk.

    Args:
        model_path: str, from command line args, points to saved model
        data_path: str, data to make predictions on

    We check if the supplied path is valid and if the packages match needed
        to run the pretrained model.

    """
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model_dict = torch.load(model_path, map_location=torch.device("cpu"))
    else:
        raise exceptions.ParamError(f"the model at {model_path} does not exist.")

    if not os.path.exists(data_path):
        raise exceptions.ParamError(f"the data at {data_path} does not exist.")

    return model_dict


def load_model(model_dict, device, dw):
    """Load pretrained model architecture."""
    model_args = model_dict["metadata_package"]["mod_args"]

    if model_args["model_type"] == "mthisan":
        model = mthisan.MTHiSAN(
            dw.inference_data["word_embedding"],
            dw.num_classes,
            **model_args["MTHiSAN_kwargs"],
        )

    elif model_args["model_type"] == "mtcnn":
        model = mtcnn.MTCNN(
            dw.inference_data["word_embedding"],
            dw.num_classes,
            **model_args["MTCNN_kwargs"],
        )

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_state_dict = model_dict["model_state_dict"]
    model_state_dict = {
        k.replace("module.", ""): v
        for k, v in model_state_dict.items()
        if k != " metadata_package"
    }
    model.load_state_dict(model_state_dict)
    model.eval()

    print("model loaded")

    return model


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        default="",
        help="""this is the location of the model
                                that will used to make predictions""",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        default="",
        help="""where the data will load from. The default is
                                the path saved in the model""",
    )
    parser.add_argument(
        "--model_args",
        "-args",
        type=str,
        default="",
        help="""path to specify the model_args; default is in
                                the configs/ directory""",
    )

    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    args = parser.parse_args()

    if len(args.model_path) == 0 or len(args.data_path) == 0:
        raise exceptions.ParamError(
            "Model and/or data path cannot be empty, please specify both."
        )

    # 1. validate model/data args
    print("Validating kwargs in model_args.yml file")
    data_source = "pre-generated"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = load_model_dict(args.model_path, args.data_path)

    if len(args.model_args) > 0:  # get config file from cli
        print("Reading config file from command-line args")
        if os.path.isfile(args.model_args):
            with open(args.model_args, "r", encoding="utf-8") as f_in:
                mod_args = yaml.safe_load(f_in)
            pretrained_args = False
        else:
            raise exceptions.ParamError(
                "within the Model_Suite the "
                + "model_args.yml file is needed to set "
                + "the model arguments"
            )
    else:  # use the model args file from training
        mod_args = model_dict["metadata_package"]["mod_args"]
        pretrained_args = True

    print("Validating kwargs from pretrained model ")
    model_args = validate_params.ValidateParams(args, data_source=data_source, model_args=mod_args)

    model_args.check_data_train_args(from_pretrained=pretrained_args)
    if model_args.model_args["model_type"] == "mthisan":
        model_args.hisan_arg_check()
    elif model_args.model_args["model_type"] == "mtcnn":
        model_args.mtcnn_arg_check()

    if model_args.model_args["abstain_kwargs"]["abstain_flag"]:
        model_args.check_abstain_args()

    model_args.check_data_files()

    if model_args.model_args["data_kwargs"]["reproducible"]:
        seed = model_args.model_args["data_kwargs"]["random_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    # 2. load data
    print("Loading data and creating DataLoaders")

    dw = data_utils.DataHandler(data_source, model_args.model_args)
    dw.load_folds(fold=0)
    if model_args.model_args["data_kwargs"]["data_pipeline"] == "mod_pipeline":
        dw.convert_y(inference=True)
        # maps labels to int, eg, C50 -> <some int>

    data_loaders = dw.make_torch_dataloaders(
        reproducible=model_args.model_args["data_kwargs"]["reproducible"],
        seed=seed,
        inference=True,
        clc_flag=False
    )

    if model_args.model_args["abstain_kwargs"]["abstain_flag"]:
        dac = abstention.AbstainingClassifier(model_args.model_args, device)
        # modifies n_classes and id2label dict for abstention/ntask
        dac.add_abstention_classes(dw)
        print("Running with:")
        for key in model_args.model_args["abstain_kwargs"]:
            print(key, ": ", model_args.model_args["abstain_kwargs"][key])
    else:
        dac = None

    model = load_model(model_dict, device, dw)

    # 5. score predictions from pretrained model or model just trained
    evaluator = predictions.ScoreModel(
        model_args.model_args, data_loaders, dw, model, device
    )

    # 5a. score a model
    # evaluator.score(dac=dac)

    # 5b. make predictions
    evaluator.predict(
        dw.metadata["metadata"],
        dw.dict_maps["id2label"],
        save_probs=model_args.model_args["train_kwargs"]["save_probs"],
    )

    # 5c. score and predict
    # evaluator.evaluate_model(dw.metadata['metadata'], dw.dict_maps['id2label'], dac=dac)


if __name__ == "__main__":
    main()
