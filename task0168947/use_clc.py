"""
    Top-level model building script using independent modules.
"""
import argparse
import os
import random

import torch
import yaml

import numpy as np

from validate import exceptions
from validate import validate_params
from data_loaders import data_utils
from abstention import abstention
from models import clc, mthisan, mtcnn
from predict import predictions

# needed for running on the vms
# torch.multiprocessing.set_sharing_strategy('file_system')


def create_doc_embeddings(model, model_type, data_loader, tasks, device):
    """Generate document embeddings from trained model."""
    model.eval()
    if model_type == "mtcnn":
        embed_dim = 900
    else:
        embed_dim = 400

    bs = data_loader.batch_size
    embeds = torch.empty((len(data_loader.dataset), embed_dim), device=device)
    ys = {
        task: torch.empty(len(data_loader.dataset), dtype=torch.int, device=device)
        for task in tasks
    }
    idxs = torch.empty((len(data_loader.dataset),), dtype=torch.int, device=device)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            X = batch["X"].to(device, non_blocking=True)
            _, embed = model(X, return_embeds=True)
            if embed.shape[0] == bs:
                embeds[i*bs:(i+1)*bs] = embed
                idxs[i*bs:(i+1)*bs] = batch["index"]
                for task in tasks:
                    if task != "Ntask":
                        ys[task][i*bs:(i+1)*bs] = batch[f"y_{task}"]
            else:
                embeds[-embed.shape[0]:] = embed
                idxs[-embed.shape[0]:] = batch["index"]
                for task in tasks:
                    if task != "Ntask":
                        ys[task][-embed.shape[0]:] = batch[f"y_{task}"]
    ys_np = {task: vals.cpu().numpy() for task, vals in ys.items()}
    outputs = {"X": embeds.cpu().numpy(), "y": ys_np, "index": idxs.cpu().numpy()}
    return outputs


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


def load_clc_model(clc_args, model_dict, device, dw, embed_dim=400):
    """Load pretrained model architecture."""
    model = clc.CaseLevelContext(
        dw.num_classes,
        device=device,
        doc_embed_size=embed_dim,
        **clc_args.model_args["model_kwargs"],
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

    print("clc model loaded")

    return model


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
        help="""this is the location of the clc model
                                that will used to make predictions""",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        default="",
        help="""where the data will load from, this must be specified.""",
    )
    parser.add_argument(
        "--model_args",
        "-args",
        type=str,
        default="",
        help="""path to specify the model_args; default is in
                                the model_suite directory""",
    )

    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    args = parser.parse_args()

    if len(args.model_path) == 0 or len(args.data_path) == 0:
        raise exceptions.ParamError(
            "Model and/or data path cannot be empty, please specify both."
        )

    # 1. validate model/data args
    print("Validating kwargs in clc_args.yml file")
    cache_class = [False]
    data_source = "pre-generated"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clc_model_dict = load_model_dict(args.model_path, args.data_path)

    if len(args.model_args) > 0:  # get config file from cli
        if os.path.isfile(args.model_args):
            print("Reading clc config file from cli")
            with open(args.model_args, "r", encoding="utf-8") as f_in:
                mod_args = yaml.safe_load(f_in)
            pretrained_config = False
        else:
            raise exceptions.ParamError(
                "within the Model_Suite the "
                + "model_args.yml file is needed to set "
                + "the model arguments"
            )
    else:  # use the model args file from training
        mod_args = clc_model_dict["clc_args"]
        pretrained_config = True

    base_model_args = clc_model_dict["metadata_package"]["mod_args"]
    base_model_path = mod_args["data_kwargs"]["model_path"]  # [1:]
    base_model_dict = load_model_dict(base_model_path, args.data_path)

    print("Validating kwargs from pretrained clc model ")
    clc_params = validate_params.ValidateClcParams(
        cache_class, args, data_source=data_source, model_args=mod_args
    )

    clc_params.check_data_train_args(from_pretrained=pretrained_config)

    if clc_params.model_args["abstain_kwargs"]["abstain_flag"]:
        clc_params.check_abstain_args()

    base_model_args = clc_model_dict["metadata_package"]

    print("Validating kwargs from pretrained base model")
    submodel_args = validate_params.ValidateParams(
        cache_class,
        args,
        data_source=data_source,
        model_args=base_model_args["mod_args"],
    )

    submodel_args.check_data_train_args(from_pretrained=True)

    if (
        clc_params.model_args["data_kwargs"]["tasks"]
        != submodel_args.model_args["data_kwargs"]["tasks"]
    ):
        raise exceptions.ParamError(
            "Tasks must be the same for both original and clc models"
        )

    if (
        clc_params.model_args["abstain_kwargs"]["abstain_flag"]
        != submodel_args.model_args["abstain_kwargs"]["abstain_flag"]
    ):
        raise exceptions.ParamError("DAC must be true for both original and clc models")

    if (
        clc_params.model_args["abstain_kwargs"]["ntask_flag"]
        != submodel_args.model_args["abstain_kwargs"]["ntask_flag"]
    ):
        raise exceptions.ParamError(
            "N-task must be true for both original and clc models"
        )

    if submodel_args.model_args["model_type"] == "mthisan":
        submodel_args.hisan_arg_check()
    elif submodel_args.model_args["model_type"] == "mtcnn":
        # warn("MTCNN will be deprecated; use HiSAN for all future models.", DeprecationWarning)
        submodel_args.mtcnn_arg_check()

    if submodel_args.model_args["abstain_kwargs"]["abstain_flag"]:
        submodel_args.check_abstain_args()

    submodel_args.check_data_files()

    if clc_params.model_args["data_kwargs"]["reproducible"]:
        seed = clc_params.model_args["data_kwargs"]["random_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    # 2. load data
    print("Loading data and creating DataLoaders")

    dw = data_utils.DataHandler(data_source, submodel_args.model_args, cache_class)
    dw.load_folds(fold=0, subset_frac=1.0)

    if clc_params.model_args["abstain_kwargs"]["abstain_flag"]:
        dac = abstention.AbstainingClassifier(
            clc_params.model_args, device, clc=clc_flag
        )
        # modifies n_classes and id2label dict for abstention/ntask
        dac.add_abstention_classes(dw)
        print("\nRunning with:")
        for key in clc_params.model_args["abstain_kwargs"]:
            print(key, ": ", clc_params.model_args["abstain_kwargs"][key])
    else:
        dac = None
    # create data loaders for first step of clc, creating doc embeddings
    # from the base model
    data_loaders = dw.make_torch_dataloaders(
        switch_rate=0.0,
        reproducible=clc_params.model_args["data_kwargs"]["reproducible"],
        shuffle_data=False,
        seed=seed,
        inference=True,
        clc_flag=False,
        clc_args=None,
    )

    model = load_model(base_model_dict, device, dw)
    print("Creating doc embeddings")

    outputs = {}

    for split in dw.splits:
        outputs[split] = create_doc_embeddings(
            model,
            submodel_args.model_args["model_type"],
            data_loaders[split],
            dw.tasks,
            device,
        )
    # running in inference mode from the base model
    # to create the GroupedCases class for input to the clc model
    inference_loaders = dw.make_torch_dataloaders(
        doc_embeds=outputs,
        reproducible=clc_params.model_args["data_kwargs"]["reproducible"],
        seed=seed,
        clc_flag=True,
        clc_args=clc_params.model_args,
        inference=True,
    )

    if clc_params.model_args["abstain_kwargs"]["abstain_flag"]:
        dac = abstention.AbstainingClassifier(clc_params.model_args, device)
        # modifies n_classes and id2label dict for abstention/ntask
        dac.add_abstention_classes(dw)
        print("Running with:")
        for key in clc_params.model_args["abstain_kwargs"]:
            print(key, ": ", clc_params.model_args["abstain_kwargs"][key])
    else:
        dac = None

    model = load_clc_model(
        clc_params, clc_model_dict, device, dw, embed_dim=outputs["train"]["X"].shape[1]
    )

    # 5. score predictions from pretrained model or model just trained
    evaluator = predictions.ScoreModel(
        clc_params.model_args, inference_loaders, dw, model, device, clc_flag=True
    )

    # 5a. score a model
    # evaluator.score(dac=dac)

    # 5b. make predictions
    # evaluator.predict(dw.metadata['metadata'], dw.dict_maps['id2label'])

    # 5c. score and predict
    evaluator.evaluate_model(dw.metadata["metadata"], dw.dict_maps["id2label"], dac=dac)


if __name__ == "__main__":
    main()
