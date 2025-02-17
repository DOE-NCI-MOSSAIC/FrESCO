"""
    Top-level clc model building script using independent modules.
"""
import datetime
import json
import os
import random

import torch

import numpy as np

from fresco.validate import exceptions, validate_params
from fresco.data_loaders import data_utils
from fresco.abstention import abstention
from fresco.models import mthisan, mtcnn, clc
from fresco.training import training
from fresco.predict import predictions


def create_model(params, dw, device, embed_dim):
    """Define model based on model_args.

    Args:
        params: dict of model_args file
        dw: DataHandler class
        device: torch.device, either 'cpu' or 'cuda'
        embed_dim: int, dim of doc embeddings

    Returns:
        a model

    """
    model = clc.CaseLevelContext(
        dw.num_classes,
        device=device,
        doc_embed_size=embed_dim,
        **params.model_args["model_kwargs"],
    )

    model.to(device, non_blocking=True)

    return model


def create_doc_embeddings(model, model_type, data_loader, tasks, device):
    """Generate document embeddings from trained model."""
    model.eval()
    if model_type == "mtcnn":
        embed_dim = 900
    else:
        embed_dim = 400

    bs = data_loader.batch_size
    embeds = torch.empty((len(data_loader.dataset), embed_dim), device=device)
    ys = {task: torch.empty(len(data_loader.dataset), dtype=torch.int, device=device) for task in tasks}
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


def load_model_dict(model_path, data_path=""):
    """Load pretrained model from disk.

    Args:
        model_path: str, from command line args, points to saved model
        clc_params: ValidateParams class, with model_args dict
        data_path: str or None, using data from the trained model, or different one

    We check if the supplied path is valid and if the packages match needed
        to run the pretrained model.

    """
    if os.path.exists(model_path):
        model_dict = torch.load(model_path)
    else:
        raise exceptions.ParamError("Provided model path does not exist")

    if len(data_path) > 0:
        with open(data_path + "metadata.json", "r", encoding="utf-8") as f:
            data_args = json.load(f)
    else:
        data_args = model_dict["metadata_package"]["model_metadata"]

    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
    else:
        raise exceptions.ParamError(f"the model at {model_path} does not exist.")

    # check that model args all agree
    # this needs to be looked ate carefully, not sure it deos what we want. AS-12/5
    # need to check abstention args especially
    # print(model_dict['metadata_package'].keys())
    # print(model_dict['metadata_package']['mod_args'])
    # for mod_pipeline only
    if model_dict["metadata_package"]["model_metadata"] is not None:
        mismatches = [
            mod
            for mod in model_dict["metadata_package"]["model_metadata"].keys()
            if data_args[mod] != model_dict["metadata_package"]["model_metadata"][mod]
        ]

        # check to see if the stored package matches the expected one
        if len(mismatches) > 0:
            with open("metadata_package.json", "w", encoding="utf-8") as f_out:
                json.dump(model_dict["metadata_package"], f_out, indent=2)
                raise exceptions.ParamError(
                    f'the package(s) {", ".join(mismatches)} does not match '
                    + f"the generated data in {data_path}."
                    + "\nThe needed recreation info is in metadata_package.json"
                )
    # print(XXX)
    return model_dict


def load_model(model_dict, device, dw):
    # if torch.cuda.device_count() < 2:
    #     raise exceptions.ParamError("Case level context requires > 2 gpus")

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

    model.to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model_state_dict = model_dict["model_state_dict"]
    model_state_dict = {
        k.replace("module.", ""): v for k, v in model_state_dict.items() if k != " metadata_package"
    }

    model.load_state_dict(model_state_dict)

    print("model loaded")

    return model


def save_full_model(kw_args, save_path, packages, fold):
    """Save trained model with package info metadata."""
    # save_path = save_path + f"_fold{fold}.h5"
    path = "savedmodels/" + kw_args["save_name"] + "/"
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    savename = path + kw_args["save_name"] + f"{now}_fold{fold}.h5"
    model = torch.load(save_path)
    model["metadata_package"] = packages
    model["clc_args"] = kw_args
    torch.save(model, savename)
    print(f"\nmodel file has been saved at {savename}")
    os.remove(save_path)


def run_case_level(args):
    clc_flag = True

    # 1. validate model/data args
    print("Validating kwargs in clc_args.yml file")
    cache_class = [False]
    data_source = "pre-generated"

    clc_params = validate_params.ValidateClcParams(cache_class, args, data_source=data_source)

    clc_params.check_data_train_args()

    if clc_params.model_args["abstain_kwargs"]["abstain_flag"]:
        clc_params.check_abstain_args()

    if clc_params.model_args["data_kwargs"]["reproducible"]:
        seed = clc_params.model_args["data_kwargs"]["random_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = load_model_dict(clc_params.model_args["data_kwargs"]["model_path"])

    clc_params.model_args["data_kwargs"]["data_path"] = model_dict["metadata_package"]["mod_args"][
        "data_kwargs"
    ]["data_path"]
    # clc_params.check_data_files(model_dict["metadata_package"]["mod_args"]["data_kwargs"])

    args.model_args = model_dict["metadata_package"]["mod_args"]
    print("Validating kwargs from pretrained model ")
    submodel_args = validate_params.ValidateParams(
        cache_class, args, data_source=data_source, model_args=args.model_args
    )
    # check step 1 model args
    submodel_args.check_data_train_args(from_pretrained=True)

    if clc_params.model_args["data_kwargs"]["tasks"] != submodel_args.model_args["data_kwargs"]["tasks"]:
        raise exceptions.ParamError("Tasks must be the same for both original and clc models")

    if (
        clc_params.model_args["abstain_kwargs"]["abstain_flag"]
        != submodel_args.model_args["abstain_kwargs"]["abstain_flag"]
    ):
        raise exceptions.ParamError("DAC must be true for both original and clc models")

    if (
        clc_params.model_args["abstain_kwargs"]["ntask_flag"]
        != submodel_args.model_args["abstain_kwargs"]["ntask_flag"]
    ):
        raise exceptions.ParamError("N-task must be true for both original and clc models")

    if submodel_args.model_args["model_type"] == "mthisan":
        submodel_args.hisan_arg_check()
    elif submodel_args.model_args["model_type"] == "mtcnn":
        submodel_args.mtcnn_arg_check()

    if submodel_args.model_args["abstain_kwargs"]["abstain_flag"]:
        submodel_args.check_abstain_args()

    submodel_args.check_data_files()

    # 2. load data
    print("Loading data and creating DataLoaders")

    dw = data_utils.DataHandler(data_source, submodel_args.model_args, cache_class)
    dw.load_folds(fold=0, subset_frac=clc_params.model_args["data_kwargs"]["subset_proportion"])
    if submodel_args.model_args["data_kwargs"]["data_pipeline"] == "mod_pipeline":
        dw.convert_y(inference=True)

    if clc_params.model_args["abstain_kwargs"]["abstain_flag"]:
        dac = abstention.AbstainingClassifier(clc_params.model_args, device, clc=clc_flag)
        # modifies n_classes and id2label dict for abstention/ntask
        dac.add_abstention_classes(dw)
        print("\nRunning with:")
        for key in clc_params.model_args["abstain_kwargs"]:
            print(key, ": ", clc_params.model_args["abstain_kwargs"][key])
    else:
        dac = None

    # running with inference=True here to create doc embeddings
    # for clc model, no noise added
    # since this uses the base model, no clc_args are required
    data_loaders = dw.make_torch_dataloaders(
        switch_rate=0.0,
        reproducible=clc_params.model_args["data_kwargs"]["reproducible"],
        shuffle_data=False,
        seed=seed,
        clc_flag=False,
        clc_args=None,
        inference=True,
    )

    model = load_model(model_dict, device, dw)
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

    dw.make_grouped_cases(
        outputs,
        clc_params.model_args,
        reproducible=clc_params.model_args["data_kwargs"]["reproducible"],
        seed=seed,
    )

    # 3. create a CLC model
    print("\nDefining a CLC model")
    model = create_model(clc_params, dw, device, embed_dim=outputs["train"]["X"].shape[1])

    # 4. train new model
    print("Creating model trainer")

    trainer = training.ModelTrainer(
        clc_params.model_args,
        model,
        dw,
        device=device,
        fold="clc",
        class_weights=dw.weights,
        clc=clc_flag,
    )

    print("Training a case-level model with " + f"{torch.cuda.device_count()} {device} device\n")
    trainer.fit_model(dw.grouped_cases["train"], val_loader=dw.grouped_cases["val"], dac=dac)

    # 5. score predictions from pretrained model or model just trained
    evaluator = predictions.ScoreModel(
        clc_params.model_args,
        dw.grouped_cases,
        dw,
        model,
        device,
        savepath=None,
        clc_flag=clc_flag,
    )

    # only need one of 5a, 5b, or 5c, depending on requirements
    # 5a. score a model
    evaluator.score(dac=dac)

    # 5b. make predictions
    evaluator.predict(
        dw.metadata["metadata"],
        dw.dict_maps["id2label"],
        save_probs=clc_params.model_args["train_kwargs"]["save_probs"]
    )

    # this is the default args filename and path
    with open(trainer.savepath + "clc_args.json", "w", encoding="utf-8") as f_out:
        json.dump(clc_params.model_args, f_out, indent=4)

    if len(args.model_path) == 0:
        save_full_model(
            clc_params.model_args,
            trainer.savename,
            dw.metadata["packages"],
            dw.model_args["data_kwargs"]["fold_number"],
        )


def main():
    print(
        """\nNot intended to be run as stand alone script.
          Run train_model.py to train a new model.\n"""
    )


if __name__ == "__main__":
    main()
