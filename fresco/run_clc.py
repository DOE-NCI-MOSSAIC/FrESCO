"""
    Top-level clc model building script using independent modules.
"""
import json
import os
import random
import datetime

from warnings import warn

import torch

import numpy as np

from fresco.validate import exceptions, validate_params
from fresco.data_loaders import data_utils
from fresco.abstention import abstention
from fresco.models import mthisan, mtcnn, clc
from fresco.training import training
from fresco.predict import predictions

def create_model(params, dw, device):
    """
    Define a model based on model_args.

    Args:
        params (dict): Dictionary of model_args file.
        dw (DataHandler): DataHandler class object.
        device (torch.device): Torch device, either 'cpu' or 'cuda'.

    Returns:
        A model.
    """

    model = clc.CaseLevelContext(dw.num_classes, device=device, **params.model_args['model_kwargs'])

    model.to(device, non_blocking=True)
    return model


def create_doc_embeddings(model, model_type, data_loader, device):
    """Generate document embeddings from trained model."""
    model.eval()
    if model_type == 'mtcnn':
        embed_dim  = 900
    else:
        embed_dim = 400
    embeds = np.empty((len(data_loader.dataset), embed_dim))
    bs = data_loader.batch_size
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            X = batch["X"].to(device, non_blocking=True)
            _, embed = model(X, return_embeds=True)
            if embed.shape[0] == bs:
                embeds[i*bs:(i+1)*bs, : ] = embed.cpu().numpy()
            else:
                embeds[-embed.shape[0]:, : ] = embed.cpu().numpy()
    return embeds


def load_model_dict(model_path, data_path=""):
    """
    Load pretrained model from disk.

    Args:
        model_path (str): Path to the saved model from command line args.
        valid_params (ValidateParams): ValidateParams class object with model_args dict.
        data_path (str or None): Path to data from the trained model, or None.

    We check if the supplied path is valid and if the packages match needed
    to run the pretrained model.
    """

    if os.path.exists(model_path):
        model_dict = torch.load(model_path)
    else:
        raise exceptions.ParamError("Provided model path does not exist")
    if len(data_path) > 0:
        with open(data_path + 'metadata.json', 'r', encoding='utf-8') as f:
            data_args = json.load(f)
    else:
        data_args = model_dict['metadata_package']['mod_args']

    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
    else:
        raise exceptions.ParamError(f'the model at {model_path} does not exist.')

    # check that model args all agree
    mismatches = [mod for mod in model_dict['metadata_package']['mod_args'].keys()
        if data_args[mod] != model_dict['metadata_package']['mod_args'][mod]]

    # check to see if the stored package matches the expected one
    if len(mismatches) > 0:
        with open('metadata_package.json', 'w', encoding='utf-8') as f_out:
            json.dump(model_dict['metadata_package'], f_out, indent=2)
            raise exceptions.ParamError(f'the package(s) {", ".join(mismatches)} does not match ' +
                                        f'the generated data in {data_path}.' +
                                         '\nThe needed recreation info is in metadata_package.json')

    return model_dict


def load_model(model_dict, device, dw):

    model_args = model_dict['metadata_package']['mod_args']

    if model_args['model_type'] == 'mthisan':
        model = mthisan.MTHiSAN(dw.inference_data['word_embedding'],
                                dw.num_classes,
                                **model_args['MTHiSAN_kwargs'])

    elif model_args['model_type'] == 'mtcnn':
        model = mtcnn.MTCNN(dw.inference_data['word_embedding'],
                            dw.num_classes,
                            **model_args['MTHiSAN_kwargs'])

    model.to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_dict = {k: v for k,v in model_dict.items() if k!='metadata_package'}
    try:
        model.load_state_dict(model_dict)
    except RuntimeError:
        model_dict = {k.replace('module.',''): v for k,v in model_dict.items()}
        model.load_state_dict(model_dict)

    print('model loaded')

    return model


def save_full_model(kw_args, save_path, packages, fold):
    """Save trained model with package info metadata."""
    path = 'savedmodels/' + kw_args['save_name'] + "/"
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    savename = path + kw_args['save_name'] + f"_{now}_fold{fold}_clc_full.h5"
    model = torch.load(save_path)
    model['metadata_package'] = packages
    torch.save(model, savename)
    print(f"\nFull model file has been saved at {savename}")
    os.remove(save_path)


def run_case_level(args):

    clc_flag = True
    # 1. validate model/data args
    print("Validating kwargs in clc_args.yml file")
    data_source = 'pre-generated'

    valid_params = validate_params.ValidateClcParams(args, data_source=data_source)

    valid_params.check_data_train_args()

    if valid_params.model_args['abstain_kwargs']['abstain_flag']:
        valid_params.check_abstain_args()

    if valid_params.model_args['data_kwargs']['reproducible']:
        seed = valid_params.model_args['data_kwargs']['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = load_model_dict(valid_params.model_args['data_kwargs']['model_path'])

    valid_params.model_args['data_kwargs']['data_path'] = model_dict['metadata_package']['mod_args']['data_kwargs']['data_path']
    valid_params.check_data_files(valid_params.model_args['data_kwargs']['data_path'])

    # model args of the first (one used for generating the doc embeddings) trained model
    # submodel_args = model_dict['metadata_package']['mod_args']
    # need to verify submodel args

    data_source = 'pre-generated'
    args.model_args = model_dict['metadata_package']['mod_args']
    print("Validating kwargs from pretrained model ")
    submodel_args = validate_params.ValidateParams(args,
                                                   data_source=data_source,
                                                   model_args=args.model_args)

    submodel_args.check_data_train_args(from_pretrained=True)

    if valid_params.model_args['abstain_kwargs']['abstain_flag'] != \
        submodel_args.model_args['abstain_kwargs']['abstain_flag']:
            raise exceptions.ParamError("DAC must be true for both original and clc model")
    if valid_params.model_args['abstain_kwargs']['ntask_flag'] != \
        submodel_args.model_args['abstain_kwargs']['ntask_flag']:
            raise exceptions.ParamError("N-task must be true for both original and clc model")

    if submodel_args.model_args['model_type'] == 'mthisan':
        submodel_args.hisan_arg_check()
    elif submodel_args.model_args['model_type'] == 'mtcnn':
        warn("MTCNN will be deprecated; please use HiSAN for all future models.", warnings.DeprecationWarning)
        submodel_args.mtcnn_arg_check()

    if submodel_args.model_args['abstain_kwargs']['abstain_flag']:
        submodel_args.check_abstain_args()

    submodel_args.check_data_files()

    if valid_params.model_args['data_kwargs']['reproducible']:
        seed = submodel_args.model_args['data_kwargs']['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    # 2. load data
    print("Loading data and creating DataLoaders")

    dw = data_utils.DataHandler(data_source, submodel_args.model_args)
    dw.load_folds(fold=0, subset_frac=valid_params.model_args['data_kwargs']['subset_proportion'])

    if valid_params.model_args['abstain_kwargs']['abstain_flag']:
        dac = abstention.AbstainingClassifier(valid_params.model_args, device, clc=clc_flag)
        # modifies n_classes and id2label dict for abstention/ntask
        dac.add_abstention_classes(dw)
        print('Running with:')
        for key in valid_params.model_args['abstain_kwargs']:
            print(key, ": ", valid_params.model_args['abstain_kwargs'][key])
    else:
        dac = None

    data_loaders = dw.make_torch_dataloaders(submodel_args.model_args['data_kwargs']['add_noise'],
                                             reproducible=valid_params.model_args['data_kwargs']['reproducible'],
                                             seed=seed)

    model = load_model(model_dict, device, dw)
    embeds = {k: create_doc_embeddings(model, submodel_args.model_args['model_type'], v, device)
              for k, v in data_loaders.items()}

    dw.make_grouped_cases(embeds, valid_params.model_args, device)

    # 3. create a CLC model
    print("\nDefining a CLC model")
    model = create_model(valid_params, dw, device)
    # 4. train new model
    print("Creating model trainer")

    trainer = training.ModelTrainer(valid_params.model_args, model, dw, device=device,
                                    fold='clc',class_weights=dw.weights, clc=clc_flag)

    print("Training a case-level model with " +
          f"{torch.cuda.device_count()} {device} device\n")
    trainer.fit_model(dw.grouped_cases['train'], val_loader=dw.grouped_cases['val'], dac=dac)

    # 5. score predictions from pretrained model or model just trained
    evaluator = predictions.ScoreModel(valid_params.model_args,
                                       dw.grouped_cases,
                                       model,
                                       device,
                                       clc_flag=clc_flag)

    # Only need one of 5a, 5b, or 5c, depending on requirements
    # 5a. score a model
    # evaluator.score(dac=dac)

    # 5b. make predictions
    # evaluator = predictions.ScoreModel(valid_params.model_args,
    #                                    dw.grouped_cases,
    #                                    model,
    #                                    device,
    #                                    clc_flag=clc_flag)
    # evaluator.predict(dw.dict_maps['id2label'])

    # 5c. score and predict
    # evaluator = predictions.ScoreModel(valid_params.model_args,
    #                                    dw.grouped_cases,
    #                                    model,
    #                                    device,
    #                                    clc_flag=clc_flag)
    evaluator.evaluate_model(dw.dict_maps['id2label'], dac=dac)

    # this is the default args filename and path
    with open(trainer.savepath + "clc_args.json", "w", encoding="utf-8") as f_out:
        json.dump(valid_params.model_args, f_out, indent=4)

    #save_full_model(trainer.savename,dw.metadata['packages'])
    save_full_model(valid_params.model_args,
                    trainer.savename,
                    dw.metadata['packages'],
                    dw.model_args['data_kwargs']['fold_number'])


def main():
    print("""\nNot intended to be run as stand alone script.
          Run create_model.py to train a new model.\n""")


if __name__ == "__main__":
    main()
