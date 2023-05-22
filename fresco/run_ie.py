"""
    Top-level information extraction model building script using independent modules.
"""
import argparse
import datetime
import json
import os
import random

import torch

import numpy as np

from fresco.validate import exceptions
from fresco.validate import validate_params
from fresco.data_loaders import data_utils
from fresco.abstention import abstention
from fresco.models import mthisan, mtcnn
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

    if params.model_args['model_type'] == 'mthisan':
        model = mthisan.MTHiSAN(dw.inference_data['word_embedding'],
                                dw.num_classes,
                                **params.model_args['MTHiSAN_kwargs'])

    elif params.model_args['model_type'] == 'mtcnn':
        model = mtcnn.MTCNN(dw.inference_data['word_embedding'],
                            dw.num_classes,
                            **params.model_args['MTCNN_kwargs'])
    model.to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def load_model(model_path, device, dw, valid_params):
    """
    Load a pretrained model from disk.

    Args:
        model_path (str): Path to the saved model obtained from command line args.
        device (torch.device): Torch device, either 'cuda' or 'cpu', set in the calling function.
        dw (DataHandler): DataHandler class instance for wrangling data and checking metadata.
        valid_params (ValidateParams): ValidateParams class instance with model_args dict.

    Post-condition:
        The saved model is loaded and set to eval() mode.

    Returns:
        None
    """

    data_path = dw.model_args['data_kwargs']['data_path']
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
    else:
        raise exceptions.ParamError(f'the model at {model_path} does not exist.')

    if torch.cuda.is_available():
        model_dict = torch.load(model_path)
    else:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_dict = {k.replace('module.',''):v for k,v in model_dict.items()}

    mismatches = [mod for mod in model_dict['metadata_package'].keys()
        if dw.metadata['metadata'][mod] != model_dict['metadata_package'][mod]]

    # check to see if the stored package matches the expected one
    # what is this?
    # package['match'] = len(mismatches)==0

    if len(mismatches) > 0:
        with open('metadata_package.json', 'w', encoding='utf-8') as jd:
            json.dump(model_dict['metadata_package'], jd, indent=2)
            raise exceptions.ParamError(f'the package(s) {", ".join(mismatches)} does not match ' +
                                        f'the generated data in {data_path}.' +
                                         '\nThe needed recreation info is in metadata_package.json')
    # Load Model
    if valid_params.model_args['model_type'] == 'mthisan':
        model = mthisan.MTHiSAN(dw.inference_data['word_embedding'],
                                dw.num_classes,
                                **valid_params.model_args['MTHiSAN_kwargs'])

    elif valid_params.model_args['model_type'] == 'mtcnn':
        model = mtcnn.MTCNN(dw.inference_data['word_embedding'],
                            dw.num_classes,
                            **valid_params.model_args['MTHiSAN_kwargs'])

    model.to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_dict = {k:v for k,v in model_dict.items() if k!='metadata_package'}
    # model_dict = {k.replace('module.',''):v for k,v in model_dict.items()}
    model.load_state_dict(model_dict)
    model.eval()

    print('model_loaded')

    return model


def save_full_model(kw_args, save_path, packages, fold):
    """
    Save trained model with package info metadata.
    """
    # save_path = save_path + f"_fold{fold}.h5"
    path = 'savedmodels/' + kw_args['save_name'] + "/"
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    savename = path + kw_args['save_name'] + f"_{now}_fold{fold}.h5"
    model = torch.load(save_path)
    model['metadata_package'] = packages
    torch.save(model, savename)
    print(f"\nFull model file has been saved at {savename}")
    os.remove(save_path)


def run_ie(args=None):

    # 1. validate model/data args
    print("Validating kwargs in model_args.yml file")
    data_source = 'pre-generated'

    valid_params = validate_params.ValidateParams(args, data_source=data_source)

    valid_params.check_data_train_args()

    if valid_params.model_args['model_type'] == 'mthisan':
        valid_params.hisan_arg_check()
    elif valid_params.model_args['model_type'] == 'mtcnn':
        valid_params.mtcnn_arg_check()

    if valid_params.model_args['abstain_kwargs']['abstain_flag']:
        valid_params.check_abstain_args()

    valid_params.check_data_files()

    if valid_params.model_args['data_kwargs']['reproducible']:
        seed = valid_params.model_args['data_kwargs']['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    # 2. load data
    print("Loading data and creating DataLoaders")
    dw = data_utils.DataHandler(data_source, valid_params.model_args)
    dw.load_folds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if valid_params.model_args['abstain_kwargs']['abstain_flag']:
        dac = abstention.AbstainingClassifier(valid_params.model_args, device)
        # modifies n_classes and id2label dict for abstention/ntask
        dac.add_abstention_classes(dw)
        print('\nRunning with:')
        for key in valid_params.model_args['abstain_kwargs']:
            print(key, ": ", valid_params.model_args['abstain_kwargs'][key])
    else:
        dac = None

    data_loaders = dw.make_torch_dataloaders(valid_params.model_args['data_kwargs']['add_noise'],
                                             reproducible=valid_params.model_args['data_kwargs']['reproducible'],
                                             seed=seed)

    # 3. create a model
    print("\nDefining a model")
    model = create_model(valid_params, dw, device)
    # 4. train new model
    print("Creating model trainer")
    batch = valid_params.model_args['data_kwargs']['batch_per_gpu']

    trainer = training.ModelTrainer(valid_params.model_args,
                                    model,
                                    dw,
                                    class_weights=dw.weights,
                                    device=device)

    if valid_params.model_args['abstain_kwargs']['abstain_flag']:
        print(f"Training a {valid_params.model_args['model_type']} dac model with " +
              f"{torch.cuda.device_count()} {device} device\n")
        trainer.fit_model(data_loaders['train'], val_loader=data_loaders['val'], dac=dac)
    else:
        print(f"Training a {valid_params.model_args['model_type']} model with " +
              f"{torch.cuda.device_count()} {device} device\n")
        trainer.fit_model(data_loaders['train'], val_loader=data_loaders['val'])

    # 5. score predictions from pretrained model or model just trained
    evaluator = predictions.ScoreModel(valid_params.model_args, data_loaders, model, device)

    # Only need one of 5a, 5b, or 5c, depending on requirements
    # 5a. score a model
    evaluator.score(split='test', dac=dac)

    # 5b. make predictions
    evaluator = predictions.ScoreModel(valid_params.model_args, data_loaders, model, device)
    evaluator.predict(dw.dict_maps['id2label'])

    # 5c. score and predict
    evaluator = predictions.ScoreModel(valid_params.model_args, data_loaders, model, device)
    evaluator.evaluate_model(dw.dict_maps['id2label'], dac=dac, save_probs=False)

    # this is the default args filename and path
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(trainer.savepath + f"model_args_{now}.json", "w", encoding="utf-8") as f_out:
        json.dump(valid_params.model_args, f_out, indent=4)

    if len(args.model_path) == 0:
        save_full_model(valid_params.model_args,
                        trainer.savename,
                        dw.metadata['packages'],
                        dw.model_args['data_kwargs']['fold_number'])


def main():
    print("""\nNot intended to be run as stand alone script.
          Run create_model.py to train a new model.\n""")


if __name__ == "__main__":
    main()
