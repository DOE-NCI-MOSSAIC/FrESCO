"""
    Top-level model building script using independent modules.
"""
import argparse
import json
import os
import random
import sys

import torch
import numpy as np

from fresco.validate import exceptions
from fresco.validate import validate_params
from fresco.data_loaders import data_utils
from fresco.abstention import abstention
from fresco.models import mthisan, mtcnn

from torch.profiler import profile, record_function, ProfilerActivity

def create_model(params, dw, device):
    """Define model based on model_args.

        Args:
            params: dict of model_args file
            dw: DataHandler class
            device: torch.device, either 'cpu' or 'cuda'

        Returns:
            a model

    """
    if params.model_args['model_type'] == 'mthisan':
        model = mthisan.MTHiSAN(dw.inference_data['word_embedding'],
                                dw.num_classes,
                                **params.model_args['MTHiSAN_kwargs'])

    elif params.model_args['model_type'] == 'mtcnn':
        model = mtcnn.MTCNN(dw.inference_data['word_embedding'],
                            dw.num_classes,
                            **params.model_args['MTCNN_kwargs'])
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def load_model(model_path, device, dw, valid_params):
    """Load pretrained model from disk.

        Args:
            model_path: str, from command line args, points to saved model
            device: torch.device, either cuda or cpu, set in calling function
            dw: DataHandler class, for wrangling data, and checking metadata
            valid_params: ValidateParams class, with model_args dict

        We check if the supplied path is valid and if the packages match needed
            to run the pretrained model.

        Post-condition:
            saved model loaded and set to eval() model

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

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_dict = {k:v for k,v in model_dict.items() if k!='metadata_package'}
    model_dict = {k.replace('module.',''):v for k,v in model_dict.items()}
    model.load_state_dict(model_dict)
    model.eval()

    print('model_loaded')

    return model


def save_full_model(save_path, packages, fold):
    """Save trained model with package info metadata."""
    save_path = save_path + f"_fold{fold}.h5"
    model = torch.load(save_path)
    model['metadata_package'] = packages
    torch.save(model, save_path)
    print(f"\nmodel file has been saved at {save_path}")


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', '-mp', type=str, default='',
                        help="""this is the location of the model
                                that will used to make predictions""")
    parser.add_argument('--data_path', '-dp', type=str, default='',
                        help="""where the data will load from. The default is
                                the path saved in the model""")
    parser.add_argument('--model_args', '-args', type=str, default='',
                        help="""file specifying the model_args; default is in
                                the model_suite directory""")

    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    args = parser.parse_args()

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
        print('Running with:')
        for key in valid_params.model_args['abstain_kwargs']:
            print(key ,": ", valid_params.model_args['abstain_kwargs'][key])
    else:
        dac = None

    data_loaders = dw.make_torch_dataloaders(valid_params.model_args['data_kwargs']['add_noise'],
                                             reproducible=valid_params.model_args['data_kwargs']['reproducible'],
                                             seed=seed)

    # 3. create a model
    print("\nDefining a model")
    if len(args.model_path) == 0:
        model = create_model(valid_params, dw, device)
        print(f"Device {device}")
        sample = data_loaders['train']
        # sample
        for sample in data_loaders['train']:
            pass
        X = sample["X"].to(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                model(X)
        print(prof.key_averages().table(sort_by="cpu_time_total",row_limit=50))


if __name__ == "__main__":
    main()
