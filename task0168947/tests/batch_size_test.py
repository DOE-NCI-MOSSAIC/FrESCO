"""
    Top-level model building script using independent modules.
"""
import argparse
import json
import os
import random

import torch

import numpy as np

from validate import exceptions
from validate import validate_params
from data_loaders import data_utils
from abstention import abstention
from keywords import keywords
from models import mthisan, mtcnn
from training import training
from predict import predictions

# needed for running on the vms
# torch.multiprocessing.set_sharing_strategy('file_system')


def load_model_dict(model_path, valid_params, data_path=""):
    """Load pretrained model from disk.

        Args:
            model_path: str, from command line args, points to saved model
            valid_params: ValidateParams class, with model_args dict
            data_path: str or None, using data from the trained model, or different one

        We check if the supplied path is valid and if the packages match needed
            to run the pretrained model.

    """
    if os.path.exists(model_path):
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        raise exceptions.ParamError("Provided model path does not exist")
    if len(data_path) > 0:
        with open(data_path + 'metadata.json', 'r', encoding='utf-8') as f:
            data_args = json.load(f)
    #else:
        #if 'model_metadata' in model_dict['metadata_package'].keys():
            #data_args = model_dict['metadata_package']['model_metadata']
        #elif 'Model_Suite' in model_dict['metadata_package'].keys():
            #data_args = model_dict['metadata_package']['model_metadata']

    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
    else:
        raise exceptions.ParamError(f'the model at {model_path} does not exist.')

    # check that model args all agree
    # this needs to be looked ate carefully, not sure it deos what we want. AS-12/5
    # need to check abstention args especially

    # mismatches = [mod for mod in model_dict['metadata_package']['model_metadata'].keys()
    #    if data_args[mod] != model_dict['metadata_package']['model_metadata'][mod]]
    mismatches = []
    # check to see if the stored package matches the expected one
    if len(mismatches) > 0:
        with open('metadata_package.json', 'w', encoding='utf-8') as f_out:
            json.dump(model_dict['metadata_package'], f_out, indent=2)
            raise exceptions.ParamError(f'the package(s) {", ".join(mismatches)} does not match ' +
                                        f'the generated data in {data_path}.' +
                                         '\nThe needed recreation info is in metadata_package.json')

    return model_dict


def load_model(model_dict, device, dw):

    # if torch.cuda.device_count() < 2:
    #     raise exceptions.ParamError("Case level context requires > 2 gpus")

    # model_args = model_dict['metadata_package']['Model_Suite']
    model_args = model_dict['metadata_package']['mod_args']

    if model_args['model_type'] == 'mthisan':
        model = mthisan.MTHiSAN(dw.inference_data['word_embedding'],
                                dw.num_classes,
                                **model_args['MTHiSAN_kwargs'])

    elif model_args['model_type'] == 'mtcnn':
        model = mtcnn.MTCNN(dw.inference_data['word_embedding'],
                            dw.num_classes,
                            **model_args['MTCNN_kwargs'])

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_dict = {k: v for k,v in model_dict.items() if k!='metadata_package'}
    model_dict = {k.replace('module.',''): v for k,v in model_dict.items()}
    model.load_state_dict(model_dict)

    print('model loaded')

    return model


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', '-mp', type=str, default='',
                        help="""this is the location of the model
                                that will used to make predictions""")
    parser.add_argument('--data_path', '-dp', type=str, default='',
                        help="""where the data will load from. The default is
                                the path saved in the model""")
    parser.add_argument('--model_args', '-args', type=str, default='',
                        help="""path to specify the model_args; default is in
                                the model_suite directory""")

    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    args = parser.parse_args()

    if len(args.model_path) == 0 or len(args.data_path) == 0:
        raise exceptions.ParamError("Model and/or data path cannot be empty, please specify both.")

    # 1. validate model/data args
    # print("Validating kwargs in model_args.yml file")
    cache_class = [False]
    data_source = 'pre-generated'
    # use the model args file from training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = load_model_dict(args.model_path, device)
    mod_args = model_dict['metadata_package']['mod_args']

    print("Validating kwargs from pretrained model ")
    model_args = validate_params.ValidateParams(cache_class, args,
                                                data_source=data_source,
                                                model_args=mod_args)

    model_args.check_data_train_args(from_pretrained=True)
    if model_args.model_args['model_type'] == 'mthisan':
        model_args.hisan_arg_check()
    elif model_args.model_args['model_type'] == 'mtcnn':
        model_args.mtcnn_arg_check()

    model_args.check_data_files()

    seed = model_args.model_args['data_kwargs']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. load data
    print("Loading data and creating DataLoaders")

    dw = data_utils.DataHandler(data_source, model_args.model_args, cache_class)
    dw.load_folds(fold=0)

    for n in range(8):
        batch = 2**n
        print(f"\nPredicting with batch size {batch}")
        data_loaders = dw.inference_loader(reproducible=True,
                                           seed=seed, batch_size=batch)
        dac = None

        model = load_model(model_dict, device, dw)
        savepath = f"batch_size_{batch}_"
        evaluator = predictions.ScoreModel(model_args.model_args, data_loaders, dw,
                                           model, device, savepath=savepath)

        evaluator.predict(dw.metadata['metadata'],
                          dw.dict_maps['id2label'],
                          save_probs=True)



if __name__ == "__main__":
    main()
