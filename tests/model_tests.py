import argparse
import json
import os
import random
import sys 

import pandas as pd
import numpy as np

from validate import exceptions
from validate import validate_params
from data_loaders import data_utils
from abstention import abstention
from keywords import keywords
from models import mthisan, mtcnn
from training import training
from predict import predictions

import torch

torch.backends.cudnn.benchmark = True

def get_params():
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
    args = parser.parse_args()
    cache_class = [False]
    data_source = 'pre-generated'

    valid_params = validate_params.ValidateParams(cache_class, args, data_source=data_source)

    valid_params.check_data_train_args()

    if valid_params.model_args['model_type'] == 'mthisan':
        valid_params.hisan_arg_check()
    elif valid_params.model_args['model_type'] == 'mtcnn':
        valid_params.mtcnn_arg_check()

    if valid_params.model_args['abstain_kwargs']['abstain_flag']:
        valid_params.check_abstain_args()

    valid_params.check_data_files(args.data_path)

    if valid_params.model_args['data_kwargs']['reproducible']:
        seed = valid_params.model_args['data_kwargs']['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None
    return valid_params, seed

def load_data(valid_params, seed):
    print("Loading data and creating DataLoaders")
    cache_class = [False]
    data_source = 'pre-generated'
    dw = data_utils.DataHandler(data_source, valid_params.model_args, cache_class)
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

    if valid_params.model_args['train_kwargs']['keywords']:
        kwds = keywords.Keywords(valid_params.model_args['data_kwargs']['tasks'],
                                 dw.dict_maps['id2word'],
                                 dw.dict_maps['id2label'],
                                 device)
        kwds.load_keyword_lists()
    return data_loaders, dw, device

def get_trainer(valid_params, dw, device):
    print("\nDefining a model")
    model = create_model(valid_params, dw, device)

    print("Creating model trainer")
    batch = valid_params.model_args['data_kwargs']['batch_per_gpu']

    trainer = training.ModelTrainer(valid_params.model_args,
                                    model,
                                    class_weights=dw.weights,
                                    device=device)
    return model, trainer

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

def test_loss_multiple(params, seed, data_loaders, dw, device, mode ):
    print('------testing loss multiple tasks')
    params.model_args['train_kwargs']['class_weights'] = None
    if mode=='train':
        run_train_test(params, seed, data_loaders, dw, device, 5)     
    elif mode=='evaluate':
        run_eval_test(params, seed, data_loaders, dw, device, 5)

def test_weighted_loss_multiple(params, seed, data_loaders, dw, device, mode ):
    print('------testing weighted loss multiple tasks')
    params.model_args['train_kwargs']['class_weights'] = 'weights/random_weights.pickle'
    model, trainer = get_trainer(params, dw, device)
    if mode=='train':
        run_train_test(params, seed, data_loaders, dw, device, 5)     
    elif mode=='evaluate':
        run_eval_test(params, seed, data_loaders, dw, device, 5)

def test_weighted_loss_single(params, seed, data_loaders, dw, device, mode ):
    print('------testing weighted loss single task')
    params.model_args['train_kwargs']['class_weights'] = 'weights/random_weights.pickle'
    if mode=='train':
        run_train_test(params, seed, data_loaders, dw, device, 1)     
    elif mode=='evaluate':
        run_eval_test(params, seed, data_loaders, dw, device, 1)

def test_loss_single(params, seed, data_loaders, dw, device, mode ):
    print('------testing loss single task')
    params.model_args['train_kwargs']['class_weights'] = None
    if mode=='train':
        run_train_test(params, seed, data_loaders, dw, device, 1)     
    elif mode=='evaluate':
        run_eval_test(params, seed, data_loaders, dw, device, 1)

def run_train_test(params, seed, data_loaders, dw, device, nloss):
    model, trainer = get_trainer(params, dw, device)
    assert len(trainer.loss_funs) == nloss 
    for batch in data_loaders['train']:
        loss = trainer.compute_loss(batch, dac=None)
        loss.backward()
        break

def run_eval_test(params, seed, data_loaders, dw, device, nloss):
    model, trainer = get_trainer(params, dw, device)
    evaluator = predictions.ScoreModel(params.model_args, data_loaders, model, device)
    assert len(evaluator.loss_funs) == nloss 
    evaluator.evaluate_model(dw.metadata['metadata'], dw.dict_maps['id2label'], dac=None)

if __name__=='__main__':
    ## trainer tests
    print('*****************************')
    print('***********Testing trainer...')
    # test multitask, weighted and unweighted
    params,seed = get_params()
    data_loaders, dw, device = load_data(params, seed)
    test_weighted_loss_multiple(params, seed, data_loaders, dw, device, mode='train')
    test_loss_multiple(params, seed, data_loaders, dw, device, mode='train')

    # test singletask, weighted and unweighted
    params.model_args['data_kwargs']['tasks'] = ['site']
    data_loaders, dw, device = load_data(params, seed)
    test_weighted_loss_single(params, seed, data_loaders, dw, device, mode='train')
    test_loss_single(params, seed, data_loaders, dw, device, mode='train')
    print('************...done')

    ## prediction tests
    print('***********Testing evaluate...')
    # test model predictions, weighted and unweighted
    params,seed = get_params()
    data_loaders, dw, device = load_data(params, seed)
    test_weighted_loss_multiple(params, seed, data_loaders, dw, device, mode='evaluate')
    test_loss_multiple(params, seed, data_loaders, dw, device, mode='evaluate')

    # test singletask, weighted and unweighted
    params.model_args['data_kwargs']['tasks'] = ['site']
    data_loaders, dw, device = load_data(params, seed)
    test_weighted_loss_single(params, seed, data_loaders, dw, device, mode='evaluate')
    test_loss_single(params, seed, data_loaders, dw, device, mode='evaluate')
    print('***********...done')

