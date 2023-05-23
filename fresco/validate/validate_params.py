"""

Module for sanity checking parameters for data and model building.

"""

import datetime
import json
import math
import os

import yaml

from fresco.validate import exceptions


class ValidateParams():
    """
    Class to validate model-specific parameters for MOSSAIC models.

    Args:
        cli_args: argparse list of command line args.
        data_source (str): Indicates where the data will come from. Should be one of:
            - pre-generated: data_args.yml will indicate the source.

    Post-condition: model_args dict loaded and sanity checked.
    """

    def __init__(self, cli_args,
                 data_source: str = 'pre-generated',
                 model_args: dict = None):

        if model_args is None:
            if len(cli_args.model_args) > 0:
                mod_args_file = cli_args.model_args
            else:
                mod_args_file = 'model_args.yml'
            print(cli_args.model_args)
            if os.path.isfile(mod_args_file):
                with open(mod_args_file, "r", encoding="utf-8") as f_in:
                    self.model_args = yaml.safe_load(f_in)
            else:
                raise exceptions.ParamError("within FrESCO the " +
                                            "model_args.yml file is needed to set " +
                                            "the model arguments")
        else:
            self.model_args = model_args

        if self.model_args['model_type'] not in ['mtcnn', 'mthisan']:
            raise exceptions.ParamError("model type was not found " +
                                        "to be 'mtcnn' or 'mthisan'. " +
                                        "Currently these are the only expected options.")

        if self.model_args['save_name'] == "":
            self.save_name = f'model_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        else:
            self.save_name = self.model_args['save_name']

        self.save_name = os.path.join('savedmodels', self.save_name)

        if not os.path.exists(os.path.dirname(self.save_name)):
            print(f'savepath {os.path.dirname(self.save_name)} does not exist, creating it')
            os.makedirs(os.path.dirname(self.save_name))

        fold = self.model_args['data_kwargs']['fold_number']

        if len(cli_args.data_path) > 0:
            self.model_args['data_kwargs']['data_path'] = cli_args.data_path

        if not os.path.exists(self.model_args['data_kwargs']['data_path']):
            raise exceptions.ParamError("User provided data path does not exist.")

        if isinstance(fold, int):
            self.model_args['data_kwargs']['fold_number'] = fold
        else:
            raise exceptions.ParamError("Model building does not presently support > 1 fold.")

        if (self.model_args['data_kwargs']['subset_proportion'] > 1 or
                self.model_args['data_kwargs']['subset_proportion'] <= 0):
            raise exceptions.ParamError("subset proportion must be float value between 0 and 1.")

        if not isinstance(self.model_args['data_kwargs']['batch_per_gpu'], int):
            raise exceptions.ParamError("Batch size must be an int value.")
        if self.model_args['data_kwargs']['batch_per_gpu'] < 0 or \
            self.model_args['data_kwargs']['batch_per_gpu'] > 2048:
            raise exceptions.ParamError("Batch size must be an int value between 1 and 2048.")

        if not isinstance(self.model_args['train_kwargs']['mixed_precision'], bool):
            raise exceptions.ParamError("Mixed precision must be boolean.")

    def hisan_arg_check(self):
        """
        Check and modify HiSAN specific args.

        Parameters: none

        Pre-condition: self.model_args is not None

        Post-condition:
            self.model_args['MTHiSAN_kwargs']['max_lines'] modified to be the ceiling of the doc_max_len / max_words_per_line.
            self.model_args['train_kwargs']['doc_max_len'] modified to be max_words_per_line * max_lines.
        """

        self.model_args['MTHiSAN_kwargs']['max_lines'] = \
            math.ceil(self.model_args['train_kwargs']['doc_max_len'] //
                      self.model_args['MTHiSAN_kwargs']['max_words_per_line'])

        self.model_args['train_kwargs']['doc_max_len'] = \
            self.model_args['MTHiSAN_kwargs']['max_words_per_line'] *\
            self.model_args['MTHiSAN_kwargs']['max_lines']

        if self.model_args['train_kwargs']['class_weights'] is not None:
            self.check_weights()

    def mtcnn_arg_check(self):
        """
        Check the number of filters matchesthe number of windows.
        """
        if len(self.model_args['MTCNN_kwargs']['num_filters']) != \
                len(self.model_args['MTCNN_kwargs']['window_sizes']):
            raise exceptions.ParamError("Number of filters must match the number of window_sizes.")

        if self.model_args['train_kwargs']['class_weights'] is not None:
            self.check_weights()

    def check_data_train_args(self, from_pretrained=False):
        """
        Verify arguments are appropriate for the chosen model options.

        Parameters:
            from_pretrained (bool): Checking model args from a pretrained model. Pretrained model args are different,
            some are copied from data_kwargs to train_kwargs.

        Pre-condition: self.model_args is not None.

        Post-condition: self.model_args['train_kwargs']['doc_max_len'] is updated from the data_kwargs,
        'max_lines' is added to the hisan kw_args.
        """
        schema = {'data_kwargs': ['doc_max_len', 'tasks', 'fold_number', 'data_path',
                                  'mutual_info_filter', 'mutual_info_threshold',
                                  'subset_proportion', 'add_noise', 'add_noise_flag',
                                  'multilabel', 'random_seed', 'batch_per_gpu', 'reproducible'],
                  'MTHiSAN_kwargs': ['max_words_per_line',
                                     'att_heads', 'att_dim_per_head',
                                     'att_dropout', 'bag_of_embeddings', 'embeddings_scale'],
                  'MTCNN_kwargs': ['window_sizes', 'num_filters',
                                   'dropout', 'bag_of_embeddings', 'embeddings_scale'],
                  'train_kwargs': ['max_epochs', 'patience', 'mixed_precision',
                                   'class_weights']}

        if from_pretrained:
            schema['train_kwargs'] = ['batch_per_gpu', 'class_weights', 'doc_max_len',
                                      'max_epochs', 'multilabel', 'patience',
                                      'random_seed', 'reproducible', 'mixed_precision']
            schema['MTHiSAN_kwargs'] = ['max_words_per_line', 'max_lines',
                                        'att_heads', 'att_dim_per_head',
                                        'att_dropout', 'bag_of_embeddings', 'embeddings_scale']


        model_kwds = ['MTCNN_kwargs', 'MTHiSAN_kwargs', 'Transformers_kwargs',
                      'abstain_kwargs', 'data_kwargs', 'model_type', 'save_name',
                      'task_unks', 'train_kwargs']

        if sorted(self.model_args.keys()) != model_kwds:
            print("\nReceived: ", sorted(self.model_args.keys()))
            print("Expected: ", model_kwds)
            raise exceptions.ParamError("model_arg keys do not match the schema")

        for kwrd, vals in schema.items():
            if kwrd == 'abstain_kwargs':
                continue  # these are checked in a separate function
            if sorted(self.model_args[kwrd]) != sorted(vals):
                print("\nReceived: ", sorted(self.model_args[kwrd]))
                print("Expected: ", sorted(vals))
                raise exceptions.ParamError((f"model args {kwrd} does not have " +
                                            "the expected variables"))

        # copy data kwargs to train kwds
        copy_kwds = ['doc_max_len', 'batch_per_gpu', 'random_seed', 'multilabel', 'reproducible']
        for word in copy_kwds:
            self.model_args['train_kwargs'].update([(word, self.model_args['data_kwargs'][word])])

    def check_abstain_args(self):
        """
        Verify keywords needed for abstention to work are present and valid.
        """
        abstain_kwargs = ['abstain_flag', 'alphas', 'max_abs', 'min_acc',
                          'abs_gain', 'acc_gain', 'alpha_scale',
                          'tune_mode', 'stop_limit', 'stop_metric',
                          'ntask_flag', 'ntask_tasks', 'ntask_alpha',
                          'ntask_alpha_scale', 'ntask_max_abs', 'ntask_min_acc']

        if self.model_args['train_kwargs']['class_weights'] is not None and self.model_args['abstain_kwargs']['abstain_flag']:
            raise exceptions.ParamError("Class weighting is not presently implemented for the DAC.")

        if sorted(abstain_kwargs) != sorted(self.model_args['abstain_kwargs']):
            print("\nReceived: ", sorted(self.model_args['abstain_kwargs']))
            print("Expected: ", sorted(abstain_kwargs))
            raise exceptions.ParamError(("model args abstain_kwargs does not have " +
                                        "the expected variables"))

        if set(self.model_args['abstain_kwargs']['alphas'].keys()).isdisjoint(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Alpha tasks are not a subset of the data tasks.")

        if len(self.model_args['abstain_kwargs']['alphas']) > \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of abstain alphas is greater than number of tasks.")

        if len(self.model_args['abstain_kwargs']['max_abs']) != \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of max abstain rates is different than number of tasks.")

        if len(self.model_args['abstain_kwargs']['min_acc']) != \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of min acc rates is different than number of tasks.")

        if len(self.model_args['abstain_kwargs']['alpha_scale']) != \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of alpha scales is different than number of tasks.")

        if self.model_args['abstain_kwargs']['ntask_flag'] and not\
            self.model_args['abstain_kwargs']['abstain_flag']:
            raise exceptions.ParamError("Ntask cannot be enables without Abstention")

        if self.model_args['abstain_kwargs']['ntask_flag']:
            if set(self.model_args['abstain_kwargs']['ntask_tasks']).isdisjoint(self.model_args['data_kwargs']['tasks']):
                raise exceptions.ParamError("Ntask tasks are not a subset of the data tasks.")

    def check_keyword_args(self):
        """
        Validate keyword args.
        """
        tasks = ['histology', 'laterality', 'site', 'subsite', 'behavior']
        if not set(self.model_args['data_kwargs']['tasks']).issubset(tasks):
            raise exceptions.ParamError("Keywords are only available for: " +
                                        "histology, laterality, site, subsite, behavior")

    def check_weights(self):
        """
        Validate class weights path exists.
        """
        if isinstance(self.model_args['train_kwargs']['class_weights'], str):
            path = self.model_args['train_kwargs']['class_weights']
            if not os.path.exists(path):
                raise exceptions.ParamError("Invalid path; please provide a valid path for " +
                                            "class weights")

    def check_data_files(self, data_path=None):
        """
        Verify the necessary data files exist.

        Args:
            data_path (str): From argparser, optional path to dataset.

        Note: Setting data_path will override the path set in model_args.yml.
        """
        data_files = ['data_fold.csv', 'word_embeds_fold.npy', 'id2labels_fold.json']

        if data_path is not None:  # reading from argparser
            if os.path.exists(os.path.dirname(data_path)):
                _data_path = data_path
            else:
                raise exceptions.ParamError(f'user defined data_path {data_path} does not exist, exiting')
        else:
            _data_path = self.model_args['data_kwargs']['data_path']
        fold = self.model_args['data_kwargs']['fold_number']

        tasks = self.model_args['data_kwargs']['tasks']
        with open(os.path.join(_data_path, 'id2labels_fold' + str(fold) + '.json'),
                  'r', encoding='utf-8') as f:
            tmp = json.load(f)

        id2label = {task: {int(k): str(v) for k, v in labels.items()}
                                   for task, labels in tmp.items()}
        if sorted(tasks) != sorted(id2label.keys()):
            raise exceptions.ParamError((' the tasks in model_args file must match the ' +
                                         'tasks in the data.'))

        for f_in in data_files:
            data_file = os.path.join(_data_path, f_in.replace('fold', f'fold{fold}'))
            if f_in == 'word_embeds_fold.npy' and not os.path.isfile(data_file):
                print("Word embeddings file does not exist; will default to random embeddings.")
            elif not os.path.isfile(data_file):
                raise exceptions.ParamError(f' the file {data_file} does not exist.')


class ValidateClcParams():
    """
    Class to validate model-specific parameters for MOSSAIC models.

    Args:
        cli_args: Argparse list of command line args.
        data_source (str): Indicates where the data will come from. Should be one of:
            - pre-generated: data_args.yml will indicate the source.

    Post-condition: model_args dict loaded and sanity checked.
    """

    def __init__(self, cli_args, data_source: str = 'pre-generated'):

        if len(cli_args.model_args) > 0:
            mod_args_file = cli_args.args
        else:
            mod_args_file = 'clc_args.yml'

        if os.path.isfile(mod_args_file):
            with open(mod_args_file, "r", encoding="utf-8") as f_in:
                self.model_args = yaml.safe_load(f_in)
        else:
            raise exceptions.ParamError("within the Model_Suite the " +
                                        "clc_args.yml file is needed to set " +
                                        "the model arguments")

        if self.model_args['save_name'] == "":
            self.save_name = f'model_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        else:
            self.save_name = self.model_args['save_name']

        self.save_name = os.path.join('savedmodels', self.save_name)

        if not os.path.exists(os.path.dirname(self.save_name)):
            print(f'savepath {os.path.dirname(self.save_name)} does not exist, creating it')
            os.makedirs(os.path.dirname(self.save_name))

        if (self.model_args['data_kwargs']['subset_proportion'] > 1 or
                self.model_args['data_kwargs']['subset_proportion'] <= 0):
            raise exceptions.ParamError("subset proportion must be float value between 0 and 1.")

        if not isinstance(self.model_args['train_kwargs']['batch_per_gpu'], int):
            raise exceptions.ParamError("Batch size must be an int value.")
        if self.model_args['train_kwargs']['batch_per_gpu'] < 0 or \
            self.model_args['train_kwargs']['batch_per_gpu'] > 2048:
            raise exceptions.ParamError("Batch size must be an int value between 1 and 2048.")

    def clc_arg_check(self):
        """
        Check and modify HiSAN specific args.

        Parameters: none

        Pre-condition: self.model_args is not None

        Post-condition:
            self.model_args['MTHiSAN_kwargs']['max_lines'] modified to be the ceiling of doc_max_len / max_words_per_line.
            self.model_args['train_kwargs']['doc_max_len'] modified to be max_words_per_line * max_lines.
        """

        if self.model_args['model_kwargs']['att_dropout'] > 1 or \
            self.model_args['model_kwargs']['att_dropout'] < 0:
            raise exceptions.ParamError("Attn dropout must be between 0 and 1")

        if not isinstance(self.model_args['train_kwargs']['att_heads'], int):
            raise exceptions.ParamError("Attn heads mut be an int between 1 and 16")

        if self.model_args['train_kwargs']['att_heads'] > 16 or \
            self.model_args['train_kwargs']['att_heads'] < 1:
            raise exceptions.ParamError("Attn heads mut be an int between 1 and 16")

        if not isinstance(self.model_args['train_kwargs']['att_dim_per_head'], int):
            raise exceptions.ParamError("Attn dim per head mut be an int between 1 and 16")

        if self.model_args['train_kwargs']['att_dim_per_head'] > 100 or \
            self.model_args['train_kwargs']['att_dim_per_head'] < 1:
            raise exceptions.ParamError("Attn dim per head mut be an int between 1 and 100")

    def check_data_train_args(self):
        """
        Verify arguments are appropriate for the chosen model options.

        Parameters: none

        Pre-condition: self.model_args is not None.

        Post-condition: self.model_args['train_kwargs']['doc_max_len'] is updated from the data_kwargs.
        """

        schema = {'data_kwargs': ['tasks', 'exclude_single', 'shuffle_case_order',
                                  'subset_proportion', 'model_path',
                                  'random_seed', 'reproducible'],
                  'model_kwargs': ['att_dim_per_head', 'att_heads',
                                   'att_dropout', 'forward_mask'],
                  'train_kwargs': ['batch_per_gpu', 'max_epochs', 'patience', 'mixed_precision', 'class_weights']}

        model_kwds = ['model_kwargs', 'abstain_kwargs', 'data_kwargs',
                      'save_name', 'train_kwargs']

        if sorted(self.model_args.keys()) != sorted(model_kwds):
            print("\nReceived: ", sorted(self.model_args.keys()))
            print("Expected: ", model_kwds)
            raise exceptions.ParamError("model_arg keys do not match the schema")

        for kwrd, vals in schema.items():
            if kwrd == 'abstain_kwargs':
                continue  # these are checked in a separate function
            if sorted(self.model_args[kwrd]) != sorted(vals):
                print("\nReceived: ", sorted(self.model_args[kwrd]))
                print("Expected: ", sorted(vals))
                raise exceptions.ParamError((f"model args {kwrd} does not have " +
                                            "the expected variables"))

        # copy data kwargs to train kwds
        copy_kwds = ['random_seed', 'reproducible']
        for word in copy_kwds:
            self.model_args['train_kwargs'].update([(word, self.model_args['data_kwargs'][word])])

    def check_abstain_args(self):
        """
        Verify keywords needed for abstention to work are present and valid.
        """
        abstain_kwargs = ['abstain_flag', 'alphas', 'max_abs', 'min_acc',
                          'abs_gain', 'acc_gain', 'alpha_scale',
                          'tune_mode', 'stop_limit', 'stop_metric',
                          'ntask_flag', 'ntask_tasks', 'ntask_alpha',
                          'ntask_alpha_scale', 'ntask_max_abs', 'ntask_min_acc']

        if sorted(abstain_kwargs) != sorted(self.model_args['abstain_kwargs']):
            print("\nReceived: ", sorted(self.model_args['abstain_kwargs']))
            print("Expected: ", sorted(abstain_kwargs))
            raise exceptions.ParamError(("model args abstain_kwargs does not have " +
                                        "the expected variables"))

        if set(self.model_args['abstain_kwargs']['alphas'].keys()).isdisjoint(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Alpha tasks are not a subset of the data tasks.")

        if len(self.model_args['abstain_kwargs']['alphas']) > \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of abstain alphas is greater than number of tasks.")

        if len(self.model_args['abstain_kwargs']['max_abs']) != \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of max abstain rates is different than number of tasks.")

        if len(self.model_args['abstain_kwargs']['min_acc']) != \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of min acc rates is different than number of tasks.")

        if len(self.model_args['abstain_kwargs']['alpha_scale']) != \
                len(self.model_args['data_kwargs']['tasks']):
            raise exceptions.ParamError("Number of alpha scales is different than number of tasks.")

        if self.model_args['abstain_kwargs']['ntask_flag'] and not\
            self.model_args['abstain_kwargs']['abstain_flag']:
            raise exceptions.ParamError("Ntask cannot be enables without Abstention")

        if self.model_args['abstain_kwargs']['ntask_flag']:
            if set(self.model_args['abstain_kwargs']['ntask_tasks']).isdisjoint(self.model_args['data_kwargs']['tasks']):
                raise exceptions.ParamError("Ntask tasks are not a subset of the data tasks.")

    def check_data_files(self, data_path):
        """
        Verify the necessary data files exist.

        Args:
            data_path (str): From argparser, optional path to dataset.

        Note: Setting data_path will override the path set in model_args.yml.
        """
        data_files = ['data_fold.csv', 'word_embeds_fold.npy', 'id2labels_fold.json']

        if data_path is not None:  # reading from argparser
            if os.path.exists(os.path.dirname(data_path)):
                _data_path = data_path
            else:
                raise exceptions.ParamError(f'user defined data_path {data_path} does not exist, exiting')
        else:
            _data_path = self.model_args['data_kwargs']['data_path']
        # fold = self.model_args['data_kwargs']['fold_number']
        fold = 0

        tasks = self.model_args['data_kwargs']['tasks']
        with open(os.path.join(_data_path, 'id2labels_fold' + str(fold) + '.json'),
                  'r', encoding='utf-8') as f:
            tmp = json.load(f)

        id2label = {task: {int(k): str(v) for k, v in labels.items()}
                                   for task, labels in tmp.items()}
        if sorted(tasks) != sorted(id2label.keys()):
            raise exceptions.ParamError((' the tasks in model_args file must match the ' +
                                         'tasks in the data.'))

        for f_in in data_files:
            data_file = os.path.join(_data_path, f_in.replace('fold', f'fold{fold}'))
            if f_in == 'word_embeds_fold.npy' and not os.path.isfile(data_file):
                print("Word embeddings file does not exist; will default to random embeddings.")
            elif not os.path.isfile(data_file):
                raise exceptions.ParamError(f' the file {data_file} does not exist.')
