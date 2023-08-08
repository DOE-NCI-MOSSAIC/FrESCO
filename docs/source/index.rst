.. FrESCO documentation master file, created by
   sphinx-quickstart on Thu May 18 17:04:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FrESCO: Framework for Exploring Scalable Computational Oncology
===============================================================

.. image:: logo.png
  :width: 400
  :alt: FrESCO logo


Motivation
----------

The National Cancer Institute (NCI) monitors population level cancer
trends as part of its Surveillance, Epidemiology, and End Results (SEER)
program. This program consists of state or regional level cancer
registries which collect, analyze, and annotate cancer pathology
reports. From these annotated pathology reports, each individual
registry aggregates cancer phenotype information and summary statistics
about cancer prevalence to facilitate population level monitoring of
cancer incidence. Extracting cancer phenotype from these reports is a
labor intensive task, requiring specialized knowledge about the reports
and cancer. Automating this information extraction process from cancer
pathology reports has the potential to improve not only the quality of
the data by extracting information in a consistent manner across
registries, but to improve the quality of patient outcomes by reducing
the time to assimilate new data and enabling time-sensitive applications
such as precision medicine. Here we present FrESCO: Framework for
Exploring Scalable Computational Oncology, a modular deep-learning
natural language processing (NLP) library for extracting pathology
information from clinical text documents.

Quickstart Guide
----------------

Install via pip
~~~~~~~~~~~~~~~

Install from `pypi <https://pypi.org/project/nci-fresco/>`__ using

.. code:: shell

   pip install nci-fresco

Install from source
~~~~~~~~~~~~~~~~~~~

Clone the repo

.. code:: shell

   git clone https://github.com/DOE-NCI-MOSSAIC/FrESCO.git

Load the working branch and pull in the subrepos

.. code:: shell

   cd FrESCO
   git checkout main

Setup the conda environment (the default name for the environment is
“ms39”, this can be edited in the ms39.yml file)

.. code:: shell

   conda env create -n fresco python=3.9
   conda activate fresco

Install the FrESCO library and dependencies

.. code:: shell

   pip install .

For further PyTorch instructions or more details for pytorch, head over
to the `PyTorch docs <https://pytorch.org/docs/stable/index.html>`__.

Notebooks and Examples
~~~~~~~~~~~~~~~~~~~~~~

We have supplied `example notebooks <https://github.com/DOE-NCI-MOSSAIC/FrESCO/tree/main/notebooks>`__ for each of the sample datasets
contained in the `Github repository <https://github.com/DOE-NCI-MOSSAIC/FrESCO>`__ showing model training on each dataset. We
have also supplied ``model_args`` files for each of the datasets
contained within the repo to speedup the time to ge up and running with
the codebase.

Data Preparation
~~~~~~~~~~~~~~~~

We have supplied three different datasets as examples, each must be
unzipped before any model training via the ``tar -xf dataset.tar.gz``
command from the ``data`` directory. The three datasets are: - imdb:
binary sentiment classification with the imdb dataset, - P3B3: benchmark
multi-task classification task, and - clc: case-level context multi-task
classfication data.

We have prepared ``model_args`` files for each dataset within the
``configs/`` directory. To run with your own data, the following
instructions explain the requirements for the training data. Model 
arguments are described in the subsequent section. 

Add the path to the desired dataset in the the ``data_path`` argument in
the ``configs/model_args.yml`` file. The required data files are: -
``data_fold0.csv``: Pandas dataframe with columns: - ``X``: list of
input values, of ``int`` type - ``task_n``: output for task ``n``, a
``string`` type (these are the y-values) - ``split``: one of ``train``,
``test``, or ``val`` - ``id2labels_fold0.json``: index to label
dictionary mapping for each of the string representations of the outputs
to an integer value, dict keys must match the y-values label -
``word_embeds_fold0.npy``: word embedding matrix for the vocabulary,
dimensions are ``words x embedding_dim``. If no word embedding exists,
the model will use randomly generated embeddings.

For the ``P3B3`` and ``clc`` datasets we have provided the required
files. To create these files for the ``imdb`` dataset, you may run the
``data_setup.py`` script within the ``scripts`` folder to create these
necessary files for model training.

| You will also need to set the ``tasks``, these must correspond to the
  task columns names in the ``data_fold0.csv`` file and keys in the
  ``id2labels_fold0.json`` dictionary. For example, in the P3B3 data,
  the task columns are ``task_n, n = 1,2,3,4``. Whereas the imdb data
  has the ``sentiment`` task. If using any sort of class weighting
  scheme, the keyword ``class_weights`` must be either a pickle file or
  dictionary with keys corresponding to the task and value with a
  corresponding list, or numpy array, of weights for that task.
| If the ``class_weights`` keyword is blank, corresponding to ``None``,
  no class weighting scheme will be used during training nor inference.

If working with hierarchical data, and the case-level context model is
the desired output, then the dataframe in the corresponding
``data_fold0.csv`` must contain an additional integer-valued column
``group`` where the values describe the hierarchy present within the
data. For example, all rows where ``group = 17`` are associated for the
purpose of training a case-level context model.

Model Training
~~~~~~~~~~~~~~

The ``model_args.yaml`` file controls the settings for model training.
Edit this file as desired based on your requirements and desired
outcome. The ``"save_name"`` entry controls the name used for model
checkpoints and prediction outputs; if left empty, a datetimestamp will
be used.

The following commands allow setting your GPUs, if enabled, before
training your model.

.. code:: shell

   nvidia-smi                             #check GPU utilization
   export CUDA_VISIBLE_DEVICES=0,1        #replace 0,1 with the GPUs you want to use
   echo $CUDA_VISIBLE_DEVICES             #check which GPUs you have chosen

To train the model for any information extraction task, multi-task
calssification, simply run

.. code:: shell

   python train_model.py -m ie -args ../configs/model_args.yml

from the ``scripts`` directory.

We have supplied test data for each of the model types provided.
Information extraction models may be created with either ``P3B3`` or
``imdb`` data.

Case-Level Context Model
^^^^^^^^^^^^^^^^^^^^^^^^

If you’re wanting a case-level context model, there is a two-step
process. See ``notebooks/clc_example.ipynb`` for a fully worked example.

Step 1: Create an information extraction model specifying the
``data/clc`` data directory in the ``configs/model_args.yml``. Then run

.. code:: shell

   python train_model.py -m ie -args ../configs/clc_step1.yml

from the ``scripts/`` directory.

Step 2: To train a clc model, set the ``model_path`` keyword arg to the
path of the trained model trained from step 1 step in the
``configs/clc_step2.yml`` file. Then run

.. code:: shell

   python train_model.py -m clc -args ../configs/clc_step2.yml

from the ``scripts/`` directory to train a case-level context model.

Note that the case level context model requires a pre-trained
information extraction model to be specified in the
``configs/clc_args.yml`` file. The default setting, if the ``-m``
argument is omitted, is information extraction, and which task is
specified in the ``configs/model_args.yml`` file.

Deep-Abstaining Classifier and Ntask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both information extraction and case-level context models have the
ability to incorporate abstention or Ntask. The deep abstaining
classifier (DAC) from the
`CANDLE <https://github.com/ECP-CANDLE/Candle>`__ repository, allows the
model to not make a prediction on a sample if the softmax score does not
meet a predetermined threshold specified in the min_acc keyword of the
``model_args.yml`` file. It can be tuned to meet a threshold of
accuracy, minimum number of samples abstained, or both through adapting
the values of alpha through the training process. This adapation is
automated in the code, only requires the user to specify the initial
values, tuning mode, and scaling.

Ntask is useful for multi-task learning. It creates and additional
‘task’ that predicts if the softmax scores from all of the tasks do not
meet a specified threshold within the ``model_args`` file . It has its
own parameters that are tuned during the training process to obtain a
minimum of abstained samples and maximum accuracy on the predicted
samples. Ntask may not be enabled without abstention being enabled as
well. The code will throw an exception and halt if such configuration is
passed.

Expected Results
~~~~~~~~~~~~~~~~

Training a model with the supplied default args, we see convergence
within 50-60 epochs with 0.80-0.85 accuracy on the imdb data set and in
excess of 0.90 accuracy across all tasks for the P3B3 dataset within 60
epochs or so. **NOTE:** P3B3 has a known issue training with mixed
precision and with ``DAC`` and ``NTask`` enabled. Ensure these keywords
are all ``False`` for all runs with the P3B3 dataset.

Description of model arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   save_name: # name of the file without a file extension.
   task_unks: # task unknowns; assign classes in the test set this value. Use the form below.
   task_name: 'uknown_label'

   # dataloader settings
   data_kwargs:
     tasks: ["task_1", "task_2", "task_3", "task_4"] # list of the name of tasks for multitask model
     fold_number: 0  # which datafold to load
     data_path: # path to the data
     subset_proportion: 1.0 # subset of the data to use (0-1.0)
     add_noise_flag: # Boolean, whether flip random words in the data set
     add_noise: # fraction of words to flip
     mutual_info_filter: False  # Boolean; filter data based on mutual data
     mutual_info_threshold: # threshold to use
     # copied to train_kwargs
     reproducible: True # sets all random seeds for reproducible train/test split and training features.
     random_seed: 42 # seed to use for reproducibility
     batch_per_gpu: 128 # batches per GPU
     doc_max_len: 3000  # The maximum number of words used to train the model
     multilabel: False
   
   # general training args
   model_type: "mthisan" # mtcnn or mthisan
   train_kwargs:
     max_epochs: 100
     patience: 5
     mixed_precision: False
     class_weights: #  './weights/P3B3_weights.pickle' # path, dict, or blank

   # mthisan model args
   MTHiSAN_kwargs:
     max_words_per_line: 15
     att_heads: 8
     att_dim_per_head: 50
     att_dropout: 0.1
     bag_of_embeddings: False
     embeddings_scale: 2.5

   # mtcnn model args
   MTCNN_kwargs:
     window_sizes: [3,4,5]
     num_filters: [300,300,300]
     dropout: 0.5
     bag_of_embeddings: False
     embeddings_scale: 20

   # abstain args
   abstain_kwargs:
     abstain_flag: False  # bool, whether to allow the model to not make a classification
     abs_gain: 5.0  # factor to modify abstention penalty when tuning for abstention rate
     acc_gain: 10.0  # factor to modify abstention penalty when tuning for accuracy rate
     alphas: {'task_1':5,'task_2':5,'task_3':5,'task_4':5}  # penalty for abstaining on a sample
     max_abs: {'task_1':0.8,'task_2':0.8,'task_3':0.5,'task_4':0.8}  # maximum desired abstention rates
     min_acc: {'task_1':0.975,'task_2':0.975,'task_3':0.975,'task_4':0.975}  # minimum desired accuracy rates
     alpha_scale: {'task_1':0.8,'task_2':0.8,'task_3':0.8,'task_4':0.8}  # scale factor when adapting abstention penalty
     tune_mode: 'acc'  # how to tune abstention, acc: accuracy, abs: abstention rate, abs_acc: both
     stop_limit: 0.005  # threshold for early stopping with abstention
     stop_metric: 'max'  # max: l1 metric for early stopping with abstention
     ntask_flag: False  # bool, whether to allow the model to enable Ntask
     ntask_tasks:  ["task_1", "task_2", "task_3", "task_4"]  # tasks to use for Ntask determination
     ntask_alpha: 0.1  # Ntask penalty
     ntask_alpha_scale: 0.8  # scale to modify Ntask alpha penalty
     ntask_max_abs: 0.9  # maximum desired abstention rate with Ntask
     ntask_min_acc: 0.975  # minimum desired accuracy rate with Ntask


Contributing
~~~~~~~~~~~~

Get in touch if you would like to help in writing code, example
notebooks, and documentation are essential aspects of the project. To
contribute please fork the project, make your proposed changes and
submit a pull request. We will do our best to sort out any issues and
get your contributions merged into the main branch.

If you found a bug, have questions, or are just having trouble with the
library, please open an issue in our issue tracker and we’ll try to help
resolve it.

How to Cite
~~~~~~~~~~~

Please cite our software if you use it in your work. The
bibtex entry is:

::

   @misc{osti_1958817,
   title = {FrESCO},
   author = {Spannaus, Adam and Gounley, John and Hanson, Heidi and Chandra Shekar, Mayanka and Schaefferkoetter, Noah and Mohd-Yusof, Jamaludin and Fox, Zach and USDOE},
   abstractNote = {The National Cancer Institute (NCI) monitors population level cancer trends as part of its Surveillance, Epidemiology, and End Results (SEER) program. This program consists of state or regional level cancer registries which collect, analyze, and annotate cancer pathology reports. From these annotated pathology reports, each individual registry aggregates cancer phenotype information and summary statistics about cancer prevalence to facilitate population level monitoring of cancer incidence. Extracting cancer phenotype from these reports is a labor intensive task, requiring specialized knowledge about the reports and cancer. Automating this information extraction process from cancer pathology reports has the potential to improve not only the quality of the data by extracting information in a consistent manner across registries, but to improve the quality of patient outcomes by reducing the time to assimilate new data and enabling time-sensitive applications such as precision medicine. Here we present FrESCO: Framework for Exploring Scalable Computational Oncology, a modular deep-learning natural language processing (NLP) library for extracting pathology information from clinical text documents.},
   url = {https://www.osti.gov//servlets/purl/1958817},
   doi = {10.11578/dc.20230227.2},
   url = {https://www.osti.gov/biblio/1958817}, year = {2023},
   month = {3},
   note =
   }

Documentation

.. toctree::
   :maxdepth: 4

   ../modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
