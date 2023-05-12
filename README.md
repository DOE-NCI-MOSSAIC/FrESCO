# FrESCO: Framework for Exploring Scalable Computational Oncology

## Motivation
The National Cancer Institute (NCI) monitors population level cancer trends as part of its Surveillance, Epidemiology, and End Results (SEER) program. This program consists of state or regional level cancer registries which collect, analyze, and annotate cancer pathology reports. From these annotated pathology reports, each individual registry aggregates cancer phenotype information and summary statistics about cancer prevalence to facilitate population level monitoring of cancer incidence. Extracting cancer phenotype from these reports is a labor intensive task, requiring specialized knowledge about the reports and cancer. Automating this information extraction process from cancer pathology reports has the potential to  improve not only the quality of the data by extracting information in a consistent manner across registries, but to improve the quality of patient outcomes by reducing the time to assimilate new data and enabling time-sensitive applications such as precision medicine. Here we present FrESCO: Framework for Exploring Scalable Computational Oncology, a modular deep-learning natural language processing (NLP) library for extracting pathology information from clinical text documents.


## Quickstart Guide

### Initial Setup
Clone the repo
```shell
git clone https://code.ornl.gov/mossiac/fresco
```
Load the working branch and pull in the subrepos
```shell
cd fresco
git checkout main
```
Setup the conda environment (the default name for the environment is "ms39", this can be edited in the ms39.yml file)
```shell
conda env create --file ms39.yml
conda activate ms39
```
Install PyTorch 1.13.1. In Linux CUDA enabled setup, note that your specific cudatoolkit version requirements may vary,
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
otherwise, for CPU only,
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```
Further PyTorch instructions may be found in the [PyTorch docs](https://pytorch.org/docs/stable/index.html).

### Data Preparation
The data used must be generated prior to inference. Add the path to existing data in the the 
`data_path` argument. The required data files are:
- `data_fold0.csv`: Pandas dataframe with columns:
    - `X`: list of input values, of `int` type
    - `task_n`: output for task `n`, a `string` type (these are the y-values)
    - `split`: one of `train`, `test`, or `val`
- `id2labels_fold0.json`: index to label dictionary mapping for each of the string representations of the outputs to an integer value, dict keys must match the y-values label
- `word_embeds_fold0.npy`: word embedding matrix for the vocabulary, dimensions are `words x embedding_dim` 

You will also need to set the `tasks`, these must correspond to the task columns in the `data_fold0.csv` file and keys in the `id2labels_fold0.json` dictionary.
For example, in the P3B3 data, the task columns are `task_n, n = 1,2,3,4`. Whereas the imdb data has the `sentiment` task.
If using any sort of class weighting scheme, the keyword `class_weights` must be either a pickle file or dictionary
with keys corresponding to the task and value with a corresponding list, or numpy array, of weights for that task.  
If the `class_weights` keyword is blank, corresponding to `None`, no class weighting scheme will be used during training nor inference.

If working with hierarchical data, and the case-level context model is the desired output, then the dataframe in `data_fold0.csv` must contain an additional integer-valued column `group` where the values describe the hierarchy present within the data. For example, all rows where `group = 17` are associated for the purpose of training a case-level context model.

### Model Training
The `model_args.yaml` file controls the settings for model training. Edit this file as desired based on your requirements and desired outcome.
The `"save_name"` entry controls the name used for model checkpoints and prediction outputs; if left empty, a datetimestamp will be used.

The following commands allow setting your GPUs, if enabled, before training your model.
```shell
nvidia-smi                             #check GPU utilization
export CUDA_VISIBLE_DEVICES=0,1        #replace 0,1 with the GPUs you want to use
echo $CUDA_VISIBLE_DEVICES             #check which GPUs you have chosen
```
To train the model for any information extraction task, multi-task calssification, simply run
```shell
python train_model.py -m ie
```
We have supplied test data for each of the model types provided. Information extraction models may be created with either `P3B3` or `imdb` data, and the `clc` subfolder of the data directory
containes hierarchical data for case-level context.

If you're wanting a case-level context model, there is a two-step process. 

Step 1: Create an information extraction model specifying the data in the `data/clc` directory in the `model_args.yml` file. Then run
```shell
python train_model.py -m ie 
```
Step 2: Train a clc model, set the `model_path` arg to the model trained in the previous step in the `clc_args.yml` file. Then run
```shell
python train_model.py -m clc
```
to train a case-level context model.

Note that the case level context model requires a pre-trained information extraction model to be specified in the `clc_args` file. 
The default setting, if the `-m ` argument is omitted, is information extraction, and which task is specified in the `model_args` file.

### Deep-Abstaining Classifier and Ntask

Both information extraction and case-level context models have the ability to incorporate abstention or Ntask. The deep abstaining classifier (DAC)
from the [CANDLE](https://github.com/ECP-CANDLE/Candle) repository, allows the model to
not make a prediction on a sample if the softmax score does not meet a predetermined threshold. It can be tuned to meet a threshold of accuracy, minimum number of samples
abstained, or both through adapting the values of alpha through the training process. This adapation is automated in the code, only requires the user to specify the initial
values, tuning mode, and scaling. 

Ntask is useful for multi-task learning on the P3B3 dataset. It creates and additional 'task' that predicts if the softmax scores from all
of the tasks meet a specified threshold. It has its own parameters that are tuned during the training process to obtain a minimum of abstained
samples and maximum accuracy on the predicted samples. Ntask may not be enabled without abstention being enabled as well. The code will
throw an exception and halt if such configuration is passed.

### Expected Results

Training a model with the supplied default args, we see convergence within 50-60 epochs with 0.80-0.85 accuracy on the imdb data set and
in excess of 0.90 accuracy across all tasks for the P3B3 dataset within 60 epochs or so. **NOTE:** P3B3 has a known issue training with mixed
precision enabled. Please ensure `mixed_precision: False` for all runs with the P3B3 dataset.


### Contributing

Get in touch if you would like to help in writing code, example notebooks, and documentation are essential aspects of the project. To contribute please fork the project, make your proposed changes and submit a pull request. We will do our best to sort out any issues and get your contributions merged into the main branch.

If you found a bug, have questions, or are just having trouble with the library, please open an issue in our issue tracker and we'll try to help resolve it.

### How to Cite

If you use our software in your work please cite this repository. The bibtex entry is:
```
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
```
